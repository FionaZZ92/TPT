import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from kernels.moe import *
import argparse
import torch
import time
import aiter
import triton
import triton.language as tl
import triton.testing as testing
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    _get_config_dtype_str,
)

#<=256
aiter_small_moe_config = {
    'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 64,
    "BLOCK_SIZE_K1": 64,
    "BLOCK_SIZE_K2": 64,
    'GROUP_SIZE_M': 4,
    'num_warps': 8,
    'num_stages': 2,
    'waves_per_eu': 0,
    'matrix_instr_nonkdim': 16,
    'kpack': 2,
}
#>=1024
aiter_large_moe_config = {
    'BLOCK_SIZE_M': 256,
    'BLOCK_SIZE_N': 256,
    "BLOCK_SIZE_K1": 64,
    "BLOCK_SIZE_K2": 64,
    'GROUP_SIZE_M': 8,
    'num_warps': 8,
    'num_stages': 2,
    'waves_per_eu': 0,
    'matrix_instr_nonkdim': 16,
    'kpack': 1,
}
vllm_small_moe_config = {
    'BLOCK_SIZE_M': 16,
    'BLOCK_SIZE_N': 32,
    'BLOCK_SIZE_K': 64,
    'GROUP_SIZE_M': 1,
    'num_warps': 2,
    'num_stages': 2,
    'waves_per_eu': 2,
    'matrix_instr_nonkdim': 16,
    'kpack': 1,
}


bench_config = [testing.Benchmark(
    x_names=['input_token'],
    x_vals= [1,1024,8000],
    line_arg='method',
    line_vals=['aiter_persistant','aiter', 'vllm'],
    line_names=['Aiter_p','Aiter', "VLLM"],
    styles=[('red', '-'), ('blue', '-'), ('green', '-')],
    ylabel='ms',
    plot_name='moe-benchmark',
    args={"inter_dim": 96, "hidden_size": 2048, "experts": 128, "topk": 8, "dtype": torch.bfloat16}
)]

#@testing.perf_report(bench_config)
def benchmark(input_token, method, inter_dim, hidden_size, experts, topk, dtype):
    x = torch.randn((input_token, hidden_size), dtype=dtype, device="cuda") #[M,K]
    w1 = torch.randn(
            (experts, inter_dim, hidden_size), dtype=dtype, device="cuda"
        ) #Qwen3-Omni-30B-A3B-Instruct w1[E,N/tp,K]
    w2 = torch.randn((experts, hidden_size, inter_dim), dtype=dtype, device="cuda") #w2[128,2048,768]
    intermediate = torch.zeros((input_token * topk, inter_dim), dtype=torch.float32, device="cuda") #[M*topk, N]
    gating_output = torch.randn((input_token, experts), device="cuda", dtype=dtype)
    triton_out = torch.zeros((input_token, topk, hidden_size), dtype=dtype, device="cuda")
    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    quant_dtype = None
    if dtype == torch.float8_e4m3fn and "vllm" in method:
        w1_scale = torch.randn(experts, device='cuda', dtype=torch.float32) #it is for PTPC, not block
        w2_scale = torch.randn(experts, device='cuda', dtype=torch.float32)
        a1_scale = torch.randn(1, device='cuda', dtype=torch.float32)
        a2_scale = torch.randn(1, device='cuda', dtype=torch.float32)
        w1 = w1.to(torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fn)
        quant_dtype = torch.float8_e4m3fn
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=quant_dtype,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=None,
    )
    #topk_weights, topk_ids, token_expert_indices = fused_topk(x, gating_output, topk, renormalize=True) #renorm false when use deepgemm
    softmax_vals = torch.softmax(gating_output, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=topk, dim=1)
    
    quantiles = [0.5, 0.2, 0.8]

    if "aiter" in method:
        if input_token >=1024:
            aiter_config = aiter_large_moe_config
            config_m_block = aiter_large_moe_config["BLOCK_SIZE_M"]
        else:
            aiter_config = aiter_small_moe_config
            config_m_block = aiter_small_moe_config["BLOCK_SIZE_M"]
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            config_m_block, #config["BLOCK_SIZE_M"],
            experts,
            None, #expert_map,
        )
        from aiter.ops.triton.moe_op_e2e import e2e_moe as triton_e2e_moe
        if method == "aiter_persistant":
            ms, min_ms, max_ms = testing.do_bench(lambda: triton_e2e_moe(
                    x,
                    w1,
                    w2,
                    intermediate, #intermediate [M*topk,N] if persistant, else None
                    triton_out,
                    a1_scale,
                    w1_scale,
                    w2_scale,
                    topk_weights,
                    sorted_token_ids,
                    topk_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    False, #if true: c *= topk_weights.unsqueeze(-1), if false & gelu: c = 0.5 * c * (1.0 + torch.tanh(0.7978845608 * (c + 0.044715 * c * c * c)))
                    topk,
                    False, #fp8_w8a8
                    False, #int8_w8a16
                    aiter_config,
                ),
                quantiles=quantiles,
                warmup=2, 
                rep=10,
            )
        else:
            ms, min_ms, max_ms = testing.do_bench(lambda: triton_e2e_moe(
                    x,
                    w1,
                    w2,
                    None, #intermediate [M*topk,N] if persistant, else None
                    triton_out,
                    a1_scale,
                    w1_scale,
                    w2_scale,
                    topk_weights,
                    sorted_token_ids,
                    topk_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    False, #if true: c *= topk_weights.unsqueeze(-1), if false & gelu: c = 0.5 * c * (1.0 + torch.tanh(0.7978845608 * (c + 0.044715 * c * c * c)))
                    topk,
                    False, #fp8_w8a8
                    False, #int8_w8a16
                    aiter_config,
                ),
                quantiles=quantiles,
                warmup=2, 
                rep=10,
            )
    else:
        #vllm moe
        ms, min_ms, max_ms = testing.do_bench(lambda: fused_experts(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                inplace=True,
                quant_config=None,
                allow_deep_gemm=False,
            ),
        quantiles=quantiles,
        warmup=2, 
        rep=10,
        )
    #end = time.time()
    #duration = (end-start)*10e6
    gemm1_flops = 2 * input_token * inter_dim * hidden_size * topk
    gbps = lambda ms: (2 * gemm1_flops)  / ms / 1e9
    #return gbps(ms), gbps(max_ms), gbps(min_ms)
    return ms, max_ms, min_ms

def benchmark_runner(config, test_func):
    @testing.perf_report(config)
    def benchmark_wrapper(*args, **kwargs):
        return test_func(*args, **kwargs)
    return benchmark_wrapper

def main(args:argparse.ArgumentParser):
    if args.benchmark:
        bench_func=benchmark_runner(bench_config, benchmark)
        bench_func.run(show_plots=True, print_data=True)
    else:
        run_config = [testing.Benchmark(
            x_names=['input_token'],
            x_vals= [args.input_token],
            line_arg='method',
            line_vals=[args.method],
            line_names=[args.method],
            styles=[('red', '-')],
            ylabel='ms',
            plot_name='moe-test',
            args={"inter_dim": args.inter_dim, "hidden_size": args.hidden_size, "experts": args.experts, "topk": args.topk, "dtype": torch.bfloat16}
        )]
        run_func = benchmark_runner(run_config, benchmark)
        run_func.run(show_plots=True, print_data=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,description="select test",)
    parser.add_argument("--input_token", "-i", type=int, default="1")
    parser.add_argument("--inter_dim", "-n", type=int, default="96")
    parser.add_argument("--hidden_size", "-k", type=int, default="2048")
    parser.add_argument(
        "--dtype", type=str, choices=["bfloat16", "float8_e4m3fn"], default="bfloat16"
    )
    parser.add_argument("--experts", "-E", type=int, default="128")
    parser.add_argument("--topk", type=int, default="8")
    parser.add_argument(
        "--method", type=str, choices=["aiter_persistant", "aiter", "vllm"], default="vllm"
    )
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    main(args)
