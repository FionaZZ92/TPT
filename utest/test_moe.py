import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from kernels.moe import *
import argparse
import torch
import traceback
import json
from itertools import product
import aiter
import triton
import triton.language as tl
import triton.testing as testing
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    _get_config_dtype_str,
)
import inspect
import ast
import shutil
from pathlib import Path

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
aiter_p_small_moe_config = {
    'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N1': 128,
    'BLOCK_SIZE_N2': 64,
    "BLOCK_SIZE_K1": 64,
    "BLOCK_SIZE_K2": 64,
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

tune_config = []

def hip_autotune_config(dtype, method):
    MN_size = [16,32,64,128,256]
    K_size = [32,64,128]
    group_size = [1,2,4,6,8]
    waves_size = [1,2,4]
    num_stages = [2,3,4,5]
    '''MN_size = [16,32,64,128,256]
    K_size = [64]
    group_size = [4]
    waves_size = [8]
    num_stages = [2]'''
    matrix_instr_nonkdim_range = [16] if dtype == torch.bfloat16 else []
    kpack_range = [1, 2] if dtype == torch.bfloat16 else []

    if "aiter_persistent" in method:
        tune_dict = {
            "BLOCK_SIZE_M": MN_size,
            "BLOCK_SIZE_N1": MN_size,
            "BLOCK_SIZE_N2": MN_size,
            "BLOCK_SIZE_K1": K_size,
            "BLOCK_SIZE_K2": K_size,
            "num_warps": waves_size,
            "num_stages": num_stages,
            "matrix_instr_nonkdim": matrix_instr_nonkdim_range,
            "kpack": kpack_range,
        }
    elif "aiter" in method:
        tune_dict = {
            "BLOCK_SIZE_M": MN_size,
            "BLOCK_SIZE_N": MN_size,
            "BLOCK_SIZE_K1": K_size,
            "BLOCK_SIZE_K2": K_size,
            "GROUP_SIZE_M": group_size, #Group size for L2 cache optimization.
            "num_warps": waves_size,
            "num_stages": num_stages,
            "matrix_instr_nonkdim": matrix_instr_nonkdim_range,
            "kpack": kpack_range,
        }
    else:
        tune_dict = {
            "BLOCK_SIZE_M": MN_size,
            "BLOCK_SIZE_N": MN_size,
            "BLOCK_SIZE_K": K_size,
            "GROUP_SIZE_M": group_size, #Group size for L2 cache optimization.
            "num_warps": waves_size,
            "num_stages": num_stages,
            "matrix_instr_nonkdim": matrix_instr_nonkdim_range,
            "kpack": kpack_range,
        }

    tune_config = []

    keys, values = zip(*tune_dict.items())
    for config_values in product(*values):
        config = dict(zip(keys, config_values))
        tune_config.append(config)

    #return [triton.Config(s) for s in tune_config]
    return tune_config

def check_best_config(method):
    config_file_path = f"moe_{args.method}_best_config.json"
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            print(f"=====Using configuration from {config_file_path} for MoE layer.")
            # If a configuration has been found, return it
            best_config = json.load(f)
            return best_config
    else:
        return None

def set_triton_env_vars(output_dir):
    """设置Triton环境变量以启用中间文件导出"""
    # 设置TTIR导出环境变量
    os.environ['TRITON_DUMP_KERNEL'] = '1'
    os.environ['TRITON_DUMP_PATH'] = str(output_dir) #old version
    os.environ['TRITON_COMPILER_DUMP_PATH'] = str(output_dir)
    os.environ['TRITON_ALWAYS_COMPILE'] = '1'
    os.environ['TRITON_DEBUG'] = '1'

def save_to_file(filename, content):
    try:
        abs_path = os.path.abspath(filename)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[save ast succeed] -> save into file path: {abs_path}")
    except Exception as e:
        print(f"[save ast failed] -> error {e}")

def save_cache_to_export(method):
    cache_dir = Path.home() / ".triton" / "cache"
    export_dir = f"{method}_cache/"
    #os.makedirs(export_dir, exist_ok=True)
    if cache_dir.exists():
        shutil.copytree(cache_dir, export_dir, dirs_exist_ok=True)
    else:
        print("=====Not remain ~/.triton/cache!!!=====")
    shutil.rmtree(cache_dir)

def save_ast(method):
    if "aiter" in method:
        from aiter.ops.triton.moe_op_e2e import e2e_moe as triton_e2e_moe
        kernel_source = inspect.getsource(triton_e2e_moe)
    else:
        kernel_source = inspect.getsource(fused_experts)
    kernel_ast = ast.parse(kernel_source)
    if sys.version_info >=(3, 9):
        ast_dump_str = ast.dump(kernel_ast, indent=4)
    else:
        ast_dump_str = ast.dump(kernel_ast)
    ast_filename = f"moe_{method}.ast.txt"
    save_to_file(ast_filename, ast_dump_str)
    



#@testing.perf_report(bench_config)
def runner_test(input_token, method, inter_dim, hidden_size, experts, topk, dtype, config, usebest):
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

    try:
        from aiter.ops.triton.moe_op_e2e import e2e_moe as triton_e2e_moe
        from aiter.ops.triton.moe_op_e2e import moe_set_use_persistent_kernel as triton_e2e_moe_set_use_persistent_kernel
        if method == "aiter_persistent":
            if usebest:
                config = check_best_config(method)
            if config == None: #non autotune mode
                config = aiter_p_small_moe_config
            config_m_block = config['BLOCK_SIZE_M']
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids,
                config_m_block, #config["BLOCK_SIZE_M"],
                experts,
                None, #expert_map,
            )
            triton_e2e_moe_set_use_persistent_kernel(True)
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
                    config,
                ),
                quantiles=quantiles,
                warmup=3, 
                rep=10,
            )
        elif method == "aiter":
            if usebest:
                config = check_best_config(method)
            if config == None: #non autotune mode
                if input_token >=1024:
                    config = aiter_large_moe_config
                else:
                    config = aiter_small_moe_config
            config_m_block = config['BLOCK_SIZE_M']
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids,
                config_m_block, #config["BLOCK_SIZE_M"],
                experts,
                None, #expert_map,
            )
            triton_e2e_moe_set_use_persistent_kernel(False)
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
                    config,
                ),
                quantiles=quantiles,
                warmup=3, 
                rep=10,
            )
        else:
            #if usebest:
            #    config = check_best_config(method)
            if config == None:
                if input_token == 1:
                    config = vllm_small_moe_config
                    print(f"=====Use vllm_small_moe_config:{vllm_small_moe_config}")
            ms, min_ms, max_ms = testing.do_bench(lambda: fused_experts(
                    x,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    inplace=True,
                    quant_config=None,
                    allow_deep_gemm=False,
                    config=config,
                ),
                quantiles=quantiles,
                warmup=3, 
                rep=10,
            )
    except ValueError as ve:
        print(f"config value error: {ve}")
        return None
    except triton.TritonError as e:
        print(f"triton kernel error: {e}")
        return None
    except Exception as e:
        print(f"Unexpect error: {e}")
        traceback.print_exc()
        return None
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
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float8_e4m3fn': torch.float8_e4m3fn,
    }
    if args.mode == "autotune":
        if args.usebest:
            print("=====ignore usebest for autotune=====")
        if args.cache:
            print("=====ignore cache for autotune=====")
        tune_config = hip_autotune_config(dtype_map[args.dtype], args.method)
        config_time_data = []
        for i in tune_config:
            latency = runner_test(args.input_token, 
                args.method, 
                args.inter_dim,
                args.hidden_size,
                args.experts,
                args.topk,
                dtype_map[args.dtype],
                i,
                False,
            )
            if latency != None:
                tmp = {
                    "name": i,
                    "latency": latency[0],
                }
                config_time_data.append(tmp)
        config_time_data = sorted(config_time_data, key=lambda x: x.get("latency"), reverse=False)
        #print(f"Available config:{config_time_data}")
        with open(f"moe_{args.method}_best_config.json", 'w') as f:
            json.dump(config_time_data[0].get("name"), f, indent=2)
        

    elif args.mode == "benchmark":
        if args.cache:
            print("=====ignore cache for autotune=====")
        config = None
        bench_config = [testing.Benchmark(
            x_names=['input_token'],
            x_vals= [1,1024,2048,4096,8192],
            line_arg='method',
            line_vals=['aiter_persistent','aiter', 'vllm'],
            line_names=['Aiter_p','Aiter', "VLLM"],
            styles=[('red', '-'), ('blue', '-'), ('green', '-')],
            ylabel='ms',
            plot_name='moe-benchmark',
            args={"inter_dim": args.inter_dim, "hidden_size": args.hidden_size, 
                "experts": args.experts, "topk": args.topk, 
                "dtype": dtype_map[args.dtype], "config": config, "usebest": args.usebest}
        )]
        bench_func=benchmark_runner(bench_config, runner_test)
        bench_func.run(show_plots=True, print_data=True)
    else:
        if args.cache:
            save_ast(args.method)
        config = None
        run_config = [testing.Benchmark(
            x_names=['input_token'],
            x_vals= [args.input_token],
            line_arg='method',
            line_vals=[args.method],
            line_names=[args.method],
            styles=[('red', '-')],
            ylabel='ms',
            plot_name='moe-test',
            args={"inter_dim": args.inter_dim, "hidden_size": args.hidden_size, 
                "experts": args.experts, "topk": args.topk, 
                "dtype": dtype_map[args.dtype], "config": config, "usebest": args.usebest}
        )]
        run_func = benchmark_runner(run_config, runner_test)
        run_func.run(show_plots=True, print_data=True)
        if args.cache:
            save_cache_to_export(args.method)



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
        "--method", type=str, choices=["aiter_persistent", "aiter", "vllm"], default="vllm"
    )
    #group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument(
        "--mode", type=str, choices=["run", "benchmark", "autotune"], default="run"
    )
    parser.add_argument("--usebest", action="store_true", default=True)
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()
    main(args)
