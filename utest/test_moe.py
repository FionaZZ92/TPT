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
import pathlib
import re
import fnmatch

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
    config_file_path = f"moe_{method}_best_config.json"
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            print(f"[INFO] Using configuration from {config_file_path} for MoE layer.")
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
    os.environ['TRITON_CACHE_DIR'] = "./"

def save_to_file(filename, content):
    try:
        abs_path = os.path.abspath(filename)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[INFO] save ast succeed -> save into file path: {abs_path}")
    except Exception as e:
        print(f"[ERROR] save ast failed -> error {e}")

def save_cache_to_export(method):
    cache_dir = os.getenv('TRITON_CACHE_DIR')
    if cache_dir is None:
        cache_dir = pathlib.Path.home() / ".triton" / "cache"
    export_dir = f"{method}_cache/"
    if cache_dir.exists():
        shutil.copytree(cache_dir, export_dir, dirs_exist_ok=True)
    else:
        print("[ERROR] Not remain ~/.triton/cache!!!")
    #shutil.rmtree(cache_dir)

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

def replace_files_with_pattern(root_path, file_pattern, new_pattern):
    
    # 使用lambda表达式和fnmatch进行模式匹配
    pattern_match = lambda file_name, pat: fnmatch.fnmatch(file_name, pat)
    
    #matched_files = []
    
    for file_path in root_path.rglob('*'):
        if file_path.is_file() and pattern_match(file_path.name, file_pattern):
            suffix = file_path.suffix
            escaped_pattern = re.escape(file_pattern)
            regex_pattern = escaped_pattern.replace(r'\*', '.*') 
            #'(^__grp__.*@e2e_moe_persistent_kernel_top_k_.*?|.*@e2e_moe_persistent_kernel_top_k_.*?)\\.(.+)$'
            pattern_grp = f'(^__grp__{regex_pattern}?|{regex_pattern}?)\.(.+)$'
            match = re.match(pattern_grp, str(file_path.name))
            if match:
                if match.group(1).startswith('__grp__'):
                    new_name = "__grp__" + new_pattern + suffix
                else:
                    new_name = new_pattern + suffix
                new_path = file_path.parent / new_name
                try:
                    file_path.rename(new_path)
                    print(f"[INFO] Rename: {file_path.name} -> {new_name}")
                except Exception as e:
                    print(f"[INFO] Rename failed {file_path.name}: {e}")
            else:
                print(f"[ERROR] replace cache failed.")
        
    
def compile_ir(args):
    try: 
        if args.method == "aiter_persistent":
            file_pattern = "*@e2e_moe_persistent_kernel_top_k_*"
            new_pattern = "e2e_moe_persistent_kernel"
            kwargs = aiter_p_small_moe_config
        elif args.method == "aiter":
            file_pattern = "*@e2e_moe_kernel_top_k_*"
            new_pattern = "e2e_moe_kernel"
            kwargs = aiter_small_moe_config
        else:
            file_pattern = "*@fused_moe_kernel*"
            new_pattern = "fused_moe_kernel"
            kwargs = vllm_small_moe_config
        tuned_config = check_best_config(args.method)
        if tuned_config is not None and str(args.input_token) in tuned_config:
            kwargs = tuned_config[str(args.input_token)]
        target = triton.runtime.driver.active.get_current_target()
        backend = triton.compiler.compiler.make_backend(target)
        options = backend.parse_options(kwargs)
        kernel = triton.compile(str(args.ir_file), target=target, options=options.__dict__)
        print(f"[INFO] kernel.metadata:{kernel.metadata}")
    except Exception as e:
        print(f"[ERROR] compile/run IR failed: {e}")

    
    target_directory = os.getenv('TRITON_CACHE_DIR')
    if target_directory is None:
        target_directory = pathlib.Path.home() / ".triton" / "cache"
    replace_files_with_pattern(target_directory, file_pattern, new_pattern)



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
                tuned_config = check_best_config(method)
                if tuned_config is not None and str(input_token) in tuned_config:
                    config = tuned_config[str(input_token)]
                    print(f"[INFO] Use config of input_token {input_token}")
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
            if usebest:
                config = check_best_config(method)
            if config == None:
                if input_token == 1:
                    config = vllm_small_moe_config
                    print(f"[INFO] Use vllm_small_moe_config:{vllm_small_moe_config}")
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
        print(f"[ERROR] config value error: {ve}")
        return None
    except triton.TritonError as e:
        print(f"[ERROR] triton kernel error: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpect error: {e}")
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
            print("[WARNING] ignore usebest for autotune.")
        if args.cache:
            print("[WARNING] ignore cache for autotune.")
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
        token_config = {
            f"{args.input_token}": config_time_data[0].get("name")
        }
        with open(f"moe_{args.method}_best_config.json", 'w') as f:
            json.dump(token_config, f, indent=2)
    elif args.mode == "benchmark":
        if args.cache:
            print("[WARNING] ignore cache for autotune")
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
    elif args.mode == "run":
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
    else: #run_ir
        if os.path.exists(args.ir_file) and os.path.isfile(args.ir_file):
            compile_ir(args)
            print("[INFO]Load and compile works.")
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
        else:
            print("[ERROR] Cannot find ir_file.")

            



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
        "--mode", type=str, choices=["run", "run_ir", "benchmark", "autotune"], default="run"
    )
    parser.add_argument("--usebest", action="store_true", default=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--ir_file", type=str, default="")
    args = parser.parse_args()
    main(args)
