# TPT
Triton kernel profile and debug tool among vllm &amp; aiter for internal usage

Use any image contains torch, aiter and vllm, for example:
```bash
docker pull rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210
```

## Usage:
```bash
python3 utest/test_moe.py --help
[aiter] import [module_aiter_enum] under /opt/aiter/aiter/jit/module_aiter_enum.so
INFO 12-30 10:38:08 [__init__.py:225] Automatically detected platform rocm.
[aiter] merge tuned file under model_configs/ and configs/ /opt/aiter/aiter/configs/bf16_tuned_gemm.csv:/opt/aiter/aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv
INFO:aiter:merge tuned file under model_configs/ and configs/ /opt/aiter/aiter/configs/bf16_tuned_gemm.csv:/opt/aiter/aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv
usage: test_moe.py [-h] [--input_token INPUT_TOKEN] [--inter_dim INTER_DIM] [--hidden_size HIDDEN_SIZE] [--dtype {bfloat16,float8_e4m3fn}] [--experts EXPERTS] [--topk TOPK] [--method {aiter_persistent,aiter,vllm}]
                   [--mode {run,benchmark,autotune}] [--usebest] [--cache]

select test

options:
  -h, --help            show this help message and exit
  --input_token INPUT_TOKEN, -i INPUT_TOKEN
  --inter_dim INTER_DIM, -n INTER_DIM
  --hidden_size HIDDEN_SIZE, -k HIDDEN_SIZE
  --dtype {bfloat16,float8_e4m3fn}
  --experts EXPERTS, -E EXPERTS
  --topk TOPK
  --method {aiter_persistent,aiter,vllm}
  --mode {run,benchmark,autotune}
  --usebest
  --cache
```
## benchmark_example:
```bash
root@hjbog-srdc-52:/app/users/fizhao/triton_lib/utest# python3 test_moe.py --method aiter --cache --mode run
[aiter] import [module_aiter_enum] under /opt/aiter/aiter/jit/module_aiter_enum.so
INFO 01-13 05:19:28 [__init__.py:225] Automatically detected platform rocm.
[aiter] merge tuned file under model_configs/ and configs/ /opt/aiter/aiter/configs/bf16_tuned_gemm.csv:/opt/aiter/aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv
INFO:aiter:merge tuned file under model_configs/ and configs/ /opt/aiter/aiter/configs/bf16_tuned_gemm.csv:/opt/aiter/aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv
[save ast succeed] -> save into file path: /app/users/fizhao/triton_lib/utest/moe_aiter.ast.txt
=====Using configuration from moe_aiter_best_config.json for MoE layer.
moe-test:
   input_token    aiter
0          1.0  0.08568


root@hjbog-srdc-52:/app/users/fizhao/triton_lib/utest# python3 test_moe.py --mode benchmark
[aiter] import [module_aiter_enum] under /opt/aiter/aiter/jit/module_aiter_enum.so
INFO 01-13 05:21:56 [__init__.py:225] Automatically detected platform rocm.
[aiter] merge tuned file under model_configs/ and configs/ /opt/aiter/aiter/configs/bf16_tuned_gemm.csv:/opt/aiter/aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv
INFO:aiter:merge tuned file under model_configs/ and configs/ /opt/aiter/aiter/configs/bf16_tuned_gemm.csv:/opt/aiter/aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv
=====Use vllm_small_moe_config:{'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 2, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1}
Using default MoE config. Performance might be sub-optimal! Config file not found at %s /opt/vllm/vllm/model_executor/layers/fused_moe/configs/E=128,N=96,device_name=AMD_Instinct_MI308X.json, /app/users/fizhao/triton_lib/kernels/configs/E=128,N=96,device_name=AMD_Instinct_MI308X.json
moe-benchmark:
   input_token   Aiter_p     Aiter      VLLM
0          1.0  0.114581  0.085240  0.234800
1       1024.0  0.845562  0.765422  0.399201
2       2048.0  1.405484  1.358244  0.687122
3       4096.0  2.620887  2.241407  1.286483
4       8192.0  5.007975  3.500651  2.507687
```

## IR pass optimization tool:
While you run utest with `--cache` option, you will get trtion cache with selected method kernel. Put the ttir/ttgir/mlir as input for triton compiler pass manager optimization, then you will get new IR file. Compare differences between original and new IR to find out regular compiler optimization functionalities. For example:
```bash
root@hjbog-srdc-52:/app/users/fizhao/triton_lib/common# python3 compiler_pass.py -s ../utest/aiter_persistent_cache/4XYLNMBEKC235AB34XNG7NAXVVQBMAAWUZ4TGDYM47KLA5YGGQZQ/e2e_moe_persistent_kernel.ttir
===Applying Pass...
===Executing Pass optimization...
[DONE]Generate TTIR successfully.
[DONE] TTIR saved in: new.ttir
root@hjbog-srdc-52:/app/users/fizhao/triton_lib/common# diff ../utest/aiter_persistent_cache/4XYLNMBEKC235AB34XNG7NAXVVQBMAAWUZ4TGDYM47KLA5YGGQZQ/e2e_moe_persistent_kernel.ttir new.ttir
```

You can use this ttir to compile with the config which used in "utest/test_moe.py". While specifying `--cache`, you will get compiled IR and binary file with new hash. 
```bash
root@hjbog-srdc-52:/app/users/fizhao/triton_lib/utest# python3 test_moe.py --mode run_ir --method aiter_persistent --cache --ir_file ../common/new.ttir 
[aiter] import [module_aiter_enum] under /opt/aiter/aiter/jit/module_aiter_enum.so
INFO 01-19 08:16:57 [__init__.py:225] Automatically detected platform rocm.
[aiter] merge tuned file under model_configs/ and configs/ /opt/aiter/aiter/configs/bf16_tuned_gemm.csv:/opt/aiter/aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv
INFO:aiter:merge tuned file under model_configs/ and configs/ /opt/aiter/aiter/configs/bf16_tuned_gemm.csv:/opt/aiter/aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv
=====Using configuration from moe_aiter_persistent_best_config.json for MoE layer.
[INFO]kernel.metadata:KernelMetadata(allow_flush_denorm=False, allowed_dot_input_precisions=['ieee', 'tf32'], arch='gfx942', backend_name='hip', cluster_dims=(1, 1, 1), debug=False, default_dot_input_precision='ieee', deprecated_fp8_dot_operand_dtypes=[], enable_fp_fusion=True, extern_libs=[['ocml', '/usr/local/lib/python3.10/dist-packages/triton/backends/amd/lib/ocml.bc'], ['ockl', '/usr/local/lib/python3.10/dist-packages/triton/backends/amd/lib/ockl.bc']], hash='e172b35c4fc4b97f1a63d45144de2ec8fb8a6a38687ef2b665a2d2e29545d8b6', kpack=2, launch_cooperative_grid=False, matrix_instr_nonkdim=16, max_num_imprecise_acc_default=0, name='e2e_moe_persistent_kernel_top_k_8_N_96_K_2048_EVEN_K_1_EVEN_N_0_MUL_ROUTED_WEIGHT_0_use_fp8_w8a8_0_use_int8_w8a16_0_BLOCK_SIZE_M_16_BLOCK_SIZE_N1_128_BLOCK_SIZE_N2_64_BLOCK_SIZE_K1_64_BLOCK_SIZE_K2_64_NUM_SMS_160', num_ctas=1, num_stages=2, num_warps=4, sanitize_overflow=True, schedule_hint='none', shared=18432, supported_fp8_dtypes=['fp8e4b8', 'fp8e4nv', 'fp8e5', 'fp8e5b16'], target=GPUTarget(backend='hip', arch='gfx942', warp_size=64), triton_version='3.4.0', warp_size=64, waves_per_eu=1)
Rename: @e2e_moe_persistent_kernel_top_k_8_N_96_K_2048_EVEN_K_1_EVEN_N_0_MUL_ROUTED_WEIGHT_0_use_fp8_w8a8_0_use_int8_w8a16_0_BLOCK_SIZE_M_16_BLOCK_SIZE_N1_128.json -> e2e_moe_persistent_kernel.json
Rename: @e2e_moe_persistent_kernel_top_k_8_N_96_K_2048_EVEN_K_1_EVEN_N_0_MUL_ROUTED_WEIGHT_0_use_fp8_w8a8_0_use_int8_w8a16_0_BLOCK_SIZE_M_16_BLOCK_SIZE_N1_128.amdgcn -> e2e_moe_persistent_kernel.amdgcn
Rename: @e2e_moe_persistent_kernel_top_k_8_N_96_K_2048_EVEN_K_1_EVEN_N_0_MUL_ROUTED_WEIGHT_0_use_fp8_w8a8_0_use_int8_w8a16_0_BLOCK_SIZE_M_16_BLOCK_SIZE_N1_128.hsaco -> e2e_moe_persistent_kernel.hsaco
Rename: @e2e_moe_persistent_kernel_top_k_8_N_96_K_2048_EVEN_K_1_EVEN_N_0_MUL_ROUTED_WEIGHT_0_use_fp8_w8a8_0_use_int8_w8a16_0_BLOCK_SIZE_M_16_BLOCK_SIZE_N1_128.llir -> e2e_moe_persistent_kernel.llir
Rename: __grp__@e2e_moe_persistent_kernel_top_k_8_N_96_K_2048_EVEN_K_1_EVEN_N_0_MUL_ROUTED_WEIGHT_0_use_fp8_w8a8_0_use_int8_w8a16_0_BLOCK_SIZE_M_16_BLOCK_SIZE_N1_128.json -> __grp__e2e_moe_persistent_kernel.json
Rename: @e2e_moe_persistent_kernel_top_k_8_N_96_K_2048_EVEN_K_1_EVEN_N_0_MUL_ROUTED_WEIGHT_0_use_fp8_w8a8_0_use_int8_w8a16_0_BLOCK_SIZE_M_16_BLOCK_SIZE_N1_128.ttir -> e2e_moe_persistent_kernel.ttir
Rename: @e2e_moe_persistent_kernel_top_k_8_N_96_K_2048_EVEN_K_1_EVEN_N_0_MUL_ROUTED_WEIGHT_0_use_fp8_w8a8_0_use_int8_w8a16_0_BLOCK_SIZE_M_16_BLOCK_SIZE_N1_128.ttgir -> e2e_moe_persistent_kernel.ttgir
[INFO]Load and compile works.
=====Using configuration from moe_aiter_persistent_best_config.json for MoE layer.
Use config of input_tokn 1
/usr/local/lib/python3.10/dist-packages/triton/testing.py:370: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
moe-test:
   input_token  aiter_persistent
0          1.0          0.078201
```
Replace the `.hsaco` file in original hash path, the cache saved in `~/.triton/cache` by default. Then, run it again.

```bash
root@hjbog-srdc-52:/app/users/fizhao/triton_lib/utest# cp /app/users/fizhao/triton_lib/utest/aiter_persistent_cache/4FZLGXCPYS4X6GTD2RIUJXROZD5YU2RYNB7PFNTFULJOFFKF3C3A/e2e_moe_persistent_kernel.hsaco ~/.triton/cache/4XYLNMBEKC235AB34XNG7NAXVVQBMAAWUZ4TGDYM47KLA5YGGQZQ/e2e_moe_persistent_kernel.hsaco
root@hjbog-srdc-52:/app/users/fizhao/triton_lib/utest# python3 test_moe.py --mode run --method aiter_persistent
```


## auto tune example

```bash
python3 test_moe.py --method aiter_persistent --mode autotune --input_token 1024
```

After tune the `aiter_persistent`

```bash
moe-benchmark:
   input_token   Aiter_p     Aiter      VLLM
0          1.0  0.074580  0.084640  0.232161
1       1024.0  0.661602  0.782982  0.410642
2       2048.0  1.282384  1.373924  0.691702
3       4096.0  2.429887  2.212227  1.283644
4       8192.0  4.307934  3.515332  2.501889
```
