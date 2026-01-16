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
root@hjbog-srdc-52:/app/users/fizhao/triton_lib/common# python3 compiler_pass.py -s ../utest/aiter_persistent_cache/CPX2JAPODDRSV2O46PZMO2VDHFXUHPLXWIENX3EK2C7O33E7OMUA/e2e_moe_persistent_kernel.ttir
===Applying Pass...
===Executing Pass optimization...
[DONE]Generate TTIR successfully.
[DONE] TTIR saved in: new.ttir
root@hjbog-srdc-52:/app/users/fizhao/triton_lib/common# diff ../utest/aiter_persistent_cache/CPX2JAPODDRSV2O46PZMO2VDHFXUHPLXWIENX3EK2C7O33E7OMUA/e2e_moe_persistent_kernel.ttir new.ttir
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