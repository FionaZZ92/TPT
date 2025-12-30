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
usage: test_moe.py [-h] [--input_token INPUT_TOKEN] [--inter_dim INTER_DIM] [--hidden_size HIDDEN_SIZE] [--dtype {bfloat16,float8_e4m3fn}] [--experts EXPERTS] [--topk TOPK] [--method {aiter_persistant,aiter,vllm}]
                   [--benchmark]

select test

options:
  -h, --help            show this help message and exit
  --input_token INPUT_TOKEN, -i INPUT_TOKEN
  --inter_dim INTER_DIM, -n INTER_DIM
  --hidden_size HIDDEN_SIZE, -k HIDDEN_SIZE
  --dtype {bfloat16,float8_e4m3fn}
  --experts EXPERTS, -E EXPERTS
  --topk TOPK
  --method {aiter_persistant,aiter,vllm}
  --benchmark
```
## benchmark_example:
```bash
python3 utest/test_moe.py --benchmark
[aiter] import [module_aiter_enum] under /opt/aiter/aiter/jit/module_aiter_enum.so
INFO 12-30 10:42:26 [__init__.py:225] Automatically detected platform rocm.
[aiter] merge tuned file under model_configs/ and configs/ /opt/aiter/aiter/configs/bf16_tuned_gemm.csv:/opt/aiter/aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv
INFO:aiter:merge tuned file under model_configs/ and configs/ /opt/aiter/aiter/configs/bf16_tuned_gemm.csv:/opt/aiter/aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv
Using default MoE config. Performance might be sub-optimal! Config file not found at %s /opt/vllm/vllm/model_executor/layers/fused_moe/configs/E=128,N=96,device_name=AMD_Instinct_MI308X.json, /app/users/fizhao/TPT/kernels/configs/E=128,N=96,device_name=AMD_Instinct_MI308X.json
moe-benchmark:
   input_token   Aiter_p     Aiter      VLLM
0          1.0  0.083240  0.084640  0.235281
1       1024.0  0.763662  0.773603  0.415322
2       8000.0  3.512892  3.515133  2.475088
```
