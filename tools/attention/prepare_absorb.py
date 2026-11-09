"""Extract the unmodified native latent-absorption kernel for timing controls."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
s=(a.source/'src/cuda/prefill_kernels.cu').read_text();start=s.index('extern "C" __global__ void mla_prefill_absorb_wkc_kernel(');end=s.index('/*',start)
x='#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x){return __bfloat162float(x);}\n'+s[start:end]
(a.out/'absorb-baseline.cuh').write_text(x)
