"""Extract unchanged native accumulation/selection kernels for row-band replay."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
s=(a.source/'src/cuda/prefill_kernels.cu').read_text()
expand=s[s.index('extern "C" __global__ void dsa_prefill_kpool_expand_kernel('):s.index('// Compute the exact owner score matrix')]
start=s.index('extern "C" __global__ void dsa_prefill_accumulate_gemm_scores_kernel(')
end=s.index('\n/*\n * Pack head-major',start)
(a.out/'score_kernels.cuh').write_text('#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cub/block/block_radix_sort.cuh>\n#include <cstdint>\n#include <climits>\n__device__ inline float bf16_to_float(__nv_bfloat16 x){return __bfloat162float(x);}\n'+expand+s[start:end])
