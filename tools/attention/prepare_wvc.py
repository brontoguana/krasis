"""Extract the actual native value projection and its pre-rounding equation."""
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir()
s = (a.source / 'src/cuda/decode_kernels.cu').read_text()
start = s.index('extern "C" __global__ void mla_apply_wvc(')
end = s.index('\n}', start) + 2
s = s[start:end]
f = s.replace('mla_apply_wvc(', 'mla_apply_wvc_before_rounding(', 1)
f = f.replace('__nv_bfloat16* __restrict__ output', 'float* __restrict__ output', 1)
f = f.replace('__nv_bfloat16* out_h', 'float* out_h', 1)
f = f.replace('f32_to_bf16(acc)', 'acc', 1)
prefix = '''#include <cuda_runtime.h>
#include <cuda_bf16.h>
__device__ inline float bf16_to_f32(__nv_bfloat16 x){return __bfloat162float(x);}
__device__ inline __nv_bfloat16 f32_to_bf16(float x){return __float2bfloat16_rn(x);}
'''
(a.output / 'wvc-native.cuh').write_text(prefix + s + '\n' + f + '\n')
