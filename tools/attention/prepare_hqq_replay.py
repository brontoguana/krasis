"""Preserve GEMV arithmetic while testing cooperative packed-word loads."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);p.add_argument('--aligned',action='store_true');a=p.parse_args();a.out.mkdir()
s=(a.source/'src/cuda/decode_kernels.cu').read_text();s=s[s.index('extern "C" __global__ void hqq6_decode_gemv_bf16_vec4('):s.index('extern "C" __global__ void hqq8_decode_gemv_f32(')]
new=s.replace('hqq6_decode_gemv_bf16_vec4(', 'hqq6_decode_gemv_bf16_vec4_warp_load(',1)
old='''        const unsigned char* tri = w_row + quartet * 3;
        unsigned int bits = ((unsigned int)tri[0])
            | (((unsigned int)tri[1]) << 8)
            | (((unsigned int)tri[2]) << 16);'''
rep='''        const unsigned int* words = reinterpret_cast<const unsigned int*>(w_row + (quartet - lane) * 3);
        const unsigned int word = lane < 24 ? words[lane] : 0u;
        const int bit = lane * 24;
        const unsigned int low = __shfl_sync(0xffffffffu, word, bit >> 5);
        const unsigned int high = __shfl_sync(0xffffffffu, word, min((bit >> 5) + 1, 23));
        const unsigned int bits = __funnelshift_r(low, high, bit & 31);'''
assert new.count(old)==1;new=new.replace(old,rep,1)
if a.aligned:
 aligned='''        const unsigned int* words = reinterpret_cast<const unsigned int*>(w_row);
        const int byte = quartet * 3;
        const int shift = (byte & 3) * 8;
        const unsigned int low = words[byte >> 2];
        // The final quartet starts at byte offset 1 modulo 4 and needs no
        // second load. Never read beyond the registered packed row.
        const unsigned int high = shift >= 16 ? words[(byte >> 2) + 1] : 0u;
        const unsigned int bits = __funnelshift_r(low, high, shift);'''
 new=new.replace(rep,aligned,1)
(a.out/'hqq-load.cuh').write_text('#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n'+s+new)
