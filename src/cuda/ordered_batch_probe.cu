#include <cuda_bf16.h>
// Private diagnostic: original and token-indexed unrounded arithmetic.
extern "C" __global__ void hqq6_decode_gemv_bf16_vec4(
    const unsigned char* __restrict__ packed_w,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    const unsigned short* __restrict__ input,
    unsigned short* __restrict__ output,
    int rows,
    int cols,
    int group_size,
    int packed_row_stride_bytes,
    int scales_row_stride_bytes,
    int zeros_row_stride_bytes
) {
    extern __shared__ unsigned short s_input[];
    int tid = threadIdx.x;

    for (int i = tid; i < cols; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;
    if (row >= rows) return;

    const unsigned char* w_row =
        packed_w + (long long)row * packed_row_stride_bytes;
    const float* s_row = (const float*)((const char*)scales
        + (long long)row * scales_row_stride_bytes);
    const float* z_row = (const float*)((const char*)zeros
        + (long long)row * zeros_row_stride_bytes);

    float acc = 0.0f;
    int quartets = cols >> 2;
    for (int quartet = lane; quartet < quartets; quartet += 32) {
        const unsigned char* tri = w_row + quartet * 3;
        unsigned int bits = ((unsigned int)tri[0])
            | (((unsigned int)tri[1]) << 8)
            | (((unsigned int)tri[2]) << 16);
        int col = quartet << 2;
        int group = col / group_size;
        float scale = s_row[group];
        float zero = z_row[group];

        #pragma unroll
        for (int offset = 0; offset < 4; ++offset) {
            int q = (bits >> (offset * 6)) & 0x3F;
            float x = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(&s_input[col + offset]));
            acc += ((float)q - zero) * scale * x;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    if (lane == 0) {
        __nv_bfloat16 result = __float2bfloat16(acc);
        output[row] = *reinterpret_cast<unsigned short*>(&result);
    }
}

extern "C" __global__ void hqq6_batch_gemv_bf16_vec4(
    const unsigned char* __restrict__ packed_w,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    const unsigned short* __restrict__ input,
    unsigned short* __restrict__ output,
    int rows,
    int cols,
    int group_size,
    int packed_row_stride_bytes,
    int scales_row_stride_bytes,
    int zeros_row_stride_bytes,
    unsigned long long tokens
) {
    unsigned long long token = (unsigned long long)blockIdx.z * gridDim.y + blockIdx.y;
    if (token >= tokens) return;
    input += token * (unsigned long long)cols;
    output += token * (unsigned long long)rows;
    extern __shared__ unsigned short s_input[];
    int tid = threadIdx.x;

    for (int i = tid; i < cols; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;
    if (row >= rows) return;

    const unsigned char* w_row =
        packed_w + (long long)row * packed_row_stride_bytes;
    const float* s_row = (const float*)((const char*)scales
        + (long long)row * scales_row_stride_bytes);
    const float* z_row = (const float*)((const char*)zeros
        + (long long)row * zeros_row_stride_bytes);

    float acc = 0.0f;
    int quartets = cols >> 2;
    for (int quartet = lane; quartet < quartets; quartet += 32) {
        const unsigned char* tri = w_row + quartet * 3;
        unsigned int bits = ((unsigned int)tri[0])
            | (((unsigned int)tri[1]) << 8)
            | (((unsigned int)tri[2]) << 16);
        int col = quartet << 2;
        int group = col / group_size;
        float scale = s_row[group];
        float zero = z_row[group];

        #pragma unroll
        for (int offset = 0; offset < 4; ++offset) {
            int q = (bits >> (offset * 6)) & 0x3F;
            float x = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(&s_input[col + offset]));
            acc += ((float)q - zero) * scale * x;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    if (lane == 0) {
        __nv_bfloat16 result = __float2bfloat16(acc);
        output[row] = *reinterpret_cast<unsigned short*>(&result);
    }
}


// Explicit diagnostic row reuse: preserve each row's original arithmetic order.
template<int Tile> __device__ __forceinline__ void hqq6_tile_body(
 const unsigned char* __restrict__ packed_w,const float* __restrict__ scales,
 const float* __restrict__ zeros,const unsigned short* __restrict__ input,
 unsigned short* __restrict__ output,int rows,int cols,int group_size,
 int packed_row_stride_bytes,int scales_row_stride_bytes,int zeros_row_stride_bytes,
 unsigned long long tokens) {
 unsigned long long base=((unsigned long long)blockIdx.z*gridDim.y+blockIdx.y)*Tile;
 if(base>=tokens)return; // uniform for the entire block
 extern __shared__ unsigned short tile_input[];
 int tid=threadIdx.x;
 #pragma unroll
 for(int t=0;t<Tile;++t) {
   for(int col=tid;col<cols;col+=blockDim.x) {
     tile_input[(long long)t*cols+col]=base+t<tokens ? input[(base+t)*(unsigned long long)cols+col] : 0;
   }
 }
 __syncthreads();
 int warp_id=tid>>5,lane=tid&31,warps_per_block=blockDim.x>>5;
 int row=blockIdx.x*warps_per_block+warp_id;
 if(row>=rows)return;
 const unsigned char* w_row=packed_w+(long long)row*packed_row_stride_bytes;
 const float* s_row=(const float*)((const char*)scales+(long long)row*scales_row_stride_bytes);
 const float* z_row=(const float*)((const char*)zeros+(long long)row*zeros_row_stride_bytes);
 float acc[Tile];
 #pragma unroll
 for(int t=0;t<Tile;++t)acc[t]=0.0f;
 int quartets=cols>>2;
 for(int quartet=lane;quartet<quartets;quartet+=32) {
   const unsigned char* tri=w_row+quartet*3;
   unsigned int bits=((unsigned int)tri[0])|(((unsigned int)tri[1])<<8)|(((unsigned int)tri[2])<<16);
   int col=quartet<<2,group=col/group_size;
   float scale=s_row[group],zero=z_row[group];
   #pragma unroll
   for(int offset=0;offset<4;++offset) {
     int q=(bits>>(offset*6))&0x3F;
     float weight=((float)q-zero)*scale;
     #pragma unroll
     for(int t=0;t<Tile;++t) {
       float x=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&tile_input[(long long)t*cols+col+offset]));
       acc[t]+=weight*x;
     }
   }
 }
 #pragma unroll
 for(int t=0;t<Tile;++t) {
   for(int offset=16;offset>0;offset>>=1)acc[t]+=__shfl_down_sync(0xFFFFFFFF,acc[t],offset);
   if(lane==0&&base+t<tokens) {
     __nv_bfloat16 result=__float2bfloat16(acc[t]);
     output[(base+t)*(unsigned long long)rows+row]=*reinterpret_cast<unsigned short*>(&result);
   }
 }
}
#define TILE_ENTRY(T) extern "C" __global__ void hqq6_batch_tile##T( \
 const unsigned char* packed_w,const float* scales,const float* zeros, \
 const unsigned short* input,unsigned short* output,int rows,int cols,int group_size, \
 int packed_row_stride_bytes,int scales_row_stride_bytes,int zeros_row_stride_bytes, \
 unsigned long long tokens) {hqq6_tile_body<T>(packed_w,scales,zeros,input,output,rows,cols,group_size,packed_row_stride_bytes,scales_row_stride_bytes,zeros_row_stride_bytes,tokens);}
TILE_ENTRY(2)
TILE_ENTRY(4)
#undef TILE_ENTRY

#include <mma.h>
// Experimental fixed arithmetic. No serving dispatch uses this kernel.
__device__ __forceinline__ float component_trunc(float x) {
    return __uint_as_float(__float_as_uint(x) & 0xffffe000u);
}
extern "C" __global__ void hqq6_batch_fixed_tf32x3(
 const unsigned char* packed_w,const float* scales,const float* zeros,
 const unsigned short* input,unsigned short* output,float* output_f32,
 int rows,int cols,int group_size,int packed_row_stride_bytes,
 int scales_row_stride_bytes,unsigned long long tokens) {
 namespace w=nvcuda::wmma;
 const unsigned long long token_base=((unsigned long long)blockIdx.z*gridDim.x+blockIdx.x)*32;
 const int row_base=blockIdx.y*64;
 if(token_base>=tokens||row_base>=rows)return;
 const int tid=threadIdx.x,warp=tid>>5;
 // Fixed aligned float strides, valid for every supported model geometry.
 __shared__ __align__(32) float staging[(3*64+32)*40];
 float* ah=staging;float* am=ah+64*40;float* al=am+64*40;
 float* bx=al+64*40;
 w::fragment<w::accumulator,16,16,8,float> ch,cm,cl;
 w::fill_fragment(ch,0.0f);w::fill_fragment(cm,0.0f);w::fill_fragment(cl,0.0f);
 for(int kbase=0;kbase<cols;kbase+=32) {
   // Reconstruct exactly the old F32 weight before the explicitly different decomposition.
   for(int quartet=tid;quartet<64*8;quartet+=blockDim.x) {
     const int rr=quartet/8,kk=(quartet%8)*4,row=row_base+rr,col=kbase+kk;
     float scale=0.0f,zero=0.0f;unsigned int bits=0;
     if(row<rows&&col<cols) {
       const unsigned char* q=packed_w+(long long)row*packed_row_stride_bytes+(col/4)*3;
       bits=(unsigned int)q[0]|((unsigned int)q[1]<<8)|((unsigned int)q[2]<<16);
       const int group=col/group_size;
       scale=((const float*)((const char*)scales+(long long)row*scales_row_stride_bytes))[group];
       zero=((const float*)((const char*)zeros+(long long)row*scales_row_stride_bytes))[group];
     }
     #pragma unroll
     for(int off=0;off<4;++off) {
       float weight=(row<rows&&col+off<cols)?((float)((bits>>(off*6))&63)-zero)*scale:0.0f;
       const float high=component_trunc(weight);
       const float residual=__fsub_rn(weight,high);
       const float middle=component_trunc(residual);
       const float low=component_trunc(__fsub_rn(residual,middle));
       // Converted TF32 values are used only by WMMA; never ordinary float operations.
       ah[rr*40+kk+off]=w::__float_to_tf32(high);
       am[rr*40+kk+off]=w::__float_to_tf32(middle);
       al[rr*40+kk+off]=w::__float_to_tf32(low);
     }
   }
   for(int i=tid;i<32*32;i+=blockDim.x) {
     const int token=i/32,kk=i%32,col=kbase+kk;
     float x=0.0f;
     if(token_base+token<tokens&&col<cols) {
       x=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(input+(token_base+token)*(unsigned long long)cols+col));
     }
     bx[token*40+kk]=w::__float_to_tf32(x);
   }
   __syncthreads();
   #pragma unroll
   for(int kk=0;kk<32;kk+=8) {
     w::fragment<w::matrix_a,16,16,8,w::precision::tf32,w::row_major> a;
     w::fragment<w::matrix_b,16,16,8,w::precision::tf32,w::col_major> b;
     w::load_matrix_sync(b,bx+(warp%2)*16*40+kk,40);
     w::load_matrix_sync(a,ah+(warp/2)*16*40+kk,40);
     w::mma_sync(ch,a,b,ch);
     w::load_matrix_sync(a,am+(warp/2)*16*40+kk,40);
     w::mma_sync(cm,a,b,cm);
     w::load_matrix_sync(a,al+(warp/2)*16*40+kk,40);
     w::mma_sync(cl,a,b,cl);
   }
   __syncthreads();
 }
 // Uniform fragment element combination is documented; no mapping assumptions.
 for(int i=0;i<ch.num_elements;++i)ch.x[i]=__fadd_rn(__fadd_rn(ch.x[i],cm.x[i]),cl.x[i]);
 // All staged operands are dead after the last barrier, so the output reuses storage.
 float* scratch=staging+warp*256;
 w::store_matrix_sync(scratch,ch,16,w::mem_row_major);
 __syncthreads();
 for(int i=tid;i<8*256;i+=blockDim.x) {
   const int wi=i/256,local=i%256;
   const int row=row_base+(wi/2)*16+local/16;
   const unsigned long long token=token_base+(wi%2)*16+local%16;
   if(row<rows&&token<tokens) {
     const float sum=staging[i];
     output_f32[token*(unsigned long long)rows+row]=sum;
     const __nv_bfloat16 value=__float2bfloat16_rn(sum);
     output[token*(unsigned long long)rows+row]=*reinterpret_cast<const unsigned short*>(&value);
   }
 }
}

// Unadopted fixed K8 accumulation; ordinary serving dispatch unchanged.
__device__ __forceinline__ float protected_add(float a,float b) {
 float out;asm volatile("add.rn.f32 %0, %1, %2;":"=f"(out):"f"(a),"f"(b));return out;
}
__device__ __forceinline__ float protected_sub(float a,float b) {
 float out;asm volatile("sub.rn.f32 %0, %1, %2;":"=f"(out):"f"(a),"f"(b));return out;
}
__device__ __forceinline__ void general_twosum(float a,float b,float& sum,float& error) {
 sum=protected_add(a,b);const float virtual_b=protected_sub(sum,a);
 error=protected_add(protected_sub(a,protected_sub(sum,virtual_b)),protected_sub(b,virtual_b));
}
__device__ __forceinline__ void persistent_twosum_add(float x,float& hi,float& lo) {
 float sum,error;general_twosum(hi,x,sum,error);
 const float residue=protected_add(lo,error);general_twosum(sum,residue,hi,lo);
}
extern "C" __global__ void hqq6_batch_compensated_tf32x3(
 const unsigned char* packed_w,const float* scales,const float* zeros,
 const unsigned short* input,unsigned short* output,float* output_f32,
 int rows,int cols,int group_size,int packed_row_stride_bytes,
 int scales_row_stride_bytes,unsigned long long tokens) {
 namespace w=nvcuda::wmma;
 const unsigned long long token_base=((unsigned long long)blockIdx.z*gridDim.x+blockIdx.x)*32;
 const int row_base=blockIdx.y*64;
 if(token_base>=tokens||row_base>=rows)return;
 const int tid=threadIdx.x,warp=tid>>5;
 // Fixed aligned float strides, valid for every supported model geometry.
 __shared__ __align__(32) float staging[(3*64+32)*40];
 float* ah=staging;float* am=ah+64*40;float* al=am+64*40;
 float* bx=al+64*40;
 w::fragment<w::accumulator,16,16,8,float> ch,cm,cl;
 w::fragment<w::accumulator,16,16,8,float> carry_hi,carry_lo;
 w::fill_fragment(carry_hi,0.0f);w::fill_fragment(carry_lo,0.0f);
 for(int kbase=0;kbase<cols;kbase+=32) {
   // Reconstruct exactly the old F32 weight before the explicitly different decomposition.
   for(int quartet=tid;quartet<64*8;quartet+=blockDim.x) {
     const int rr=quartet/8,kk=(quartet%8)*4,row=row_base+rr,col=kbase+kk;
     float scale=0.0f,zero=0.0f;unsigned int bits=0;
     if(row<rows&&col<cols) {
       const unsigned char* q=packed_w+(long long)row*packed_row_stride_bytes+(col/4)*3;
       bits=(unsigned int)q[0]|((unsigned int)q[1]<<8)|((unsigned int)q[2]<<16);
       const int group=col/group_size;
       scale=((const float*)((const char*)scales+(long long)row*scales_row_stride_bytes))[group];
       zero=((const float*)((const char*)zeros+(long long)row*scales_row_stride_bytes))[group];
     }
     #pragma unroll
     for(int off=0;off<4;++off) {
       float weight=(row<rows&&col+off<cols)?((float)((bits>>(off*6))&63)-zero)*scale:0.0f;
       const float high=component_trunc(weight);
       const float residual=__fsub_rn(weight,high);
       const float middle=component_trunc(residual);
       const float low=component_trunc(__fsub_rn(residual,middle));
       // Converted TF32 values are used only by WMMA; never ordinary float operations.
       ah[rr*40+kk+off]=w::__float_to_tf32(high);
       am[rr*40+kk+off]=w::__float_to_tf32(middle);
       al[rr*40+kk+off]=w::__float_to_tf32(low);
     }
   }
   for(int i=tid;i<32*32;i+=blockDim.x) {
     const int token=i/32,kk=i%32,col=kbase+kk;
     float x=0.0f;
     if(token_base+token<tokens&&col<cols) {
       x=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(input+(token_base+token)*(unsigned long long)cols+col));
     }
     bx[token*40+kk]=w::__float_to_tf32(x);
   }
   __syncthreads();
   #pragma unroll
   for(int kk=0;kk<32;kk+=8) {
     w::fill_fragment(ch,0.0f);w::fill_fragment(cm,0.0f);w::fill_fragment(cl,0.0f);
     w::fragment<w::matrix_a,16,16,8,w::precision::tf32,w::row_major> a;
     w::fragment<w::matrix_b,16,16,8,w::precision::tf32,w::col_major> b;
     w::load_matrix_sync(b,bx+(warp%2)*16*40+kk,40);
     w::load_matrix_sync(a,ah+(warp/2)*16*40+kk,40);
     w::mma_sync(ch,a,b,ch);
     w::load_matrix_sync(a,am+(warp/2)*16*40+kk,40);
     w::mma_sync(cm,a,b,cm);
     w::load_matrix_sync(a,al+(warp/2)*16*40+kk,40);
     w::mma_sync(cl,a,b,cl);
     for(int i=0;i<ch.num_elements;++i) {
       persistent_twosum_add(ch.x[i],carry_hi.x[i],carry_lo.x[i]);
       persistent_twosum_add(cm.x[i],carry_hi.x[i],carry_lo.x[i]);
       persistent_twosum_add(cl.x[i],carry_hi.x[i],carry_lo.x[i]);
     }
   }
   __syncthreads();
 }
 // Uniform fragment element combination is documented; no mapping assumptions.
 for(int i=0;i<ch.num_elements;++i)ch.x[i]=protected_add(carry_hi.x[i],carry_lo.x[i]);
 // All staged operands are dead after the last barrier, so the output reuses storage.
 float* scratch=staging+warp*256;
 w::store_matrix_sync(scratch,ch,16,w::mem_row_major);
 __syncthreads();
 for(int i=tid;i<8*256;i+=blockDim.x) {
   const int wi=i/256,local=i%256;
   const int row=row_base+(wi/2)*16+local/16;
   const unsigned long long token=token_base+(wi%2)*16+local%16;
   if(row<rows&&token<tokens) {
     const float sum=staging[i];
     output_f32[token*(unsigned long long)rows+row]=sum;
     const __nv_bfloat16 value=__float2bfloat16_rn(sum);
     output[token*(unsigned long long)rows+row]=*reinterpret_cast<const unsigned short*>(&value);
   }
 }
}

__device__ __forceinline__ void magnitude_ordered_twosum(float a,float b,float& sum,float& error) {
 sum=protected_add(a,b);
 const bool a_larger=(__float_as_uint(a)&0x7fffffffU)>=(__float_as_uint(b)&0x7fffffffU);
 const float large=a_larger?a:b,small=a_larger?b:a;
 error=protected_sub(small,protected_sub(sum,large));
 // general_twosum's exact-zero error is positive zero, including (-0,-0).
 if((__float_as_uint(error)&0x7fffffffU)==0)error=0.0f;
}
__device__ __forceinline__ void persistent_ordered_twosum_add(float x,float& hi,float& lo) {
 float sum,error;magnitude_ordered_twosum(hi,x,sum,error);
 const float residue=protected_add(lo,error);magnitude_ordered_twosum(sum,residue,hi,lo);
}
extern "C" __global__ void hqq6_batch_ordered_tf32x3(
 const unsigned char* packed_w,const float* scales,const float* zeros,
 const unsigned short* input,unsigned short* output,float* output_f32,
 int rows,int cols,int group_size,int packed_row_stride_bytes,
 int scales_row_stride_bytes,unsigned long long tokens) {
 namespace w=nvcuda::wmma;
 const unsigned long long token_base=((unsigned long long)blockIdx.z*gridDim.x+blockIdx.x)*32;
 const int row_base=blockIdx.y*64;
 if(token_base>=tokens||row_base>=rows)return;
 const int tid=threadIdx.x,warp=tid>>5;
 // Fixed aligned float strides, valid for every supported model geometry.
 __shared__ __align__(32) float staging[(3*64+32)*40];
 float* ah=staging;float* am=ah+64*40;float* al=am+64*40;
 float* bx=al+64*40;
 w::fragment<w::accumulator,16,16,8,float> ch,cm,cl;
 w::fragment<w::accumulator,16,16,8,float> carry_hi,carry_lo;
 w::fill_fragment(carry_hi,0.0f);w::fill_fragment(carry_lo,0.0f);
 for(int kbase=0;kbase<cols;kbase+=32) {
   // Reconstruct exactly the old F32 weight before the explicitly different decomposition.
   for(int quartet=tid;quartet<64*8;quartet+=blockDim.x) {
     const int rr=quartet/8,kk=(quartet%8)*4,row=row_base+rr,col=kbase+kk;
     float scale=0.0f,zero=0.0f;unsigned int bits=0;
     if(row<rows&&col<cols) {
       const unsigned char* q=packed_w+(long long)row*packed_row_stride_bytes+(col/4)*3;
       bits=(unsigned int)q[0]|((unsigned int)q[1]<<8)|((unsigned int)q[2]<<16);
       const int group=col/group_size;
       scale=((const float*)((const char*)scales+(long long)row*scales_row_stride_bytes))[group];
       zero=((const float*)((const char*)zeros+(long long)row*scales_row_stride_bytes))[group];
     }
     #pragma unroll
     for(int off=0;off<4;++off) {
       float weight=(row<rows&&col+off<cols)?((float)((bits>>(off*6))&63)-zero)*scale:0.0f;
       const float high=component_trunc(weight);
       const float residual=__fsub_rn(weight,high);
       const float middle=component_trunc(residual);
       const float low=component_trunc(__fsub_rn(residual,middle));
       // Converted TF32 values are used only by WMMA; never ordinary float operations.
       ah[rr*40+kk+off]=w::__float_to_tf32(high);
       am[rr*40+kk+off]=w::__float_to_tf32(middle);
       al[rr*40+kk+off]=w::__float_to_tf32(low);
     }
   }
   for(int i=tid;i<32*32;i+=blockDim.x) {
     const int token=i/32,kk=i%32,col=kbase+kk;
     float x=0.0f;
     if(token_base+token<tokens&&col<cols) {
       x=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(input+(token_base+token)*(unsigned long long)cols+col));
     }
     bx[token*40+kk]=w::__float_to_tf32(x);
   }
   __syncthreads();
   #pragma unroll
   for(int kk=0;kk<32;kk+=8) {
     w::fill_fragment(ch,0.0f);w::fill_fragment(cm,0.0f);w::fill_fragment(cl,0.0f);
     w::fragment<w::matrix_a,16,16,8,w::precision::tf32,w::row_major> a;
     w::fragment<w::matrix_b,16,16,8,w::precision::tf32,w::col_major> b;
     w::load_matrix_sync(b,bx+(warp%2)*16*40+kk,40);
     w::load_matrix_sync(a,ah+(warp/2)*16*40+kk,40);
     w::mma_sync(ch,a,b,ch);
     w::load_matrix_sync(a,am+(warp/2)*16*40+kk,40);
     w::mma_sync(cm,a,b,cm);
     w::load_matrix_sync(a,al+(warp/2)*16*40+kk,40);
     w::mma_sync(cl,a,b,cl);
     for(int i=0;i<ch.num_elements;++i) {
       persistent_ordered_twosum_add(ch.x[i],carry_hi.x[i],carry_lo.x[i]);
       persistent_ordered_twosum_add(cm.x[i],carry_hi.x[i],carry_lo.x[i]);
       persistent_ordered_twosum_add(cl.x[i],carry_hi.x[i],carry_lo.x[i]);
     }
   }
   __syncthreads();
 }
 // Uniform fragment element combination is documented; no mapping assumptions.
 for(int i=0;i<ch.num_elements;++i)ch.x[i]=protected_add(carry_hi.x[i],carry_lo.x[i]);
 // All staged operands are dead after the last barrier, so the output reuses storage.
 float* scratch=staging+warp*256;
 w::store_matrix_sync(scratch,ch,16,w::mem_row_major);
 __syncthreads();
 for(int i=tid;i<8*256;i+=blockDim.x) {
   const int wi=i/256,local=i%256;
   const int row=row_base+(wi/2)*16+local/16;
   const unsigned long long token=token_base+(wi%2)*16+local%16;
   if(row<rows&&token<tokens) {
     const float sum=staging[i];
     output_f32[token*(unsigned long long)rows+row]=sum;
     const __nv_bfloat16 value=__float2bfloat16_rn(sum);
     output[token*(unsigned long long)rows+row]=*reinterpret_cast<const unsigned short*>(&value);
   }
 }
}
