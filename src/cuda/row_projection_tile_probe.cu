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
