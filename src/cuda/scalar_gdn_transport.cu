// Scalar-gate native-FLA64 owner transport. No vector-KDA parameterization.
// Metadata is a registered per-sequence device allocation, not host state.
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

namespace {
constexpr int BT = 64; // Native FLA arithmetic block, never a memory budget.
enum Meta { Cursor, Pending, Epoch, Poison, Inflight, WorkRows, NewRows, Reserved, MetaWords };
__device__ bool finite_bf16(uint16_t x) { return (x & 0x7f80u) != 0x7f80u; }
__device__ void poison(uint64_t* m) { atomicExch(reinterpret_cast<unsigned long long*>(m+Poison),1ull); }
}

extern "C" __global__ void scalar_gdn_begin(
    uint64_t* meta, uint64_t expected_start, const int32_t* decode_position,
    int new_rows, int capacity_rows, int decode) {
    if(blockIdx.x || threadIdx.x) return;
    uint64_t expected=expected_start;
    // Explicit domains: 0 prefill, 1 dynamic device-position decode,
    // 2 host-position eager decode. Rust rejects domain2 during graph capture.
    if(decode<0 || decode>2) {poison(meta);return;}
    if(decode==1) {
        if(!decode_position || *decode_position<0) {poison(meta);return;}
        expected=static_cast<uint64_t>(*decode_position);
    }
    if(meta[Poison] || meta[Inflight] || meta[Cursor]!=expected ||
       meta[Pending]!=meta[Cursor]%BT || new_rows<=0 || capacity_rows<BT ||
       static_cast<uint64_t>(new_rows)>UINT64_MAX-meta[Cursor] ||
       meta[Pending]+static_cast<uint64_t>(new_rows)>static_cast<uint64_t>(capacity_rows) ||
       (decode && new_rows!=1)) {poison(meta);return;}
    meta[Inflight]=1;
    meta[WorkRows]=meta[Pending]+static_cast<uint64_t>(new_rows);
    meta[NewRows]=static_cast<uint64_t>(new_rows);
}

// Caller validates all actual backing extents before enqueue. Each plane has
// a separate real element stride, including unequal key/value dimensions.
extern "C" __global__ void scalar_gdn_stage_bf16(
    uint16_t* staged, const uint16_t* pending, const uint16_t* live,
    uint64_t* meta, int row_elements, int padded_rows) {
    const uint64_t i=static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const uint64_t cells=static_cast<uint64_t>(row_elements)*padded_rows;
    if(i>=cells) return;
    const uint64_t row=i/row_elements,col=i%row_elements;
    uint16_t value=0;
    if(!meta[Poison] && row<meta[WorkRows]) {
        value=row<meta[Pending]?pending[row*row_elements+col]:live[(row-meta[Pending])*row_elements+col];
        if(!finite_bf16(value)) poison(meta);
    }
    staged[i]=value;
}
extern "C" __global__ void scalar_gdn_stage_gate(
    float* staged, const float* pending, const float* live,
    uint64_t* meta, int heads, int padded_rows) {
    const uint64_t i=static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const uint64_t cells=static_cast<uint64_t>(heads)*padded_rows;
    if(i>=cells) return;
    const uint64_t row=i/heads,col=i%heads;
    float value=0;
    if(!meta[Poison] && row<meta[WorkRows]) {
        value=row<meta[Pending]?pending[row*heads+col]:live[(row-meta[Pending])*heads+col];
        if(!isfinite(value)) poison(meta);
    }
    staged[i]=value;
}
extern "C" __global__ void scalar_gdn_cumsum(
    float* cumulative, const float* raw, uint64_t* meta, int heads, int padded_rows) {
    // One serial FP32 sum per head/native block: matches the actual169
    // producer's independently verified order. Padding raw gates are zero.
    const int head=blockIdx.x,chunk=blockIdx.y;
    if(threadIdx.x || head>=heads || chunk*BT>=padded_rows) return;
    float sum=0;
    for(int row=chunk*BT;row<(chunk+1)*BT;row++) {
        sum+=raw[static_cast<uint64_t>(row)*heads+head];
        cumulative[static_cast<uint64_t>(row)*heads+head]=sum;
        if(!isfinite(sum)) poison(meta);
    }
}

extern "C" __global__ void scalar_gdn_retain_bf16(
    uint16_t* pending, const uint16_t* staged, const uint64_t* meta,
    int row_elements) {
    const uint64_t i=static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=static_cast<uint64_t>(BT)*row_elements || meta[Poison]) return;
    const uint64_t row=i/row_elements,col=i%row_elements;
    const uint64_t count=meta[WorkRows]%BT,start=meta[WorkRows]-count;
    pending[i]=row<count?staged[(start+row)*row_elements+col]:0;
}
extern "C" __global__ void scalar_gdn_retain_gate(
    float* pending, const float* staged, const uint64_t* meta, int heads) {
    const uint64_t i=static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=static_cast<uint64_t>(BT)*heads || meta[Poison]) return;
    const uint64_t row=i/heads,col=i%heads;
    const uint64_t count=meta[WorkRows]%BT,start=meta[WorkRows]-count;
    pending[i]=row<count?staged[(start+row)*heads+col]:0;
}
extern "C" __global__ void scalar_gdn_emit(
    uint16_t* emitted, const uint16_t* output, const uint64_t* meta,
    int row_elements, int emitted_rows) {
    const uint64_t i=static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=static_cast<uint64_t>(emitted_rows)*row_elements) return;
    // NaN on poisoned state is deliberate rejection, never fabricated data.
    emitted[i]=meta[Poison]?0x7fc1u:output[meta[Pending]*row_elements+i];
}
extern "C" __global__ void scalar_gdn_commit_carry(
    float* boundary, const float* completed, const uint64_t* meta,
    uint64_t state_elements, int decode) {
    const uint64_t i=static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=state_elements || meta[Poison]) return;
    if((!decode && meta[WorkRows]>=BT) || (decode && meta[WorkRows]==BT))
        boundary[i]=completed[i];
}
extern "C" __global__ void scalar_gdn_commit(uint64_t* meta) {
    if(blockIdx.x || threadIdx.x || meta[Poison]) return;
    if(!meta[Inflight] || !meta[NewRows] || meta[WorkRows]!=meta[Pending]+meta[NewRows]) {poison(meta);return;}
    meta[Cursor]+=meta[NewRows];
    meta[Pending]=meta[WorkRows]%BT;
    meta[Epoch]++;
    meta[Inflight]=0;
}

// Persistent device-level semantic health. Publication reads this only after
// the actual whole-phase event; errors stay sticky until verified reset/restore.
extern "C" __global__ void scalar_gdn_collect_health(
    const uint64_t* meta, uint64_t* health) {
    if(blockIdx.x || threadIdx.x) return;
    if(meta[Poison] || meta[Inflight] || meta[Pending]!=meta[Cursor]%BT)
        atomicOr(reinterpret_cast<unsigned long long*>(health),1ull);
}
