// Chunk-parallel, bit-exact rANS decoder for expert-contiguous Marlin backing.
//
// One block owns one output stripe. Each of 128 lanes carries an independent
// rANS state and writes strided bytes; across a warp, every output iteration is
// coalesced. Packed-weight tasks decode two 16-symbol nibbles per byte. Scale
// tasks decode one 256-symbol byte with separate tables for BF16 low/high bytes.

#include <stdint.h>

static constexpr uint32_t kRansScaleBits = 12;
static constexpr uint32_t kRansMask = (1u << kRansScaleBits) - 1u;
static constexpr uint32_t kRansLowerBound = 1u << 23;
static constexpr uint32_t kCodecLanes = 128;
static constexpr uint32_t kTableAlphabet = 256;
static constexpr uint32_t kDecodeSlots = 1u << kRansScaleBits;
static constexpr uint32_t kHeaderWords = 8;
static constexpr uint32_t kTaskWords = 4;

__device__ __forceinline__ uint32_t load_progress_acquire_device(
    const uint32_t* progress) {
    uint32_t value;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(progress)
                 : "memory");
    return value;
}

__device__ __forceinline__ uint8_t decode_symbol(
    uint32_t& state,
    const uint8_t*& input,
    const uint16_t* decode_symbols,
    const uint16_t* frequencies,
    const uint16_t* starts,
    uint32_t table) {
    const uint32_t slot = state & kRansMask;
    const uint32_t symbol = decode_symbols[table * kDecodeSlots + slot];
    const uint32_t frequency = frequencies[table * kTableAlphabet + symbol];
    const uint32_t start = starts[table * kTableAlphabet + symbol];
    state = frequency * (state >> kRansScaleBits) + slot - start;
    while (state < kRansLowerBound) {
        state = (state << 8) | *input++;
    }
    return static_cast<uint8_t>(symbol);
}

extern "C" __global__ void decode_expert_rans(
    const uint8_t* blob,
    uint8_t* output,
    const uint16_t* decode_symbols,
    const uint16_t* frequencies,
    const uint16_t* starts,
    uint32_t task_start,
    uint32_t task_count) {
    if (blockDim.x != kCodecLanes) {
        return;
    }
    const uint32_t* header = reinterpret_cast<const uint32_t*>(blob);
    const uint32_t total_tasks = header[3];
    if (blockIdx.x >= task_count) {
        return;
    }
    const uint32_t task_index = task_start + blockIdx.x;
    if (task_index >= total_tasks) {
        return;
    }
    const uint32_t task_offset = header[4];
    const uint32_t lane_offsets_offset = header[5];
    const uint32_t* task = reinterpret_cast<const uint32_t*>(blob + task_offset)
        + task_index * kTaskWords;
    const uint32_t output_offset = task[0];
    const uint32_t output_bytes = task[1];
    const uint32_t lane_offsets_index = task[2];
    const uint32_t mode = task[3];
    const uint32_t lane = threadIdx.x;
    const uint32_t* lane_offsets =
        reinterpret_cast<const uint32_t*>(blob + lane_offsets_offset);
    const uint8_t* input = blob + lane_offsets[lane_offsets_index + lane];
    uint32_t state = static_cast<uint32_t>(input[0])
        | (static_cast<uint32_t>(input[1]) << 8)
        | (static_cast<uint32_t>(input[2]) << 16)
        | (static_cast<uint32_t>(input[3]) << 24);
    input += sizeof(uint32_t);

    const uint32_t output_end = output_offset + output_bytes;
    for (uint32_t output_index = output_offset + lane;
         output_index < output_end;
         output_index += kCodecLanes) {
        if (mode == 0) {
            const uint8_t low = decode_symbol(
                state, input, decode_symbols, frequencies, starts, 0);
            const uint8_t high = decode_symbol(
                state, input, decode_symbols, frequencies, starts, 0);
            output[output_index] = low | static_cast<uint8_t>(high << 4);
        } else {
            const uint32_t table = 1u + (lane & 1u);
            output[output_index] = decode_symbol(
                state, input, decode_symbols, frequencies, starts, table);
        }
    }
}

// One launch covers the entire expert while task-aligned compressed ranges
// arrive behind it. The copy stream publishes an exclusive ready-task count
// in device memory after each complete range. The copy stream writes the
// counter with a pinned four-byte H2D transfer, so no publisher kernel can be
// starved by the waiting decoder blocks. Device-scope acquire ordering makes
// the preceding payload DMA visible before a block reads lane streams.
extern "C" __global__ void decode_expert_rans_streaming(
    const uint8_t* blob,
    uint8_t* output,
    const uint16_t* decode_symbols,
    const uint16_t* frequencies,
    const uint16_t* starts,
    const uint32_t* ready_tasks,
    uint32_t task_count) {
    if (blockDim.x != kCodecLanes || blockIdx.x >= task_count) {
        return;
    }
    __shared__ uint32_t task_ready;
    if (threadIdx.x == 0) {
        const uint32_t required = blockIdx.x + 1u;
        uint32_t observed;
        do {
            observed = load_progress_acquire_device(ready_tasks);
#if __CUDA_ARCH__ >= 700
            if (observed < required) {
                __nanosleep(64);
            }
#endif
        } while (observed < required);
        task_ready = observed;
    }
    __syncthreads();
    if (task_ready <= blockIdx.x) {
        return;
    }

    const uint32_t* header = reinterpret_cast<const uint32_t*>(blob);
    const uint32_t total_tasks = header[3];
    const uint32_t task_index = blockIdx.x;
    if (task_index >= total_tasks) {
        return;
    }
    const uint32_t task_offset = header[4];
    const uint32_t lane_offsets_offset = header[5];
    const uint32_t* task = reinterpret_cast<const uint32_t*>(blob + task_offset)
        + task_index * kTaskWords;
    const uint32_t output_offset = task[0];
    const uint32_t output_bytes = task[1];
    const uint32_t lane_offsets_index = task[2];
    const uint32_t mode = task[3];
    const uint32_t lane = threadIdx.x;
    const uint32_t* lane_offsets =
        reinterpret_cast<const uint32_t*>(blob + lane_offsets_offset);
    const uint8_t* input = blob + lane_offsets[lane_offsets_index + lane];
    uint32_t state = static_cast<uint32_t>(input[0])
        | (static_cast<uint32_t>(input[1]) << 8)
        | (static_cast<uint32_t>(input[2]) << 16)
        | (static_cast<uint32_t>(input[3]) << 24);
    input += sizeof(uint32_t);

    const uint32_t output_end = output_offset + output_bytes;
    for (uint32_t output_index = output_offset + lane;
         output_index < output_end;
         output_index += kCodecLanes) {
        if (mode == 0) {
            const uint8_t low = decode_symbol(
                state, input, decode_symbols, frequencies, starts, 0);
            const uint8_t high = decode_symbol(
                state, input, decode_symbols, frequencies, starts, 0);
            output[output_index] = low | static_cast<uint8_t>(high << 4);
        } else {
            const uint32_t table = 1u + (lane & 1u);
            output[output_index] = decode_symbol(
                state, input, decode_symbols, frequencies, starts, table);
        }
    }
}
