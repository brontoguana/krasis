// Expand a complete expert into its normal Marlin staging layout. Weight
// codes select eight original INT4 levels; BF16 scales copy without conversion.
extern "C" __global__ void streamed_cold_int3_expand(
    const unsigned char* src, unsigned int* dst,
    unsigned int w13_words, unsigned int w13_scale_words,
    unsigned int w2_words, unsigned int w2_scale_words, unsigned int codebook) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = w13_words + w13_scale_words + w2_words + w2_scale_words;
    if (i >= total) return;
    unsigned int offset, word = 0;
    if (i < w13_words) { offset = i * 3; }
    else if (i < w13_words + w13_scale_words) {
        offset = w13_words * 3 + (i - w13_words) * 4;
        dst[i] = *reinterpret_cast<const unsigned int*>(src + offset); return;
    } else if (i < w13_words + w13_scale_words + w2_words) {
        offset = w13_words * 3 + w13_scale_words * 4
            + (i - w13_words - w13_scale_words) * 3;
    } else {
        offset = (w13_words + w2_words) * 3 + w13_scale_words * 4
            + (i - w13_words - w13_scale_words - w2_words) * 4;
        dst[i] = *reinterpret_cast<const unsigned int*>(src + offset); return;
    }
    unsigned int packed = src[offset] | (static_cast<unsigned int>(src[offset+1]) << 8)
        | (static_cast<unsigned int>(src[offset+2]) << 16);
    #pragma unroll
    for (int j = 0; j < 8; ++j) word |= ((codebook >> (((packed >> (j*3)) & 7)*4)) & 15) << (j*4);
    dst[i] = word;
}
