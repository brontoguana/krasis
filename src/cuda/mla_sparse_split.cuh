// Sparse K6 split execution. Each CTA owns one head and at most128 selections.
// Packed equations and cache bytes are unchanged; only softmax reduction order
// changes. Workspace belongs to the decode graph and is stream ordered.
extern "C" __global__ void mla_sparse_k6_split(
    float* __restrict__ partial, const float* __restrict__ q,
    const float* __restrict__ qp, const unsigned char* __restrict__ c,
    const unsigned char* __restrict__ p, const int* __restrict__ indices,
    float scale, int dim, int rope, int capacity,
    int sequence_length, const int* __restrict__ sequence_pointer
) {
    const int h=blockIdx.x, split=blockIdx.y, tid=threadIdx.x;
    const int seq=sequence_pointer ? *sequence_pointer : sequence_length;
    const int count=min(max(seq,0),capacity), begin=split*128;
    const int splits=(capacity+127)/128;
    extern __shared__ float shared[];
    float* query=shared; float* pe=query+dim; float* weights=pe+rope;
    __shared__ float reductions[8];
    for(int d=tid;d<dim;d+=256)query[d]=q[h*dim+d];
    for(int d=tid;d<rope;d+=256)pe[d]=qp[h*rope+d];
    __syncthreads();
    const int slot=begin+tid;
    int pos=-1;float score=-1e30f;
    if(tid<128 && slot<count){
        pos=indices[slot];
        if(pos>=0 && pos<seq){
            score=mla_dot_k6(c,pos,dim,query);
            if(rope)score+=mla_dot_k6(p,pos,rope,pe);
            score*=scale;
        }
    }
    if(tid<128)weights[tid]=score;
    float m=score;
    for(int o=16;o;o>>=1)m=fmaxf(m,__shfl_down_sync(0xffffffff,m,o));
    if((tid&31)==0)reductions[tid>>5]=m;
    __syncthreads();
    if(tid==0){float x=reductions[0];for(int w=1;w<8;w++)x=fmaxf(x,reductions[w]);reductions[0]=x;}
    __syncthreads();m=reductions[0];
    __syncthreads(); // Every warp reads the maximum before reductions is reused.
    float weight=0;
    if(tid<128){if(pos>=0 && pos<seq)weight=__expf(weights[tid]-m);weights[tid]=weight;}
    float sum=weight;
    for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
    if((tid&31)==0)reductions[tid>>5]=sum;
    __syncthreads();
    float* dst=partial+((long long)h*splits+split)*(dim+2);
    if(tid==0){float s=0;for(int w=0;w<8;w++)s+=reductions[w];dst[dim]=m;dst[dim+1]=s;}
    for(int d=tid;d<dim;d+=256){
        float acc=0;
        for(int i=0;i<min(128,max(count-begin,0));i++){
            const int at=indices[begin+i];
            if(at>=0 && at<seq)acc+=weights[i]*mla_load_k6_value(c,at,dim,d);
        }
        dst[d]=acc;
    }
}
extern "C" __global__ void mla_sparse_k6_split_warp(
    float* __restrict__ partial, const float* __restrict__ q,
    const float* __restrict__ qp, const unsigned char* __restrict__ c,
    const unsigned char* __restrict__ p, const int* __restrict__ indices,
    float scale, int dim, int rope, int capacity,
    int sequence_length, const int* __restrict__ sequence_pointer
) {
    const int h=blockIdx.x, split=blockIdx.y, tid=threadIdx.x;
    const int seq=sequence_pointer ? *sequence_pointer : sequence_length;
    const int count=min(max(seq,0),capacity), begin=split*128;
    const int splits=(capacity+127)/128;
    extern __shared__ float shared[];
    float* query=shared; float* pe=query+dim; float* weights=pe+rope;
    __shared__ float reductions[8];
    for(int d=tid;d<dim;d+=256)query[d]=q[h*dim+d];
    for(int d=tid;d<rope;d+=256)pe[d]=qp[h*rope+d];
    __syncthreads();
    for(int item=tid/32;item<128;item+=8){
        const int slot=begin+item, lane=tid&31;
        const int pos=slot<count?indices[slot]:-1;
        float dot=0, pe_dot=0;
        if(pos>=0 && pos<seq){
            for(int d=lane;d<dim;d+=32){
                const unsigned char* block=c+((long long)pos*(dim/16)+d/16)*14;
                const float step=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(block));
                dot+=query[d]*step*(float)(mla_unpack_k6(block+2,d&15)-32);
            }
            for(int d=lane;d<rope;d+=32){
                const unsigned char* block=p+((long long)pos*(rope/16)+d/16)*14;
                const float step=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(block));
                pe_dot+=pe[d]*step*(float)(mla_unpack_k6(block+2,d&15)-32);
            }
        }
        for(int off=16;off;off>>=1){
            dot+=__shfl_down_sync(0xffffffff,dot,off);
            pe_dot+=__shfl_down_sync(0xffffffff,pe_dot,off);
        }
        if(lane==0)weights[item]=(pos>=0 && pos<seq)?(dot+pe_dot)*scale:-1e30f;
    }
    __syncthreads();
    const int slot=begin+tid;
    const int pos=tid<128 && slot<count?indices[slot]:-1;
    const float score=tid<128?weights[tid]:-1e30f;
    float m=score;
    for(int o=16;o;o>>=1)m=fmaxf(m,__shfl_down_sync(0xffffffff,m,o));
    if((tid&31)==0)reductions[tid>>5]=m;
    __syncthreads();
    if(tid==0){float x=reductions[0];for(int w=1;w<8;w++)x=fmaxf(x,reductions[w]);reductions[0]=x;}
    __syncthreads();m=reductions[0];
    __syncthreads(); // Every warp reads the maximum before reductions is reused.
    float weight=0;
    if(tid<128){if(pos>=0 && pos<seq)weight=__expf(weights[tid]-m);weights[tid]=weight;}
    float sum=weight;
    for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
    if((tid&31)==0)reductions[tid>>5]=sum;
    __syncthreads();
    float* dst=partial+((long long)h*splits+split)*(dim+2);
    if(tid==0){float s=0;for(int w=0;w<8;w++)s+=reductions[w];dst[dim]=m;dst[dim+1]=s;}
    for(int d=tid;d<dim;d+=256){
        float acc=0;
        for(int i=0;i<min(128,max(count-begin,0));i++){
            const int at=indices[begin+i];
            if(at>=0 && at<seq)acc+=weights[i]*mla_load_k6_value(c,at,dim,d);
        }
        dst[d]=acc;
    }
}
extern "C" __global__ void mla_sparse_k6_combine(
    float* __restrict__ out, const float* __restrict__ partial,
    int dim, int splits
) {
    const int h=blockIdx.x,tid=threadIdx.x;
    const float* base=partial+(long long)h*splits*(dim+2);
    float m=-1e30f;
    for(int s=0;s<splits;s++)m=fmaxf(m,base[s*(dim+2)+dim]);
    float den=0;
    for(int s=0;s<splits;s++)den+=__expf(base[s*(dim+2)+dim]-m)*base[s*(dim+2)+dim+1];
    for(int d=tid;d<dim;d+=blockDim.x){
        float num=0;
        for(int s=0;s<splits;s++)num+=__expf(base[s*(dim+2)+dim]-m)*base[s*(dim+2)+d];
        out[h*dim+d]=den>0?num/den:0;
    }
}

// Parallel score production with exactly the original per-score operation order.
extern "C" __global__ void mla_sparse_k6_scores(
    float* __restrict__ scores, const float* __restrict__ q,
    const float* __restrict__ qp, const unsigned char* __restrict__ c,
    const unsigned char* __restrict__ p, const int* __restrict__ indices,
    float scale, int dim, int rope, int capacity,
    int sequence_length, const int* __restrict__ sequence_pointer
) {
    const int h=blockIdx.x, tid=threadIdx.x, slot=blockIdx.y*256+tid;
    const int seq=sequence_pointer?*sequence_pointer:sequence_length;
    extern __shared__ float query[];
    for(int d=tid;d<dim;d+=256)query[d]=q[h*dim+d];
    for(int d=tid;d<rope;d+=256)query[dim+d]=qp[h*rope+d];
    __syncthreads();
    if(slot>=capacity)return;
    float score=-1e30f;
    if(slot<seq){int pos=indices[slot];if(pos>=0 && pos<seq){
        score=mla_dot_k6(c,pos,dim,query);
        if(rope)score+=mla_dot_k6(p,pos,rope,query+dim);
        score*=scale;
    }}
    scores[(long long)h*capacity+slot]=score;
}

extern "C" __global__ void mla_sparse_k6_scored_output(
    float* __restrict__ output, const float* __restrict__ scores,
    const unsigned char* __restrict__ ckv_cache,
    const int* __restrict__ selected_indices, const float* __restrict__ unpacked_values, int dim, int capacity,
    int sequence_length, const int* __restrict__ sequence_pointer
) {
    const int seq_len = sequence_pointer ? *sequence_pointer : sequence_length;
    const int selected_count = min(max(seq_len,0), capacity);
    const int kv_lora_rank = dim, num_heads = gridDim.x;
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (num_threads + warpSize - 1) / warpSize;

    extern __shared__ float smem[];
    float* smem_reduce = smem;
    float* smem_weights = smem_reduce + num_warps;
    if (selected_count <= 0) {
        float* out_head = output + h * kv_lora_rank;
        for (int d = tid; d < kv_lora_rank; d += num_threads) {
            out_head[d] = 0.0f;
        }
        return;
    }

    float local_max = -1e30f;
    for (int slot = tid; slot < selected_count; slot += num_threads) {
        int pos = selected_indices[slot];
        float score = -1e30f;
        if (pos >= 0 && pos < seq_len) {
            score = scores[(long long)h * capacity + slot];
        }
        if (selected_count <= MLA_TILE_SIZE) {
            smem_weights[slot] = score;
        }
        local_max = fmaxf(local_max, score);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float maximum = smem_reduce[0];
        for (int warp = 1; warp < num_warps; warp++) {
            maximum = fmaxf(maximum, smem_reduce[warp]);
        }
        smem_reduce[0] = maximum;
    }
    __syncthreads();
    float global_max = smem_reduce[0];

    float local_sum = 0.0f;
    float v_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int tile_start = 0; tile_start < selected_count; tile_start += MLA_TILE_SIZE) {
        int tile_end = min(tile_start + MLA_TILE_SIZE, selected_count);
        int tile_len = tile_end - tile_start;

        for (int ti = tid; ti < tile_len; ti += num_threads) {
            int pos = selected_indices[tile_start + ti];
            float weight = 0.0f;
            if (pos >= 0 && pos < seq_len) {
                float score;
                if (selected_count <= MLA_TILE_SIZE) {
                    score = smem_weights[tile_start + ti];
                } else {
                    score = scores[(long long)h * capacity + tile_start + ti];
                }
                weight = __expf(score - global_max);
            }
            smem_weights[ti] = weight;
            local_sum += weight;
        }
        __syncthreads();

        int di = 0;
        for (int d = tid; d < kv_lora_rank; d += num_threads) {
            float acc = 0.0f;
            for (int ti = 0; ti < tile_len; ti++) {
                int pos = selected_indices[tile_start + ti];
                if (pos >= 0 && pos < seq_len) {
                    acc += smem_weights[ti] *
                        (unpacked_values ? unpacked_values[(long long)(tile_start + ti) * dim + d] : mla_load_k6_value(ckv_cache, pos, kv_lora_rank, d));
                }
            }
            v_acc[di++] += acc;
        }
        __syncthreads();
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int warp = 0; warp < num_warps; warp++) {
            sum += smem_reduce[warp];
        }
        smem_reduce[0] = sum;
    }
    __syncthreads();
    float inv_sum = 1.0f / smem_reduce[0];

    float* out_head = output + h * kv_lora_rank;
    int di = 0;
    for (int d = tid; d < kv_lora_rank; d += num_threads) {
        out_head[d] = v_acc[di++] * inv_sum;
    }
}

// Unpack each selected value once. BF16 scale times a signed6-bit integer is
// exactly representable in FP32; the original weighted accumulation is retained.
extern "C" __global__ void mla_sparse_k6_unpack_values(
    float* __restrict__ values, const unsigned char* __restrict__ cache,
    const int* __restrict__ indices, int dim, int capacity,
    int sequence_length, const int* __restrict__ sequence_pointer
) {
    const int seq=sequence_pointer?*sequence_pointer:sequence_length;
    const int slot=blockIdx.x;
    const int pos=slot<min(seq,capacity)?indices[slot]:-1;
    for(int d=threadIdx.x;d<dim;d+=blockDim.x)
        values[(long long)slot*dim+d]=(pos>=0 && pos<seq)?mla_load_k6_value(cache,pos,dim,d):0.0f;
}
