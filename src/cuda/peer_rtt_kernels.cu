// Phase-0 transport feasibility kernel for peer-expert serving.
//
// A single persistent block on the peer GPU polls a sequence word in portable
// mapped host memory, copies one request payload to the response mailbox, and
// publishes completion with system scope.  The host-side Rust benchmark puts
// the request word behind the primary GPU's D2H on the same CUDA stream, so the
// measured interval contains the real no-P2P request and response path without
// a per-iteration peer launch.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void peer_mailbox_round_trip(
    volatile uint32_t* control,
    const uint32_t* request,
    uint32_t* response,
    uint32_t message_words,
    uint32_t iterations) {
    for (uint32_t sequence = 1; sequence <= iterations; ++sequence) {
        if (threadIdx.x == 0) {
            while (control[0] != sequence) {
                __nanosleep(32);
            }
        }
        __syncthreads();

        for (uint32_t index = threadIdx.x; index < message_words;
             index += blockDim.x) {
            response[index] = request[index];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            __threadfence_system();
            control[1] = sequence;
        }
        __syncthreads();
    }
}

// Graph-compatible primary-side ordering node for devices/drivers which do
// not advertise CUDA stream memory operations.  The request D2H precedes this
// kernel on the primary stream; the response H2D follows it.  Capturing this
// node therefore provides the required cross-device dependency without a host
// callback or a Python synchronization point.
extern "C" __global__ void primary_mailbox_publish_and_wait(
    volatile uint32_t* control,
    uint32_t sequence) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        control[0] = sequence;
        __threadfence_system();
        while (control[1] != sequence) {
            __nanosleep(32);
        }
    }
}
