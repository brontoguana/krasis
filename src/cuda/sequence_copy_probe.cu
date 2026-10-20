// Explicit queue-liveness control only. No model arithmetic or throughput claim.
extern "C" __global__ void retained_copy_queue_delay(unsigned long long cycles) {
 const unsigned long long start=clock64();
 while(clock64()-start<cycles) { asm volatile(""); }
}
