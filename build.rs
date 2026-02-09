fn main() {
    // Link libnuma for NUMA-aware memory allocation.
    // If libnuma is not present, numa_available() will return -1 at runtime.
    println!("cargo:rustc-link-lib=numa");
}
