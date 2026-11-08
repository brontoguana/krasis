#pragma once

// CUDA 13.0/13.1 conflicts with glibc 2.42+ when _GNU_SOURCE enables the new
// C23 rsqrt declarations. Sidecar builds therefore disable _GNU_SOURCE before
// CUDA's injected headers are parsed. Recent libstdc++ configurations assume
// GNU pthread clock APIs are available, however, so their optional timed-wait
// paths must also be disabled for that translation unit.
#if defined(__linux__) && defined(__has_include)
#if __has_include(<bits/c++config.h>)
#include <bits/c++config.h>
#undef _GLIBCXX_USE_PTHREAD_COND_CLOCKWAIT
#undef _GLIBCXX_USE_PTHREAD_MUTEX_CLOCKLOCK
#endif
#endif
