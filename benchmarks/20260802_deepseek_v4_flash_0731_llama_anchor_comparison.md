# DeepSeek-V4-Flash-0731 independent llama.cpp quality anchor

Exact-token CPU-only diagnostic; no runtime source changed. Values reconstructed from 4-decimal cumulative PPL rows.

| Window | Scored targets | llama.cpp | Scalar | abs diff | Ordinary GEMM | abs diff | Best GEMM | abs diff | Closest |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2047 | 2.8605 | 3.1127 | 0.2522 | 3.1664 | 0.3059 | 3.1347 | 0.2742 | scalar |
| 2 | 1024 | 1.7565 | 2.2163 | 0.4598 | 2.2265 | 0.4700 | 2.2111 | 0.4546 | best_gemm |
| 3 | 1024 | 1.9476 | 2.6156 | 0.6680 | 2.5787 | 0.6311 | 2.6491 | 0.7015 | ordinary_gemm |
| 4 | 1024 | 1.9779 | 2.5040 | 0.5261 | 2.4985 | 0.5206 | 2.5469 | 0.5690 | ordinary_gemm |
| 5 | 1024 | 3.6411 | 4.1037 | 0.4625 | 4.1582 | 0.5171 | 4.0714 | 0.4303 | best_gemm |
| 6 | 1024 | 5.1402 | 5.3073 | 0.1671 | 5.4369 | 0.2967 | 5.4265 | 0.2864 | scalar |
| 7 | 1024 | 4.7052 | 4.8795 | 0.1742 | 4.9741 | 0.2689 | 4.9089 | 0.2037 | scalar |
| 8 | 1024 | 6.3695 | 6.2935 | 0.0759 | 6.3778 | 0.0084 | 6.2887 | 0.0808 | ordinary_gemm |
| 9 | 1024 | 2.7802 | 3.0964 | 0.3162 | 3.0545 | 0.2743 | 3.0854 | 0.3052 | ordinary_gemm |
| 10 | 1024 | 3.7659 | 3.7536 | 0.0123 | 3.7212 | 0.0447 | 3.7183 | 0.0476 | scalar |

Aggregate over 11263 scored targets: llama.cpp 3.1703; scalar 3.5398 (abs diff 0.3695); ordinary GEMM 3.5624 (abs diff 0.3921); best GEMM 3.5553 (abs diff 0.3850).

Classification: ambiguous. Scalar is closest in aggregate, but llama.cpp is far below all three and per-window winners are mixed. Do not use this result alone to promote either GEMM path.
