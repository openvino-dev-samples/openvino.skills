# Qwen/Qwen3-8B (dense decoder, GQA, SwiGLU) — Per-token-size Kernel Tables

## Platform: BMG

### DECODE — per-kv-size (M=1)

#### BMG — decode kv=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0155 | 36 | 0.558 | 918 | 114.8 | 25.3% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0016 | 36 | 0.056 | 918 | 114.8 | 25.3% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0012 | 36 | 0.044 | 918 | 114.8 | 25.3% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **12.29 ms**

#### BMG — decode kv=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0295 | 36 | 1.063 | 1017 | 127.2 | 28.1% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0021 | 36 | 0.074 | 1017 | 127.2 | 28.1% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **12.77 ms**

#### BMG — decode kv=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.0272 | 36 | 0.980 | 2092 | 261.5 | 57.7% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0033 | 36 | 0.119 | 2092 | 261.5 | 57.7% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **12.73 ms**

#### BMG — decode kv=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.0613 | 36 | 2.208 | 1886 | 235.7 | 52.0% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0080 | 36 | 0.286 | 1886 | 235.7 | 52.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **14.13 ms**

#### BMG — decode kv=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.1577 | 36 | 5.678 | 1516 | 189.5 | 41.8% | memory |
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0172 | 36 | 0.618 | 1516 | 189.5 | 41.8% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **17.93 ms**

#### BMG — decode kv=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.2749 | 36 | 9.897 | 1707 | 213.4 | 47.1% | memory |
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0371 | 36 | 1.334 | 1707 | 213.4 | 47.1% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **22.87 ms**

#### BMG — decode kv=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.4957 | 36 | 17.846 | 1866 | 233.2 | 51.5% | memory |
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0773 | 36 | 2.782 | 1866 | 233.2 | 51.5% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **32.26 ms**

#### BMG — decode kv=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.9355 | 36 | 33.678 | 1951 | 243.8 | 53.8% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.1629 | 36 | 5.865 | 1951 | 243.8 | 53.8% | memory |
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **51.18 ms**

### PREFILL — per-S-size (compute-bound uses INT8 XMX peak = 233.5 TOPS)

#### BMG — prefill S=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.6982 | 36 | 25.136 | 126515 | 103.9 | 54.2% | compute |
| fc_up | `gemm_kernel` | 0.6599 | 36 | 23.756 | 147347 | 121.0 | 63.1% | compute |
| fc_gate | `gemm_kernel` | 0.6595 | 36 | 23.743 | 147438 | 121.1 | 63.2% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 0.4006 | 36 | 14.423 | 40561 | 79.2 | 17.4% | compute |
| fc_qkv | `gemm_kernel` | 0.3525 | 36 | 12.690 | 131245 | 118.5 | 56.2% | compute |
| fc_o | `gemm_kernel` | 0.2340 | 36 | 8.424 | 125642 | 123.7 | 53.8% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 0.1165 | 36 | 4.196 | 126515 | 103.9 | 54.2% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.0402 | 36 | 1.447 | 131245 | 118.5 | 56.2% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.0397 | 36 | 1.429 | 147347 | 121.0 | 63.1% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.0396 | 36 | 1.426 | 147438 | 121.1 | 63.2% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.0395 | 36 | 1.421 | 125642 | 123.7 | 53.8% | compute |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0229 | 36 | 0.825 | 40561 | 79.2 | 17.4% | compute |

**Total inference time (this stage)** ≈ **118.91 ms**

#### BMG — prefill S=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 1.2896 | 36 | 46.427 | 135421 | 77.7 | 58.0% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 1.2685 | 36 | 45.666 | 51600 | 50.4 | 22.1% | compute |
| fc_gate | `gemm_kernel` | 1.2268 | 36 | 44.166 | 158123 | 90.7 | 67.7% | compute |
| fc_up | `gemm_kernel` | 1.2256 | 36 | 44.123 | 158269 | 90.8 | 67.8% | compute |
| fc_qkv | `gemm_kernel` | 0.6180 | 36 | 22.249 | 148264 | 97.1 | 63.5% | compute |
| fc_o | `gemm_kernel` | 0.4204 | 36 | 15.133 | 138196 | 101.7 | 59.2% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 0.2327 | 36 | 8.377 | 135421 | 77.7 | 58.0% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.0772 | 36 | 2.779 | 148264 | 97.1 | 63.5% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.0769 | 36 | 2.770 | 158123 | 90.7 | 67.7% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.0769 | 36 | 2.770 | 158269 | 90.8 | 67.8% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.0769 | 36 | 2.769 | 138196 | 101.7 | 59.2% | compute |

**Total inference time (this stage)** ≈ **237.23 ms**

#### BMG — prefill S=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 4.7889 | 36 | 172.402 | 55841 | 27.3 | 23.9% | compute |
| fc_down | `gemm_kernel` | 2.4756 | 36 | 89.120 | 140574 | 63.2 | 60.2% | compute |
| fc_gate | `gemm_kernel` | 2.4572 | 36 | 88.459 | 157969 | 71.0 | 67.7% | compute |
| fc_up | `gemm_kernel` | 2.4306 | 36 | 87.502 | 159643 | 71.8 | 68.4% | compute |
| fc_qkv | `gemm_kernel` | 1.2139 | 36 | 43.700 | 150954 | 80.1 | 64.7% | compute |
| fc_o | `gemm_kernel` | 0.8093 | 36 | 29.133 | 143070 | 87.6 | 61.3% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 0.4575 | 36 | 16.471 | 140574 | 63.2 | 60.2% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.1529 | 36 | 5.506 | 157969 | 71.0 | 67.7% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.1521 | 36 | 5.477 | 159643 | 71.8 | 68.4% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.1518 | 36 | 5.466 | 150954 | 80.1 | 64.7% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.1514 | 36 | 5.450 | 143070 | 87.6 | 61.3% | compute |

**Total inference time (this stage)** ≈ **548.68 ms**

#### BMG — prefill S=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 18.4221 | 36 | 663.197 | 58941 | 14.4 | 25.2% | compute |
| fc_down | `gemm_kernel` | 4.8381 | 36 | 174.173 | 143513 | 55.6 | 61.5% | compute |
| fc_up | `gemm_kernel` | 4.7928 | 36 | 172.541 | 161850 | 62.7 | 69.3% | compute |
| fc_gate | `gemm_kernel` | 4.7758 | 36 | 171.929 | 162381 | 62.9 | 69.6% | compute |
| fc_qkv | `gemm_kernel` | 2.3868 | 36 | 85.925 | 153318 | 71.9 | 65.7% | compute |
| fc_o | `gemm_kernel` | 1.5883 | 36 | 57.180 | 145411 | 80.0 | 62.3% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 0.9079 | 36 | 32.685 | 143513 | 55.6 | 61.5% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.3026 | 36 | 10.892 | 162381 | 62.9 | 69.6% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.3025 | 36 | 10.890 | 153318 | 71.9 | 65.7% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.3023 | 36 | 10.881 | 161850 | 62.7 | 69.3% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.3020 | 36 | 10.872 | 145411 | 80.0 | 62.3% | compute |

**Total inference time (this stage)** ≈ **1401.17 ms**

#### BMG — prefill S=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 72.7683 | 36 | 2619.660 | 60061 | 7.3 | 25.7% | compute |
| fc_gate | `gemm_kernel` | 9.5848 | 36 | 345.053 | 161841 | 57.7 | 69.3% | compute |
| fc_down | `gemm_kernel` | 9.5400 | 36 | 343.439 | 145245 | 51.8 | 62.2% | compute |
| fc_up | `gemm_kernel` | 9.5379 | 36 | 343.365 | 162603 | 58.0 | 69.6% | compute |
| fc_qkv | `gemm_kernel` | 4.7293 | 36 | 170.254 | 154583 | 67.7 | 66.2% | compute |
| fc_o | `gemm_kernel` | 3.1405 | 36 | 113.060 | 146801 | 76.2 | 62.9% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 1.8151 | 36 | 65.344 | 145245 | 51.8 | 62.2% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.6058 | 36 | 21.811 | 161841 | 57.7 | 69.3% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.6053 | 36 | 21.790 | 154583 | 67.7 | 66.2% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.6050 | 36 | 21.779 | 162603 | 58.0 | 69.6% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.6044 | 36 | 21.757 | 146801 | 76.2 | 62.9% | compute |

**Total inference time (this stage)** ≈ **4087.31 ms**

#### BMG — prefill S=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 302.6424 | 36 | 10895.125 | 57950 | 3.5 | 24.8% | compute |
| fc_down | `gemm_kernel` | 19.0891 | 36 | 687.208 | 145112 | 49.5 | 62.2% | compute |
| fc_up | `gemm_kernel` | 19.0827 | 36 | 686.976 | 162495 | 55.4 | 69.6% | compute |
| fc_gate | `gemm_kernel` | 19.0444 | 36 | 685.597 | 162847 | 55.5 | 69.7% | compute |
| fc_qkv | `gemm_kernel` | 9.4680 | 36 | 340.849 | 154470 | 65.2 | 66.2% | compute |
| fc_o | `gemm_kernel` | 6.2401 | 36 | 224.642 | 147659 | 74.4 | 63.2% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 3.6418 | 36 | 131.104 | 145112 | 49.5 | 62.2% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 1.2167 | 36 | 43.801 | 162495 | 55.4 | 69.6% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 1.2111 | 36 | 43.600 | 162847 | 55.5 | 69.7% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 1.2089 | 36 | 43.520 | 154470 | 65.2 | 66.2% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 1.2062 | 36 | 43.424 | 147659 | 74.4 | 63.2% | compute |

**Total inference time (this stage)** ≈ **13825.85 ms**

## Platform: PTL

### DECODE — per-kv-size (M=1)

#### PTL — decode kv=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_up | `gemm_kernel` | 0.2559 | 36 | 9.211 | 393 | 102.3 | 97.5% | memory |
| fc_down | `gemm_kernel` | 0.2555 | 36 | 9.197 | 394 | 102.5 | 97.6% | memory |
| fc_gate | `gemm_kernel` | 0.2552 | 36 | 9.187 | 394 | 102.6 | 97.7% | memory |
| lm_head | `gemm_kernel` | 6.0300 | 1 | 6.030 | 206 | 104.9 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.652 | 390 | 101.3 | 96.5% | memory |
| fc_o | `gemm_kernel` | 0.0873 | 36 | 3.141 | 385 | 100.1 | 95.3% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0511 | 36 | 1.838 | 289 | 36.1 | 34.4% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.225 | 10 | 5.2 | 5.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.139 | 2 | 12.7 | 12.1% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0039 | 36 | 0.139 | 289 | 36.1 | 34.4% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0031 | 36 | 0.112 | 289 | 36.1 | 34.4% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.104 | 4 | 25.6 | 24.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.087 | 14 | 6.8 | 6.5% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0021 | 36 | 0.077 | 19 | 7.7 | 7.3% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.8 | 26.5% | memory |

**Total inference time (this stage)** ≈ **44.36 ms**

#### PTL — decode kv=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_up | `gemm_kernel` | 0.2559 | 36 | 9.211 | 393 | 102.3 | 97.5% | memory |
| fc_down | `gemm_kernel` | 0.2555 | 36 | 9.197 | 394 | 102.5 | 97.6% | memory |
| fc_gate | `gemm_kernel` | 0.2552 | 36 | 9.187 | 394 | 102.6 | 97.7% | memory |
| lm_head | `gemm_kernel` | 6.0300 | 1 | 6.030 | 206 | 104.9 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.652 | 390 | 101.3 | 96.5% | memory |
| fc_o | `gemm_kernel` | 0.0873 | 36 | 3.141 | 385 | 100.1 | 95.3% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0843 | 36 | 3.033 | 365 | 45.7 | 43.5% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.225 | 10 | 5.2 | 5.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.139 | 2 | 12.7 | 12.1% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0038 | 36 | 0.137 | 365 | 45.7 | 43.5% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.104 | 4 | 25.6 | 24.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.087 | 14 | 6.8 | 6.5% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0021 | 36 | 0.077 | 19 | 7.7 | 7.3% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.8 | 26.5% | memory |

**Total inference time (this stage)** ≈ **45.44 ms**

#### PTL — decode kv=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_up | `gemm_kernel` | 0.2559 | 36 | 9.211 | 393 | 102.3 | 97.5% | memory |
| fc_down | `gemm_kernel` | 0.2555 | 36 | 9.197 | 394 | 102.5 | 97.6% | memory |
| fc_gate | `gemm_kernel` | 0.2552 | 36 | 9.187 | 394 | 102.6 | 97.7% | memory |
| lm_head | `gemm_kernel` | 6.0300 | 1 | 6.030 | 206 | 104.9 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.652 | 390 | 101.3 | 96.5% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.1003 | 36 | 3.611 | 613 | 76.6 | 72.9% | memory |
| fc_o | `gemm_kernel` | 0.0873 | 36 | 3.141 | 385 | 100.1 | 95.3% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.225 | 10 | 5.2 | 5.0% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0054 | 36 | 0.193 | 613 | 76.6 | 72.9% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.139 | 2 | 12.7 | 12.1% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.104 | 4 | 25.6 | 24.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.087 | 14 | 6.8 | 6.5% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0021 | 36 | 0.077 | 19 | 7.7 | 7.3% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.8 | 26.5% | memory |

**Total inference time (this stage)** ≈ **46.07 ms**

#### PTL — decode kv=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_up | `gemm_kernel` | 0.2559 | 36 | 9.211 | 393 | 102.3 | 97.5% | memory |
| fc_down | `gemm_kernel` | 0.2555 | 36 | 9.197 | 394 | 102.5 | 97.6% | memory |
| fc_gate | `gemm_kernel` | 0.2552 | 36 | 9.187 | 394 | 102.6 | 97.7% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.1933 | 36 | 6.961 | 644 | 80.5 | 76.7% | memory |
| lm_head | `gemm_kernel` | 6.0300 | 1 | 6.030 | 206 | 104.9 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.652 | 390 | 101.3 | 96.5% | memory |
| fc_o | `gemm_kernel` | 0.0873 | 36 | 3.141 | 385 | 100.1 | 95.3% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0110 | 36 | 0.397 | 644 | 80.5 | 76.7% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.225 | 10 | 5.2 | 5.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.139 | 2 | 12.7 | 12.1% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.104 | 4 | 25.6 | 24.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.087 | 14 | 6.8 | 6.5% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0021 | 36 | 0.077 | 19 | 7.7 | 7.3% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.8 | 26.5% | memory |

**Total inference time (this stage)** ≈ **49.63 ms**

#### PTL — decode kv=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.3773 | 36 | 13.583 | 660 | 82.5 | 78.6% | memory |
| fc_up | `gemm_kernel` | 0.2559 | 36 | 9.211 | 393 | 102.3 | 97.5% | memory |
| fc_down | `gemm_kernel` | 0.2555 | 36 | 9.197 | 394 | 102.5 | 97.6% | memory |
| fc_gate | `gemm_kernel` | 0.2552 | 36 | 9.187 | 394 | 102.6 | 97.7% | memory |
| lm_head | `gemm_kernel` | 6.0300 | 1 | 6.030 | 206 | 104.9 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.652 | 390 | 101.3 | 96.5% | memory |
| fc_o | `gemm_kernel` | 0.0873 | 36 | 3.141 | 385 | 100.1 | 95.3% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0238 | 36 | 0.858 | 660 | 82.5 | 78.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.225 | 10 | 5.2 | 5.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.139 | 2 | 12.7 | 12.1% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.104 | 4 | 25.6 | 24.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.087 | 14 | 6.8 | 6.5% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0021 | 36 | 0.077 | 19 | 7.7 | 7.3% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.8 | 26.5% | memory |

**Total inference time (this stage)** ≈ **56.71 ms**

#### PTL — decode kv=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.7529 | 36 | 27.104 | 661 | 82.6 | 78.7% | memory |
| fc_up | `gemm_kernel` | 0.2559 | 36 | 9.211 | 393 | 102.3 | 97.5% | memory |
| fc_down | `gemm_kernel` | 0.2555 | 36 | 9.197 | 394 | 102.5 | 97.6% | memory |
| fc_gate | `gemm_kernel` | 0.2552 | 36 | 9.187 | 394 | 102.6 | 97.7% | memory |
| lm_head | `gemm_kernel` | 6.0300 | 1 | 6.030 | 206 | 104.9 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.652 | 390 | 101.3 | 96.5% | memory |
| fc_o | `gemm_kernel` | 0.0873 | 36 | 3.141 | 385 | 100.1 | 95.3% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0511 | 36 | 1.840 | 661 | 82.6 | 78.7% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.225 | 10 | 5.2 | 5.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.139 | 2 | 12.7 | 12.1% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.104 | 4 | 25.6 | 24.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.087 | 14 | 6.8 | 6.5% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0021 | 36 | 0.077 | 19 | 7.7 | 7.3% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.8 | 26.5% | memory |

**Total inference time (this stage)** ≈ **71.21 ms**

#### PTL — decode kv=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 1.5092 | 36 | 54.330 | 662 | 82.7 | 78.8% | memory |
| fc_up | `gemm_kernel` | 0.2559 | 36 | 9.211 | 393 | 102.3 | 97.5% | memory |
| fc_down | `gemm_kernel` | 0.2555 | 36 | 9.197 | 394 | 102.5 | 97.6% | memory |
| fc_gate | `gemm_kernel` | 0.2552 | 36 | 9.187 | 394 | 102.6 | 97.7% | memory |
| lm_head | `gemm_kernel` | 6.0300 | 1 | 6.030 | 206 | 104.9 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.652 | 390 | 101.3 | 96.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.1050 | 36 | 3.779 | 662 | 82.7 | 78.8% | memory |
| fc_o | `gemm_kernel` | 0.0873 | 36 | 3.141 | 385 | 100.1 | 95.3% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.225 | 10 | 5.2 | 5.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.139 | 2 | 12.7 | 12.1% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.104 | 4 | 25.6 | 24.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.087 | 14 | 6.8 | 6.5% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0021 | 36 | 0.077 | 19 | 7.7 | 7.3% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.8 | 26.5% | memory |

**Total inference time (this stage)** ≈ **100.38 ms**

#### PTL — decode kv=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 3.0534 | 36 | 109.921 | 660 | 82.5 | 78.5% | memory |
| fc_up | `gemm_kernel` | 0.2559 | 36 | 9.211 | 393 | 102.3 | 97.5% | memory |
| fc_down | `gemm_kernel` | 0.2555 | 36 | 9.197 | 394 | 102.5 | 97.6% | memory |
| fc_gate | `gemm_kernel` | 0.2552 | 36 | 9.187 | 394 | 102.6 | 97.7% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.1949 | 36 | 7.017 | 660 | 82.5 | 78.5% | memory |
| lm_head | `gemm_kernel` | 6.0300 | 1 | 6.030 | 206 | 104.9 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.652 | 390 | 101.3 | 96.5% | memory |
| fc_o | `gemm_kernel` | 0.0873 | 36 | 3.141 | 385 | 100.1 | 95.3% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.225 | 10 | 5.2 | 5.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.139 | 2 | 12.7 | 12.1% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.104 | 4 | 25.6 | 24.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.087 | 14 | 6.8 | 6.5% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0021 | 36 | 0.077 | 19 | 7.7 | 7.3% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.8 | 26.5% | memory |

**Total inference time (this stage)** ≈ **159.21 ms**

### PREFILL — per-S-size (compute-bound uses INT8 XMX peak = 118.0 TOPS)

#### PTL — prefill S=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_gate | `gemm_kernel` | 1.5820 | 36 | 56.953 | 60723 | 49.9 | 51.5% | compute |
| fc_down | `gemm_kernel` | 1.5670 | 36 | 56.412 | 54162 | 44.5 | 45.9% | compute |
| fc_up | `gemm_kernel` | 1.4895 | 36 | 53.621 | 64865 | 53.3 | 55.0% | compute |
| fc_qkv | `gemm_kernel` | 0.7302 | 36 | 26.288 | 62099 | 56.1 | 52.6% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 0.6196 | 36 | 22.307 | 25364 | 49.5 | 21.5% | compute |
| fc_o | `gemm_kernel` | 0.5026 | 36 | 18.094 | 56804 | 55.9 | 48.2% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 0.3362 | 36 | 12.102 | 54162 | 44.5 | 45.9% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.1155 | 36 | 4.158 | 60723 | 49.9 | 51.5% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.1023 | 36 | 3.682 | 56804 | 55.9 | 48.2% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.0997 | 36 | 3.590 | 62099 | 56.1 | 52.6% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.0996 | 36 | 3.587 | 64865 | 53.3 | 55.0% | compute |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0577 | 36 | 2.077 | 25364 | 49.5 | 21.5% | compute |

**Total inference time (this stage)** ≈ **262.87 ms**

#### PTL — prefill S=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 2.9729 | 36 | 107.026 | 58029 | 33.3 | 49.2% | compute |
| fc_gate | `gemm_kernel` | 2.8141 | 36 | 101.307 | 68513 | 39.3 | 58.1% | compute |
| fc_up | `gemm_kernel` | 2.8007 | 36 | 100.826 | 68800 | 39.5 | 58.3% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 2.2826 | 36 | 82.173 | 28615 | 27.9 | 24.3% | compute |
| fc_qkv | `gemm_kernel` | 1.4281 | 36 | 51.412 | 62346 | 40.8 | 52.9% | compute |
| fc_o | `gemm_kernel` | 0.9868 | 36 | 35.524 | 56732 | 41.8 | 48.1% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 0.5797 | 36 | 20.871 | 58029 | 33.3 | 49.2% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.2252 | 36 | 8.108 | 62346 | 40.8 | 52.9% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.2245 | 36 | 8.083 | 56732 | 41.8 | 48.1% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.1958 | 36 | 7.048 | 68800 | 39.5 | 58.3% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.1950 | 36 | 7.019 | 68513 | 39.3 | 58.1% | compute |

**Total inference time (this stage)** ≈ **529.40 ms**

#### PTL — prefill S=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 8.8177 | 36 | 317.437 | 30378 | 14.8 | 25.8% | compute |
| fc_down | `gemm_kernel` | 5.7702 | 36 | 207.728 | 59250 | 26.6 | 50.2% | compute |
| fc_gate | `gemm_kernel` | 5.5674 | 36 | 200.426 | 69104 | 31.1 | 58.6% | compute |
| fc_up | `gemm_kernel` | 5.4587 | 36 | 196.514 | 70524 | 31.7 | 59.8% | compute |
| fc_qkv | `gemm_kernel` | 2.7573 | 36 | 99.262 | 65633 | 34.8 | 55.6% | compute |
| fc_o | `gemm_kernel` | 1.8688 | 36 | 67.278 | 60924 | 37.3 | 51.6% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 1.1887 | 36 | 42.793 | 59250 | 26.6 | 50.2% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.3992 | 36 | 14.371 | 69104 | 31.1 | 58.6% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.3877 | 36 | 13.958 | 70524 | 31.7 | 59.8% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.3871 | 36 | 13.935 | 60924 | 37.3 | 51.6% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.3838 | 36 | 13.817 | 65633 | 34.8 | 55.6% | compute |

**Total inference time (this stage)** ≈ **1187.52 ms**

#### PTL — prefill S=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 34.8375 | 36 | 1254.150 | 31097 | 7.6 | 26.4% | compute |
| fc_down | `gemm_kernel` | 11.4309 | 36 | 411.512 | 57475 | 22.3 | 48.7% | compute |
| fc_up | `gemm_kernel` | 11.0415 | 36 | 397.495 | 69336 | 26.9 | 58.8% | compute |
| fc_gate | `gemm_kernel` | 11.0064 | 36 | 396.232 | 69573 | 27.0 | 59.0% | compute |
| fc_qkv | `gemm_kernel` | 5.4634 | 36 | 196.682 | 66134 | 31.0 | 56.1% | compute |
| fc_o | `gemm_kernel` | 3.7608 | 36 | 135.387 | 59888 | 33.0 | 50.8% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 2.9169 | 36 | 105.009 | 57475 | 22.3 | 48.7% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.8517 | 36 | 30.662 | 69336 | 26.9 | 58.8% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.8463 | 36 | 30.468 | 69573 | 27.0 | 59.0% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.8291 | 36 | 29.848 | 59888 | 33.0 | 50.8% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.7712 | 36 | 27.764 | 66134 | 31.0 | 56.1% | compute |

**Total inference time (this stage)** ≈ **3015.21 ms**

#### PTL — prefill S=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 138.5833 | 36 | 4988.998 | 31501 | 3.8 | 26.7% | compute |
| fc_down | `gemm_kernel` | 22.5891 | 36 | 813.206 | 58775 | 21.0 | 49.8% | compute |
| fc_gate | `gemm_kernel` | 22.1070 | 36 | 795.851 | 68562 | 24.4 | 58.1% | compute |
| fc_up | `gemm_kernel` | 21.6913 | 36 | 780.887 | 69883 | 24.9 | 59.2% | compute |
| fc_qkv | `gemm_kernel` | 11.2362 | 36 | 404.502 | 62624 | 27.4 | 53.1% | compute |
| fc_o | `gemm_kernel` | 7.4059 | 36 | 266.611 | 58148 | 30.2 | 49.3% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 5.4717 | 36 | 196.981 | 58775 | 21.0 | 49.8% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 2.0486 | 36 | 73.749 | 58148 | 30.2 | 49.3% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 1.9482 | 36 | 70.136 | 68562 | 24.4 | 58.1% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 1.9319 | 36 | 69.547 | 62624 | 27.4 | 53.1% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 1.9091 | 36 | 68.726 | 69883 | 24.9 | 59.2% | compute |

**Total inference time (this stage)** ≈ **8529.20 ms**

#### PTL — prefill S=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 671.5162 | 36 | 24174.582 | 26116 | 1.6 | 22.1% | compute |
| fc_up | `gemm_kernel` | 55.7676 | 36 | 2007.634 | 55119 | 18.8 | 46.7% | compute |
| fc_down | `gemm_kernel` | 46.0086 | 36 | 1656.309 | 58286 | 19.9 | 49.4% | compute |
| fc_gate | `gemm_kernel` | 43.9443 | 36 | 1581.995 | 69099 | 23.6 | 58.6% | compute |
| fc_qkv | `gemm_kernel` | 21.6273 | 36 | 778.583 | 65004 | 27.5 | 55.1% | compute |
| fc_o | `gemm_kernel` | 14.3138 | 36 | 515.297 | 60882 | 30.7 | 51.6% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 10.5836 | 36 | 381.010 | 58286 | 19.9 | 49.4% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 4.0767 | 36 | 146.761 | 55119 | 18.8 | 46.7% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 3.7922 | 36 | 136.521 | 69099 | 23.6 | 58.6% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 3.7458 | 36 | 134.850 | 60882 | 30.7 | 51.6% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 3.7446 | 36 | 134.807 | 65004 | 27.5 | 55.1% | compute |

**Total inference time (this stage)** ≈ **31648.35 ms**

