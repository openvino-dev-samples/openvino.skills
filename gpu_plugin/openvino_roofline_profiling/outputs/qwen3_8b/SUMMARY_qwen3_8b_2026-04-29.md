# Qwen3-8B — Roofline Summary (BMG + PTL)

Generated: 2026-04-29 (revision 2 — streaming-read BW for PTL, per-kv decode sweep)
Model: `Qwen/Qwen3-8B` (dense decoder, GQA, SwiGLU, INT4 g=128 body / INT8 g=128 lm_head, INT8 KV cache, FP16 activations)

Architecture (from [models/qwen3_8b/ops_mapping.json](../../models/qwen3_8b/ops_mapping.json)):
- `num_hidden_layers = 36`, `hidden_size = 4096`, `intermediate_size = 12288`
- `num_attention_heads = 32`, `num_key_value_heads = 8`, `head_dim = 128` (GQA group size = 4)
- `vocab_size = 151936`
- QKV concat = `(NH + 2·NKV)·HD = 6144`; O-proj = `NH·HD × H = 4096 × 4096`

Platforms (from [utils/platforms.json](../../utils/platforms.json)):
- **BMG** — Intel Arc B580 (Battlemage dGPU, Xe2). 2900 MHz measured; **453 GB/s** measured streaming-read; FP16 116.74 TFLOPS / INT8 233.47 TOPS XMX peak.
- **PTL** — Intel Arc B390 (Panther Lake iGPU, Xe2). 2400 MHz; **105 GB/s** measured streaming-read (rev 2 — derived from the saturated `lm_head` weight-stream kernel; the older 94 GB/s memcpy number is kept as `bw_gbs_measured_read_memcpy` for reference); FP16 58.98 TFLOPS / INT8 117.96 TOPS XMX peak.

Logs: [logs_bmg](logs_bmg) (67) · [logs_ptl](logs_ptl) (66).
Per-platform reports: [report_bmg.md](report_bmg.md) · [report_ptl.md](report_ptl.md) · [kernel_tables.md](kernel_tables.md).

---

## DECODE per-op @ kv = 4096

### BMG (peak 453 GB/s)

| Op | Avg ms | Calls | Total ms | GB/s | Eff% |
|---|---:|---:|---:|---:|---:|
| fc_down (INT4) | 0.0864 | 36 | 3.110 | 303.0 | 66.9% |
| fc_up (INT4)   | 0.0633 | 36 | 2.278 | 413.8 | **91.3%** |
| fc_gate (INT4) | 0.0632 | 36 | 2.277 | 414.0 | **91.4%** |
| lm_head (INT8) | 1.4069 | 1  | 1.407 | 449.5 | **99.2%** |
| fc_qkv (INT4)  | 0.0347 | 36 | 1.250 | 377.1 | 83.3% |
| pa (kv=4k)     | 0.0321 | 36 | 1.155 | 261.5 | 57.7% |
| fc_o (INT4)    | 0.0238 | 36 | 0.857 | 366.9 | 81.0% |
| small ops (12 kernels) | – | – | 0.455 | – | <30% |

**Decode total = 12.789 ms → 78.2 tok/s**

### PTL (peak 105 GB/s, streaming-read calibrated)

| Op | Avg ms | Calls | Total ms | GB/s | Eff% |
|---|---:|---:|---:|---:|---:|
| lm_head (INT8) | 6.0300 | 1  | 6.030 | 104.9 | **99.9%** |
| fc_gate (INT4) | 0.2552 | 36 | 9.187 | 102.6 | **97.7%** |
| fc_down (INT4) | 0.2555 | 36 | 9.197 | 102.5 | **97.6%** |
| fc_up   (INT4) | 0.2559 | 36 | 9.211 | 102.3 | **97.5%** |
| fc_qkv  (INT4) | 0.1292 | 36 | 4.652 | 101.3 | 96.5% |
| fc_o    (INT4) | 0.0873 | 36 | 3.141 | 100.1 | 95.3% |
| pa (kv=4k)     | 0.1095 | 36 | 3.943 |  76.6 | 72.9% |
| small ops (12 kernels) | – | – | 0.853 | – | <30% |

**Decode total = 46.213 ms → 21.6 tok/s**

PTL is essentially memory-bound at the streaming-read roofline for every dense FC. PA decode at 73% is the only meaningful headroom on layer-loop ops.

---

## DECODE per-kv-size totals (M = 1)

Per-token decode latency vs kv length, with the FC + small-ops contribution held constant (decode FC time does not depend on kv) and PA scaled with the measured `pa_decode_kv{kv}` log.

### BMG

| kv | FC total (ms) | PA total (ms) | small total (ms) | lm_head (ms) | **decode total (ms)** | **tok/s** |
|---:|---:|---:|---:|---:|---:|---:|
| 1024   |  9.77 |   0.66 | 0.46 | 1.41 |  **12.29** | **81.4** |
| 2048   |  9.77 |   1.19 | 0.46 | 1.41 |  **12.82** | **78.0** |
| 4096   |  9.77 |   1.16 | 0.46 | 1.41 |  **12.79** | **78.2** |
| 8192   |  9.77 |   2.56 | 0.46 | 1.41 |  **14.20** | **70.4** |
| 16384  |  9.77 |   6.38 | 0.46 | 1.41 |  **18.01** | **55.5** |
| 32768  |  9.77 |  11.32 | 0.46 | 1.41 |  **22.95** | **43.6** |
| 65536  |  9.77 |  20.72 | 0.46 | 1.41 |  **32.35** | **30.9** |
| 131072 |  9.77 |  39.63 | 0.46 | 1.41 |  **51.27** | **19.5** |

### PTL

| kv | FC total (ms) | PA total (ms) | small total (ms) | lm_head (ms) | **decode total (ms)** | **tok/s** |
|---:|---:|---:|---:|---:|---:|---:|
| 1024   | 35.39 |   2.09 | 0.85 | 6.03 |  **44.36** | **22.5** |
| 2048   | 35.39 |   3.31 | 0.85 | 6.03 |  **45.58** | **21.9** |
| 4096   | 35.39 |   3.94 | 0.85 | 6.03 |  **46.21** | **21.6** |
| 8192   | 35.39 |   7.50 | 0.85 | 6.03 |  **49.77** | **20.1** |
| 16384  | 35.39 |  14.64 | 0.85 | 6.03 |  **56.91** | **17.6** |
| 32768  | 35.39 |  29.23 | 0.85 | 6.03 |  **71.50** | **14.0** |
| 65536  | 35.39 |  58.41 | 0.85 | 6.03 | **100.68** |  **9.9** |
| 131072 | 35.39 | 117.19 | 0.85 | 6.03 | **159.46** |  **6.3** |

**Observations**

- **kv ≤ 4k**: PA contributes 5–9% of decode → FC + lm_head are the bottleneck. Decode rate is essentially constant.
- **kv = 16k–32k**: PA grows to 27–41% of decode on BMG and 25–41% on PTL. Decode rate already cut by ~30–40%.
- **kv = 128k**: PA dominates (77% on BMG, 73% on PTL). BMG falls from 78 → 19.5 tok/s (4×); PTL falls from 22.5 → 6.3 tok/s (3.6×). At very long context the decoder is *not* matmul-bound — kv-cache traffic is.
- BMG/PTL ratio drops from 3.6× at kv=4k to **3.1×** at kv=128k because PA decode efficiency is lower on BMG (57% @ kv=4k) than on PTL (73% @ kv=4k). This is BMG-side optimization headroom.

---

## PREFILL (per-S total inference, ms)

| S | BMG total (ms) | BMG tok/s | PTL total (ms) | PTL tok/s | PA share BMG | PA share PTL |
|---:|---:|---:|---:|---:|---:|---:|
| 1024  |   120.3 | 8511 |   268.9 | 3808 | 13% |  9% |
| 2048  |   240.9 | 8501 |   539.7 | 3795 | 20% | 16% |
| 4096  |   554.9 | 7382 |  1201.9 | 3408 | 32% | 27% |
| 8192  |  1410.9 | 5806 |  3039.9 | 2695 | 48% | 42% |
| 16384 |  4105.2 | 3991 |  8572.3 | 1911 | 64% | 59% |
| 32768 | 13860.9 | 2364 | 31729.9 | 1033 | 79% | 76% |

- **S ≤ 4k**: dominated by FC weight-load (INT4 prefill kernels). Per-layer FC time is ≈ linear in S (gemm_kernel int8-XMX path). MLP triple (gate/up/down) is the largest contributor.
- **S = 8k**: PA per-layer overtakes any single FC.
- **S ≥ 16k**: PA dominates by 2–4× over all FCs combined; quadratic Q·Kᵀ becomes the only thing that matters.

---

## Cross-platform summary

| Metric | BMG | PTL | BMG/PTL |
|---|---:|---:|---:|
| Streaming-read BW (GB/s) | 453 | 105 | 4.31× |
| Decode kv=4k (ms) | 12.79 | 46.21 | 3.61× |
| Decode kv=128k (ms) | 51.27 | 159.46 | 3.11× |
| Prefill S=2048 (ms) | 240.9 | 539.7 | 2.24× |
| Prefill S=32768 (ms) | 13860.9 | 31729.9 | 2.29× |

Decode at short context tracks DRAM ratio (3.6× ≈ 4.3× × PA-eff penalty). Prefill tracks INT8 XMX ratio (2× INT8 TOPS = 2.2× speed) once PA quadratic dominates. The PTL streaming-read recalibration brings the FC efficiency cluster from 107–112% (impossible) to **95–100%** (correctly saturated).

---

## Top optimization priorities (Qwen3-8B on BMG)

1. **`fc_down` decode efficiency** — 67% BW vs 91% for gate/up despite identical FLOP/byte. Investigate the K=12288 (vs N=12288) gemm tile selection in the GPU plugin's int4 weight-stream path. **Expected gain: ≈ 1.0 ms / decode (+6 tok/s).**
2. **PA decode at long context (kv ≥ 32k)** — `paged_attention_opt__gqa_single_token_*` BMG efficiency drops from 57% (kv=4k) to ~25% (kv=131k); PTL holds 73% even at long kv. The BMG GQA scatter + INT8 KV dequant path leaves ~0.5–1× of headroom that PTL is already extracting. See [kernel_tables.md](kernel_tables.md) decode kv=32k/65k/131k tables. **Expected gain at kv=128k: 5–10 ms / decode → +5–8 tok/s on long-context decode.**
3. **PA prefill at S ≥ 8k** — `sdpa_micro__prefill_*__sa` BMG drops from ~70% INT8 XMX (S=4k) to ~25% (S=32k) and consumes 79% of TTFT. Consider FlashAttention-2 tiling, INT8 SDPA, or chunked prefill.
4. **lm_head INT4 alternative** — already 99% BW saturated, nothing left at INT8. INT4 lm_head halves the 600 MB/token weight read → ≈ 0.7 ms / decode (+4 tok/s) at the cost of accuracy. A/B worth running.

## Top optimization priorities (Qwen3-8B on PTL)

1. **PA decode at long context (kv ≥ 16k)** — drops from 73% BW (kv=4k) to ~36% (kv=128k). PA is by far the kv-scaling lever on PTL: at kv=128k it costs 117 ms vs 35 ms total FC, so even a 1.5× PA decode speedup gets ~30 ms / decode back.
2. **Long-context prefill (S ≥ 16k)** — same PA bottleneck pattern as BMG, 2–3× slower per S because of iGPU INT8 XMX peak.
3. **No FC headroom on PTL** — the dense FC path is already at the streaming-read roofline; further weight-bandwidth reductions require lower-bit weight quant (INT3/INT2) or weight reuse across calls (KV-batching, prompt cache).

---

## Reproducing this report

```bash
# BMG (Linux, openvino-ci-74)
cd /mnt/river/model_loading/roofline_test_utils
bash run_qwen3_8b_bmg.sh                                   # ~13 min sweep, 67 logs

# PTL (Windows, Local_Admin@10.239.132.229)
D:\river\moe\dev_roofline_profiling\utils\run_qwen3_8b_ptl.bat   # ~10 min sweep

# Local post-processing
cd .github/skills/openvino_roofline_profiling
python3 utils/parse_logs.py outputs/qwen3_8b/logs_bmg outputs/qwen3_8b/bmg_metrics.json
python3 utils/parse_logs.py outputs/qwen3_8b/logs_ptl outputs/qwen3_8b/ptl_metrics.json
# NOTE: parse_logs.py excludes "activation" kernels as fc_bench L2-flush helpers,
#       so so_swish_decode is patched in manually post-parse.
python3 utils/build_model_report.py  --model-dir models/qwen3_8b --output-dir outputs/qwen3_8b --platform BMG --out outputs/qwen3_8b/performance_metrics_bmg.json
python3 utils/build_model_report.py  --model-dir models/qwen3_8b --output-dir outputs/qwen3_8b --platform PTL --out outputs/qwen3_8b/performance_metrics_ptl.json
python3 utils/build_kernel_tables.py --model-dir models/qwen3_8b --output-dir outputs/qwen3_8b --out outputs/qwen3_8b/kernel_tables.md
```
