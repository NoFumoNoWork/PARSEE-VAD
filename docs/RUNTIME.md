# Hardware and Runtime Measurement

This document records the public runtime protocol and the final direct full-set
numbers used by the paper. Machine usernames, hostnames, private filesystem paths,
and internal experiment directories are intentionally omitted.

## Reference Environment

| Component | Recorded value |
|---|---|
| GPU | NVIDIA RTX 5880 Ada Generation, 49,140 MiB VRAM |
| CPU | Intel Xeon Gold 6526Y, 2 sockets, 16 cores/socket |
| RAM | 503 GiB |
| OS | Ubuntu 22.04.4 LTS |
| Python | 3.11.15 |
| PyTorch | 2.9.1+cu128 |
| Transformers | 5.13.1 |
| CUDA runtime | 12.8 |
| NVIDIA driver | 575.57.08 |
| Precision | bfloat16 |
| Backbone | Qwen3.5-9B |
| Attention backend | Installed Transformers/Qwen default; an exact named backend was not recorded in the final runtime artifact |

The Qwen3.5 linear patch-embedding bypass in `src/qwen/linear_patch_bypass.py` was
enabled for these measurements.

## Final MSAD 512sq Full-Set Measurement

Each variant was run as a separate single-GPU process over all 3,607 MSAD test
decision windows from 360 videos.

| Variant | Windows | Mean total (s/window) | Median total | P95 total | Mean model | Median model | P95 model | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2 only | 3,607 | 1.2711 | 1.1902 | 1.6866 | 0.8016 | 0.7782 | 0.9472 | 0 |
| Dense all-probe | 3,607 | 1.5813 | 1.4994 | 1.9875 | 1.0729 | 1.0455 | 1.2388 | 0 |
| PARSEE routed | 3,607 | 1.3723 | 1.3784 | 1.7410 | 0.8841 | 0.8376 | 1.0521 | 0 |

The direct dense-to-routed end-to-end reduction is:

```text
(1.5813 - 1.3723) / 1.5813 = 13.2%
```

The specialist-query reduction is a separate quantity. Dense execution performs two
specialist queries per window (`7,214` total), whereas routed execution performed
`1,202` P3 and `837` P4 queries (`2,039` total):

```text
1 - 2039 / 7214 = 71.7%
```

Do not describe the 71.7% specialist-query reduction as a 71.7% latency reduction.

## Measurement Protocol

- nine frames per decision: `[-80,-70,-60,-50,-40,-30,-20,-10,0]`;
- 512sq budget: `max_pixels=262144`, `min_pixels=1`, aspect preserving, no forced upsampling;
- one decision window processed at a time;
- videos processed sequentially within a video;
- PARSEE temporal state reset at every video boundary;
- dataset-level sharding, when used, parallelizes independent videos only;
- wall-clock sections measured with `time.perf_counter()`;
- CUDA synchronized around visual-prefix and proposition-tail model calls;
- full-set summaries include frame decode, resize, visual prefill, proposition tails,
  and Python/orchestration overhead;
- no warm-up exclusion was recorded for the final full-set pass;
- one complete full-set pass was recorded per variant.

The source code records per-window timing fields including frame decode, image resize,
shared visual prefill, proposition-tail timing, and total window time.

## Reproducing a Routed Runtime Run

After supplying the public-safe decision manifests and model/dataset environment
variables:

```bash
python -m scripts.run_pixel_budget_sweep \
  --config configs/workflows/parsee_final.yaml \
  --dataset msad \
  --budget 512sq \
  --run-id runtime_msad_512 \
  --no-wait
```

The run directory snapshots the workflow, prompt, and model configuration used for
that execution. The raw private runtime artifacts used to produce the paper numbers
are not part of the source release.
