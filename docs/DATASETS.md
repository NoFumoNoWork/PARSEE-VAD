# Datasets and Manifest Contract

Raw benchmark videos are external to the repository. Set:

```bash
PARSEE_UCF_ROOT
PARSEE_MSAD_ROOT
PARSEE_XD_ROOT
PARSEE_UBNORMAL_ROOT
```

## Decision Manifest

The sweep consumes one CSV per dataset. Shared code relies on these stable fields:

```text
video_id       stable video identifier
category       dataset category / Normal where applicable
video_path     path relative to the dataset root (preferred)
anchor         decision anchor frame
current_frames semicolon-separated ordered frame indices
decision_index within-video decision index
gt             anchor-level diagnostic label when available
```

UCF/MSAD manifests may additionally contain `gt_original`, `gt_msad`, `label_scope`, `video`, `total_frames`, `frame_count`, or interval metadata. Dataset-specific extra columns are allowed.

Expected files for the public sweep are:

```text
data/manifests/ucf_test_windows.csv
data/manifests/msad_test_windows.csv
data/manifests/xd_violence_test_windows.csv
data/manifests/ubnormal_test_windows.csv
```

Do not commit raw videos or manifests containing private absolute machine paths.

## Official Evaluation Inputs

The frame-level evaluator expects these external official-annotation inputs:

### UCF-Crime

`PARSEE_UCF_GT_MANIFEST` points to a CSV with one metadata row per video (duplicates are tolerated) containing:

```text
video
total_frames
original_intervals
```

`original_intervals` is parsed as inclusive anomaly start/end pairs.

### MSAD

`PARSEE_MSAD_GT_MANIFEST` points to a CSV containing at least `video` (or `video_id`) and `frame_count` (or `total_frames`).

`PARSEE_MSAD_ANNOTATION_CSV` points to the anomaly annotation table containing:

```text
name
starting frame of anomaly
ending frame of anomaly
```

The evaluation protocol treats these intervals as zero-based and inclusive.

### XD-Violence

`PARSEE_XD_ANNOTATION_TXT` points to the official test annotation text. Remaining integer tokens are parsed as anomaly start/end pairs. Video frame counts are read from the corresponding video files under `PARSEE_XD_ROOT`.

### UBnormal

Official test masks are read from the dataset tree under `PARSEE_UBNORMAL_ROOT`. Normal videos receive all-zero GT. For abnormal videos, non-empty `*_gt.png` masks mark anomalous frames.

The repository includes the 158 abnormal and 53 normal official-test name lists used by `scripts/build_ubnormal_manifest.py` (211 test videos total).

## Frame-Level Alignment

For anchors `a0 < a1 < ...`:

```text
score(a0) -> [0, a0]
score(ak) -> [a(k-1)+1, ak]   for k > 0
last score -> [last_anchor+1, video_end]
```

The assignment is piecewise constant. It intentionally does not linearly interpolate between anchors.
