# Project 1 — OpenCLIP ViT-B/32 image encoder in NumPy

A from-scratch NumPy implementation of the **image tower of the final project**
(OpenCLIP ViT-B/32). No PyTorch, no OpenCLIP package, no GPU, no network: only
array indexing, `reshape`/`transpose`, matrix products, reductions and
broadcasting. Model parameters are supplied by the course and are never trained.

Each 224x224 RGB image becomes one unit-length 512-dimensional vector:

```
[B,3,224,224] -> patchify -> [B,49,3072] -> embed -> [B,50,768]
              -> 12 x TransformerBlock -> [B,50,768]
              -> CLS + post-Norm + projection -> [B,512]
```

## Requirements

Python 3.9 or later.

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` pins `numpy`, `Pillow` and `imageio`; `imageio` is only used
by the optional classroom demo `practice/lecture-demos/02_imageio_images.py`.

## What is in this repository, and what is not

Included: `image_encoder.py` (the assessed file), the supplied helpers and
command-line scripts (`helpers.py`, `runtime.py`, `transformer_steps.py`,
`preprocess_image.py`, `run.py`, `check.py`), the configuration
(`config.json`, `requirements.txt`), the report source under `report/` with its
architecture figure, the lecture companion under `practice/lecture-demos/`, and
the solutions to the twelve NumPy exercises in `practice/numpy_homework.py`.
`PPT-CODE-MAP.txt` and `SOURCE-MANIFEST.json` are course scaffolding notes and
are kept as supplied.

**Not included, by assignment rule:** `weights/image_encoder.npz` (326 MB, 152
visual tensors), `assets/reference_outputs.npz`, and the ten course images in
`assets/images/`. The assignment explicitly asks students not to resubmit the
large parameter file or the reference data.

Two more files are deliberately kept out. `report/project1-report.pdf` is the
compiled report; it is submitted separately, and its cover carries personal
details, so only the source stays here. `practice/project1_exercises.pdf` is
the course exercise sheet, which is course material rather than student work —
the solutions written against it are tracked.

To run the checks, copy the course-provided `weights/` and `assets/`
directories next to `image_encoder.py` — copy them whole, including both
`manifest.json` files, because `check.py` uses them to fingerprint the
checkpoint and the reference arrays and stops at the first stage when they are
absent. Paths are resolved relative to the scripts, so no other change is
needed.

The original assignment description is kept as `COURSE-README.md`.

## How to run

Run these from the repository root, in this order. Each stage prints the shape,
the maximum and mean absolute error, and the tolerances applied.

```bash
python check.py --stage assets        # checkpoint fingerprint: 152 tensors
python check.py --stage preprocess    # supplied resize/crop/normalise
python check.py --stage patches       # TODO-01
python check.py --stage tokens        # TODO-05
python check.py --stage block0        # TODO-02 / 03 / 04, first block only
python check.py --stage last_tokens   # all 12 blocks
python check.py --stage output        # TODO-06, ten images
python check.py --stage batch         # reordered and duplicated images
```

Or all of them at once:

```bash
python check.py --stage all
python run.py --output output_tokens.npz
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

`run.py` writes `ids` and `output_tokens` in the same order as its inputs and
exits non-zero if the result is not finite `float32` with shape `[B,512]`.

The twelve NumPy exercises need neither the checkpoint nor the images; they run
on their own and print their results:

```bash
python practice/numpy_homework.py
```

## Results on the supplied ten images

| Stage | Shape | max abs error | mean abs error | atol | Result |
| --- | --- | --- | --- | --- | --- |
| preprocess | (10, 3, 224, 224) | 0.0e+00 | 0.0e+00 | 2e-06 | PASS |
| patches | (2, 49, 3072) | 0.0e+00 | 0.0e+00 | 2e-04 | PASS |
| tokens | (2, 50, 768) | 4.005e-05 | 1.289e-06 | 2e-04 | PASS |
| block0 | (2, 50, 768) | 7.063e-06 | 3.250e-07 | 2e-04 | PASS |
| last_tokens | (2, 50, 768) | 4.768e-05 | 2.696e-06 | 2e-04 | PASS |
| output | (10, 512) | 5.439e-07 | 8.311e-08 | 2e-05 | PASS |
| batch | (3, 512) | 4.619e-07 | 7.632e-08 | 2e-05 | PASS |

All ten output rows have `| ||x|| - 1 | = 0`, and repeating the same image
twice returns bit-identical rows.

## What was implemented

`image_encoder.py` only; every other file is supplied.

| TODO | Method | Operation |
| --- | --- | --- |
| 01 | `ImageEncoder.patchify` | `[B,3,224,224]` → `[B,49,3072]`, row-major patches, C-H-W per patch |
| 02 | `MultiHeadAttention.split_heads` | `[B,N,D]` → `[B,H,N,D/H]` |
| 03 | `MultiHeadAttention.__call__` | scaled scores, weighted values, head merge, output projection |
| 04 | `TransformerBlock.__call__` | two residual additions and the MLP |
| 05 | `ImageEncoder.embed` | bias-free patch projection, CLS prepend, learned positions |
| 06 | `ImageEncoder.__call__` | CLS select, post-Norm, 768→512 projection, unit scaling |

## Conventions that decide correctness

1. Attention and MLP linear weights are stored `[out,in]`, so every one of them
   is multiplied as `x @ W.T`. `attn.out_proj.weight` is `[D,D]`; the rule is
   the same.
2. `visual.proj` is the single exception: it is stored `[in,out] = [768,512]`
   and must **not** be transposed.
3. The patch projection has no bias and uses `patch_weight.T` after
   `visual.conv1.weight` is reshaped from `[768,3,32,32]` to `[768,3072]`.
4. Attention `k` is 4D here, so only its last two axes are exchanged
   (`k.transpose(0,1,3,2)`); `k.T` would reverse every axis.
5. Everything stays `float32`; the checks reject a `float64` result.
6. Heads are regrouped as `transpose(0,2,1,3).reshape(B,N,D)`, the exact inverse
   of the split.

## Limitations

- `GELU` is supplied as a scalar `math.erf` loop, so a 10-image forward pass
  takes seconds rather than milliseconds. Vectorising it would change the
  reference-matching arithmetic, so it is left untouched.
- FP32 accumulation order differs from PyTorch, which is why small errors remain
  instead of exact zeros beyond the preprocessing stage.
- The encoder holds all 12 blocks in memory at once and processes a whole batch
  in one call; no streaming or gradient support is provided (none is needed).

## Report

`report/main.tex` is the experiment report (SZU template, compiled with
XeLaTeX on Overleaf) and `report/figs/vit_overview.png` is the architecture
figure it includes. The compiled PDF is submitted separately and is not tracked
here; `main.tex` and the figure are enough to rebuild it.
