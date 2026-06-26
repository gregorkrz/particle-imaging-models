# Model zoo

A catalog of the models registered in pimm, grouped by role: **backbones**,
**pretraining** methods, and **segmentation / detection** heads. Each entry lists
the registry `type` string you drop into a config's `model.type` (or
`model.backbone.type`), its task, and the paper it comes from.

:::{note}
This is a catalog, not a benchmark table — no performance numbers are quoted
here. For training recipes, browse the config families listed at the bottom of
this page. For how `type` strings are resolved, see
{doc}`../getting_started/concepts`.
:::

## Pretrained checkpoints

Published Panda checkpoints live on the Hugging Face Hub as **consolidated
exports** (`model.safetensors` + `config.json`), so the architecture
travels with the weights — {py:func}`~pimm.from_pretrained` rebuilds the model with no
config needed.

```{list-table}
:header-rows: 1
:widths: 34 42 24

* - Repo
  - Task
  - `model.type`
* - [`deeplearnphysics/panda-base`](https://huggingface.co/deeplearnphysics/panda-base)
  - Pretrained PT-v3m2 encoder — fine-tune downstream tasks from this
  - `PT-v3m2`
* - [`deeplearnphysics/panda-semantic`](https://huggingface.co/deeplearnphysics/panda-semantic)
  - Semantic segmentation — 5 classes (shower, track, michel, delta, led)
  - `DefaultSegmentorV2`
* - [`deeplearnphysics/panda-particle`](https://huggingface.co/deeplearnphysics/panda-particle)
  - Panoptic **particle ID** — 6 classes (photon, electron, muon, pion, proton, led)
  - `detector-v4`
* - [`deeplearnphysics/panda-interaction`](https://huggingface.co/deeplearnphysics/panda-interaction)
  - Panoptic **interaction / vertex** grouping — 2 classes
  - `detector-v4`
```

**Load a trained model for inference** — one call (see {doc}`../models/index` for
building the input and reading the output; the Panda detectors also need
`postprocess()`, shown in {doc}`../tutorials/panda_detector`):

```python
import pimm
model = pimm.from_pretrained("deeplearnphysics/panda-semantic", device="cuda")
```

**Fine-tune a downstream task** — `panda-base` is the pretrained PT-v3m2 encoder
to build on. The fine-tune recipe (copy a config, load the encoder via a
`CheckpointLoader` remap, train) is in {doc}`../tutorials/byo_dataset_semseg` and
{doc}`../tutorials/panda_detector`.

:::{note}
A bare `hf://<repo>` resolves to the repo's `model.safetensors`; the explicit form
is `hf://<repo>/<file>`. A fine-tune config's `CheckpointLoader` rewrites a
checkpoint's keys onto the model's `backbone.*` — the remap rule must match the
checkpoint's key layout, so confirm the load reports no missing backbone keys.
:::

## Versioning convention

Models use **`vXmY`** naming: **version `X`, mode `Y`**. A new *version* marks a
large architectural change; a new *mode* marks a smaller variant of the same
version (e.g. `PT-v3m1` is the base Point Transformer V3, `PT-v3m2` is the
Sonata-flavored variant). The same scheme applies to heads (`SpUNet-v1m1`,
`detector-v3m2`, `Sonata-v1m2`, ...).

## Backbones

Backbones go under `model.backbone.type`; the segmentation/detection wrapper
(e.g. {py:class}`~pimm.models.default.DefaultSegmentorV2`) goes under `model.type`.

```{list-table}
:header-rows: 1
:widths: 26 22 30 22

* - Model
  - `type`
  - Notes
  - Paper
* - Point Transformer V3
  - `PT-v3m1`
  - Base PTv3 backbone (serialized attention, optional FlashAttention).
  - [arXiv:2312.10035](https://arxiv.org/abs/2312.10035)
* - Point Transformer V3 (Sonata)
  - `PT-v3m2`
  - PTv3 variant used with Sonata pretraining; encoder-only & upcast modes.
  - [arXiv:2312.10035](https://arxiv.org/abs/2312.10035)
* - Point Transformer V2
  - `PT-v2m1`, `PT-v2m2`, `PT-v2m3`
  - Grouped vector attention; `m3` adds point-data norm.
  - [arXiv:2210.05666](https://arxiv.org/abs/2210.05666)
* - Point Transformer V1
  - `PointTransformer-Seg{26,38,50}`, `-Cls{26,38,50}`, `-PartSeg{26,38,50}`
  - Original Point Transformer, seg / cls / part-seg depths.
  - [arXiv:2012.09164](https://arxiv.org/abs/2012.09164)
* - SparseUNet (SpConv)
  - `SpUNet-v1m1`, `SpUNet-v1m2`, `SpUNet-v1m3`
  - SpConv-based UNet; `m2` BN-momentum, `m3` point-data norm.
  - [spconv](https://github.com/traveller59/spconv)
* - MinkUNet
  - `MinkUNet*` (e.g. `MinkUNet34C`)
  - MinkowskiEngine sparse-conv UNet family.
  - [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
* - LitePT
  - `LitePT`
  - Lightweight point-transformer backbone with a PointROPE CUDA extension.
  - —
```

## Pretraining

Self-supervised methods that produce a backbone checkpoint to fine-tune
downstream tasks (see {doc}`../checkpoints/index` for fine-tune mechanics).

```{list-table}
:header-rows: 1
:widths: 26 24 28 22

* - Method
  - `type`
  - Objective
  - Paper
* - Sonata / Panda SSL
  - `Sonata-v1m1`, `Sonata-v1m2`, `Sonata-DINO-v1m1`
  - DINO-style discriminative SSL: teacher–student with online prototype
    clustering.
  - [Sonata arXiv:2503.16429](https://arxiv.org/abs/2503.16429),
    [Panda arXiv:2512.01324](https://arxiv.org/abs/2512.01324)
* - PoLAr-MAE
  - `PoLAr-MAE`
  - Masked autoencoder with chamfer + energy reconstruction losses.
  - [arXiv:2502.02558](https://arxiv.org/abs/2502.02558)
```

## Segmentation & detection

Task heads that wrap a backbone. Semantic segmentors emit per-point class logits;
the Panda Detector and PointGroup emit instances / panoptic output.

```{list-table}
:header-rows: 1
:widths: 28 24 26 22

* - Model
  - `type`
  - Task
  - Paper
* - Default semantic segmentor
  - `DefaultSegmentorV2`, `DefaultSegmentorV3`
  - Semantic segmentation (per-point classes).
  - —
* - DINO-enhanced segmentor
  - `DINOEnhancedSegmentor`
  - Semantic segmentation with an auxiliary SSL objective.
  - [Sonata arXiv:2503.16429](https://arxiv.org/abs/2503.16429)
* - PoLAr-MAE semantic segmentor
  - `PoLArMAE-SemSeg`
  - Semantic segmentation on a PoLAr-MAE encoder.
  - [arXiv:2502.02558](https://arxiv.org/abs/2502.02558)
* - PointGroup
  - `PG-v1m1`
  - Clustering-based instance segmentation.
  - [PointGroup](https://github.com/dvlab-research/PointGroup)
* - Panda Detector
  - `detector-v4`
  - Unified panoptic / instance detection (Mask2Former-style, low-energy aware);
    per-query PID, with optional vertex / momentum / custom heads.
  - [arXiv:2512.01324](https://arxiv.org/abs/2512.01324)
* - Default classifier
  - `DefaultClassifier`
  - Event/point classification head.
  - —
```

:::{seealso}
Some `type` names accept aliases — for example `detector-v3` and `detector-v3m1`
register the same class. When a config inherits a model and changes its shape,
remember that swapping `model.type` usually wants `_delete_=True` on the `model`
dict (see {doc}`../configuration/index`).
:::

## Config families

Ready-made recipes live under `configs/`. **Not every `type` above ships a
config** — entries without one (e.g. the `detector-v3*` variants and
`DefaultSegmentorV3`) are building blocks you wire up in your own config. The
families with runnable recipes:

```{list-table}
:header-rows: 1
:widths: 34 66

* - Directory
  - Contents
* - `configs/panda/pretrain/`
  - Sonata / Panda SSL and MAE pretraining recipes on PILArNet-M.
* - `configs/panda/semseg/`
  - Semantic-segmentation fine-tunes (encoder-only, decoder, frozen/full-tune
    variants).
* - `configs/panda/panseg/`
  - Panda Detector panoptic / instance fine-tunes (PID, vertex, momentum, joint).
* - `configs/_base_/`
  - Shared base configs, including `default_runtime.py`.
```

A config-path target for `--train.config` is one of these files with the
`configs/` prefix and `.py` suffix optional, e.g.
`panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft`.

### Reading a config name

Config filenames encode the recipe. Reading
`panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft`:

```{list-table}
:header-rows: 1
:widths: 26 74

* - Token
  - Meaning
* - `pretrain` / `semseg` / `panseg`
  - task: SSL pretraining / semantic segmentation / panoptic (detector)
* - `pt-v3m2`
  - backbone (`PT-v3m2`); `sonata`, `detector-v4`, etc. name the model
* - `pilarnet`
  - dataset (`PILArNetH5Dataset`)
* - `ft`
  - fine-tune: load a pretrained backbone (vs. training it cold)
* - `5cls` / `pid` / `vtx`
  - task variant: 5-class semseg / particle-ID / interaction-vertex
* - `fft` / `dec` / `lin` / `scratch`
  - regime: full fine-tune / decoder-only (frozen encoder) / linear probe (frozen backbone) / random-init baseline
* - `smallmask`, `1m`, `amp`, `seed0`
  - misc knobs (mask schedule, event count, AMP, seed)
```

## See also

- {doc}`../configuration/index` — wire a `type` into `model.type` and override it.
- {doc}`cli` — launch a config family target with `pimm launch` / `pimm submit`.
- {doc}`../datasets/index` — PILArNet-M and the packed input each model expects.
- {doc}`../checkpoints/index` — fine-tune from a pretrained backbone.
