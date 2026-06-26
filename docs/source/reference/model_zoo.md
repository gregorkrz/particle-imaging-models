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

Self-supervised methods that produce a backbone checkpoint to warm-start
downstream tasks (see {doc}`../checkpoints/index` for warm-start mechanics).

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
  - `detector-v1m1`, `detector-v1m2`, `detector-v3` / `detector-v3m1`,
    `detector-v3m2`, `detector-v3m3`, `detector-v4`
  - Panoptic / instance detection (Mask2Former-style, low-energy aware);
    optional PID, vertex, momentum heads.
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

Ready-made recipes live under `configs/`. The most useful families:

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
`panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-enc-upcast-fft`.

## See also

- {doc}`../configuration/index` — wire a `type` into `model.type` and override it.
- {doc}`cli` — launch a config family target with `pimm launch` / `pimm submit`.
- {doc}`../datasets/index` — PILArNet-M and the packed input each model expects.
- {doc}`../checkpoints/index` — warm-start a fine-tune from a pretrained backbone.
