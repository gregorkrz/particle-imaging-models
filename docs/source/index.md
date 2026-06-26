---
sd_hide_title: true
---

# pimm — Particle Imaging Models

:::{div} pimm-hero

# Particle Imaging Models

[Foundation-model research for neutrino & particle-imaging detectors.]{.pimm-tagline}

```{button-ref} getting_started/installation
:ref-type: myst
:color: primary
:class: sd-px-4 sd-fs-6
Get started
```
```{button-link} https://github.com/DeepLearnPhysics/particle-imaging-models
:color: secondary
:outline:
:class: sd-px-4 sd-fs-6
View on GitHub
```

:::

---

**pimm** adapts modern deep-learning and computer-vision methods to event
reconstruction in neutrino detectors. It gives you self-supervised
pre-training, semantic and panoptic segmentation, exact-resume distributed
training, and a launch layer that runs the *same* code locally, on a multi-GPU
node, or across many HPC nodes.

## What's inside

- {doc}`Quick start <getting_started/index>` — install from source or a
  container, run your first job locally, and learn the three ideas (packed
  tensors, registries, configs) that make pimm tick.
- {doc}`Distributed training <distributed/index>` — one launch path for
  single-GPU, multi-GPU, and multi-node. DDP and FSDP2, deterministic
  checkpointing, exact mid-epoch resume across world-size changes.
- {doc}`Scientific computing <hpc/index>` — interactive vs batch submission,
  site profiles (S3DF, NERSC), QOS, requeue chaining, environment variables, job
  monitoring, and resuming.
- {doc}`Checkpoints <checkpoints/index>` — one atomic, reshardable checkpoint
  format for every parallelism. Save-cadence hooks, manual export, and automatic
  Hugging Face upload.
- {doc}`Using trained models <models/index>` — load any export with
  `pimm.from_pretrained`, fine-tune from the Hub, and reproduce the
  exact dataset format a model expects.
- {doc}`Datasets & transforms <datasets/index>` — the packed point-cloud
  contract, PILArNet-M, multimodal LArTPC/Water-Cherenkov readers, transform
  pipelines, and bringing your own data.
- {doc}`Hooks <hooks/index>` — the lifecycle hook system (logging, diagnostics,
  evaluators, checkpoint savers) and how to write your own.
- {doc}`Evaluation <evaluation/index>` — in-loop evaluators, probe suites for
  SSL, and final testing with `test.sh`.
- {doc}`Tutorials <tutorials/index>` 
- {doc}`API reference <api/index>`

## Integrated works

- **Backbones** — [PTv3](https://arxiv.org/abs/2312.10035),
  [PTv2](https://arxiv.org/abs/2210.05666),
  [PTv1](https://arxiv.org/abs/2012.09164), and SparseUNet (SpConv/Minkowski).
- **Pre-training** — [Sonata](https://arxiv.org/abs/2503.16429) /
  [Panda](https://arxiv.org/abs/2512.01324) discriminative SSL and
  [PoLAr-MAE](https://arxiv.org/abs/2502.02558) masked autoencoding.
- **Segmentation** — semantic segmentation, PointGroup, and the
  [Panda Detector](https://arxiv.org/abs/2512.01324) panoptic model.
- **Datasets** — [PILArNet-M](https://arxiv.org/abs/2502.02558), multimodal
  JAXTPC, Water-Cherenkov (LUCiD).

:::{seealso}
Start with **{doc}`getting_started/installation`**, then the
**{doc}`getting_started/quickstart`**. Coming from another Pointcept-style
codebase? Skim **{doc}`getting_started/concepts`** for what pimm does
differently.
:::

```{toctree}
:hidden:
:maxdepth: 2

Getting started <getting_started/index>
Datasets <datasets/index>
Trained models <models/index>
Tutorials <tutorials/index>
Configuration <configuration/index>
Hooks <hooks/index>
Checkpoints <checkpoints/index>
Evaluation <evaluation/index>
Distributed training <distributed/index>
HPC <hpc/index>
Reference <reference/index>
API reference <api/index>
```
