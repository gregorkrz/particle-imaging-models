# Tutorials

Each tutorial goes from raw data to a trained, evaluated model.

- {doc}`Training a semantic segmentation model <byo_dataset_semseg>` - **Start here.** Wrap your own point clouds in a pimm dataset, write a config, and train a PTv3 backbone for per-point semantic segmentation - locally, then on multiple GPUs. Optionally fine-tune from a pretrained Sonata backbone.
- {doc}`Panda panoptic detector <panda_detector>` - Perform panoptic segmentation, which outputs per-instance masks + per-instance information with the Panda Detector, fine-tuned from a frozen Sonata encoder.
- {doc}`PEFT <panda_detector_peft>` - Parameter-efficient fine-tuning of Panda Detector on a new dataset, using LoRA for the PTv3 backbone.

:::{seealso}
New to the codebase? Read {doc}`../getting_started/concepts` first - the
tutorials assume you know about packed tensors, registries, and the config
system. For the data-side mechanics in isolation, see
{doc}`../research_ecosystem/contributing_a_dataset`.
:::

```{toctree}
:hidden:

byo_dataset_semseg
panda_detector
panda_detector_peft
```
