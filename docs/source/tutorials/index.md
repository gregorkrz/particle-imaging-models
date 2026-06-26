# Tutorials

End-to-end walkthroughs that tie the rest of the documentation together. Each one
goes from raw data to a trained, evaluated model.

- {doc}`Bring your own dataset → PTv3 semantic segmentation <byo_dataset_semseg>` — **Start here.** Wrap your own LArTPC-style point clouds in a pimm dataset, write a config, and train a PTv3 backbone for per-point semantic segmentation — locally, then on multiple GPUs. Optionally fine-tune from a pretrained Sonata backbone.
- {doc}`Panda panoptic detector <panda_detector>` — **Advanced.** Go from semantic to *panoptic*: per-instance masks + PID with the Mask2Former-style Panda Detector, fine-tuned from a frozen Sonata encoder and run with requeue chaining on HPC.

:::{seealso}
New to the codebase? Read {doc}`../getting_started/concepts` first — the
tutorials assume you know about packed tensors, registries, and the config
system. For the data-side mechanics in isolation, see
{doc}`../datasets/bring_your_own`.
:::

```{toctree}
:hidden:

byo_dataset_semseg
panda_detector
```
