# Research ecosystem

- {doc}`Using & fine-tuning models <using_trained_models>` - load any export
  with `pimm.from_pretrained`, reproduce the exact input a model requires, and
  fine-tune from a checkpoint or the Hub.
- {doc}`Publishing your model <publishing_a_model>` - export your trained weights
  and push them to the Hub so others can use them.
- {doc}`Contributing a model <contributing_a_model>` - register a new
  architecture (and, rarely, a custom trainer like `GRPOTrainer`) on the pimm
  substrate.
- {doc}`Contributing a hook <contributing_a_hook>` - extend the training
  lifecycle with logging, diagnostics, evaluators, or savers.
- {doc}`Contributing a dataset <contributing_a_dataset>` - write a reader and a
  dataset class that emit the packed-batch format.
- {doc}`Contributing a transform <contributing_a_transform>` - preprocess raw
  data, add augmentations, or build the multi-view machinery a training method
  needs.

:::{seealso}
Everything here builds on three ideas - **registries**, **packed tensors**, and
the **one-forward trainer rule**. If you haven't yet, read
{doc}`../getting_started/concepts`. The end-to-end story (custom data → custom
model → trained) is the tutorial {doc}`../tutorials/byo_dataset_semseg`.
:::

```{toctree}
:hidden:

using_trained_models
publishing_a_model
contributing_a_model
contributing_a_hook
contributing_a_dataset
contributing_a_transform
```
