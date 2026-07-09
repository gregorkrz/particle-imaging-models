# Parameter-efficient fine-tuning

**Goal.** Adapt a large pretrained backbone to a new task or detector while training only a tiny fraction of its weights.
Instead of unfreezing the whole encoder, you freeze it and train small low-rank adapters inside its attention blocks.
This keeps the pretrained representation intact, cuts optimizer memory, and makes it cheap to keep many task-specific variants around a single shared backbone.

This tutorial assumes you're comfortable with the fine-tuning loop from {doc}`byo_dataset_semseg` - the config, launch, and evaluation patterns are identical.
The only thing that changes is that the model is wrapped in a LoRA adapter.

## How LoRA works in pimm

pimm ships one PEFT primitive: the `LoRAAdapter` model (registered in the `MODELS` registry, referenced by `type="LoRAAdapter"`).
It wraps an already-built model - a segmentor, classifier, or detector, anything `build_model` produces - and does three things:

1. Freezes every parameter of the wrapped model.
2. Injects a low-rank update into each targeted attention projection, replacing that `nn.Linear` in place with a `LoRALinear`.
3. Optionally re-enables a few whitelisted modules (typically the task head) so they train from scratch alongside the adapters.

Each `LoRALinear` computes `y = W0 x + b0 + (alpha / rank) * (B @ A) x`, where `A` and `B` are the only trainable parameters.
`A` is randomly initialized and `B` is initialized to zero, so at the start of training the adapter is an exact no-op and the wrapped model reproduces its pretrained outputs bit-for-bit.
Because `LoRALinear` subclasses `nn.Linear` and reuses the original `weight` / `bias`, the pretrained checkpoint still loads straight into the adapted layers - only the new `lora_A` / `lora_B` keys are added.

## Configuration

`LoRAAdapter` takes the following keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `model` | (required) | Config of the inner model to build and adapt. |
| `rank` | `8` | LoRA rank for every injected adapter. |
| `alpha` | `16.0` | Scaling numerator; the effective scale is `alpha / rank`. |
| `dropout` | `0.0` | Dropout applied to the LoRA branch input. |
| `target_modules` | `("attn.qkv", "attn.proj")` | A `Linear` is adapted when its dotted name equals or ends with `.<entry>`. The default targets PT-v3 attention projections. |
| `trainable_keywords` | `()` | Parameters whose name contains any of these substrings stay trainable. Empty means only the LoRA parameters train. |

Take the semantic-segmentation model from {doc}`byo_dataset_semseg` (section 3) and wrap it.
The inner `model` dict is exactly what you'd write for a full fine-tune - you just nest it under the adapter:

```python
# --- model: PTv3 segmentor, LoRA on the encoder attention, head trained ---
model = dict(
    type="LoRAAdapter",
    rank=8,
    alpha=16,
    dropout=0.0,
    target_modules=("attn.qkv", "attn.proj"),   # PT-v3 attention projections
    trainable_keywords=("seg_head",),           # keep the readout trainable
    model=dict(
        type="DefaultSegmentorV2",
        num_classes=num_classes,
        backbone_out_channels=1232,
        backbone=dict(
            type="PT-v3m2",
            in_channels=4,
            # ... same backbone args as the full fine-tune config ...
            enc_mode=True,
            freeze_encoder=False,               # the adapter handles freezing
        ),
        criteria=[
            dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
            dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0 / 20.0,
                 ignore_index=-1),
        ],
        mlp_head=False,
    ),
)
```

On startup the adapter logs how many projections it wrapped and the fraction of parameters left trainable, so you can confirm the backbone is frozen.

### One gotcha: the checkpoint remap

Wrapping the model shifts every submodule one level deeper - the backbone now lives at `module.model.backbone` instead of `module.backbone`.
If you warm-start from a pretrained backbone (as in section 8 of {doc}`byo_dataset_semseg`), the `CheckpointLoader` `replacement` has to point at the new location:

```python
# add to cfg.hooks (replaces the plain CheckpointLoader)
dict(
    type="CheckpointLoader",
    keywords="module.student.backbone",
    replacement="module.model.backbone",   # note: model.backbone, not backbone
),
```

A remap that matches zero parameters raises, so a silent random-init can't slip through.

## Train and evaluate

Everything downstream is unchanged.
Launch, resume, and evaluate exactly as in the full fine-tune:

```bash
pimm launch --train.config mytpc/semseg-ptv3-lora --run.name semseg-lora \
  --train.weight hf://deeplearnphysics/panda-base
```

Because only the adapters and the head carry gradients, you can usually afford a larger batch or a higher learning rate than a full fine-tune.
Everything else - the evaluator, the checkpoint saver, the OneCycle schedule - behaves as it does in {doc}`byo_dataset_semseg`.

## Where to go next

- {doc}`byo_dataset_semseg` - the full fine-tuning loop (dataset, config, launch, evaluate) that this page builds on; section 8 covers warm-starting from a pretrained backbone.
- {doc}`../research_ecosystem/using_trained_models` - load a base checkpoint such as `deeplearnphysics/panda-base` with {py:func}`~pimm.from_pretrained`.
- {doc}`../checkpoints/saving_and_loading` - how `CheckpointLoader` keyword remapping works in detail.
