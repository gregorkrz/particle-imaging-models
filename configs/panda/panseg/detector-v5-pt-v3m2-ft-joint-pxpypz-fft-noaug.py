"""Joint detector-v5 (px/py/pz) WITHOUT random geometric augmentations.

Ablation of detector-v5-pt-v3m2-ft-joint-pxpypz-fft: the train transform drops
the RandomRotate (z/x/y) and RandomFlip steps so the network sees only the
canonical event orientation. Everything else -- model, data, losses, hooks --
is inherited unchanged from the base config.

Because there are no rotations/flips, the momentum_vec target is never mixed
across components; the `aux_vector_keys` declaration and GridSample handling are
harmless no-ops here but are kept identical to the base so the collected keys
match.
"""

_base_ = ["./detector-v5-pt-v3m2-ft-joint-pxpypz-fft.py"]

# Must be restated here: the base config's module-level locals (target_keys,
# grid_size) are not visible to child configs -- only the merged config dict is
# inherited. Keep these in sync with the base.
grid_size = 0.001
target_keys = (
    "segment_pid",
    "instance_particle",
    "segment_interaction",
    "instance_interaction",
    "momentum",
    "momentum_vec",
    "vertex",
    "is_primary",
)

# Same as the base train transform but with the three RandomRotate steps and the
# RandomFlip removed. Merge semantics replace the list wholesale, so val/test
# transforms (which had no augmentations) are inherited untouched.
transform = [
    dict(type="Update", keys_dict={"aux_vector_keys": ["momentum_vec"]}),
    dict(
        type="NormalizeCoord",
        center=[384.0, 384.0, 384.0],
        scale=768.0 * 3**0.5 / 2,
    ),
    dict(type="LogTransform", min_val=1.0e-2, max_val=20.0, keys=("energy",)),
    dict(type="MomentumTransform", keys=("momentum",)),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", *target_keys),
        feat_keys=("coord", "energy"),
    ),
]

data = dict(train=dict(transform=transform))

# Distinguish the run in W&B from the augmented baseline.
hooks_override = {
    "WandbNamer": {"extra": "joint-pxpypz-fft-noaug"},
}
