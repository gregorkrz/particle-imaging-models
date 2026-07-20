_base_ = ["../../_base_/default_runtime.py"]

# misc custom setting
batch_size = 48*3  # bs: total bs in all gpus
num_worker = 24
val_batch_size = 1
mix_prob = 0.0
clip_grad = 1.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
matmul_precision = "high"
seed = 0
evaluate = True
find_unused_parameters = False

# Weights & Biases specific settings
use_wandb = True
wandb_project = "PanSeg-Joint-Sonata-PILArNet-M"

# scheduler settings
epoch = 20
eval_epoch = 20

# Warm-start (full fine-tune) from the published Panda Particle detector.
# Pinned to the same commit the sibling ft-pid-fft-detector config uses. The
# CheckpointLoader hook below loads this into the full UnifiedDetector; the new
# momentum_vec head has no counterpart in the checkpoint and stays at init.
weight = "hf://DeepLearnPhysics/panda-particle@bd90792dfe83cd05b437b719564b311f0a0b785a"

# Per-component momentum regression loss weight. px/py/pz are signed GeV values
# regressed in linear space (no MomentumTransform), so they share the default
# SmoothL1RegressionLoss but with a modest weight to balance against the mask
# and classification terms.
momentum_component_loss_weight = 1.0

# detector-v5 model settings
model = dict(
    type="detector-v5",
    eval_label="particle",
    label_configs=dict(
        particle=dict(
            num_queries=32,
            num_classes=6,  # photon, electron, muon, pion, proton, led
            instance_key="instance_particle",
            segment_key="segment_pid",
            stuff_classes=[5],
            use_stuff_head=True,
            loss_weight=1.0,
            overlap=False,
            query_heads=[
                # Momentum magnitude is log10-transformed in the data pipeline.
                # LED clusters carry no true momentum (sentinel -1); drop them
                # from the regression loss.
                dict(name="momentum", use_class_logits=True, sentinel=-1.0),
                # Signed momentum vector (px, py, pz) in GeV, regressed in linear
                # space -- NOT passed through MomentumTransform (log10, would
                # discard the sign). Kept as a single dim=3 target so it rotates
                # correctly under the geometric augmentations (a rotation mixes
                # the components). Maps to the `momentum_vec` dataloader key.
                # The all-(-1) LED sentinel is dropped from the loss; empirically
                # LED is the only class with undefined momentum.
                dict(
                    name="momentum_vec",
                    dim=3,
                    use_class_logits=True,
                    loss_weight=momentum_component_loss_weight,
                    sentinel=-1.0,
                ),
                # PILArNet v3 stores the interaction vertex on each particle.
                dict(name="vertex", dim=3),
                dict(name="is_primary", kind="categorical", num_classes=2),
            ],
            criterion=dict(
                type="FastUnifiedInstanceLoss",
                cost_mask=1.0,
                cost_dice=1.0,
                cost_class=1.0,
                loss_weight_focal=2.0,
                loss_weight_dice=5.0,
                cls_weight_matched=2.0,
                cls_weight_noobj=0.5,
                focal_alpha=0.25,
                focal_gamma=2.0,
                aux_loss_weight=1.0,
                num_points=100_000,
            ),
        ),
        interaction=dict(
            num_queries=12,
            num_classes=2,  # stuff, thing
            instance_key="instance_interaction",
            segment_key="segment_interaction",
            stuff_classes=[0],
            use_stuff_head=True,
            loss_weight=1.0,
            overlap=False,
            query_heads=[dict(name="vertex", dim=3)],
            criterion=dict(
                type="FastUnifiedInstanceLoss",
                cost_mask=1.0,
                cost_dice=1.0,
                cost_class=1.0,
                loss_weight_focal=2.0,
                loss_weight_dice=5.0,
                cls_weight_matched=2.0,
                cls_weight_noobj=0.5,
                focal_alpha=0.25,
                focal_gamma=2.0,
                aux_loss_weight=1.0,
                num_points=100_000,
            ),
        ),
    ),
    supervise_attn_mask=True,
    backbone=dict(
        type="PT-v3m2",
        in_channels=4,  # [xyz, energy]
        order=("hilbert", "hilbert-trans", "z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 9, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(256, 256, 256, 256),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        layer_scale=1e-5,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=True,  # encoder + decoder
        freeze_encoder=False,
    ),
    full_in_channels=1232,
    mlp_point_proj=True,
    hidden_channels=256,
    num_heads=16,
    depth=3,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.0,
    layer_scale=None,
    pre_norm=True,
    enable_flash=True,
    upcast_attention=False,
    upcast_softmax=False,
    pos_emb=True,
    postprocess=dict(
        stuff_threshold=0.5,
        mask_threshold=0.5,
        conf_threshold=0.5,
        nms_kernel="gaussian",
        nms_sigma=2.0,
        nms_pre=-1,
        nms_max=-1,
        min_points=2,
        fill_uncovered=False,
    ),
)

lr_decay = 0.97
base_lr = 2e-4
base_wd = 0.01
backbone_mult = 1.0

# encoder/decoder depths
enc_depths = model["backbone"]["enc_depths"]
dec_depth = model["depth"]

param_dicts = []

# encoder: smallest LR at first encoder block
for e in range(len(enc_depths)):
    for b in range(enc_depths[e]):
        exp = (sum(enc_depths) - sum(enc_depths[:e]) - b - 1) + dec_depth
        param_dicts.append(
            dict(
                keyword=f"enc{e}.block{b}.",
                lr=base_lr * (lr_decay**exp) * backbone_mult,
            )
        )

# decoder: highest LR at last decoder block
for b in range(dec_depth):
    exp = dec_depth - b - 1
    param_dicts.append(
        dict(
            keyword=f"decoder.blocks.{b}.",
            lr=base_lr * (lr_decay**exp),
        )
    )

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.025,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
grid_size = 0.001  # ~ 0.001/(1 / (768.0 * 3**0.5 / 2))
target_keys = (
    "segment_pid",
    "instance_particle",
    "segment_interaction",
    "instance_interaction",
    "momentum",
    # Signed momentum vector (N, 3) from the _pxpypz dataset (decode.py packs
    # px/py/pz, sentinel-filled where truth momentum is absent). Declared in
    # aux_vector_keys below so the geometric augmentations rotate/flip it.
    "momentum_vec",
    "vertex",
    "is_primary",
)
transform = [
    # Declare momentum_vec as a magnitude-bearing vector target so RandomRotate/
    # RandomFlip rotate it with the point cloud (linear part only, no centering,
    # no renormalization). NormalizeCoord does NOT touch aux_vector_keys, so the
    # momentum vector is correctly left un-scaled/un-translated.
    dict(type="Update", keys_dict={"aux_vector_keys": ["momentum_vec"]}),
    dict(
        type="NormalizeCoord",
        center=[384.0, 384.0, 384.0],
        scale=768.0 * 3**0.5 / 2,
    ),
    dict(type="LogTransform", min_val=1.0e-2, max_val=20.0, keys=("energy",)),
    # Only the magnitude head is log-compressed; the momentum vector stays
    # linear/signed.
    dict(type="MomentumTransform", keys=("momentum",)),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", *target_keys),
        feat_keys=("coord", "energy"),
    ),
]
test_transform = [
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

particle_names = ["photon", "electron", "muon", "pion", "proton", "led"]
interaction_names = ["stuff", "thing"]
data = dict(
    num_classes=6,
    ignore_index=-1,
    names=particle_names,
    interaction_names=interaction_names,
    train=dict(
        type="PILArNetH5Dataset",
        revision="v3",
        split="train",
        # data_root="/path/to/pilarnet-m/",
        transform=transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1_000_000,  # override via --options data.train.max_len=X
        remove_low_energy_scatters=False,
    ),
    val=dict(
        type="PILArNetH5Dataset",
        revision="v3",
        split="val",
        # data_root="/path/to/pilarnet-m/",
        transform=test_transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1000,
        remove_low_energy_scatters=False,
    ),
    test=dict(
        type="PILArNetH5Dataset",
        revision="v3",
        split="test",
        # data_root="/path/to/pilarnet-m/",
        transform=test_transform,
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1000,
        remove_low_energy_scatters=False,
    ),
)

# hook
hooks = [
    dict(
        type="WandbNamer",
        keys=("model.type", "data.train.max_len", "amp_dtype", "seed"),
        extra="joint-pxpypz-fft",
    ),
    dict(
        type="WeightDecayExclusion",
        exclude_bias_from_wd=True,
        exclude_norm_from_wd=True,
        exclude_gamma_from_wd=True,
        exclude_token_from_wd=True,
        exclude_ndim_1_from_wd=True,
    ),
    dict(
        type="CheckpointLoader",
        # Warm-start from the published Panda Particle detector (see `weight`
        # above). It is a full UnifiedDetector export, so its keys already match
        # this model -- no renaming needed. strict=False so the new momentum_vec
        # head (absent from the checkpoint) is left at its init instead of
        # raising on missing keys.
        replacements={},
        strict=False,
    ),
    dict(
        type="ParameterCounter",
        show_details=True,
        show_gradients=False,
        sort_by_params=False,
        min_params=1,
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="InstanceSegmentationEvaluator",
        every_n_steps=1000,
        stuff_threshold=0.5,
        mask_threshold=0.5,
        iou_thresh=0.5,
        class_names={
            "particle": particle_names[:-1],
            "interaction": interaction_names,
        },
        require_class_for_match=False,
        labels=("particle", "interaction"),
        primary_label="particle",
    ),
    dict(type="CheckpointSaver", save_freq=None, evaluator_every_n_steps=1000),
    dict(type="FinalEvaluator", test_last=True),
]

# InstanceSegTester evaluates the model's eval_label (particle here).
test = dict(
    type="InstanceSegTester",
    class_names=particle_names[:-1],
    stuff_classes=[5],
    require_class_for_match=False,
)
