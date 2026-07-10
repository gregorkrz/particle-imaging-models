import os
from pathlib import Path

import pytest


DATA_ROOT = os.environ.get("JAXTPC_DATA_ROOT")
if not DATA_ROOT or not Path(DATA_ROOT).expanduser().is_dir():
    pytest.skip(
        "Set JAXTPC_DATA_ROOT to a JAXTPC dataset directory",
        allow_module_level=True,
    )
DATA_ROOT = str(Path(DATA_ROOT).expanduser().resolve())

import numpy as np
import torch
import torch.nn as nn

from pimm.datasets.jaxtpc_dataset import JAXTPCDataset
from pimm.datasets.utils import collate_fn


pytestmark = pytest.mark.external_data
MAX_LEN = 4


def check(condition, message):
    assert condition, message


def make_ds(**kwargs):
    defaults = dict(
        data_root=DATA_ROOT,
        split="",
        dataset_name="sim",
        max_len=MAX_LEN,
    )
    defaults.update(kwargs)
    return JAXTPCDataset(**defaults)


def test_seg_only():
    data = make_ds(modalities=("seg",)).get_data(0)
    check(data["coord"].shape[1] == 3, "coord is 3D")
    check(data["energy"].shape[1] == 1, "energy has one feature")
    check("segment" not in data, "segment absent without labels")


def test_seg_labl():
    data = make_ds(modalities=("seg", "labl"), label_key="particle").get_data(0)
    check(data["coord"].shape[1] == 3, "coord is 3D")
    check("segment" in data, "segment present")
    check(data["segment"].shape[0] == data["coord"].shape[0], "labels align")


def test_resp_only():
    data = make_ds(modalities=("resp",)).get_data(0)
    check(data["coord"].shape[1] == 2, "coord is 2D")
    check("plane_id" in data, "plane_id present")
    check("segment" not in data, "segment absent")
    check(len(np.unique(data["plane_id"])) > 1, "multiple planes present")


def test_resp_corr_labl():
    data = make_ds(
        modalities=("resp", "corr", "labl"),
        label_key="particle",
    ).get_data(0)
    check(data["coord"].shape[1] == 2, "coord is 2D")
    check("segment" in data, "segment present")
    check("instance" in data, "instance present")
    check("plane_id" in data, "plane_id present")
    check(any(key.startswith("plane.") for key in data), "plane keys present")
    _, counts = np.unique(data["coord"], axis=0, return_counts=True)
    check(np.sum(counts > 1) > 0, "overlapping pixels present")


def test_seg_resp_corr_labl():
    data = make_ds(
        modalities=("seg", "resp", "corr", "labl"),
        label_key="particle",
    ).get_data(0)
    check(data["coord"].shape[1] == 3, "coord is 3D")
    check("segment" in data, "3D segment present")
    check("resp_coord" in data, "resp_coord present")
    check(data["resp_coord"].shape[1] == 2, "resp_coord is 2D")
    check("corr_coord" in data, "corr_coord present")
    check("corr_segment" in data, "corr_segment present")
    check("corr_instance" in data, "corr_instance present")
    check(any(key.startswith("plane.") for key in data), "plane keys present")


def test_resp_corr():
    data = make_ds(modalities=("resp", "corr")).get_data(0)
    check(data["coord"].shape[1] == 2, "coord is 2D")
    check("segment" not in data, "segment absent without labels")
    check(any(key.startswith("corr.") for key in data), "corr keys present")


def test_volume_filter():
    all_data = make_ds(modalities=("resp",)).get_data(0)
    volume_data = make_ds(modalities=("resp",), volume=0).get_data(0)
    check(
        volume_data["coord"].shape[0] < all_data["coord"].shape[0],
        "volume filter reduces points",
    )


def test_different_label_keys():
    for label_key in ("particle", "cluster", "interaction"):
        data = make_ds(
            modalities=("seg", "labl"),
            label_key=label_key,
        ).get_data(0)
        check(len(np.unique(data["segment"])) > 1, f"{label_key} has multiple classes")


def test_pipeline_3d():
    transform = [
        dict(type="NormalizeCoord", center=[0, 0, 0], scale=4000.0),
        dict(
            type="GridSample",
            grid_size=0.001,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment"),
            feat_keys=("coord", "energy"),
        ),
    ]
    ds = make_ds(
        modalities=("seg", "labl"),
        label_key="particle",
        min_deposits=1024,
        transform=transform,
    )
    batch = collate_fn([ds[0], ds[1]])
    check(batch["coord"].shape[1] == 3, "pipeline is 3D")
    check(len(batch["offset"]) == 2, "offset correct")


def test_pipeline_2d():
    transform = [
        dict(
            type="GridSample",
            grid_size=1.0,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment", "instance"),
            feat_keys=("coord", "energy"),
        ),
    ]
    ds = make_ds(
        modalities=("resp", "corr", "labl"),
        label_key="particle",
        transform=transform,
    )
    batch = collate_fn([ds[0], ds[1]])
    check(batch["coord"].shape[1] == 2, "pipeline is 2D")
    check(len(batch["offset"]) == 2, "offset correct")

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        persistent_workers=False,
    )
    check(next(iter(loader))["coord"].shape[1] == 2, "dataloader is 2D")


def test_toy_model():
    transform = [
        dict(
            type="GridSample",
            grid_size=1.0,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment"),
            feat_keys=("coord", "energy"),
        ),
    ]
    ds = make_ds(
        modalities=("resp", "corr", "labl"),
        label_key="particle",
        transform=transform,
    )
    batch = collate_fn([ds[0], ds[1]])
    model = nn.Linear(batch["feat"].shape[1], 5)
    logits = model(batch["feat"])
    loss = nn.CrossEntropyLoss(ignore_index=-1)(logits, batch["segment"].long())
    loss.backward()
    check(logits.shape[1] == 5, "model has five logits")
    check(model.weight.grad is not None, "gradients computed")
