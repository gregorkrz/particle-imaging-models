import os
from pathlib import Path

import pytest


DATA_ROOT = os.environ.get("LUCID_DATA_ROOT")
if not DATA_ROOT or not Path(DATA_ROOT).expanduser().is_dir():
    pytest.skip(
        "Set LUCID_DATA_ROOT to a LUCiD dataset directory",
        allow_module_level=True,
    )
DATA_ROOT = str(Path(DATA_ROOT).expanduser().resolve())

import numpy as np
import torch
import torch.nn as nn

from pimm.datasets.lucid_dataset import LUCiDDataset
from pimm.datasets.utils import collate_fn


pytestmark = pytest.mark.external_data


def check(condition, message):
    assert condition, message


def make_ds(**kwargs):
    defaults = dict(data_root=DATA_ROOT, split="", dataset_name="wc", max_len=4)
    defaults.update(kwargs)
    return LUCiDDataset(**defaults)


def test_sensor_response():
    ds = make_ds(modalities=("sensor",), output_mode="response", include_labels=False)
    data = ds.get_data(0)
    check("coord" in data, "coord present")
    check("energy" in data, f"energy: {data['energy'].shape}")
    check("time" in data, "time present")
    check("segment" not in data, "no segment for SSL")
    check(data["coord"].shape[0] > 1000, "many sensors present")


def test_sensor_labels():
    ds = make_ds(modalities=("sensor",), output_mode="labels", include_labels=True)
    data = ds.get_data(0)
    check("coord" in data, "coord present")
    check("segment" in data, "segment present")
    check("instance" in data, "instance present")
    check(data["coord"].shape[0] > 0, "sparse entries present")
    check(len(np.unique(data["instance"])) > 1, "multiple instances present")
    check(len(np.unique(data["segment"])) >= 1, "categories present")


def test_sensor_separate():
    ds = make_ds(modalities=("sensor",), output_mode="separate")
    data = ds.get_data(0)
    check("pmt_pe" in data, "pmt_pe present")
    check("pmt_t" in data, "pmt_t present")
    check("pp_sensor_idx" in data, "pp_sensor_idx present")
    check("pp_category" in data, "pp_category present")
    check("coord" not in data, "no top-level coord")


def test_seg_only():
    data = make_ds(modalities=("seg",)).get_data(0)
    check(data["coord"].shape[1] == 3, "coord is 3D")
    check(data["energy"].shape[1] == 1, "energy has one feature")
    check("track_ids" in data, "track_ids present")
    check("pdg" in data, "pdg present")


def test_mixed_separate():
    data = make_ds(modalities=("seg", "sensor"), output_mode="separate").get_data(0)
    check(any(key.startswith("seg3d.") for key in data), "seg3d keys present")
    check("pmt_pe" in data, "pmt_pe present")


def test_pipeline_response():
    transform = [
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord",), feat_keys=("coord", "energy", "time")),
    ]
    ds = make_ds(
        modalities=("sensor",),
        output_mode="response",
        include_labels=False,
        transform=transform,
    )
    sample = ds[0]
    check(sample["feat"].shape[1] == sample["coord"].shape[1] + 2, "feature width")
    check("offset" in sample, "offset present")

    batch = collate_fn([ds[0], ds[1]])
    first_size = ds.get_data(0)["coord"].shape[0]
    check(batch["coord"].shape[0] > first_size, "batch combines events")
    check(len(batch["offset"]) == 2, "two offsets present")


def test_pipeline_labels():
    transform = [
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "segment", "instance"),
            feat_keys=("coord", "energy"),
        ),
    ]
    ds = make_ds(
        modalities=("sensor",),
        output_mode="labels",
        include_labels=True,
        transform=transform,
    )
    batch = collate_fn([ds[0], ds[1]])
    check("segment" in batch, "segment in batch")
    check(len(batch["offset"]) == 2, "offset correct")

    model = nn.Linear(batch["feat"].shape[1], max(4, len(torch.unique(batch["segment"]))))
    loss = nn.CrossEntropyLoss(ignore_index=-1)(
        model(batch["feat"]),
        batch["segment"].long(),
    )
    loss.backward()
    check(model.weight.grad is not None, "gradients computed")


def test_pipeline_seg():
    transform = [
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord",), feat_keys=("coord", "energy")),
    ]
    ds = make_ds(modalities=("seg",), transform=transform)
    batch = collate_fn([ds[0], ds[1]])
    check(batch["coord"].shape[1] == 3, "batch is 3D")
    check(len(batch["offset"]) == 2, "offset correct")


def test_dataloader():
    transform = [
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord",), feat_keys=("coord", "energy")),
    ]
    ds = make_ds(
        modalities=("sensor",),
        output_mode="response",
        include_labels=False,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        persistent_workers=False,
    )
    batch = next(iter(loader))
    check(batch["coord"].shape[0] > 0, "dataloader batch is nonempty")
