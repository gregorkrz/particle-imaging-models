import pytest


pytestmark = pytest.mark.external_model


def _assert_bare_uri_loads(uri, monkeypatch):
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")

    import pimm

    model = pimm.from_pretrained(uri, device="cpu")
    assert not model.training
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_bare_semantic_uri(monkeypatch):
    _assert_bare_uri_loads(
        "hf://DeepLearnPhysics/panda-semantic",
        monkeypatch,
    )


def test_bare_particle_uri(monkeypatch):
    _assert_bare_uri_loads(
        "hf://DeepLearnPhysics/panda-particle",
        monkeypatch,
    )
