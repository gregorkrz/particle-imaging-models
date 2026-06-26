=============
API reference
=============

Reference for pimm's public Python API, generated from the source docstrings
with Sphinx ``autodoc`` (the same setup torchrl uses). For narrative guides,
start with :doc:`../getting_started/index`.

.. note::

   The API reference imports ``pimm``, so build it from the project environment
   (the one that can ``import pimm``), e.g.
   ``conda run -n pointcept-torch2.5.0-cu12.4 make -C docs html``. The
   per-registry pages are regenerated on every build from the live registries.

Registries
==========

pimm builds everything from config dictionaries with a ``type`` key, resolved
through small registries (see :doc:`../getting_started/concepts`). Each page
below is generated from the live registry, grouped by role, and lists every
registered ``type`` with a link to its autodoc page.

- :doc:`Models <registry/models>` — models & backbones (``model = dict(type=...)``).
- :doc:`Datasets <registry/datasets>` — dataset classes (``data.train = dict(type=...)``).
- :doc:`Transforms <registry/transforms>` — transform pipeline steps.
- :doc:`Hooks <registry/hooks>` — training lifecycle hooks.
- :doc:`Losses <registry/losses>` — loss functions for ``build_criteria``.
- :doc:`Trainers <registry/trainers>` — trainer classes (``train.type``).

Core API
========

Hand-curated reference for the functions and classes that are not built from a
registry — model loading/export and the shared building blocks.

- :doc:`Loading & export <loading>` — :func:`~pimm.from_pretrained`,
  :func:`~pimm.save_pretrained`, :func:`~pimm.push_to_hub`, and the state-dict
  helpers.
- :doc:`Core building blocks <core>` — builders, the ``Point`` structure, the
  ``Config`` loader, the ``Registry``, and the checkpoint/resume primitives.

.. toctree::
   :hidden:
   :caption: Registries

   Models <registry/models>
   Datasets <registry/datasets>
   Transforms <registry/transforms>
   Hooks <registry/hooks>
   Losses <registry/losses>
   Trainers <registry/trainers>

.. toctree::
   :hidden:
   :caption: Core API

   loading
   core
