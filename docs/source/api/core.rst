====================
Core building blocks
====================

The shared abstractions used across pimm: the config-driven builders, the
``Point`` structure, the config loader, the registry, and the
checkpoint/resume primitives.

Builders
========

Construct registered objects from config dictionaries. See the per-registry
pages (:doc:`Models <registry/models>`, :doc:`Datasets <registry/datasets>`, …)
for what each registry contains.

.. currentmodule:: pimm.models.builder

.. autosummary::
   :toctree: generated
   :nosignatures:

   build_model

.. currentmodule:: pimm.datasets.builder

.. autosummary::
   :toctree: generated
   :nosignatures:

   build_dataset

.. currentmodule:: pimm.models.losses.builder

.. autosummary::
   :toctree: generated
   :nosignatures:

   build_criteria

Point structures & base modules
===============================

.. currentmodule:: pimm.models.utils.structure

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   Point

.. currentmodule:: pimm.models.modules

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   PointModule
   PointSequential
   PointModel

Configuration & registry
========================

.. currentmodule:: pimm.utils.config

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   Config

.. currentmodule:: pimm.utils.registry

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   Registry

Checkpointing & resume
======================

The checkpoint manager and the resume-state schema. See
:doc:`../checkpoints/index`.

.. currentmodule:: pimm.utils.checkpoints

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   CheckpointManager

.. currentmodule:: pimm.engines._train_utils

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   TrainState

Data loading & collation
=========================

The collate functions that pack ragged point-cloud samples into a batch, and the
checkpointable sampler that lets the training loader resume mid-epoch. See
:doc:`../research_ecosystem/contributing_a_dataset`.

.. currentmodule:: pimm.datasets.utils

.. autosummary::
   :toctree: generated
   :nosignatures:

   collate_fn

.. currentmodule:: pimm.datasets.stateful

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   StatefulRandomSampler

Distributed helpers
===================

Process-group setup and small collectives (:mod:`pimm.utils.comm`).

.. currentmodule:: pimm.utils.comm

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_world_size
   get_rank
   is_main_process
   synchronize
   all_gather
   reduce_dict
