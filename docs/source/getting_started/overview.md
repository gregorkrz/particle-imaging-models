# What is pimm?

pimm trains deep-learning models on sparse detector data from high-energy-physics experiments.
It provides model backbones, self-supervised pretraining recipes, and task models for fine-tuning, along with the data loaders, training loop, and launch tooling around them.
The same command that trains on one GPU scales to many GPUs across several nodes.

In pimm, an event is a variable-length set of hits, each with a position and per-hit features such as charge or time.
This representation is detector-neutral: liquid-argon TPCs, water-Cherenkov detectors, and wire-plane readouts all produce data in this form, and the same models run on all of them.

## Foundation models

Supervised training in HEP has usually leaned on simulation: experiments invest in detailed detector simulations to generate the large labeled samples that training from scratch requires, and model quality then depends on how faithful the simulation is.
Foundation models loosen that dependence.
A backbone pretrained on raw, unlabeled events learns the structure of real data directly, so the labeled sample for a downstream task can be small enough to produce by hand - a few hundred or a few thousand events - and still reach good performance.

## Pretraining

Experiments record far more events than ever get labeled.
pimm's pretraining recipes use those unlabeled events directly: a backbone is trained self-supervised, for example by masking part of an event and training the network to reconstruct it.
The result is a backbone whose learned features transfer across tasks, and often across detectors.

## Fine-tuning

To solve a specific task, you attach a task head to a pretrained backbone and train on your labeled events.
Typical tasks are semantic segmentation (a class label per hit), panoptic reconstruction (hits grouped into individual particles and interactions), and event classification or particle ID.
Because the backbone already carries general features, fine-tuning needs substantially fewer labeled events than training from scratch.

You do not have to pretrain your own backbone.
Published checkpoints on the Hugging Face Hub can be loaded with one call and fine-tuned directly; {doc}`../research_ecosystem/using_trained_models` lists them.

## Next

- {doc}`installation` - get pimm running, in a container or from source.
