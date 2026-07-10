# pimm documentation

Source for the pimm documentation site, published at
<https://deeplearnphysics.org/particle-imaging-models/stable/>.

Built with [Sphinx](https://www.sphinx-doc.org), [MyST
Markdown](https://myst-parser.readthedocs.io), and the [PyData Sphinx
Theme](https://pydata-sphinx-theme.readthedocs.io). The narrative guides need
only the doc dependencies, but the **API reference uses `autodoc`** (and
`gen_api.py` enumerates the live registries), so a full `make html` **imports
`pimm`** and must run in the full project environment or the project image.
See `.github/workflows/docs.yml` for the CI build.

## Build

```bash
uv sync --locked --group docs
uv run --group docs make -C docs html
```

The first build is slow (autodoc imports the package and generates a page per
registered class); subsequent builds are incremental.

> Narrative-only preview without importing `pimm`: temporarily exclude the
> `api/` tree (e.g. `sphinx-build -b html source build/html -D exclude_patterns=api/**`).
> The full site — including the API reference — needs the project environment.

Then open `docs/build/html/index.html`, or serve it:

```bash
make -C docs serve   # http://localhost:8000
```

## Layout

```text
docs/
  Makefile             # make html / clean / serve / linkcheck
  source/
    conf.py            # Sphinx + theme configuration
    index.md           # landing page
    _static/           # logo, custom.css
    getting_started/   # install, quickstart, mental model
    distributed/       # DDP / FSDP2 / multi-node
    hpc/               # Slurm, sites, chaining, monitoring, resuming
    checkpoints/       # formats, hooks, export, Hugging Face
    models/            # from_pretrained, data format for inference
    configuration/     # the Python config system
    datasets/          # datasets, transforms, packed format
    hooks/             # the hook system
    evaluation/        # evaluators and testing
    tutorials/         # end-to-end walkthroughs
    reference/         # CLI reference, model zoo
```

## Conventions

- Pages are Markdown (MyST). Use `:::{note}`-style admonitions and
  `sphinx-design` directives (`grid`, `card`, `tab-set`) for richer layout.
- Keep execution details (Slurm, accounts, containers) in the HPC pages and
  model/training behavior in the configuration/datasets pages, mirroring the
  repo's own launch-YAML vs Python-config split.
- The site tracks the code. When launch flags, config keys, or checkpoint
  formats change, update the matching page.
