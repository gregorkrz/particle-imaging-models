# Releasing pimm

## Versions

Stable releases use `vMAJOR.MINOR.PATCH` tags matching the `version` in `pyproject.toml`, which is the source of truth.
Bump MAJOR for incompatible changes, MINOR for backward-compatible features, PATCH for fixes.

## Cut a release

1. Create the draft and curate its notes:

```bash
gh release create vX.Y.Z --draft --generate-notes --title "pimm vX.Y.Z" --target main
```

2. Open a PR that bumps the `pyproject.toml` version and refreshes the lock (`uv lock`).
3. Merge it to `main`.

Merging the version bump triggers the Release workflow, which verifies the lock and build, creates the annotated `vX.Y.Z` tag on the merged commit, builds the version-tagged containers, and publishes the draft.

## Retry

Re-run the failed jobs, or dispatch Release manually with `tag` set to `vX.Y.Z` and `ref` set to the release commit SHA (or the existing tag).
Never move a published tag; roll back with a new PATCH release.

## Native wheels

Native wheel tags (`pimm-wheels-*`) are an independent stream, rebuilt only when the CUDA/PyTorch/Python/architecture stack changes.
Run the Native wheel draft release workflow manually; application releases never rebuild them.
