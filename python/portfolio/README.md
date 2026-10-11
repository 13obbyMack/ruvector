# RuVector Python package portfolio

Focused Python APIs backed by existing RuVector Rust libraries. Release candidates, not yet verified as published.

| Distribution | Import | Rust implementation | Task |
| :--- | :--- | :--- | :--- |
| ruvector | ruvector | ruvector-core, ruvector-rabitq | Vector search, collection API, CLI and MCP |
| ruvector-graph | ruvector_graph | ruvector-graph | Property graphs and bounded traversal |
| ruvector-cluster | ruvector_cluster | ruvector-cluster-rag | Embedding clustering and cohesion |
| ruvector-rerank | ruvector_rerank | ruvector-attention | Attention weights and candidate ordering |

All three new distributions depend on the same exact native runtime, `ruvector==0.1.2`. Installing all three installs that runtime once. They add useful APIs rather than duplicating Rust binaries. They do not reduce native download size. KGE, quantization and independent native wheels remain future work, not claimed features.

## Build and test

```bash
python -m pip install build maturin pytest numpy twine
maturin build --manifest-path crates/ruvector-py/Cargo.toml --out /tmp/native
python -m pip install /tmp/native/*.whl
for package in python/portfolio/ruvector-*; do
  python -m build "$package"
  python -m pip install "$package"/dist/*.whl
done
python -m pytest python/portfolio/tests -q
```

## Automatic publishing

Merging a new portfolio version into `main` automatically builds, tests, audits and publishes the exact tested artifacts. The first addition of `manifest.json` counts as a new version. Documentation edits without a version change do not publish. An explicit matching `python-portfolio-v0.1.0` tag is also supported. No recurring task is required. PRs build and test but cannot publish. The release tag must point to a commit on main. Configure the `pypi` GitHub environment to accept protected main and release tags. For unattended releases, configure the environment without a manual reviewer, retaining branch protections and CI gates.

Authentication supports an environment secret named `PYPI_API_TOKEN`. When present, only the isolated publish step receives it; otherwise OIDC Trusted Publishing is used. Token publishing does not generate OIDC attestations. Configure the token directly from its protected ruOS binding into the GitHub environment using authenticated GitHub tooling; never print it or put it in command arguments. This credential configuration has not yet been completed.

For OIDC, register each package's PyPI Trusted Publisher with owner `ruvnet`, repository `RuVector`, workflow `python-portfolio.yml`, environment `pypi`. Confirm name ownership or availability on PyPI; a missing public project is not proof that its name is allocatable. Publish the required runtime version first. The release gate also tests against the published runtime wheel, avoiding accidental dependence on unreleased native APIs.

The native runtime keeps its existing `python-v*` wheel workflow. Portfolio packages share a version and release tag, but PyPI publication is not transactional: a partial upload requires checking receipts and publishing only the missing artifacts. Never overwrite releases or silently skip conflicting files.

## Search discovery

Each package includes a descriptive title, PyPI summary, keywords, repository/documentation links, explicit problem statement, installation command, executable example and limitations. Animated SVGs include accessible titles, descriptions and reduced motion support; static fallback content remains readable. Search rankings are not guaranteed. Prefer measured user adoption and successful installs over package count.

After publishing, Linux, macOS and Windows jobs install the exact versions directly from PyPI in fresh virtual environments and execute portfolio tests and examples. A verification failure is reported as a failed release workflow; uploaded versions are immutable and require a corrective release, not an overwrite.
