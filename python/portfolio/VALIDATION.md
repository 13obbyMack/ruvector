# Validation receipt

Date: 2026-10-11
Base: 07933730fac709457e0c5f41444c9a4adc505aee
Platform: Linux x86_64, CPython 3.12.14, rustc 1.99.0
Cargo.lock SHA256: 88e1234ffa1178a8e86bc3b819931ef5d1be2a557c887e121b8ff4a62aea9604

## Measured results

* Three Python wheels and three sdists built with hatchling 1.27.0.
* All six pass twine 7.0.0 strict checks and release artifact validation.
* 70 tests pass against PyPI ruvector 0.1.2: portfolio contracts plus existing graph, clustering and attention/GNN tests. All three README examples execute during collection.
* 50 Rust unit tests pass for ruvector-py. The local runtime has a broken libpython symlink; a temporary linker shim to the existing libpython3.12.so.1.0 enabled testing without changing repository code.
* Source native wheel built with maturin 1.15.0; the same 70 tests pass against it.
* Clean environment installation from all three sdists: 22 portfolio tests pass, plus the three executable examples.
* pip-audit 2.10.1: no known vulnerabilities after upgrading build environment pip from 25.0.1 to 26.2.1. Local unpublished distributions cannot be looked up by name in the advisory service; source review covers their added code.
* cargo-audit 0.22.2: all-target runtime/build dependency closure passes with `--deny unsound`. 222 distinct name/version pairs, 224 lockfile records. Yank status is explicitly outside this scan.
* RustSec database commit: 7eebec69c352c7191b1f13eb95dd510eeca5d1de, updated 2026-10-09T10:12:02+02:00, 1296 advisories.
* SVG XML and workflow YAML parse; clustering SVG rendered and visually inspected. Animation timing was not browser tested.

## Findings and coverage limits

Full workspace audit reports zero vulnerability entries, but flags unsound lru 0.12.5 and 0.16.4 (RUSTSEC-2026-0253) and unmaintained rustls-pemfile (RUSTSEC-2025-0134). The Python SDK dependency tree instead contains patched lru 0.18.2 through ruvector-graph. The release gate derives its scope from `cargo tree --locked --target all --edges normal,build`, checks every selected package against Cargo.lock, and removes only inactive dependency edges. No advisory IDs are ignored.

Local native build initially failed on a damaged HNSW archive; cleaning that crate and rebuilding with two jobs succeeded. The locally built wheel targets manylinux_2_34 and is only test evidence, not a replacement for the existing manylinux_2_28 distribution pipeline.

No PyPI uploads, publisher registration, protected environment configuration or cross-platform release verification have been performed. Proposed project names returned HTTP 404 on PyPI, which does not prove registration availability or ownership. CI completion is distinct from these local measurements.

## Acceptance command

After installing the built wheels, run:

```bash
python -m pytest python/portfolio/tests -q
python python/portfolio/validate_release.py /path/to/dist
python python/portfolio/audit_native.py /tmp/portfolio-native.lock
```

Before release, require all CI gates, registered publisher identities, protected release tags, exact artifact versions and installation from PyPI in fresh supported-platform environments.
