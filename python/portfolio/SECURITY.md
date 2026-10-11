# Security scope

Assets: release identity, package integrity, host memory, user vectors and graph data.

Entry points: numeric arrays, graph identifiers, graph traversal limits, package build inputs and GitHub release tags. No new network, shell execution, credential or unsafe deserialization API is introduced by these wrappers.

Input boundaries reject nonfinite vectors, shape mismatch and invalid budgets before entering native kernels. Native code remains a trust boundary. Array conversion can allocate before size validation, and graph neighbor enumeration materializes a list. Run untrusted jobs under external memory/CPU limits; this is not a sandbox. Attention float32 overflow is rejected on output.

Supply chain gates: pinned action commits, read only build permissions, separate publishing job with an environment scoped PyPI token or OIDC, immutable artifacts, exact native runtime dependency, main ancestry and version checks, built wheel tests, clean sdist rebuild, strict README checks, Python dependency audit, Cargo audit. Audits fail closed. The Cargo gate derives the all-target normal/build dependency closure of ruvector-py from the checked-in lockfile, checks every tree record is present, and rejects vulnerabilities and unsoundness warnings. Dev-only and unrelated workspace crates are outside shipped-wheel scope; full workspace findings remain recorded in VALIDATION.md. Registry yank status is not covered by this advisory scan. A clean advisory scan is not proof of absence of vulnerabilities.

Report issues privately through the repository's security reporting channel. Release publication requires a configured PyPI token or Trusted Publisher and a protected GitHub environment. Token presence is checked without displaying its value, and only the publishing action receives it. Merged version changes can publish unattended; protect main against unreviewed changes. Never place API tokens in source, README, command arguments or logs.
