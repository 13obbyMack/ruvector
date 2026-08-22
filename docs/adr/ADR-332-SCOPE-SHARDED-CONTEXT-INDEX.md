# ADR-332: Scope-Sharded Context Vector Index

**Status:** Proposed

**Date:** 2026-08-22
**Owners:** RuVector maintainers

## Context

RVM now defines capability-governed `ruv://` names for resources, memories,
and skills. Its authorization boundary runs before a retrieval backend is
called. RuVector must preserve that boundary once the authorized request
reaches vector retrieval.

The existing `VectorDB::search` traverses one ANN graph and applies optional
metadata filters after candidates are returned. That behavior is useful for
ordinary filtering but cannot provide tenant isolation. Traversing a shared
graph can make latency, result quality, and resource use depend on objects the
caller was not authorized to enumerate.

## Decision

Add `ruvector-context`, a small persistent crate that physically separates
every exact structured context scope into its own `VectorDB` shard.

A scope contains exact authority, tenant, subject kind, subject ID,
collection, and path segments. Scope components are validated as a strict
ASCII contract. The on-disk filename is SHA-256 over a domain-separated,
length-delimited encoding; tenant and path names never enter filesystem paths.

Search accepts an already authorized root scope. It performs an exact
namespace lookup, selects only lexicographic descendant path shards, checks a
fanout ceiling, and only then invokes ANN search. Results from selected shards
are merged into one globally ranked bounded top K.

Each shard stores its complete scope manifest inside the same persistent
vector database. Recovery verifies that the manifest hashes to the shard
filename before the shard becomes queryable. Missing, malformed, renamed, or
configuration-incompatible shards fail closed.

RVM remains responsible for authenticating actors and capabilities. RuVector
does not parse capability handles or infer authority from a URI.

## Invariants

1. A search for namespace A never calls an ANN index belonging to namespace B.
2. A path-prefix search calls only exact shards at or below that structured
   prefix; text-prefix comparison is forbidden.
3. Fanout and top K limits are validated before the first ANN call.
4. Point IDs are immutable. An identical replay is idempotent; different
   vector bytes under the same ID are rejected.
5. Scope strings never become filesystem path components.
6. Recovery authenticates the scope-to-filename binding before serving data.
7. Exact-scope erasure closes and removes the persistent shard while holding
   the catalog write lock.

## Pseudocode

```text
search(authorized_root, vector, k):
    validate vector dimensions and k
    namespace = catalog.exact_lookup(authorized_root.namespace)
    shards = namespace.lexicographic_descendants(authorized_root.path)
    require shards.count <= max_search_shards
    for shard in shards:
        candidates = shard.ann_search(vector, k)
        bounded_global_top_k.merge(candidates)
    return bounded_global_top_k.sorted()
```

Failure case: if two descendant shards exceed a configured fanout of one, the
method returns `ScopeFanout` and every shard search counter remains zero.

## Alternatives

### Filter a shared HNSW result set

Rejected. It traverses unauthorized candidates before filtering and can lose
authorized results when the initial K is dominated by another tenant.

### Prefix point IDs inside one index

Rejected. Identifier prefixes do not constrain ANN graph traversal.

### One index per tenant only

Rejected. A caller authorized for one subject or path could still influence
and observe traversal over sibling scopes in the same tenant.

### Let RuVector validate RVM capabilities

Rejected. It would duplicate kernel policy and couple the vector engine to one
runtime. The typed RVM adapter authorizes first; this crate preserves physical
retrieval isolation afterward.

## Security and Operations

The configured root directory is trusted deployment state. Untrusted scope
components are validated and represented only by a digest on disk. Resource
ceilings bound scopes, descendant fanout, and returned results. Non-finite
vectors are rejected at ingress.

The reference persistence is process-local REDB. Multi-host replication,
distributed admission control, encrypted disks, and KMS-backed erasure remain
deployment responsibilities. RVM separately owns immutable RVF objects,
linearizable aliases, signed receipt draining, and execution capabilities.

## Acceptance Evidence

The upstream crate must demonstrate:

1. Cross-tenant search returns only the authorized tenant and leaves every
   other tenant shard search counter at zero.
2. Segment-aware path selection excludes textual siblings such as `a` and
   `alpha`.
3. Restart reconstructs scope manifests and vectors.
4. Fanout refusal occurs before any ANN call.
5. Immutable replay, conflict rejection, and exact-scope erasure behave
   deterministically.
6. Formatting, unit tests, Clippy, dependency audit, and changed-file secret
   scan pass before merge.
