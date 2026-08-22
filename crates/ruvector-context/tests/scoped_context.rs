//! Integration tests for persistent, scope-isolated context indexes.

use ruvector_context::{
    ContextIndexError, ContextIndexOptions, ContextNamespace, ContextPoint, ContextScope,
    ScopedContextIndex,
};
use ruvector_core::types::DbOptions;
use ruvector_core::DistanceMetric;

fn namespace(tenant: &str) -> ContextNamespace {
    ContextNamespace::new("context.example", tenant, "agent", "reader", "memory").unwrap()
}

fn scope(tenant: &str, path: &[&str]) -> ContextScope {
    ContextScope::new(
        namespace(tenant),
        path.iter().map(|value| (*value).to_string()).collect(),
    )
    .unwrap()
}

fn options(max_search_shards: usize) -> ContextIndexOptions {
    ContextIndexOptions {
        vector: DbOptions {
            dimensions: 3,
            distance_metric: DistanceMetric::Euclidean,
            storage_path: String::new(),
            hnsw_config: None,
            quantization: None,
        },
        max_scopes: 16,
        max_search_shards,
        max_results: 8,
    }
}

fn point(id: &str, vector: [f32; 3]) -> ContextPoint {
    ContextPoint {
        id: id.to_string(),
        vector: vector.to_vec(),
    }
}

#[test]
fn cross_tenant_search_never_touches_other_tenant_index() {
    let temp = tempfile::tempdir().unwrap();
    let index = ScopedContextIndex::open(temp.path(), options(8)).unwrap();
    let acme = scope("acme", &["project"]);
    let other = scope("other", &["project"]);
    index
        .insert(&acme, point("acme:one", [1.0, 0.0, 0.0]))
        .unwrap();
    index
        .insert(&other, point("other:one", [1.0, 0.0, 0.0]))
        .unwrap();

    let matches = index.search(&acme, &[1.0, 0.0, 0.0], 4).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].id, "acme:one");
    assert_eq!(index.scope_stats(&acme).unwrap().unwrap().searches, 1);
    assert_eq!(index.scope_stats(&other).unwrap().unwrap().searches, 0);
}

#[test]
fn path_prefix_selects_descendants_without_touching_siblings() {
    let temp = tempfile::tempdir().unwrap();
    let index = ScopedContextIndex::open(temp.path(), options(8)).unwrap();
    let root = scope("acme", &["project"]);
    let child = scope("acme", &["project", "doc"]);
    let sibling = scope("acme", &["project-archive"]);
    index
        .insert(&child, point("child", [0.0, 1.0, 0.0]))
        .unwrap();
    index
        .insert(&sibling, point("sibling", [0.0, 1.0, 0.0]))
        .unwrap();

    let matches = index.search(&root, &[0.0, 1.0, 0.0], 4).unwrap();
    assert_eq!(
        matches
            .iter()
            .map(|item| item.id.as_str())
            .collect::<Vec<_>>(),
        vec!["child"]
    );
    assert_eq!(index.scope_stats(&child).unwrap().unwrap().searches, 1);
    assert_eq!(index.scope_stats(&sibling).unwrap().unwrap().searches, 0);
}

#[test]
fn restart_recovers_hash_bound_scope_and_vectors() {
    let temp = tempfile::tempdir().unwrap();
    let target = scope("acme", &["durable"]);
    {
        let index = ScopedContextIndex::open(temp.path(), options(8)).unwrap();
        index
            .insert(&target, point("revision:view", [0.0, 0.0, 1.0]))
            .unwrap();
    }
    let recovered = ScopedContextIndex::open(temp.path(), options(8)).unwrap();
    let matches = recovered.search(&target, &[0.0, 0.0, 1.0], 1).unwrap();
    assert_eq!(matches[0].id, "revision:view");
}

#[test]
fn immutable_replay_is_idempotent_but_conflict_is_refused() {
    let temp = tempfile::tempdir().unwrap();
    let index = ScopedContextIndex::open(temp.path(), options(8)).unwrap();
    let target = scope("acme", &["immutable"]);
    index
        .insert(&target, point("same", [1.0, 0.0, 0.0]))
        .unwrap();
    index
        .insert(&target, point("same", [1.0, 0.0, 0.0]))
        .unwrap();
    assert!(matches!(
        index.insert(&target, point("same", [0.0, 1.0, 0.0])),
        Err(ContextIndexError::ImmutableConflict)
    ));
}

#[test]
fn fanout_failure_occurs_before_any_shard_search() {
    let temp = tempfile::tempdir().unwrap();
    let index = ScopedContextIndex::open(temp.path(), options(1)).unwrap();
    let root = ContextScope::root(namespace("acme"));
    let one = scope("acme", &["one"]);
    let two = scope("acme", &["two"]);
    index.insert(&one, point("one", [1.0, 0.0, 0.0])).unwrap();
    index.insert(&two, point("two", [0.0, 1.0, 0.0])).unwrap();
    assert!(matches!(
        index.search(&root, &[1.0, 0.0, 0.0], 1),
        Err(ContextIndexError::ScopeFanout {
            actual: 2,
            maximum: 1
        })
    ));
    assert_eq!(index.scope_stats(&one).unwrap().unwrap().searches, 0);
    assert_eq!(index.scope_stats(&two).unwrap().unwrap().searches, 0);
}

#[test]
fn exact_scope_erasure_removes_persistent_shard() {
    let temp = tempfile::tempdir().unwrap();
    let target = scope("acme", &["erase"]);
    let index = ScopedContextIndex::open(temp.path(), options(8)).unwrap();
    index
        .insert(&target, point("erase-me", [1.0, 0.0, 0.0]))
        .unwrap();
    assert!(index.erase_scope(&target).unwrap());
    assert_eq!(index.scope_count().unwrap(), 0);
    drop(index);
    assert_eq!(
        ScopedContextIndex::open(temp.path(), options(8))
            .unwrap()
            .scope_count()
            .unwrap(),
        0
    );
}

#[test]
fn erased_scope_recreation_never_reuses_the_unlinked_database() {
    let temp = tempfile::tempdir().unwrap();
    let target = scope("acme", &["erase-recreate"]);
    let index = ScopedContextIndex::open(temp.path(), options(8)).unwrap();
    index
        .insert(&target, point("old", [1.0, 0.0, 0.0]))
        .unwrap();
    assert!(index.erase_scope(&target).unwrap());
    index
        .insert(&target, point("new", [0.0, 1.0, 0.0]))
        .unwrap();

    let matches = index.search(&target, &[1.0, 0.0, 0.0], 8).unwrap();
    assert_eq!(
        matches
            .iter()
            .map(|item| item.id.as_str())
            .collect::<Vec<_>>(),
        vec!["new"]
    );
}

#[test]
fn non_finite_vectors_fail_before_index_access() {
    let temp = tempfile::tempdir().unwrap();
    let index = ScopedContextIndex::open(temp.path(), options(8)).unwrap();
    let target = scope("acme", &["finite"]);
    assert!(matches!(
        index.insert(&target, point("nan", [f32::NAN, 0.0, 0.0])),
        Err(ContextIndexError::NonFiniteVector)
    ));
    assert!(matches!(
        index.search(&target, &[f32::INFINITY, 0.0, 0.0], 1),
        Err(ContextIndexError::NonFiniteVector)
    ));
    assert_eq!(index.scope_count().unwrap(), 0);
}
