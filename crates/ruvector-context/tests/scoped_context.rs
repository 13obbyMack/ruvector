//! Integration tests for persistent, scope-isolated context indexes.
//!
//! Every behavioural test runs against both vector engines. Isolation claims
//! about approximate-nearest-neighbor traversal are meaningless when the
//! process contains no ANN index at all, so `hnsw_config: None` alone cannot
//! establish them.

use ruvector_context::{
    ContextIndexError, ContextIndexOptions, ContextNamespace, ContextPoint, ContextScope,
    ScopedContextIndex,
};
use ruvector_core::types::{DbOptions, HnswConfig};
use ruvector_core::DistanceMetric;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
enum Engine {
    Flat,
    Hnsw,
}

impl Engine {
    fn hnsw_config(self) -> Option<HnswConfig> {
        match self {
            Self::Flat => None,
            Self::Hnsw => Some(HnswConfig {
                max_elements: 1_024,
                ..HnswConfig::default()
            }),
        }
    }
}

/// Generate one `flat` and one `hnsw` test per declared body.
macro_rules! engine_tests {
    ($(fn $name:ident($engine:ident: Engine) $body:block)*) => {
        $(
            mod $name {
                use super::*;

                fn run($engine: Engine) $body

                #[test]
                fn flat() {
                    run(Engine::Flat);
                }

                #[test]
                fn hnsw() {
                    run(Engine::Hnsw);
                }
            }
        )*
    };
}

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

fn options(engine: Engine, max_search_shards: usize) -> ContextIndexOptions {
    ContextIndexOptions {
        vector: DbOptions {
            dimensions: 3,
            distance_metric: DistanceMetric::Euclidean,
            storage_path: String::new(),
            hnsw_config: engine.hnsw_config(),
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

/// Shard filenames are opaque hashes, so tests recover them from the directory
/// rather than recomputing the private hash construction.
fn shard_files(root: &Path) -> Vec<PathBuf> {
    let mut found = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "redb")
        })
        .collect::<Vec<_>>();
    found.sort();
    found
}

/// Build a shard for `target` in a throwaway root and return its file path.
fn isolated_shard(engine: Engine, root: &Path, target: &ContextScope, id: &str) -> PathBuf {
    let index = ScopedContextIndex::open(root, options(engine, 8)).unwrap();
    index.insert(target, point(id, [1.0, 0.0, 0.0])).unwrap();
    drop(index);
    let files = shard_files(root);
    assert_eq!(files.len(), 1);
    files.into_iter().next().unwrap()
}

engine_tests! {
    fn cross_tenant_search_never_touches_other_tenant_index(engine: Engine) {
        let temp = tempfile::tempdir().unwrap();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
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

    fn path_prefix_selects_descendants_without_touching_siblings(engine: Engine) {
        let temp = tempfile::tempdir().unwrap();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
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

    fn restart_recovers_hash_bound_scope_and_vectors(engine: Engine) {
        let temp = tempfile::tempdir().unwrap();
        let target = scope("acme", &["durable"]);
        {
            let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
            index
                .insert(&target, point("revision:view", [0.0, 0.0, 1.0]))
                .unwrap();
        }
        let recovered = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        let matches = recovered.search(&target, &[0.0, 0.0, 1.0], 1).unwrap();
        assert_eq!(matches[0].id, "revision:view");
    }

    fn immutable_replay_is_idempotent_but_conflict_is_refused(engine: Engine) {
        let temp = tempfile::tempdir().unwrap();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
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

    fn deleted_point_ids_are_reusable_with_different_bytes(engine: Engine) {
        let temp = tempfile::tempdir().unwrap();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        let target = scope("acme", &["reuse"]);
        index
            .insert(&target, point("id", [1.0, 0.0, 0.0]))
            .unwrap();
        assert!(index.delete_point(&target, "id").unwrap());
        index
            .insert(&target, point("id", [0.0, 1.0, 0.0]))
            .unwrap();
        assert_eq!(index.scope_stats(&target).unwrap().unwrap().vectors, 1);
    }

    fn fanout_failure_occurs_before_any_shard_search(engine: Engine) {
        let temp = tempfile::tempdir().unwrap();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 1)).unwrap();
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

    fn exact_scope_erasure_removes_persistent_shard(engine: Engine) {
        let temp = tempfile::tempdir().unwrap();
        let target = scope("acme", &["erase"]);
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        index
            .insert(&target, point("erase-me", [1.0, 0.0, 0.0]))
            .unwrap();
        assert!(index.erase_scope(&target).unwrap());
        assert_eq!(index.scope_count().unwrap(), 0);
        assert!(shard_files(temp.path()).is_empty());
        drop(index);
        assert_eq!(
            ScopedContextIndex::open(temp.path(), options(engine, 8))
                .unwrap()
                .scope_count()
                .unwrap(),
            0
        );
    }

    fn erased_scope_recreation_never_reuses_the_unlinked_database(engine: Engine) {
        let temp = tempfile::tempdir().unwrap();
        let target = scope("acme", &["erase-recreate"]);
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
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

    fn non_finite_vectors_fail_before_index_access(engine: Engine) {
        let temp = tempfile::tempdir().unwrap();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
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

    fn planted_foreign_shard_is_refused_and_never_laundered(engine: Engine) {
        let attacker = scope("attacker", &["docs"]);
        let victim = scope("victim", &["docs"]);

        // A real, well-formed shard belonging to the attacker's scope.
        let attacker_root = tempfile::tempdir().unwrap();
        let planted =
            isolated_shard(engine, attacker_root.path(), &attacker, "attacker-planted-row");

        // The victim scope's opaque filename, learned from a throwaway root.
        let probe_root = tempfile::tempdir().unwrap();
        let victim_file = isolated_shard(engine, probe_root.path(), &victim, "probe");
        let victim_filename = victim_file.file_name().unwrap().to_owned();

        // Plant it while the victim's index is already open, so the binding
        // check performed at open time never sees the file.
        let temp = tempfile::tempdir().unwrap();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        std::fs::copy(&planted, temp.path().join(&victim_filename)).unwrap();

        let refused = index.insert(&victim, point("victim-own-row", [1.0, 0.0, 0.0]));
        let Err(ContextIndexError::ShardAdoption(reported)) = refused else {
            panic!("create adopted a pre-existing shard: {refused:?}");
        };
        assert_eq!(Some(reported.as_str()), victim_filename.to_str());
        assert!(!reported.contains(std::path::MAIN_SEPARATOR));
        assert!(index.search(&victim, &[1.0, 0.0, 0.0], 8).unwrap().is_empty());
        assert_eq!(index.scope_count().unwrap(), 0);

        // Nothing was laundered: after a restart the planted file is still
        // bound to the attacker's scope and is quarantined, not adopted.
        drop(index);
        let restarted = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        assert_eq!(
            restarted.quarantined_shards(),
            &[victim_filename.to_str().unwrap().to_string()]
        );
        assert_eq!(restarted.scope_count().unwrap(), 0);
        assert!(restarted
            .search(&victim, &[1.0, 0.0, 0.0], 8)
            .unwrap()
            .is_empty());
    }

    fn second_handle_on_one_root_fails_loudly(engine: Engine) {
        let temp = tempfile::tempdir().unwrap();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        assert!(matches!(
            ScopedContextIndex::open(temp.path(), options(engine, 8)),
            Err(ContextIndexError::RootLocked)
        ));

        // Dropping the handle releases the advisory lock.
        drop(index);
        ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
    }

    fn lock_file_left_by_a_crashed_process_does_not_brick_the_root(engine: Engine) {
        let temp = tempfile::tempdir().unwrap();
        // A process that exits without dropping its handle leaves the lock file
        // behind, but the kernel released the advisory lock on exit. The
        // leftover file must be reused, and never mistaken for a shard.
        std::fs::write(temp.path().join(".lock"), b"").unwrap();

        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        assert!(index.quarantined_shards().is_empty());
        let target = scope("acme", &["after-crash"]);
        index
            .insert(&target, point("row", [1.0, 0.0, 0.0]))
            .unwrap();
        assert_eq!(index.scope_count().unwrap(), 1);
    }

    fn unloadable_shard_is_quarantined_without_denying_open(engine: Engine) {
        let temp = tempfile::tempdir().unwrap();
        let good = scope("acme", &["healthy"]);
        {
            let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
            index
                .insert(&good, point("healthy-row", [1.0, 0.0, 0.0]))
                .unwrap();
        }
        let junk_name = format!("{}.redb", "ab".repeat(32));
        std::fs::write(temp.path().join(&junk_name), b"not a vector database").unwrap();

        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        assert_eq!(index.quarantined_shards(), &[junk_name]);
        assert_eq!(index.scope_count().unwrap(), 1);
        let matches = index.search(&good, &[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(matches[0].id, "healthy-row");
    }

    fn unbounded_max_results_is_refused_instead_of_panicking(engine: Engine) {
        let temp = tempfile::tempdir().unwrap();
        let mut unbounded = options(engine, 8);
        unbounded.max_results = usize::MAX;
        assert!(matches!(
            ScopedContextIndex::open(temp.path(), unbounded),
            Err(ContextIndexError::InvalidConfiguration(_))
        ));
    }
}

/// A failed unlink must not be reported as an erase.
///
/// Unix only: the failure is produced by revoking write permission on the
/// index root, which a process running as root would bypass.
#[cfg(unix)]
#[test]
fn failed_shard_unlink_does_not_report_a_phantom_erase() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let target = scope("acme", &["erase-fails"]);
    let index = ScopedContextIndex::open(temp.path(), options(Engine::Flat, 8)).unwrap();
    index
        .insert(&target, point("gdpr-erase-me", [1.0, 0.0, 0.0]))
        .unwrap();

    let original = std::fs::metadata(temp.path()).unwrap().permissions();
    let mut readonly = original.clone();
    readonly.set_mode(0o555);
    std::fs::set_permissions(temp.path(), readonly).unwrap();
    let probe = temp.path().join("probe");
    let enforced = std::fs::File::create(&probe).is_err();
    let erased = enforced.then(|| index.erase_scope(&target));
    std::fs::set_permissions(temp.path(), original).unwrap();
    let _ = std::fs::remove_file(&probe);

    let Some(erased) = erased else {
        return; // running as root; the permission bits are not enforced
    };
    assert!(erased.is_err(), "unlink failure reported as success");
    assert_eq!(index.scope_count().unwrap(), 1);
    assert_eq!(index.scope_stats(&target).unwrap().unwrap().vectors, 1);
    let matches = index.search(&target, &[1.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(matches[0].id, "gdpr-erase-me");
    assert_eq!(shard_files(temp.path()).len(), 1);
}
