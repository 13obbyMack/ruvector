//! Persistent scope-sharded vector index.

use crate::{ContextIndexError, ContextNamespace, ContextScope, Result};
use ruvector_core::types::DbOptions;
use ruvector_core::{SearchQuery, VectorDB, VectorEntry};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

const MANIFEST_KEY: &str = "ruvector_context_scope_v1";
const MAX_POINT_ID_BYTES: usize = 512;

/// Resource and vector-engine configuration for a scoped context index.
#[derive(Debug, Clone)]
pub struct ContextIndexOptions {
    /// Base configuration copied into every exact-scope vector shard.
    pub vector: DbOptions,
    /// Maximum exact-scope shards this process may open or create.
    pub max_scopes: usize,
    /// Maximum descendant shards one search may touch.
    pub max_search_shards: usize,
    /// Maximum top K accepted from a caller.
    pub max_results: usize,
}

impl ContextIndexOptions {
    /// Validate resource bounds before opening persistent state.
    pub fn validate(&self) -> Result<()> {
        if self.vector.dimensions == 0 {
            return Err(ContextIndexError::InvalidConfiguration(
                "vector dimensions must be nonzero",
            ));
        }
        if self.max_scopes == 0 || self.max_search_shards == 0 || self.max_results == 0 {
            return Err(ContextIndexError::InvalidConfiguration(
                "resource limits must be nonzero",
            ));
        }
        Ok(())
    }
}

/// Immutable point registered in one exact context scope.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextPoint {
    /// Stable opaque identifier, normally derived from revision and view.
    pub id: String,
    /// Embedding in the configured vector space.
    pub vector: Vec<f32>,
}

/// One globally ranked match from an authorized scope subtree.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextMatch {
    /// Stable opaque point identifier.
    pub id: String,
    /// Distance score where lower values rank first.
    pub score: f32,
    /// Exact physical scope containing the point.
    pub scope: ContextScope,
}

/// Operational counters for one exact scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeStats {
    /// Number of vectors currently retained in the shard.
    pub vectors: usize,
    /// Number of ANN searches issued to this shard since process start.
    pub searches: u64,
}

struct Shard {
    db: VectorDB,
    searches: AtomicU64,
}

type PathShards = BTreeMap<Vec<String>, Arc<Shard>>;
type NamespaceShards = BTreeMap<ContextNamespace, PathShards>;

/// Persistent vector index physically partitioned by exact context scope.
pub struct ScopedContextIndex {
    root: PathBuf,
    options: ContextIndexOptions,
    shards: RwLock<NamespaceShards>,
}

impl ScopedContextIndex {
    /// Open an index directory and recover every hash-bound scope shard.
    pub fn open(root: impl AsRef<Path>, options: ContextIndexOptions) -> Result<Self> {
        options.validate()?;
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        let mut shards = BTreeMap::new();
        for entry in std::fs::read_dir(&root)? {
            let path = entry?.path();
            if !is_shard_path(&path) {
                continue;
            }
            let (scope, shard) = load_shard(&path, &options)?;
            let filename = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            if filename != scope.shard_filename() {
                return Err(ContextIndexError::CorruptShard(filename.to_string()));
            }
            shards
                .entry(scope.namespace().clone())
                .or_insert_with(BTreeMap::new)
                .insert(scope.path().to_vec(), Arc::new(shard));
        }
        let index = Self {
            root,
            options,
            shards: RwLock::new(shards),
        };
        if index.scope_count()? > index.options.max_scopes {
            return Err(ContextIndexError::ScopeCapacity {
                maximum: index.options.max_scopes,
            });
        }
        Ok(index)
    }

    /// Insert an immutable point, returning success for an identical replay.
    pub fn insert(&self, scope: &ContextScope, point: ContextPoint) -> Result<()> {
        validate_point(&point, self.options.vector.dimensions)?;
        let mut all = self
            .shards
            .write()
            .map_err(|_| ContextIndexError::LockPoisoned)?;
        let exists = all
            .get(scope.namespace())
            .and_then(|paths| paths.get(scope.path()))
            .cloned();
        let shard = if let Some(shard) = exists {
            shard
        } else {
            if count_scopes(&all) >= self.options.max_scopes {
                return Err(ContextIndexError::ScopeCapacity {
                    maximum: self.options.max_scopes,
                });
            }
            let shard = Arc::new(create_shard(&self.root, scope, &self.options)?);
            all.entry(scope.namespace().clone())
                .or_insert_with(BTreeMap::new)
                .insert(scope.path().to_vec(), Arc::clone(&shard));
            shard
        };
        if let Some(existing) = shard.db.get(&point.id)? {
            return if existing.vector == point.vector {
                Ok(())
            } else {
                Err(ContextIndexError::ImmutableConflict)
            };
        }
        shard.db.insert(VectorEntry {
            id: Some(point.id),
            vector: point.vector,
            metadata: None,
        })?;
        Ok(())
    }

    /// Search only exact shards at or below an already authorized root scope.
    pub fn search(
        &self,
        root: &ContextScope,
        vector: &[f32],
        k: usize,
    ) -> Result<Vec<ContextMatch>> {
        if k == 0 || k > self.options.max_results {
            return Err(ContextIndexError::ResultLimit {
                requested: k,
                maximum: self.options.max_results,
            });
        }
        if vector.len() != self.options.vector.dimensions {
            return Err(ruvector_core::RuvectorError::DimensionMismatch {
                expected: self.options.vector.dimensions,
                actual: vector.len(),
            }
            .into());
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(ContextIndexError::NonFiniteVector);
        }
        let all = self
            .shards
            .read()
            .map_err(|_| ContextIndexError::LockPoisoned)?;
        let Some(paths) = all.get(root.namespace()) else {
            return Ok(Vec::new());
        };
        let selected = paths
            .range(root.path().to_vec()..)
            .take_while(|(path, _)| path.starts_with(root.path()))
            .collect::<Vec<_>>();
        if selected.len() > self.options.max_search_shards {
            return Err(ContextIndexError::ScopeFanout {
                actual: selected.len(),
                maximum: self.options.max_search_shards,
            });
        }
        let mut matches = Vec::with_capacity(k);
        for (path, shard) in selected {
            shard.searches.fetch_add(1, Ordering::Relaxed);
            let results = shard.db.search(SearchQuery {
                vector: vector.to_vec(),
                k,
                filter: None,
                ef_search: None,
            })?;
            let scope = ContextScope::new(root.namespace().clone(), path.clone())?;
            for result in results {
                push_top_match(
                    &mut matches,
                    k,
                    ContextMatch {
                        id: result.id,
                        score: result.score,
                        scope: scope.clone(),
                    },
                );
            }
        }
        matches.sort_unstable_by(compare_matches);
        Ok(matches)
    }

    /// Delete one point from one exact scope.
    pub fn delete_point(&self, scope: &ContextScope, id: &str) -> Result<bool> {
        validate_point_id(id)?;
        let all = self
            .shards
            .read()
            .map_err(|_| ContextIndexError::LockPoisoned)?;
        let Some(shard) = all
            .get(scope.namespace())
            .and_then(|paths| paths.get(scope.path()))
        else {
            return Ok(false);
        };
        shard.db.delete(id).map_err(Into::into)
    }

    /// Erase one exact scope and its persistent vector shard.
    pub fn erase_scope(&self, scope: &ContextScope) -> Result<bool> {
        let mut all = self
            .shards
            .write()
            .map_err(|_| ContextIndexError::LockPoisoned)?;
        let Some(paths) = all.get_mut(scope.namespace()) else {
            return Ok(false);
        };
        let Some(shard) = paths.remove(scope.path()) else {
            return Ok(false);
        };
        if !paths.is_empty() {
            drop(shard);
        } else {
            all.remove(scope.namespace());
            drop(shard);
        }
        std::fs::remove_file(self.root.join(scope.shard_filename()))?;
        Ok(true)
    }

    /// Return counters for one exact scope without exposing other scopes.
    pub fn scope_stats(&self, scope: &ContextScope) -> Result<Option<ScopeStats>> {
        let all = self
            .shards
            .read()
            .map_err(|_| ContextIndexError::LockPoisoned)?;
        let Some(shard) = all
            .get(scope.namespace())
            .and_then(|paths| paths.get(scope.path()))
        else {
            return Ok(None);
        };
        Ok(Some(ScopeStats {
            vectors: shard.db.len()?,
            searches: shard.searches.load(Ordering::Relaxed),
        }))
    }

    /// Number of exact-scope shards currently open.
    pub fn scope_count(&self) -> Result<usize> {
        let all = self
            .shards
            .read()
            .map_err(|_| ContextIndexError::LockPoisoned)?;
        Ok(count_scopes(&all))
    }
}

fn create_shard(root: &Path, scope: &ContextScope, options: &ContextIndexOptions) -> Result<Shard> {
    let path = root.join(scope.shard_filename());
    let mut vector = options.vector.clone();
    vector.storage_path = path.to_string_lossy().into_owned();
    let db = VectorDB::new(vector)?;
    let manifest = serde_json::to_string(scope)?;
    if let Err(error) = db.save_config_value(MANIFEST_KEY, &manifest) {
        drop(db);
        let _ = std::fs::remove_file(&path);
        return Err(error.into());
    }
    Ok(Shard {
        db,
        searches: AtomicU64::new(0),
    })
}

fn load_shard(path: &Path, options: &ContextIndexOptions) -> Result<(ContextScope, Shard)> {
    let mut vector = options.vector.clone();
    vector.storage_path = path.to_string_lossy().into_owned();
    let db = VectorDB::new(vector)?;
    if db.options().dimensions != options.vector.dimensions
        || db.options().distance_metric != options.vector.distance_metric
    {
        return Err(ContextIndexError::CorruptShard(path.display().to_string()));
    }
    let manifest = db
        .load_config_value(MANIFEST_KEY)?
        .ok_or_else(|| ContextIndexError::CorruptShard(path.display().to_string()))?;
    let scope: ContextScope = serde_json::from_str(&manifest)
        .map_err(|_| ContextIndexError::CorruptShard(path.display().to_string()))?;
    Ok((
        scope,
        Shard {
            db,
            searches: AtomicU64::new(0),
        },
    ))
}

fn validate_point(point: &ContextPoint, dimensions: usize) -> Result<()> {
    validate_point_id(&point.id)?;
    if point.vector.len() != dimensions {
        return Err(ruvector_core::RuvectorError::DimensionMismatch {
            expected: dimensions,
            actual: point.vector.len(),
        }
        .into());
    }
    if point.vector.iter().any(|value| !value.is_finite()) {
        return Err(ContextIndexError::NonFiniteVector);
    }
    Ok(())
}

fn validate_point_id(id: &str) -> Result<()> {
    if id.is_empty()
        || id.len() > MAX_POINT_ID_BYTES
        || !id.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b':')
        })
    {
        return Err(ContextIndexError::InvalidPointId);
    }
    Ok(())
}

fn count_scopes(shards: &NamespaceShards) -> usize {
    shards.values().map(BTreeMap::len).sum()
}

fn is_shard_path(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    name.len() == 69
        && name.ends_with(".redb")
        && name[..64]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn compare_matches(left: &ContextMatch, right: &ContextMatch) -> core::cmp::Ordering {
    left.score
        .partial_cmp(&right.score)
        .unwrap_or(core::cmp::Ordering::Equal)
        .then_with(|| left.id.cmp(&right.id))
        .then_with(|| left.scope.cmp(&right.scope))
}

fn push_top_match(matches: &mut Vec<ContextMatch>, limit: usize, candidate: ContextMatch) {
    if matches.len() < limit {
        matches.push(candidate);
        return;
    }
    let worst = matches
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| compare_matches(left, right))
        .map_or(0, |(index, _)| index);
    if compare_matches(&candidate, &matches[worst]).is_lt() {
        matches[worst] = candidate;
    }
}
