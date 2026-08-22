//! Shard files: naming, creation, recovery, and the root lock.
//!
//! The vector engine backing one exact scope is wrapped here so the per-scope
//! isolation counter cannot be bypassed, and so the rules that decide whether a
//! file on disk may be treated as this scope's shard live in one place.

use crate::{ContextIndexError, ContextIndexOptions, ContextScope, Result};
use fs2::FileExt;
use ruvector_core::types::DbOptions;
use ruvector_core::{SearchQuery, SearchResult, VectorDB, VectorEntry};
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

const MANIFEST_KEY: &str = "ruvector_context_scope_v1";

/// Name of the exclusive root lock file held for the lifetime of one index.
///
/// It is deliberately not a valid shard name, so `is_shard_name` skips it.
const ROOT_LOCK_FILENAME: &str = ".lock";

/// Handle to the vector engine backing one exact context scope.
///
/// The wrapped [`VectorDB`] is private to this module, so the per-scope
/// isolation counter cannot be bypassed by accident: [`Shard::ann_search`] is
/// the only route to the approximate-nearest-neighbor index and it always
/// counts the traversal. Touching a shard without counting does not compile.
pub(crate) struct Shard {
    db: VectorDB,
    searches: AtomicU64,
}

impl Shard {
    /// Wrap an opened vector database as a countable shard.
    pub(crate) fn new(db: VectorDB) -> Self {
        Self {
            db,
            searches: AtomicU64::new(0),
        }
    }

    /// Configuration the engine actually opened with, stored config included.
    pub(crate) fn options(&self) -> &DbOptions {
        self.db.options()
    }

    /// Fetch one stored point by identifier.
    pub(crate) fn get(&self, id: &str) -> Result<Option<VectorEntry>> {
        Ok(self.db.get(id)?)
    }

    /// Store one point.
    pub(crate) fn insert(&self, entry: VectorEntry) -> Result<()> {
        self.db.insert(entry)?;
        Ok(())
    }

    /// Remove one point, reporting whether it was present.
    pub(crate) fn delete(&self, id: &str) -> Result<bool> {
        Ok(self.db.delete(id)?)
    }

    /// Number of points currently retained.
    pub(crate) fn len(&self) -> Result<usize> {
        Ok(self.db.len()?)
    }

    /// Whether the shard holds no points at all.
    pub(crate) fn is_empty(&self) -> Result<bool> {
        Ok(self.db.is_empty()?)
    }

    /// Persist one engine-level configuration value.
    pub(crate) fn save_config_value(&self, key: &str, value: &str) -> Result<()> {
        Ok(self.db.save_config_value(key, value)?)
    }

    /// Read back one engine-level configuration value.
    pub(crate) fn load_config_value(&self, key: &str) -> Result<Option<String>> {
        Ok(self.db.load_config_value(key)?)
    }

    /// Number of ANN traversals issued to this shard since it was opened.
    pub(crate) fn searches(&self) -> u64 {
        self.searches.load(Ordering::Relaxed)
    }

    /// The only accessor that reaches the ANN index; every call is counted.
    pub(crate) fn ann_search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        self.searches.fetch_add(1, Ordering::Relaxed);
        Ok(self.db.search(query)?)
    }
}

pub(crate) fn acquire_root_lock(root: &Path) -> Result<File> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join(ROOT_LOCK_FILENAME))?;
    match file.try_lock_exclusive() {
        Ok(()) => Ok(file),
        Err(error) if error.raw_os_error() == fs2::lock_contended_error().raw_os_error() => {
            Err(ContextIndexError::RootLocked)
        }
        Err(error) => Err(error.into()),
    }
}

pub(crate) fn create_shard(
    root: &Path,
    scope: &ContextScope,
    options: &ContextIndexOptions,
) -> Result<Shard> {
    let filename = scope.shard_filename();
    let path = root.join(&filename);
    // `VectorDB::new` opens an existing file, adopts its stored configuration
    // and keeps its stored vectors. Writing this scope's manifest over such a
    // file would relabel another scope's data as this one's, so a pre-existing
    // file is refused and never removed.
    if path.try_exists()? {
        return Err(ContextIndexError::ShardAdoption(filename));
    }
    let mut vector = options.vector.clone();
    vector.storage_path = path.to_string_lossy().into_owned();
    let shard = Shard::new(VectorDB::new(vector)?);
    // Defence in depth for the window between the check and the open: a shard
    // this call created is empty, carries no manifest, and opened with exactly
    // the requested vector configuration.
    if shard.load_config_value(MANIFEST_KEY)?.is_some()
        || !shard.is_empty()?
        || shard.options().dimensions != options.vector.dimensions
        || shard.options().distance_metric != options.vector.distance_metric
    {
        drop(shard);
        return Err(ContextIndexError::ShardAdoption(filename));
    }
    let manifest = serde_json::to_string(scope)?;
    if let Err(error) = shard.save_config_value(MANIFEST_KEY, &manifest) {
        // Only reachable when this call created the file, because adoption is
        // refused above, so this cannot delete another scope's shard.
        drop(shard);
        let _ = std::fs::remove_file(&path);
        return Err(error);
    }
    Ok(shard)
}

pub(crate) fn load_shard(
    path: &Path,
    filename: &str,
    options: &ContextIndexOptions,
) -> Result<(ContextScope, Shard)> {
    let mut vector = options.vector.clone();
    vector.storage_path = path.to_string_lossy().into_owned();
    let shard = Shard::new(VectorDB::new(vector)?);
    if shard.options().dimensions != options.vector.dimensions
        || shard.options().distance_metric != options.vector.distance_metric
    {
        return Err(ContextIndexError::CorruptShard(filename.to_string()));
    }
    let manifest = shard
        .load_config_value(MANIFEST_KEY)?
        .ok_or_else(|| ContextIndexError::CorruptShard(filename.to_string()))?;
    let scope: ContextScope = serde_json::from_str(&manifest)
        .map_err(|_| ContextIndexError::CorruptShard(filename.to_string()))?;
    Ok((scope, shard))
}

/// Shard filename for a directory entry, or `None` if it is not a shard.
pub(crate) fn shard_name(path: &Path) -> Option<String> {
    let name = path.file_name().and_then(|name| name.to_str())?;
    is_shard_name(name).then(|| name.to_string())
}

fn is_shard_name(name: &str) -> bool {
    // The root lock is not a shard; its name cannot collide with one, but the
    // exclusion is stated rather than inferred from the length check.
    name != ROOT_LOCK_FILENAME
        && name.len() == 69
        && name.ends_with(".redb")
        && name[..64]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_lock_is_never_mistaken_for_a_shard() {
        assert!(!is_shard_name(ROOT_LOCK_FILENAME));
        assert!(is_shard_name(&format!("{}.redb", "a".repeat(64))));
        assert!(!is_shard_name(&format!("{}.redb", "g".repeat(64))));
    }
}
