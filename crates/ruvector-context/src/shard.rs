//! Shard files: naming, creation, recovery, and the root lock.
//!
//! The vector engine backing one exact scope is wrapped here so the per-scope
//! isolation counter cannot be bypassed, and so the rules that decide whether a
//! file on disk may be treated as this scope's shard live in one place.
//!
//! Every path this module hands to the vector engine is one it created itself
//! under a name it claimed atomically. The engine's API is path-based and it
//! creates whatever file the path resolves to, so a symlink anywhere on that
//! path would let an attacker choose where a tenant's vectors are written.

use crate::{ContextIndexError, ContextIndexOptions, ContextScope, Result};
use fs4::TryLockError;
use ruvector_core::types::DbOptions;
use ruvector_core::{SearchQuery, SearchResult, VectorDB, VectorEntry};
use std::fs::File;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MANIFEST_KEY: &str = "ruvector_context_scope_v1";

/// Name of the exclusive root lock file held for the lifetime of one index.
///
/// It is deliberately not a valid shard name, so `is_shard_name` skips it.
const ROOT_LOCK_FILENAME: &str = ".lock";

/// Prefix of the scratch name a shard is built under before it is published.
const TEMP_PREFIX: &str = ".create-";

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
    fn new(db: VectorDB) -> Self {
        Self {
            db,
            searches: AtomicU64::new(0),
        }
    }

    /// Configuration the engine actually opened with, stored config included.
    fn options(&self) -> &DbOptions {
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
    fn is_empty(&self) -> Result<bool> {
        Ok(self.db.is_empty()?)
    }

    /// Persist one engine-level configuration value.
    fn save_config_value(&self, key: &str, value: &str) -> Result<()> {
        Ok(self.db.save_config_value(key, value)?)
    }

    /// Read back one engine-level configuration value.
    fn load_config_value(&self, key: &str) -> Result<Option<String>> {
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

/// Open the exclusive root lock, refusing to follow a symlink at its path.
pub(crate) fn acquire_root_lock(root: &Path) -> Result<File> {
    let path = root.join(ROOT_LOCK_FILENAME);
    let file = match lock_open_options().open(&path) {
        Ok(file) => file,
        Err(error) => return Err(classify_lock_open_error(&path, error)),
    };
    // Called through the trait explicitly: `File` grew an inherent `try_lock`
    // in Rust 1.89, and letting method resolution pick it would mean this code
    // locks via std on new toolchains and via fs4 on the 1.77 MSRV.
    match fs4::FileExt::try_lock(&file) {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => Err(ContextIndexError::RootLocked),
        Err(TryLockError::Error(error)) => Err(error.into()),
    }
}

fn lock_open_options() -> std::fs::OpenOptions {
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        // A symlink planted at the lock path would otherwise let an attacker
        // choose which file this index opens for writing, and which lock the
        // single-handle guarantee actually rests on.
        options.custom_flags(libc::O_NOFOLLOW);
    }
    options
}

fn classify_lock_open_error(path: &Path, error: std::io::Error) -> ContextIndexError {
    if std::fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_symlink()) {
        return ContextIndexError::UnsafeRootLock;
    }
    error.into()
}

/// Build a new shard for `scope` and publish it under its hash-bound name.
///
/// The shard is built under a scratch name this call claimed with `O_EXCL`,
/// then linked into place. `link(2)` fails with `EEXIST` against anything
/// already at the destination — a regular file, a directory, or a symlink,
/// dangling or not — and never follows it. That is what makes adoption
/// impossible: there is no stat-then-create window for a planted name to slip
/// through, and a planted symlink cannot redirect a tenant's vectors to a path
/// outside the index root.
pub(crate) fn create_shard(
    root: &Path,
    scope: &ContextScope,
    options: &ContextIndexOptions,
) -> Result<Shard> {
    let filename = scope.shard_filename();
    let scratch = claim_scratch_path(root)?;
    let shard = match build_shard(&scratch, scope, options) {
        Ok(shard) => shard,
        Err(error) => {
            let _ = std::fs::remove_file(&scratch);
            return Err(error);
        }
    };
    if let Err(error) = std::fs::hard_link(&scratch, root.join(&filename)) {
        drop(shard);
        let _ = std::fs::remove_file(&scratch);
        return Err(match error.kind() {
            ErrorKind::AlreadyExists => ContextIndexError::ShardAdoption(filename),
            _ => error.into(),
        });
    }
    // The engine holds the inode open, so dropping the scratch name leaves the
    // shard reachable only under the name its scope hashes to.
    let _ = std::fs::remove_file(&scratch);
    Ok(shard)
}

fn build_shard(path: &Path, scope: &ContextScope, options: &ContextIndexOptions) -> Result<Shard> {
    let mut vector = options.vector.clone();
    vector.storage_path = path.to_string_lossy().into_owned();
    let shard = Shard::new(VectorDB::new(vector)?);
    // The scratch file was created empty under a name this call claimed with
    // `O_EXCL`, so the engine cannot have adopted anything. Assert it anyway:
    // silent adoption is the single failure this whole path exists to prevent.
    if shard.load_config_value(MANIFEST_KEY)?.is_some()
        || !shard.is_empty()?
        || engine_fingerprint(shard.options())? != engine_fingerprint(&options.vector)?
    {
        return Err(ContextIndexError::ShardAdoption(scope.shard_filename()));
    }
    shard.save_config_value(MANIFEST_KEY, &serde_json::to_string(scope)?)?;
    Ok(shard)
}

/// Claim an unused scratch name inside `root` with `O_CREAT | O_EXCL`.
///
/// The exclusive create is what matters, not the name's unpredictability: a
/// pre-planted name — symlink included — fails the create and costs only a
/// retry, so an attacker cannot steer the build by guessing.
fn claim_scratch_path(root: &Path) -> Result<PathBuf> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    for _ in 0..16 {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos() as u64);
        let nonce = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            "{TEMP_PREFIX}{:016x}{nonce:016x}",
            stamp ^ u64::from(std::process::id())
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(_) => return Ok(path),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(std::io::Error::new(
        ErrorKind::AlreadyExists,
        "could not claim a scratch name for a new context shard",
    )
    .into())
}

pub(crate) fn load_shard(
    path: &Path,
    filename: &str,
    options: &ContextIndexOptions,
) -> Result<(ContextScope, Shard)> {
    let mut vector = options.vector.clone();
    vector.storage_path = path.to_string_lossy().into_owned();
    let shard = Shard::new(VectorDB::new(vector)?);
    if engine_fingerprint(shard.options())? != engine_fingerprint(&options.vector)? {
        return Err(ContextIndexError::CorruptShard(filename.to_string()));
    }
    let manifest = shard
        .load_config_value(MANIFEST_KEY)?
        .ok_or_else(|| ContextIndexError::CorruptShard(filename.to_string()))?;
    let scope: ContextScope = serde_json::from_str(&manifest)
        .map_err(|_| ContextIndexError::CorruptShard(filename.to_string()))?;
    Ok((scope, shard))
}

/// Full engine configuration, minus the storage path.
///
/// `VectorDB::new` adopts a stored database's `hnsw_config` and `quantization`
/// along with its dimensions and metric, so comparing only the latter two lets
/// an adopted file dictate engine parameters — lossy quantization, or a
/// degenerate graph. The storage path is excluded because it legitimately
/// differs between the scratch name a shard is built under and its published
/// name.
fn engine_fingerprint(options: &DbOptions) -> Result<String> {
    let normalized = DbOptions {
        storage_path: String::new(),
        ..options.clone()
    };
    Ok(serde_json::to_string(&normalized)?)
}

/// Whether a directory entry is a shard file belonging to this index.
pub(crate) fn is_shard_name(name: &str) -> bool {
    // The root lock and scratch files are not shards; their names cannot
    // collide with one, but the exclusions are stated rather than inferred
    // from the length check.
    name != ROOT_LOCK_FILENAME
        && !is_scratch_name(name)
        && name.len() == 69
        && name.ends_with(".redb")
        && name[..64]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Whether a directory entry is a scratch file from an unfinished create.
pub(crate) fn is_scratch_name(name: &str) -> bool {
    name.starts_with(TEMP_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_names_are_never_mistaken_for_shards() {
        assert!(!is_shard_name(ROOT_LOCK_FILENAME));
        assert!(!is_shard_name(&format!("{TEMP_PREFIX}{}", "a".repeat(61))));
        assert!(is_shard_name(&format!("{}.redb", "a".repeat(64))));
        assert!(!is_shard_name(&format!("{}.redb", "g".repeat(64))));
    }

    #[test]
    fn scratch_names_are_claimed_exclusively_and_distinctly() {
        let root = tempfile::tempdir().unwrap();
        let first = claim_scratch_path(root.path()).unwrap();
        let second = claim_scratch_path(root.path()).unwrap();
        assert_ne!(first, second);
        for path in [&first, &second] {
            let name = path.file_name().unwrap().to_str().unwrap();
            assert!(is_scratch_name(name));
            assert!(!is_shard_name(name));
        }
    }
}
