//! Workspace state: artifact lineages, view staging, and the fail-closed
//! staleness gate (ADR-318 §Decision 3).

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::anchor::{AnchorReceipt, EvidenceGrade, TransitionAnchor, WorkspaceTransitionRecord};
use crate::artifact::{ArtifactRef, ContentHash};
use crate::view::{StagedView, ViewKind};

/// The bound-vs-head detail of a rejected view (boxed inside
/// [`WorkspaceError`] to keep the error type small on the `Ok` path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewConflict {
    /// Lineage the view referenced.
    pub artifact_id: String,
    /// The exact version the view was derived from.
    pub bound: ArtifactRef,
    /// The artifact's current live head.
    pub head: ArtifactRef,
}

/// Errors returned by [`WorkspaceState`]. Every variant is a hard rejection:
/// per ADR-318, staleness and binding violations are never soft warnings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkspaceError {
    /// The referenced artifact lineage does not exist in this workspace.
    UnknownArtifact(String),
    /// The view is bound to a content hash that is no longer the artifact's
    /// live head — the ADR-318 §Decision 3 structural staleness rejection.
    /// Any operation carrying this view must recompute against the head.
    StaleView(Box<ViewConflict>),
    /// The view's hash matches the live head but its revision id does not —
    /// a corrupt or forged binding, rejected outright.
    BindingMismatch(Box<ViewConflict>),
    /// The ADR-312 anchor refused the transition record; per the
    /// "no anchor, no transition" contract the commit was NOT applied.
    AnchorRejected(String),
}

impl core::fmt::Display for WorkspaceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WorkspaceError::UnknownArtifact(id) => write!(f, "unknown artifact lineage {id:?}"),
            WorkspaceError::StaleView(c) => write!(
                f,
                "stale view of {:?}: bound to {} (rev {}), live head is {} (rev {}) — recompute required",
                c.artifact_id,
                c.bound.content_hash,
                c.bound.revision_id,
                c.head.content_hash,
                c.head.revision_id
            ),
            WorkspaceError::BindingMismatch(c) => write!(
                f,
                "binding mismatch for {:?}: view claims (hash {}, rev {}), head is (hash {}, rev {})",
                c.artifact_id,
                c.bound.content_hash,
                c.bound.revision_id,
                c.head.content_hash,
                c.head.revision_id
            ),
            WorkspaceError::AnchorRejected(msg) => {
                write!(f, "anchor rejected transition (commit aborted): {msg}")
            }
        }
    }
}

impl std::error::Error for WorkspaceError {}

/// One artifact's lineage: its live head and the full hash history.
#[derive(Debug, Clone)]
struct ArtifactLineage {
    head: ArtifactRef,
    /// Every committed hash, oldest first; last entry equals `head.content_hash`.
    history: Vec<ContentHash>,
}

/// The workspace: artifact lineages plus the validation gate that makes
/// stale views structurally invalid.
///
/// Generic over the [`TransitionAnchor`] so callers (WP15's acceptance
/// harness, RVF/RVM integrations) can supply a real ADR-312/322C anchor;
/// defaults are available via [`WorkspaceState::with_noop_anchor`].
pub struct WorkspaceState<A: TransitionAnchor> {
    anchor: A,
    lineages: HashMap<String, ArtifactLineage>,
    /// (record, receipt) audit log of every anchored transition.
    transitions: Vec<(WorkspaceTransitionRecord, AnchorReceipt)>,
    next_sequence: u64,
    next_view_id: u64,
}

impl WorkspaceState<crate::anchor::NoopAnchor> {
    /// Workspace with the in-process [`crate::anchor::NoopAnchor`].
    pub fn with_noop_anchor() -> Self {
        Self::new(crate::anchor::NoopAnchor)
    }
}

impl<A: TransitionAnchor> WorkspaceState<A> {
    /// Create a workspace that anchors every transition through `anchor`.
    pub fn new(anchor: A) -> Self {
        Self {
            anchor,
            lineages: HashMap::new(),
            transitions: Vec::new(),
            next_sequence: 0,
            next_view_id: 0,
        }
    }

    /// Commit a write of `content` to `artifact_id`, assigning a content
    /// hash and the lineage's next revision id (ADR-318 §Decision 1).
    ///
    /// Committing bytes identical to the live head is a no-op returning the
    /// existing head (no revision bump, no anchor record): the version did
    /// not change, so no view becomes stale.
    ///
    /// The transition record is pushed through the ADR-312 anchor **before**
    /// the workspace mutates; if the anchor rejects, the commit is aborted
    /// and the lineage is unchanged ("no anchor, no transition").
    pub fn commit(
        &mut self,
        artifact_id: &str,
        content: &[u8],
        actor_id: &str,
    ) -> Result<ArtifactRef, WorkspaceError> {
        let new_hash = ContentHash::of(content);
        let prior = self.lineages.get(artifact_id).map(|l| l.head.clone());
        if let Some(ref head) = prior {
            if head.content_hash == new_hash {
                return Ok(head.clone());
            }
        }
        let new_revision = prior.as_ref().map_or(1, |h| h.revision_id + 1);
        let record = WorkspaceTransitionRecord {
            sequence: self.next_sequence,
            timestamp_ns: now_ns(),
            artifact_id: artifact_id.to_string(),
            prior_hash: prior.as_ref().map(|h| h.content_hash),
            prior_revision: prior.as_ref().map(|h| h.revision_id),
            new_hash,
            new_revision,
            actor_id: actor_id.to_string(),
            // The workspace derived new_hash from the artifact's actual
            // bytes above — recomputed, not asserted.
            evidence_grade: EvidenceGrade::Recomputed,
        };
        let receipt = self
            .anchor
            .anchor(&record)
            .map_err(|e| WorkspaceError::AnchorRejected(e.0))?;
        // Anchor accepted: apply the mutation.
        self.next_sequence += 1;
        let head = ArtifactRef {
            artifact_id: artifact_id.to_string(),
            content_hash: new_hash,
            revision_id: new_revision,
        };
        let lineage = self
            .lineages
            .entry(artifact_id.to_string())
            .or_insert_with(|| ArtifactLineage {
                head: head.clone(),
                history: Vec::new(),
            });
        lineage.head = head.clone();
        lineage.history.push(new_hash);
        self.transitions.push((record, receipt));
        Ok(head)
    }

    /// The artifact's live head (exact current version), if it exists.
    pub fn head(&self, artifact_id: &str) -> Option<&ArtifactRef> {
        self.lineages.get(artifact_id).map(|l| &l.head)
    }

    /// Derive a view of `artifact_id`'s **current** version, binding it to
    /// the head's exact hash + revision (ADR-318 §Decision 2). `payload` is
    /// the view's own derived content (parser output, diff text, ...).
    pub fn stage_view(
        &mut self,
        kind: ViewKind,
        artifact_id: &str,
        payload: &[u8],
    ) -> Result<StagedView, WorkspaceError> {
        let head = self
            .lineages
            .get(artifact_id)
            .map(|l| l.head.clone())
            .ok_or_else(|| WorkspaceError::UnknownArtifact(artifact_id.to_string()))?;
        let view = StagedView {
            view_id: self.next_view_id,
            kind,
            artifact: head,
            payload_hash: ContentHash::of(payload),
        };
        self.next_view_id += 1;
        Ok(view)
    }

    /// Validate that `view` is still current: its bound content hash must
    /// equal the artifact's live head hash (and the revision ids must
    /// agree). Any mismatch is a hard rejection (ADR-318 §Decision 3).
    pub fn validate(&self, view: &StagedView) -> Result<(), WorkspaceError> {
        let id = &view.artifact.artifact_id;
        let head = self
            .lineages
            .get(id)
            .map(|l| &l.head)
            .ok_or_else(|| WorkspaceError::UnknownArtifact(id.clone()))?;
        let hash_ok = view.artifact.content_hash == head.content_hash;
        let rev_ok = view.artifact.revision_id == head.revision_id;
        match (hash_ok, rev_ok) {
            (true, true) => Ok(()),
            (false, _) => Err(WorkspaceError::StaleView(Box::new(ViewConflict {
                artifact_id: id.clone(),
                bound: view.artifact.clone(),
                head: head.clone(),
            }))),
            (true, false) => Err(WorkspaceError::BindingMismatch(Box::new(ViewConflict {
                artifact_id: id.clone(),
                bound: view.artifact.clone(),
                head: head.clone(),
            }))),
        }
    }

    /// Run `op` only if `view` is still current — the fail-closed gate.
    ///
    /// If validation fails, `op` is **never invoked** and the error is
    /// returned: an operation carrying a stale view cannot execute
    /// (ADR-318 §Decision 3, "enforced structurally at read time").
    pub fn execute<T>(
        &self,
        view: &StagedView,
        op: impl FnOnce(&ArtifactRef) -> T,
    ) -> Result<T, WorkspaceError> {
        self.validate(view)?;
        // `validate` guaranteed the lineage exists and matches.
        let head = self
            .head(&view.artifact.artifact_id)
            .expect("validated lineage must exist");
        Ok(op(head))
    }

    /// Audit log of every anchored transition, oldest first.
    pub fn transitions(&self) -> &[(WorkspaceTransitionRecord, AnchorReceipt)] {
        &self.transitions
    }

    /// Full committed hash history for an artifact, oldest first.
    pub fn history(&self, artifact_id: &str) -> Option<&[ContentHash]> {
        self.lineages.get(artifact_id).map(|l| l.history.as_slice())
    }
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
