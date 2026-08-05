//! Runtime-owned sequence-state metadata and session-cache primitives.
//!
//! Model setup may allocate sequence state through another allocator (currently
//! PyTorch for several attention backends), but it must register the metadata
//! read from each real allocation here. Request-time inventory, measurement,
//! snapshot, and restore paths consume this Rust-owned registry and do not call
//! back into Python.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const SESSION_SNAPSHOT_FORMAT_VERSION: u32 = 2;
pub const PREFILL_STAGE_K_KIND: &str = "gqa_prefill_stage_k";
pub const PREFILL_STAGE_V_KIND: &str = "gqa_prefill_stage_v";

pub fn is_prefill_stage_kind(kind: &str) -> bool {
    kind == PREFILL_STAGE_K_KIND || kind == PREFILL_STAGE_V_KIND
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SequenceStateGrowth {
    /// The complete allocation is required regardless of current token count.
    Fixed,
    /// Contiguous rows become live as logical tokens are appended.
    TokenRows {
        logical_tokens_per_row: usize,
        capacity_rows: usize,
        row_bytes: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceStateAllocation {
    pub name: String,
    pub kind: String,
    pub layer_idx: Option<usize>,
    pub device_ordinal: usize,
    pub ptr: u64,
    pub storage_bytes: usize,
    pub dtype: String,
    pub element_size: usize,
    pub shape: Vec<usize>,
    /// Byte strides, not element strides.
    pub strides_bytes: Vec<usize>,
    pub growth: SequenceStateGrowth,
}

impl SequenceStateAllocation {
    pub fn used_bytes(&self, logical_tokens: usize) -> usize {
        match self.growth {
            SequenceStateGrowth::Fixed => self.storage_bytes,
            SequenceStateGrowth::TokenRows {
                logical_tokens_per_row,
                capacity_rows,
                row_bytes,
            } => {
                let rows = logical_tokens
                    .div_ceil(logical_tokens_per_row)
                    .min(capacity_rows);
                rows.saturating_mul(row_bytes).min(self.storage_bytes)
            }
        }
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("sequence-state allocation name must not be empty".to_string());
        }
        if self.kind.trim().is_empty() {
            return Err(format!("{}: state kind must not be empty", self.name));
        }
        if self.ptr == 0 {
            return Err(format!("{}: device pointer must not be zero", self.name));
        }
        if self.storage_bytes == 0 || self.element_size == 0 {
            return Err(format!(
                "{}: storage_bytes and element_size must be positive",
                self.name
            ));
        }
        if self.dtype.trim().is_empty() {
            return Err(format!("{}: dtype must not be empty", self.name));
        }
        if self.shape.is_empty() || self.shape.len() != self.strides_bytes.len() {
            return Err(format!(
                "{}: shape and byte strides must have the same positive rank",
                self.name
            ));
        }
        let logical_bytes = self
            .shape
            .iter()
            .try_fold(self.element_size, |bytes, &dim| bytes.checked_mul(dim))
            .ok_or_else(|| format!("{}: logical tensor byte count overflow", self.name))?;
        if logical_bytes != self.storage_bytes {
            return Err(format!(
                "{}: real tensor metadata is inconsistent: shape bytes={} storage_bytes={}",
                self.name, logical_bytes, self.storage_bytes
            ));
        }
        match self.growth {
            SequenceStateGrowth::Fixed => {}
            SequenceStateGrowth::TokenRows {
                logical_tokens_per_row,
                capacity_rows,
                row_bytes,
            } => {
                if logical_tokens_per_row == 0 || capacity_rows == 0 || row_bytes == 0 {
                    return Err(format!(
                        "{}: token-row metadata values must be positive",
                        self.name
                    ));
                }
                let expected = capacity_rows
                    .checked_mul(row_bytes)
                    .ok_or_else(|| format!("{}: token-row byte count overflow", self.name))?;
                if expected != self.storage_bytes {
                    return Err(format!(
                        "{}: token-row bytes={} do not equal real storage_bytes={}",
                        self.name, expected, self.storage_bytes
                    ));
                }
                if self.strides_bytes.first().copied() != Some(row_bytes) {
                    return Err(format!(
                        "{}: first byte stride {:?} does not equal contiguous row_bytes={}",
                        self.name,
                        self.strides_bytes.first(),
                        row_bytes
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SequenceStateRegistry {
    allocations: Vec<SequenceStateAllocation>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SequenceStateInventory<'a> {
    pub logical_tokens: usize,
    pub allocation_count: usize,
    pub allocated_bytes: usize,
    pub used_bytes: usize,
    /// Rust-owned absolute position fields which are part of a snapshot.
    pub host_position_bytes: usize,
    pub allocations: Vec<SequenceStateInventoryItem<'a>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SequenceStateInventoryItem<'a> {
    #[serde(flatten)]
    pub allocation: &'a SequenceStateAllocation,
    pub used_bytes: usize,
}

impl SequenceStateRegistry {
    pub fn register(&mut self, allocation: SequenceStateAllocation) -> Result<(), String> {
        allocation.validate()?;
        if self
            .allocations
            .iter()
            .any(|existing| existing.name == allocation.name)
        {
            return Err(format!(
                "duplicate sequence-state allocation name {:?}",
                allocation.name
            ));
        }
        let new_begin = allocation.ptr;
        let new_end = new_begin
            .checked_add(allocation.storage_bytes as u64)
            .ok_or_else(|| format!("{}: pointer range overflow", allocation.name))?;
        for existing in &self.allocations {
            if existing.device_ordinal != allocation.device_ordinal {
                continue;
            }
            let existing_end = existing
                .ptr
                .checked_add(existing.storage_bytes as u64)
                .ok_or_else(|| format!("{}: pointer range overflow", existing.name))?;
            if new_begin < existing_end && existing.ptr < new_end {
                return Err(format!(
                    "sequence-state allocations overlap on GPU {}: {:?} and {:?}",
                    allocation.device_ordinal, existing.name, allocation.name
                ));
            }
        }
        self.allocations.push(allocation);
        self.allocations.sort_by(|left, right| {
            (left.device_ordinal, left.ptr, left.name.as_str()).cmp(&(
                right.device_ordinal,
                right.ptr,
                right.name.as_str(),
            ))
        });
        Ok(())
    }

    pub fn clear(&mut self) {
        self.allocations.clear();
    }

    pub fn allocations(&self) -> &[SequenceStateAllocation] {
        &self.allocations
    }

    pub fn contains_ptr(&self, ptr: u64) -> bool {
        self.allocations
            .iter()
            .any(|allocation| allocation.ptr == ptr)
    }

    pub fn has_non_rewindable_state(&self) -> bool {
        self.allocations
            .iter()
            .any(|allocation| matches!(allocation.growth, SequenceStateGrowth::Fixed))
    }

    pub fn compatibility_layout_value(&self) -> serde_json::Value {
        serde_json::Value::Array(
            self.allocations
                .iter()
                .map(|allocation| {
                    serde_json::json!({
                        "name": allocation.name,
                        "kind": allocation.kind,
                        "layer_idx": allocation.layer_idx,
                        "device_ordinal": allocation.device_ordinal,
                        "storage_bytes": allocation.storage_bytes,
                        "dtype": allocation.dtype,
                        "element_size": allocation.element_size,
                        "shape": allocation.shape,
                        "strides_bytes": allocation.strides_bytes,
                        "growth": allocation.growth,
                    })
                })
                .collect(),
        )
    }

    pub fn snapshot_blob_memory_cost_estimate(
        &self,
        logical_tokens: usize,
    ) -> Result<usize, String> {
        self.allocations
            .iter()
            .try_fold(0usize, |total, allocation| {
                let used_bytes = allocation.used_bytes(logical_tokens);
                if used_bytes == 0 {
                    return Ok(total);
                }
                let metadata = std::mem::size_of::<SequenceStateBlob>()
                    .checked_add(allocation.name.len())
                    .and_then(|value| value.checked_add(allocation.kind.len()))
                    .and_then(|value| value.checked_add(allocation.dtype.len()))
                    .and_then(|value| {
                        value.checked_add(
                            allocation
                                .shape
                                .len()
                                .saturating_mul(std::mem::size_of::<usize>()),
                        )
                    })
                    .and_then(|value| {
                        value.checked_add(
                            allocation
                                .strides_bytes
                                .len()
                                .saturating_mul(std::mem::size_of::<usize>()),
                        )
                    })
                    .and_then(|value| value.checked_add(used_bytes))
                    .ok_or_else(|| {
                        format!("{} snapshot memory-cost estimate overflow", allocation.name)
                    })?;
                total.checked_add(metadata).ok_or_else(|| {
                    "sequence-state snapshot memory-cost estimate overflow".to_string()
                })
            })
    }

    pub fn validate_complete_names(&self) -> Result<(), String> {
        let unique: HashSet<&str> = self
            .allocations
            .iter()
            .map(|item| item.name.as_str())
            .collect();
        if unique.len() != self.allocations.len() {
            return Err("sequence-state inventory contains duplicate names".to_string());
        }
        Ok(())
    }

    pub fn inventory(&self, logical_tokens: usize) -> SequenceStateInventory<'_> {
        let allocations: Vec<_> = self
            .allocations
            .iter()
            .map(|allocation| SequenceStateInventoryItem {
                used_bytes: allocation.used_bytes(logical_tokens),
                allocation,
            })
            .collect();
        SequenceStateInventory {
            logical_tokens,
            allocation_count: allocations.len(),
            allocated_bytes: allocations
                .iter()
                .map(|item| item.allocation.storage_bytes)
                .sum(),
            used_bytes: allocations.iter().map(|item| item.used_bytes).sum(),
            // kv_current_pos + rope_position_delta. Consumed token ids are
            // stored in SessionSnapshot and therefore scale separately.
            host_position_bytes: std::mem::size_of::<usize>() + std::mem::size_of::<i32>(),
            allocations,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerOwnership {
    pub device_ordinal: usize,
    pub physical_device_id: String,
    pub compute_capability_major: u32,
    pub compute_capability_minor: u32,
    pub total_memory_bytes: u64,
    pub layer_start: usize,
    pub layer_end: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeviceCompatibilityMaterial {
    pub ownership: LayerOwnership,
    pub model_num_layers: usize,
    pub expert_quantization: String,
    pub attention_quantization: String,
    pub kv_format: String,
    pub kv_key_bits: u8,
    pub kv_value_bits: u8,
    pub state_layout: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionCompatibilitySignature {
    pub snapshot_format_version: u32,
    pub runtime_version: String,
    pub model_identity: String,
    pub model_revision: Option<String>,
    pub tokenizer_sha256: [u8; 32],
    pub chat_template_sha256: [u8; 32],
    /// Canonical, runtime-produced expert quantization configuration.
    pub expert_quantization: String,
    /// Canonical, runtime-produced attention quantization configuration.
    pub attention_quantization: String,
    pub kv_format: String,
    pub kv_key_bits: u8,
    pub kv_value_bits: u8,
    pub model_num_layers: usize,
    pub topology: Vec<LayerOwnership>,
    /// Digest of the ordered live allocation layout (names, types, shapes,
    /// strides, ownership, and growth modes), not of allocation addresses.
    pub state_layout_sha256: [u8; 32],
}

impl SessionCompatibilitySignature {
    pub fn validate(&self) -> Result<(), String> {
        if self.snapshot_format_version != SESSION_SNAPSHOT_FORMAT_VERSION {
            return Err(format!(
                "snapshot format version {} is incompatible with runtime version {}",
                self.snapshot_format_version, SESSION_SNAPSHOT_FORMAT_VERSION
            ));
        }
        for (field, value) in [
            ("runtime_version", self.runtime_version.as_str()),
            ("model_identity", self.model_identity.as_str()),
            ("expert_quantization", self.expert_quantization.as_str()),
            (
                "attention_quantization",
                self.attention_quantization.as_str(),
            ),
            ("kv_format", self.kv_format.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(format!("session signature {field} must not be empty"));
            }
        }
        if self.kv_key_bits == 0 || self.kv_value_bits == 0 {
            return Err("session signature KV bit widths must be positive".to_string());
        }
        for (field, digest) in [
            ("tokenizer_sha256", &self.tokenizer_sha256),
            ("chat_template_sha256", &self.chat_template_sha256),
            ("state_layout_sha256", &self.state_layout_sha256),
        ] {
            if digest.iter().all(|&byte| byte == 0) {
                return Err(format!(
                    "session signature {field} must contain a computed digest"
                ));
            }
        }
        if self.topology.is_empty() {
            return Err("session signature GPU topology must not be empty".to_string());
        }
        if self.model_num_layers == 0 {
            return Err("session signature model layer count must be positive".to_string());
        }
        let mut ordinals = HashSet::new();
        let mut ranges: Vec<_> = self.topology.iter().collect();
        ranges.sort_by_key(|owner| (owner.layer_start, owner.layer_end));
        let mut expected_start = 0usize;
        for owner in ranges {
            if owner.physical_device_id.trim().is_empty() {
                return Err(format!(
                    "session signature GPU {} has no physical identity",
                    owner.device_ordinal
                ));
            }
            if owner.total_memory_bytes == 0 {
                return Err(format!(
                    "session signature GPU {} reports zero total memory",
                    owner.device_ordinal
                ));
            }
            if owner.layer_start >= owner.layer_end {
                return Err(format!(
                    "session signature GPU {} has invalid layer range [{}, {})",
                    owner.device_ordinal, owner.layer_start, owner.layer_end
                ));
            }
            if !ordinals.insert(owner.device_ordinal) {
                return Err(format!(
                    "session signature repeats GPU ordinal {}",
                    owner.device_ordinal
                ));
            }
            if owner.layer_start != expected_start {
                return Err(format!(
                    "session signature layer ownership is not contiguous: expected layer {}, got range [{}, {})",
                    expected_start, owner.layer_start, owner.layer_end
                ));
            }
            expected_start = owner.layer_end;
        }
        if expected_start != self.model_num_layers {
            return Err(format!(
                "session signature layer ownership ends at {}, model has {} layers",
                expected_start, self.model_num_layers
            ));
        }
        Ok(())
    }

    pub fn heap_bytes(&self) -> usize {
        let mut bytes = self.runtime_version.capacity()
            + self.model_identity.capacity()
            + self
                .model_revision
                .as_ref()
                .map_or(0, |revision| revision.capacity())
            + self.expert_quantization.capacity()
            + self.attention_quantization.capacity()
            + self.kv_format.capacity()
            + self
                .topology
                .capacity()
                .saturating_mul(std::mem::size_of::<LayerOwnership>());
        bytes = bytes.saturating_add(self.topology.iter().fold(0usize, |total, owner| {
            total.saturating_add(owner.physical_device_id.capacity())
        }));
        bytes
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceSequencePosition {
    pub device_ordinal: usize,
    pub kv_absolute_position: usize,
    pub rope_position_delta: i32,
    pub rope_absolute_position: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SequenceStateBlob {
    pub allocation_name: String,
    pub kind: String,
    pub layer_idx: Option<usize>,
    pub device_ordinal: usize,
    pub dtype: String,
    pub element_size: usize,
    pub shape: Vec<usize>,
    pub strides_bytes: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl SequenceStateBlob {
    pub fn validate(&self) -> Result<(), String> {
        if self.allocation_name.trim().is_empty()
            || self.kind.trim().is_empty()
            || self.dtype.trim().is_empty()
        {
            return Err("sequence-state blob names, kind, and dtype must not be empty".to_string());
        }
        if self.shape.is_empty() || self.shape.len() != self.strides_bytes.len() {
            return Err(format!(
                "{}: snapshot shape/stride ranks differ",
                self.allocation_name
            ));
        }
        if self.bytes.is_empty() {
            return Err(format!(
                "{}: snapshot state blob must not be empty",
                self.allocation_name
            ));
        }
        if self.element_size == 0 {
            return Err(format!(
                "{}: snapshot element size must be positive",
                self.allocation_name
            ));
        }
        let logical_bytes = self
            .shape
            .iter()
            .try_fold(self.element_size, |bytes, &dim| bytes.checked_mul(dim));
        if logical_bytes != Some(self.bytes.len()) {
            return Err(format!(
                "{}: snapshot shape bytes {:?} do not equal blob bytes {}",
                self.allocation_name,
                logical_bytes,
                self.bytes.len()
            ));
        }
        Ok(())
    }

    fn heap_bytes(&self) -> usize {
        self.allocation_name
            .capacity()
            .saturating_add(self.kind.capacity())
            .saturating_add(self.dtype.capacity())
            .saturating_add(
                self.shape
                    .capacity()
                    .saturating_mul(std::mem::size_of::<usize>()),
            )
            .saturating_add(
                self.strides_bytes
                    .capacity()
                    .saturating_mul(std::mem::size_of::<usize>()),
            )
            .saturating_add(self.bytes.capacity())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub compatibility: SessionCompatibilitySignature,
    pub consumed_token_ids: Vec<u32>,
    pub positions: Vec<DeviceSequencePosition>,
    pub state_blobs: Vec<SequenceStateBlob>,
}

impl SessionSnapshot {
    pub fn validate(&self) -> Result<(), String> {
        self.compatibility.validate()?;
        if self.consumed_token_ids.is_empty() {
            return Err("session snapshot must contain consumed token IDs".to_string());
        }
        if self.positions.len() != self.compatibility.topology.len() {
            return Err(format!(
                "session snapshot has {} device positions for {} topology entries",
                self.positions.len(),
                self.compatibility.topology.len()
            ));
        }
        let topology_ordinals: HashSet<_> = self
            .compatibility
            .topology
            .iter()
            .map(|owner| owner.device_ordinal)
            .collect();
        let mut position_ordinals = HashSet::new();
        let consumed_tokens = self.consumed_token_ids.len();
        let mut shared_position: Option<(usize, i32, i64)> = None;
        for position in &self.positions {
            if !topology_ordinals.contains(&position.device_ordinal) {
                return Err(format!(
                    "snapshot position refers to unknown GPU {}",
                    position.device_ordinal
                ));
            }
            if !position_ordinals.insert(position.device_ordinal) {
                return Err(format!(
                    "snapshot repeats position for GPU {}",
                    position.device_ordinal
                ));
            }
            let expected_rope = i64::try_from(position.kv_absolute_position)
                .ok()
                .and_then(|value| value.checked_add(i64::from(position.rope_position_delta)))
                .ok_or_else(|| {
                    format!(
                        "snapshot GPU {} absolute RoPE position overflows",
                        position.device_ordinal
                    )
                })?;
            if expected_rope != position.rope_absolute_position {
                return Err(format!(
                    "snapshot GPU {} RoPE position {} does not equal KV position {} plus delta {}",
                    position.device_ordinal,
                    position.rope_absolute_position,
                    position.kv_absolute_position,
                    position.rope_position_delta
                ));
            }
            if position.kv_absolute_position != consumed_tokens {
                return Err(format!(
                    "snapshot GPU {} KV position {} does not equal consumed token count {}",
                    position.device_ordinal, position.kv_absolute_position, consumed_tokens
                ));
            }
            let values = (
                position.kv_absolute_position,
                position.rope_position_delta,
                position.rope_absolute_position,
            );
            if let Some(expected) = shared_position {
                if values != expected {
                    return Err(format!(
                        "snapshot GPU {} position {:?} disagrees with multi-GPU position {:?}",
                        position.device_ordinal, values, expected
                    ));
                }
            } else {
                shared_position = Some(values);
            }
        }
        let mut allocation_names = HashSet::new();
        for blob in &self.state_blobs {
            blob.validate()?;
            if !topology_ordinals.contains(&blob.device_ordinal) {
                return Err(format!(
                    "{} refers to unknown GPU {}",
                    blob.allocation_name, blob.device_ordinal
                ));
            }
            if !allocation_names.insert((blob.device_ordinal, blob.allocation_name.as_str())) {
                return Err(format!(
                    "session snapshot repeats allocation {:?} on GPU {}",
                    blob.allocation_name, blob.device_ordinal
                ));
            }
        }
        if self.state_blobs.is_empty() {
            return Err("session snapshot must contain state blobs".to_string());
        }
        Ok(())
    }

    /// Heap bytes already owned by the completed snapshot, using actual vector
    /// and string capacities rather than a bytes-per-token estimate.
    pub fn memory_cost_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.compatibility.heap_bytes())
            .saturating_add(
                self.consumed_token_ids
                    .capacity()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(
                self.positions
                    .capacity()
                    .saturating_mul(std::mem::size_of::<DeviceSequencePosition>()),
            )
            .saturating_add(
                self.state_blobs
                    .capacity()
                    .saturating_mul(std::mem::size_of::<SequenceStateBlob>()),
            )
            .saturating_add(self.state_blobs.iter().fold(0usize, |bytes, blob| {
                bytes.saturating_add(blob.heap_bytes())
            }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SnapshotId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RamReservationId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixLookupResult {
    Hit {
        snapshot_id: SnapshotId,
        matched_tokens: usize,
    },
    SignatureMismatch {
        matched_tokens: usize,
    },
    NoMatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivePrefixPlan {
    Append { matched_tokens: usize },
    TruncateKvAndAppend { matched_tokens: usize },
    RequiresBoundarySnapshot { matched_tokens: usize },
    NoReusablePrefix,
    NoSuffixToCompute,
}

pub fn common_token_prefix(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

pub fn plan_active_prefix(
    active_tokens: &[u32],
    request_tokens: &[u32],
    has_non_rewindable_state: bool,
) -> ActivePrefixPlan {
    let matched_tokens = common_token_prefix(active_tokens, request_tokens);
    if matched_tokens == 0 {
        return ActivePrefixPlan::NoReusablePrefix;
    }
    if matched_tokens == request_tokens.len() {
        return ActivePrefixPlan::NoSuffixToCompute;
    }
    if matched_tokens == active_tokens.len() {
        return ActivePrefixPlan::Append { matched_tokens };
    }
    if has_non_rewindable_state {
        ActivePrefixPlan::RequiresBoundarySnapshot { matched_tokens }
    } else {
        ActivePrefixPlan::TruncateKvAndAppend { matched_tokens }
    }
}

pub fn ram_prefix_is_longer(active_reusable_tokens: usize, ram_matched_tokens: usize) -> bool {
    ram_matched_tokens > active_reusable_tokens
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PrefixFingerprint {
    token_count: usize,
    hash: u64,
}

const PREFIX_HASH_OFFSET: u64 = 0xcbf29ce484222325;
const PREFIX_HASH_PRIME: u64 = 0x100000001b3;

fn prefix_hash_extend(mut hash: u64, token: u32) -> u64 {
    for byte in token.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PREFIX_HASH_PRIME);
    }
    hash
}

fn token_fingerprint(tokens: &[u32]) -> PrefixFingerprint {
    let hash = tokens.iter().fold(PREFIX_HASH_OFFSET, |hash, &token| {
        prefix_hash_extend(hash, token)
    });
    PrefixFingerprint {
        token_count: tokens.len(),
        hash,
    }
}

#[derive(Default)]
struct PrefixIndex {
    candidates: HashMap<PrefixFingerprint, Vec<SnapshotId>>,
    indexed_lengths: BTreeMap<usize, usize>,
}

impl PrefixIndex {
    fn insert(&mut self, id: SnapshotId, tokens: &[u32]) {
        let fingerprint = token_fingerprint(tokens);
        self.candidates.entry(fingerprint).or_default().push(id);
        *self.indexed_lengths.entry(tokens.len()).or_default() += 1;
    }

    fn remove(&mut self, id: SnapshotId, tokens: &[u32]) -> Result<(), String> {
        let fingerprint = token_fingerprint(tokens);
        let candidates = self.candidates.get(&fingerprint).ok_or_else(|| {
            format!(
                "session prefix index is missing fingerprint for snapshot {}",
                id.0
            )
        })?;
        let position = candidates
            .iter()
            .position(|candidate| *candidate == id)
            .ok_or_else(|| format!("session prefix index is missing snapshot {}", id.0))?;
        let count = *self.indexed_lengths.get(&tokens.len()).ok_or_else(|| {
            format!(
                "session prefix index is missing token length {}",
                tokens.len()
            )
        })?;
        let next_count = count
            .checked_sub(1)
            .ok_or_else(|| "session prefix length accounting underflow".to_string())?;

        let candidates = self
            .candidates
            .get_mut(&fingerprint)
            .ok_or_else(|| "session prefix index changed during removal".to_string())?;
        candidates.swap_remove(position);
        if candidates.is_empty() {
            self.candidates.remove(&fingerprint);
        }
        if next_count == 0 {
            self.indexed_lengths.remove(&tokens.len());
        } else if let Some(count) = self.indexed_lengths.get_mut(&tokens.len()) {
            *count = next_count;
        }
        Ok(())
    }

    fn query_fingerprints(&self, tokens: &[u32]) -> Vec<PrefixFingerprint> {
        let mut fingerprints = Vec::with_capacity(self.indexed_lengths.len());
        let mut lengths = self
            .indexed_lengths
            .range(..=tokens.len())
            .map(|(&length, _)| length)
            .peekable();
        let mut next_length = lengths.next();
        let mut hash = PREFIX_HASH_OFFSET;
        for (index, &token) in tokens.iter().enumerate() {
            hash = prefix_hash_extend(hash, token);
            let token_count = index + 1;
            if next_length == Some(token_count) {
                fingerprints.push(PrefixFingerprint { token_count, hash });
                next_length = lengths.next();
                if next_length.is_none() {
                    break;
                }
            }
        }
        fingerprints
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MemoryAvailability {
    pub host_available_bytes: u64,
    pub cgroup_available_bytes: Option<u64>,
    pub effective_available_bytes: u64,
}

pub trait MemoryAvailabilityProbe: Send + Sync {
    fn availability(&self) -> Result<MemoryAvailability, String>;
}

#[derive(Default)]
pub struct SystemMemoryAvailabilityProbe;

fn parse_mem_available_bytes(text: &str) -> Result<u64, String> {
    let line = text
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))
        .ok_or_else(|| "/proc/meminfo has no MemAvailable entry".to_string())?;
    let mut fields = line.split_whitespace();
    let _name = fields.next();
    let kib = fields
        .next()
        .ok_or_else(|| "MemAvailable has no value".to_string())?
        .parse::<u64>()
        .map_err(|error| format!("parse MemAvailable: {error}"))?;
    if fields.next() != Some("kB") {
        return Err("MemAvailable is not expressed in kB".to_string());
    }
    kib.checked_mul(1024)
        .ok_or_else(|| "MemAvailable byte count overflows".to_string())
}

fn parse_cgroup_limit(value: &str) -> Result<Option<u64>, String> {
    let value = value.trim();
    if value == "max" {
        Ok(None)
    } else {
        value
            .parse::<u64>()
            .map(Some)
            .map_err(|error| format!("parse cgroup memory limit {value:?}: {error}"))
    }
}

fn cgroup_available_at(path: &Path) -> Result<Option<u64>, String> {
    let max_path = path.join("memory.max");
    let current_path = path.join("memory.current");
    if !max_path.exists() || !current_path.exists() {
        return Ok(None);
    }
    let Some(limit) = parse_cgroup_limit(
        &fs::read_to_string(&max_path)
            .map_err(|error| format!("read {}: {error}", max_path.display()))?,
    )?
    else {
        return Ok(None);
    };
    let current = fs::read_to_string(&current_path)
        .map_err(|error| format!("read {}: {error}", current_path.display()))?
        .trim()
        .parse::<u64>()
        .map_err(|error| format!("parse {}: {error}", current_path.display()))?;
    Ok(Some(limit.saturating_sub(current)))
}

fn cgroup_v1_available_at(path: &Path) -> Result<Option<u64>, String> {
    let limit_path = path.join("memory.limit_in_bytes");
    let usage_path = path.join("memory.usage_in_bytes");
    if !limit_path.exists() || !usage_path.exists() {
        return Ok(None);
    }
    let limit = fs::read_to_string(&limit_path)
        .map_err(|error| format!("read {}: {error}", limit_path.display()))?
        .trim()
        .parse::<u64>()
        .map_err(|error| format!("parse {}: {error}", limit_path.display()))?;
    let usage = fs::read_to_string(&usage_path)
        .map_err(|error| format!("read {}: {error}", usage_path.display()))?
        .trim()
        .parse::<u64>()
        .map_err(|error| format!("parse {}: {error}", usage_path.display()))?;
    Ok(Some(limit.saturating_sub(usage)))
}

#[cfg(target_os = "linux")]
fn linux_cgroup_available_bytes() -> Result<Option<u64>, String> {
    let cgroup = fs::read_to_string("/proc/self/cgroup")
        .map_err(|error| format!("read /proc/self/cgroup: {error}"))?;
    let unified_relative = cgroup.lines().find_map(|line| {
        let mut fields = line.splitn(3, ':');
        let hierarchy = fields.next()?;
        let controllers = fields.next()?;
        let path = fields.next()?;
        (hierarchy == "0" && controllers.is_empty()).then(|| path.trim_start_matches('/'))
    });
    let root = PathBuf::from("/sys/fs/cgroup");
    if let Some(relative) = unified_relative {
        let nested = root.join(relative);
        if let Some(available) = cgroup_available_at(&nested)? {
            return Ok(Some(available));
        }
    }
    if let Some(available) = cgroup_available_at(&root)? {
        return Ok(Some(available));
    }

    let memory_relative = cgroup.lines().find_map(|line| {
        let mut fields = line.splitn(3, ':');
        let _hierarchy = fields.next()?;
        let controllers = fields.next()?;
        let path = fields.next()?;
        controllers
            .split(',')
            .any(|controller| controller == "memory")
            .then(|| path.trim_start_matches('/'))
    });
    let v1_root = root.join("memory");
    if let Some(relative) = memory_relative {
        let nested = v1_root.join(relative);
        if let Some(available) = cgroup_v1_available_at(&nested)? {
            return Ok(Some(available));
        }
    }
    cgroup_v1_available_at(&v1_root)
}

#[cfg(target_os = "linux")]
impl MemoryAvailabilityProbe for SystemMemoryAvailabilityProbe {
    fn availability(&self) -> Result<MemoryAvailability, String> {
        let host_available_bytes = parse_mem_available_bytes(
            &fs::read_to_string("/proc/meminfo")
                .map_err(|error| format!("read /proc/meminfo: {error}"))?,
        )?;
        let cgroup_available_bytes = linux_cgroup_available_bytes()?;
        let effective_available_bytes = cgroup_available_bytes
            .map_or(host_available_bytes, |value| {
                value.min(host_available_bytes)
            });
        Ok(MemoryAvailability {
            host_available_bytes,
            cgroup_available_bytes,
            effective_available_bytes,
        })
    }
}

#[cfg(target_os = "windows")]
impl MemoryAvailabilityProbe for SystemMemoryAvailabilityProbe {
    fn availability(&self) -> Result<MemoryAvailability, String> {
        #[repr(C)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
        }
        let mut status = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
            return Err("GlobalMemoryStatusEx failed".to_string());
        }
        Ok(MemoryAvailability {
            host_available_bytes: status.avail_phys,
            cgroup_available_bytes: None,
            effective_available_bytes: status.avail_phys,
        })
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
impl MemoryAvailabilityProbe for SystemMemoryAvailabilityProbe {
    fn availability(&self) -> Result<MemoryAvailability, String> {
        Err("RAM-backed session cache memory probing is not implemented on this platform".into())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct RamSessionStoreStats {
    pub resident_snapshots: usize,
    pub resident_bytes: usize,
    pub reserved_bytes: usize,
    pub evictions: u64,
    pub last_effective_available_bytes: u64,
    pub last_budget_bytes: usize,
}

struct RamSessionEntry {
    snapshot: Arc<SessionSnapshot>,
    memory_cost_bytes: usize,
    last_access: u64,
}

pub struct RamSessionStore {
    probe: Arc<dyn MemoryAvailabilityProbe>,
    budget_fraction: f64,
    initial_budget_bytes: usize,
    entries: HashMap<SnapshotId, RamSessionEntry>,
    prefix_index: PrefixIndex,
    reservations: HashMap<RamReservationId, usize>,
    resident_bytes: usize,
    reserved_bytes: usize,
    access_clock: u64,
    next_snapshot_id: u64,
    next_reservation_id: u64,
    evictions: u64,
    last_availability: MemoryAvailability,
    last_budget_bytes: usize,
}

impl RamSessionStore {
    pub fn new(
        budget_fraction: f64,
        probe: Arc<dyn MemoryAvailabilityProbe>,
    ) -> Result<Self, String> {
        if !budget_fraction.is_finite() || budget_fraction <= 0.0 || budget_fraction > 1.0 {
            return Err(format!(
                "session cache RAM fraction must be finite and in (0, 1], got {budget_fraction}"
            ));
        }
        let availability = probe.availability()?;
        let initial_budget_bytes =
            fraction_bytes(availability.effective_available_bytes, budget_fraction)?;
        if initial_budget_bytes == 0 {
            return Err("session cache RAM fraction produced a zero-byte budget".to_string());
        }
        Ok(Self {
            probe,
            budget_fraction,
            initial_budget_bytes,
            entries: HashMap::new(),
            prefix_index: PrefixIndex::default(),
            reservations: HashMap::new(),
            resident_bytes: 0,
            reserved_bytes: 0,
            access_clock: 0,
            next_snapshot_id: 1,
            next_reservation_id: 1,
            evictions: 0,
            last_availability: availability,
            last_budget_bytes: initial_budget_bytes,
        })
    }

    fn refresh_budget(&mut self) -> Result<usize, String> {
        let availability = self.probe.availability()?;
        let resident_bytes = u64::try_from(self.resident_bytes)
            .map_err(|_| "session resident byte count does not fit u64".to_string())?;
        let controlled_baseline = availability
            .effective_available_bytes
            .saturating_add(resident_bytes);
        let live_budget = fraction_bytes(controlled_baseline, self.budget_fraction)?;
        let budget = self.initial_budget_bytes.min(live_budget);
        self.last_availability = availability;
        self.last_budget_bytes = budget;
        Ok(budget)
    }

    pub fn reserve(&mut self, required_bytes: usize) -> Result<RamReservationId, String> {
        self.reserve_protecting(required_bytes, &[])
    }

    pub fn reserve_protecting(
        &mut self,
        required_bytes: usize,
        protected: &[SnapshotId],
    ) -> Result<RamReservationId, String> {
        if required_bytes == 0 {
            return Err("session cache reservation must be positive".to_string());
        }
        let budget = self.refresh_budget()?;
        if required_bytes > budget {
            return Err(format!(
                "session snapshot requires {} bytes but the runtime RAM budget is {} bytes",
                required_bytes, budget
            ));
        }
        while self
            .resident_bytes
            .saturating_add(self.reserved_bytes)
            .saturating_add(required_bytes)
            > budget
        {
            self.evict_lru_excluding(protected)?;
        }
        let id = RamReservationId(self.next_reservation_id);
        let next_reservation_id = self
            .next_reservation_id
            .checked_add(1)
            .ok_or_else(|| "session reservation ID exhausted".to_string())?;
        let next_reserved_bytes = self
            .reserved_bytes
            .checked_add(required_bytes)
            .ok_or_else(|| "session reservation byte accounting overflow".to_string())?;
        self.reservations.insert(id, required_bytes);
        self.next_reservation_id = next_reservation_id;
        self.reserved_bytes = next_reserved_bytes;
        Ok(id)
    }

    pub fn cancel_reservation(&mut self, id: RamReservationId) -> Result<(), String> {
        let bytes = self
            .reservations
            .remove(&id)
            .ok_or_else(|| format!("unknown session cache reservation {}", id.0))?;
        self.reserved_bytes = self
            .reserved_bytes
            .checked_sub(bytes)
            .ok_or_else(|| "session reservation byte accounting underflow".to_string())?;
        Ok(())
    }

    /// Increase an existing reservation after a temporary runtime allocation
    /// exposes an additional exact state size. This keeps admission based on
    /// current cgroup-aware availability and evicts only through the normal LRU
    /// policy; callers must extend before allocating the pageable snapshot.
    pub fn extend_reservation(
        &mut self,
        id: RamReservationId,
        additional_bytes: usize,
        protected: &[SnapshotId],
    ) -> Result<(), String> {
        if additional_bytes == 0 {
            return Ok(());
        }
        let existing = *self
            .reservations
            .get(&id)
            .ok_or_else(|| format!("unknown session cache reservation {}", id.0))?;
        let expanded = existing
            .checked_add(additional_bytes)
            .ok_or_else(|| "session reservation byte accounting overflow".to_string())?;
        let budget = self.refresh_budget()?;
        if expanded > budget {
            return Err(format!(
                "expanded session snapshot requires {} bytes but the runtime RAM budget is {} bytes",
                expanded, budget
            ));
        }
        while self
            .resident_bytes
            .saturating_add(self.reserved_bytes)
            .saturating_add(additional_bytes)
            > budget
        {
            self.evict_lru_excluding(protected)?;
        }
        self.reservations.insert(id, expanded);
        self.reserved_bytes = self
            .reserved_bytes
            .checked_add(additional_bytes)
            .ok_or_else(|| "session reservation byte accounting overflow".to_string())?;
        Ok(())
    }

    pub fn commit(
        &mut self,
        reservation: RamReservationId,
        snapshot: SessionSnapshot,
    ) -> Result<SnapshotId, String> {
        snapshot.validate()?;
        let actual_bytes = snapshot.memory_cost_bytes();
        let reserved = *self
            .reservations
            .get(&reservation)
            .ok_or_else(|| format!("unknown session cache reservation {}", reservation.0))?;
        if actual_bytes > reserved {
            return Err(format!(
                "completed session snapshot owns {} bytes but reservation {} covers only {} bytes",
                actual_bytes, reservation.0, reserved
            ));
        }
        let next_access_clock = self
            .access_clock
            .checked_add(1)
            .ok_or_else(|| "session cache access clock exhausted".to_string())?;
        let id = SnapshotId(self.next_snapshot_id);
        let next_snapshot_id = self
            .next_snapshot_id
            .checked_add(1)
            .ok_or_else(|| "session snapshot ID exhausted".to_string())?;
        let next_resident_bytes = self
            .resident_bytes
            .checked_add(actual_bytes)
            .ok_or_else(|| "session resident byte accounting overflow".to_string())?;
        self.cancel_reservation(reservation)?;
        self.access_clock = next_access_clock;
        self.next_snapshot_id = next_snapshot_id;
        self.prefix_index.insert(id, &snapshot.consumed_token_ids);
        self.entries.insert(
            id,
            RamSessionEntry {
                snapshot: Arc::new(snapshot),
                memory_cost_bytes: actual_bytes,
                last_access: self.access_clock,
            },
        );
        self.resident_bytes = next_resident_bytes;
        Ok(id)
    }

    pub fn longest_prefix(
        &mut self,
        tokens: &[u32],
        compatibility: &SessionCompatibilitySignature,
    ) -> Result<PrefixLookupResult, String> {
        let fingerprints = self.prefix_index.query_fingerprints(tokens);
        let mut incompatible_match = None;
        let mut hit = None;
        for fingerprint in fingerprints.into_iter().rev() {
            let Some(candidates) = self.prefix_index.candidates.get(&fingerprint) else {
                continue;
            };
            for &id in candidates.iter().rev() {
                let Some(entry) = self.entries.get(&id) else {
                    continue;
                };
                if entry.snapshot.consumed_token_ids.as_slice()
                    != &tokens[..fingerprint.token_count]
                {
                    continue;
                }
                if &entry.snapshot.compatibility == compatibility {
                    hit = Some((id, fingerprint.token_count));
                    break;
                }
                incompatible_match = Some(
                    incompatible_match
                        .unwrap_or(0usize)
                        .max(fingerprint.token_count),
                );
            }
            if hit.is_some() {
                break;
            }
        }
        if let Some((snapshot_id, matched_tokens)) = hit {
            self.touch(snapshot_id)?;
            Ok(PrefixLookupResult::Hit {
                snapshot_id,
                matched_tokens,
            })
        } else if let Some(matched_tokens) = incompatible_match {
            Ok(PrefixLookupResult::SignatureMismatch { matched_tokens })
        } else {
            Ok(PrefixLookupResult::NoMatch)
        }
    }

    fn touch(&mut self, id: SnapshotId) -> Result<(), String> {
        let next = self
            .access_clock
            .checked_add(1)
            .ok_or_else(|| "session cache access clock exhausted".to_string())?;
        let entry = self
            .entries
            .get_mut(&id)
            .ok_or_else(|| format!("unknown session snapshot {}", id.0))?;
        self.access_clock = next;
        entry.last_access = next;
        Ok(())
    }

    /// Return a cheap immutable lease on a canonical pageable snapshot.
    /// The store keeps its accounting and prefix-index entry until eviction;
    /// eviction refuses to remove a snapshot while any runtime lease exists.
    pub fn get(&mut self, id: SnapshotId) -> Result<Option<Arc<SessionSnapshot>>, String> {
        if !self.entries.contains_key(&id) {
            return Ok(None);
        }
        self.touch(id)?;
        Ok(self
            .entries
            .get(&id)
            .map(|entry| Arc::clone(&entry.snapshot)))
    }

    pub fn remove(&mut self, id: SnapshotId) -> Result<Option<SessionSnapshot>, String> {
        let Some(entry) = self.entries.get(&id) else {
            return Ok(None);
        };
        if Arc::strong_count(&entry.snapshot) != 1 {
            return Err(format!(
                "session snapshot {} is leased by an active request",
                id.0
            ));
        }
        let next_resident_bytes = self
            .resident_bytes
            .checked_sub(entry.memory_cost_bytes)
            .ok_or_else(|| "session resident byte accounting underflow".to_string())?;
        self.prefix_index
            .remove(id, &entry.snapshot.consumed_token_ids)?;
        let entry = self
            .entries
            .remove(&id)
            .ok_or_else(|| format!("session snapshot {} disappeared during removal", id.0))?;
        self.resident_bytes = next_resident_bytes;
        let snapshot = Arc::try_unwrap(entry.snapshot)
            .map_err(|_| format!("session snapshot {} acquired a lease during removal", id.0))?;
        Ok(Some(snapshot))
    }

    fn evict_lru_excluding(&mut self, protected: &[SnapshotId]) -> Result<SnapshotId, String> {
        let protected: HashSet<_> = protected.iter().copied().collect();
        let id = self
            .entries
            .iter()
            .filter(|(id, entry)| {
                !protected.contains(id) && Arc::strong_count(&entry.snapshot) == 1
            })
            .min_by_key(|(id, entry)| (entry.last_access, id.0))
            .map(|(id, _)| *id)
            .ok_or_else(|| {
                format!(
                    "session RAM budget is exhausted by {} reserved bytes; no unprotected, unleased committed snapshot can be evicted",
                    self.reserved_bytes,
                )
            })?;
        self.remove(id)?
            .ok_or_else(|| format!("session snapshot {} disappeared during LRU eviction", id.0))?;
        self.evictions = self.evictions.saturating_add(1);
        Ok(id)
    }

    pub fn stats(&self) -> RamSessionStoreStats {
        RamSessionStoreStats {
            resident_snapshots: self.entries.len(),
            resident_bytes: self.resident_bytes,
            reserved_bytes: self.reserved_bytes,
            evictions: self.evictions,
            last_effective_available_bytes: self.last_availability.effective_available_bytes,
            last_budget_bytes: self.last_budget_bytes,
        }
    }
}

fn fraction_bytes(available: u64, fraction: f64) -> Result<usize, String> {
    let bytes = (available as f64 * fraction).floor();
    if !bytes.is_finite() || bytes < 0.0 || bytes > usize::MAX as f64 {
        return Err(format!(
            "session RAM budget overflow: available={available} fraction={fraction}"
        ));
    }
    Ok(bytes as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn rows(name: &str, ptr: u64) -> SequenceStateAllocation {
        SequenceStateAllocation {
            name: name.to_string(),
            kind: "gqa_k".to_string(),
            layer_idx: Some(3),
            device_ordinal: 0,
            ptr,
            storage_bytes: 4096,
            dtype: "uint8".to_string(),
            element_size: 1,
            shape: vec![16, 256],
            strides_bytes: vec![256, 1],
            growth: SequenceStateGrowth::TokenRows {
                logical_tokens_per_row: 1,
                capacity_rows: 16,
                row_bytes: 256,
            },
        }
    }

    #[test]
    fn token_rows_use_only_the_live_prefix() {
        let allocation = rows("layer3.k", 0x1000);
        assert_eq!(allocation.used_bytes(0), 0);
        assert_eq!(allocation.used_bytes(7), 7 * 256);
        assert_eq!(allocation.used_bytes(99), 4096);
    }

    #[test]
    fn compressed_rows_round_up_by_runtime_ratio() {
        let mut allocation = rows("layer3.compressed", 0x1000);
        allocation.growth = SequenceStateGrowth::TokenRows {
            logical_tokens_per_row: 4,
            capacity_rows: 16,
            row_bytes: 256,
        };
        assert_eq!(allocation.used_bytes(1), 256);
        assert_eq!(allocation.used_bytes(4), 256);
        assert_eq!(allocation.used_bytes(5), 512);
    }

    #[test]
    fn registry_rejects_overlapping_real_allocations() {
        let mut registry = SequenceStateRegistry::default();
        registry.register(rows("a", 0x1000)).unwrap();
        let error = registry.register(rows("b", 0x1800)).unwrap_err();
        assert!(error.contains("overlap"));
    }

    #[test]
    fn registry_rejects_formula_metadata_disagreeing_with_storage() {
        let mut allocation = rows("bad", 0x1000);
        allocation.storage_bytes += 1;
        assert!(allocation.validate().unwrap_err().contains("shape bytes"));
    }

    struct TestMemoryProbe {
        available: AtomicU64,
        cgroup_available: Option<u64>,
    }

    impl TestMemoryProbe {
        fn new(available: u64) -> Self {
            Self {
                available: AtomicU64::new(available),
                cgroup_available: None,
            }
        }
    }

    impl MemoryAvailabilityProbe for TestMemoryProbe {
        fn availability(&self) -> Result<MemoryAvailability, String> {
            let host = self.available.load(Ordering::SeqCst);
            let effective = self.cgroup_available.map_or(host, |limit| limit.min(host));
            Ok(MemoryAvailability {
                host_available_bytes: host,
                cgroup_available_bytes: self.cgroup_available,
                effective_available_bytes: effective,
            })
        }
    }

    fn signature() -> SessionCompatibilitySignature {
        SessionCompatibilitySignature {
            snapshot_format_version: SESSION_SNAPSHOT_FORMAT_VERSION,
            runtime_version: env!("CARGO_PKG_VERSION").to_string(),
            model_identity: "test-model".to_string(),
            model_revision: Some("revision".to_string()),
            tokenizer_sha256: [1; 32],
            chat_template_sha256: [2; 32],
            expert_quantization: "int4:g128:amax".to_string(),
            attention_quantization: "hqq4:g128".to_string(),
            kv_format: "k4v4".to_string(),
            kv_key_bits: 4,
            kv_value_bits: 4,
            model_num_layers: 1,
            topology: vec![LayerOwnership {
                device_ordinal: 0,
                physical_device_id: "GPU-test".to_string(),
                compute_capability_major: 12,
                compute_capability_minor: 0,
                total_memory_bytes: 1024,
                layer_start: 0,
                layer_end: 1,
            }],
            state_layout_sha256: [3; 32],
        }
    }

    fn snapshot(seed: u32, payload_bytes: usize) -> SessionSnapshot {
        SessionSnapshot {
            compatibility: signature(),
            consumed_token_ids: vec![seed, seed + 1],
            positions: vec![DeviceSequencePosition {
                device_ordinal: 0,
                kv_absolute_position: 2,
                rope_position_delta: 0,
                rope_absolute_position: 2,
            }],
            state_blobs: vec![SequenceStateBlob {
                allocation_name: "layer0.k".to_string(),
                kind: "gqa_k".to_string(),
                layer_idx: Some(0),
                device_ordinal: 0,
                dtype: "uint8".to_string(),
                element_size: 1,
                shape: vec![payload_bytes],
                strides_bytes: vec![1],
                bytes: vec![seed as u8; payload_bytes],
            }],
        }
    }

    fn snapshot_with_tokens(tokens: &[u32], payload_bytes: usize) -> SessionSnapshot {
        let mut value = snapshot(tokens[0], payload_bytes);
        value.consumed_token_ids = tokens.to_vec();
        value.positions[0].kv_absolute_position = tokens.len();
        value.positions[0].rope_absolute_position = tokens.len() as i64;
        value
    }

    fn commit_snapshot(store: &mut RamSessionStore, snapshot: SessionSnapshot) -> SnapshotId {
        let bytes = snapshot.memory_cost_bytes();
        let reservation = store.reserve(bytes).unwrap();
        store.commit(reservation, snapshot).unwrap()
    }

    #[test]
    fn signature_requires_complete_non_overlapping_layer_ownership() {
        let mut value = signature();
        value.model_num_layers = 2;
        assert!(value.validate().unwrap_err().contains("ends at"));
        value.topology.push(LayerOwnership {
            device_ordinal: 1,
            physical_device_id: "GPU-test-2".to_string(),
            compute_capability_major: 8,
            compute_capability_minor: 0,
            total_memory_bytes: 512,
            layer_start: 1,
            layer_end: 2,
        });
        value.validate().unwrap();
        value.topology[1].layer_start = 0;
        assert!(value.validate().unwrap_err().contains("not contiguous"));
    }

    #[test]
    fn snapshot_validates_positions_and_actual_blob_shape() {
        let mut value = snapshot(7, 128);
        value.validate().unwrap();
        value.positions[0].rope_absolute_position = 3;
        assert!(value.validate().unwrap_err().contains("RoPE position"));
        value.positions[0].rope_absolute_position = 2;
        value.state_blobs[0].shape[0] += 1;
        assert!(value.validate().unwrap_err().contains("shape bytes"));
    }

    #[test]
    fn snapshot_requires_exact_consumed_boundary_and_multi_gpu_position_agreement() {
        let mut value = snapshot(7, 128);
        value.positions[0].kv_absolute_position = 1;
        value.positions[0].rope_absolute_position = 1;
        assert!(value
            .validate()
            .unwrap_err()
            .contains("consumed token count"));

        value = snapshot(7, 128);
        value.compatibility.model_num_layers = 2;
        value.compatibility.topology.push(LayerOwnership {
            device_ordinal: 1,
            physical_device_id: "GPU-test-2".to_string(),
            compute_capability_major: 8,
            compute_capability_minor: 0,
            total_memory_bytes: 512,
            layer_start: 1,
            layer_end: 2,
        });
        value.positions.push(DeviceSequencePosition {
            device_ordinal: 1,
            kv_absolute_position: 2,
            rope_position_delta: 1,
            rope_absolute_position: 3,
        });
        assert!(value.validate().unwrap_err().contains("disagrees"));
    }

    #[test]
    fn ram_store_reserves_before_allocation_and_evicts_true_lru() {
        let sample = snapshot(1, 256);
        let cost = sample.memory_cost_bytes();
        let probe = Arc::new(TestMemoryProbe::new((cost * 2) as u64));
        let mut store = RamSessionStore::new(1.0, probe).unwrap();
        let first = commit_snapshot(&mut store, sample);
        let second = commit_snapshot(&mut store, snapshot(3, 256));
        assert_eq!(store.stats().resident_snapshots, 2);
        assert!(store.get(first).unwrap().is_some());
        let third = commit_snapshot(&mut store, snapshot(5, 256));
        assert!(store.get(second).unwrap().is_none());
        assert!(store.get(first).unwrap().is_some());
        assert!(store.get(third).unwrap().is_some());
        assert_eq!(store.stats().evictions, 1);
    }

    #[test]
    fn snapshot_lease_blocks_eviction_until_request_releases_it() {
        let first_snapshot = snapshot(1, 256);
        let cost = first_snapshot.memory_cost_bytes();
        let probe = Arc::new(TestMemoryProbe::new((cost * 2) as u64));
        let mut store = RamSessionStore::new(1.0, probe).unwrap();
        let first = commit_snapshot(&mut store, first_snapshot);
        let second = commit_snapshot(&mut store, snapshot(3, 256));
        let lease = store.get(first).unwrap().unwrap();

        let third = commit_snapshot(&mut store, snapshot(5, 256));
        assert!(store.get(first).unwrap().is_some());
        assert!(store.get(second).unwrap().is_none());
        assert!(store.get(third).unwrap().is_some());
        assert!(store.remove(first).unwrap_err().contains("leased"));

        drop(lease);
        assert!(store.remove(first).unwrap().is_some());
    }

    #[test]
    fn incremental_reservation_never_evicts_its_base_snapshot() {
        let first_snapshot = snapshot(1, 256);
        let cost = first_snapshot.memory_cost_bytes();
        let probe = Arc::new(TestMemoryProbe::new((cost * 2) as u64));
        let mut store = RamSessionStore::new(1.0, probe).unwrap();
        let first = commit_snapshot(&mut store, first_snapshot);
        let second = commit_snapshot(&mut store, snapshot(3, 256));

        let reservation = store.reserve_protecting(cost, &[first]).unwrap();
        assert!(store.get(first).unwrap().is_some());
        assert!(store.get(second).unwrap().is_none());
        assert!(store.reserve_protecting(cost, &[first]).is_err());
        store.cancel_reservation(reservation).unwrap();
    }

    #[test]
    fn extending_reservation_uses_live_budget_and_preserves_protected_snapshot() {
        let first_snapshot = snapshot(1, 256);
        let cost = first_snapshot.memory_cost_bytes();
        let budget = cost * 5 / 2;
        let probe = Arc::new(TestMemoryProbe::new(budget as u64));
        let mut store = RamSessionStore::new(1.0, probe).unwrap();
        let first = commit_snapshot(&mut store, first_snapshot);
        let second = commit_snapshot(&mut store, snapshot(3, 256));

        let initial = cost / 4;
        let additional = cost / 2;
        let reservation = store.reserve_protecting(initial, &[first]).unwrap();
        store
            .extend_reservation(reservation, additional, &[first])
            .unwrap();

        assert!(store.get(first).unwrap().is_some());
        assert!(store.get(second).unwrap().is_none());
        assert_eq!(store.stats().reserved_bytes, initial + additional);
        assert_eq!(store.stats().evictions, 1);
        store.cancel_reservation(reservation).unwrap();
        assert_eq!(store.stats().reserved_bytes, 0);
    }

    #[test]
    fn prefix_index_selects_longest_exact_compatible_boundary() {
        let short = snapshot_with_tokens(&[1, 2], 64);
        let long = snapshot_with_tokens(&[1, 2, 3, 4], 64);
        let cost = short.memory_cost_bytes() + long.memory_cost_bytes();
        let probe = Arc::new(TestMemoryProbe::new((cost * 4) as u64));
        let mut store = RamSessionStore::new(1.0, probe).unwrap();
        let short_id = commit_snapshot(&mut store, short);
        let long_id = commit_snapshot(&mut store, long);

        assert_eq!(
            store.longest_prefix(&[1, 2, 3, 4, 5], &signature()),
            Ok(PrefixLookupResult::Hit {
                snapshot_id: long_id,
                matched_tokens: 4,
            })
        );
        let mut incompatible = signature();
        incompatible.model_identity = "other-model".to_string();
        assert_eq!(
            store.longest_prefix(&[1, 2, 3, 4, 5], &incompatible),
            Ok(PrefixLookupResult::SignatureMismatch { matched_tokens: 4 })
        );
        store.remove(long_id).unwrap().unwrap();
        assert_eq!(
            store.longest_prefix(&[1, 2, 3, 4, 5], &signature()),
            Ok(PrefixLookupResult::Hit {
                snapshot_id: short_id,
                matched_tokens: 2,
            })
        );
    }

    #[test]
    fn active_prefix_plan_never_rewinds_recurrent_state() {
        assert_eq!(
            plan_active_prefix(&[1, 2], &[1, 2, 3], true),
            ActivePrefixPlan::Append { matched_tokens: 2 }
        );
        assert_eq!(
            plan_active_prefix(&[1, 2, 9], &[1, 2, 3], false),
            ActivePrefixPlan::TruncateKvAndAppend { matched_tokens: 2 }
        );
        assert_eq!(
            plan_active_prefix(&[1, 2, 9], &[1, 2, 3], true),
            ActivePrefixPlan::RequiresBoundarySnapshot { matched_tokens: 2 }
        );
        assert_eq!(
            plan_active_prefix(&[1, 2, 3], &[1, 2], false),
            ActivePrefixPlan::NoSuffixToCompute
        );
        assert_eq!(
            plan_active_prefix(&[1, 2], &[9, 2], false),
            ActivePrefixPlan::NoReusablePrefix
        );
    }

    #[test]
    fn ram_prefix_must_beat_active_shared_template_prefix() {
        assert!(ram_prefix_is_longer(4, 4_150));
        assert!(!ram_prefix_is_longer(4_150, 4));
        assert!(!ram_prefix_is_longer(4_150, 4_150));
    }

    #[test]
    fn failed_commit_keeps_reservation_visible_until_cancelled() {
        let value = snapshot(1, 64);
        let actual = value.memory_cost_bytes();
        let probe = Arc::new(TestMemoryProbe::new((actual * 4) as u64));
        let mut store = RamSessionStore::new(1.0, probe).unwrap();
        let reservation = store.reserve(actual - 1).unwrap();
        let error = store.commit(reservation, value).unwrap_err();
        assert!(error.contains("covers only"));
        assert_eq!(store.stats().reserved_bytes, actual - 1);
        store.cancel_reservation(reservation).unwrap();
        assert_eq!(store.stats().reserved_bytes, 0);
    }

    #[test]
    fn memory_parsers_use_memavailable_and_cgroup_remaining_bytes() {
        assert_eq!(
            parse_mem_available_bytes("MemTotal: 99 kB\nMemAvailable: 42 kB\n").unwrap(),
            42 * 1024
        );
        assert_eq!(parse_cgroup_limit("max\n").unwrap(), None);
        assert_eq!(parse_cgroup_limit("1000\n").unwrap(), Some(1000));

        let path = std::env::temp_dir().join(format!(
            "krasis-session-cache-cgroup-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("memory.max"), "1000\n").unwrap();
        fs::write(path.join("memory.current"), "250\n").unwrap();
        assert_eq!(cgroup_available_at(&path).unwrap(), Some(750));
        fs::write(path.join("memory.limit_in_bytes"), "2000\n").unwrap();
        fs::write(path.join("memory.usage_in_bytes"), "400\n").unwrap();
        assert_eq!(cgroup_v1_available_at(&path).unwrap(), Some(1600));
        fs::remove_dir_all(path).unwrap();
    }
}
