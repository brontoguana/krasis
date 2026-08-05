//! Runtime-owned sequence-state metadata and session-cache primitives.
//!
//! Model setup may allocate sequence state through another allocator (currently
//! PyTorch for several attention backends), but it must register the metadata
//! read from each real allocation here. Request-time inventory, measurement,
//! snapshot, and restore paths consume this Rust-owned registry and do not call
//! back into Python.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

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

    fn validate(&self) -> Result<(), String> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
