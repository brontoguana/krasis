//! Invalid restored ranges cannot be consumed until an explicit completed
//! restore covers the same allocation identity and every affected byte.
use crate::session_cache::SequenceStateAllocation;
use std::cell::Cell;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Range {
    identity: String,
    start: u64,
    end: u64,
}
impl Range {
    fn from_allocation(allocation: &SequenceStateAllocation, bytes: usize) -> Result<Self, String> {
        allocation.validate()?;
        if bytes == 0 || bytes > allocation.storage_bytes {
            return Err(format!("invalid restore range for {}", allocation.name));
        }
        let end = allocation
            .ptr
            .checked_add(
                u64::try_from(bytes).map_err(|_| "restore extent exceeds CUDA address space")?,
            )
            .ok_or("restore range address overflow")?;
        let identity = serde_json::to_string(&(
            &allocation.name,
            &allocation.kind,
            allocation.layer_idx,
            allocation.device_ordinal,
            &allocation.dtype,
            allocation.element_size,
            &allocation.shape,
            &allocation.strides_bytes,
            &allocation.growth,
        ))
        .map_err(|e| e.to_string())?;
        Ok(Self {
            identity,
            start: allocation.ptr,
            end,
        })
    }
    fn covers(&self, other: &Self) -> bool {
        self.identity == other.identity && self.start <= other.start && self.end >= other.end
    }
}

#[derive(Default)]
pub(crate) struct RestoreValidity {
    dirty: AtomicBool,
    ranges: Mutex<Vec<Range>>,
}
impl RestoreValidity {
    pub(crate) fn require_ready(&self) -> Result<(), String> {
        if !self.dirty.load(Ordering::Acquire) {
            return Ok(());
        }
        let ranges = self
            .ranges
            .lock()
            .map_err(|_| "sequence restore validity lock poisoned")?;
        if !ranges.is_empty() {
            return Err(format!("sequence state is invalid after an interrupted restore ({} affected allocation ranges); complete an explicit matching restore or restart Krasis before continuing", ranges.len()));
        }
        Ok(())
    }
    fn mark_invalid(&self, ranges: Vec<Range>) {
        // No CUDA work or callback executes while this lock is held.
        let mut invalid = self.ranges.lock().unwrap_or_else(|_| {
            eprintln!("KRASIS FATAL: sequence restore validity lock poisoned");
            std::process::exit(1)
        });
        for range in ranges {
            if !invalid.contains(&range) {
                invalid.push(range);
            }
        }
        self.dirty.store(!invalid.is_empty(), Ordering::Release);
    }
    fn repair(&self, restored: &[Range]) -> Result<(), String> {
        let mut invalid = self
            .ranges
            .lock()
            .map_err(|_| "sequence restore validity lock poisoned")?;
        invalid.retain(|range| !restored.iter().any(|good| good.covers(range)));
        self.dirty.store(!invalid.is_empty(), Ordering::Release);
        Ok(())
    }
    pub(crate) fn begin<'a>(
        &'a self,
        allocations: &[(SequenceStateAllocation, usize)],
    ) -> Result<RestoreMutation<'a>, String> {
        let ranges = allocations
            .iter()
            .map(|(a, n)| Range::from_allocation(a, *n))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RestoreMutation {
            validity: self,
            ranges,
            enqueued: Cell::new(false),
            completed: false,
        })
    }
}
pub(crate) struct RestoreMutation<'a> {
    validity: &'a RestoreValidity,
    ranges: Vec<Range>,
    enqueued: Cell<bool>,
    completed: bool,
}
impl RestoreMutation<'_> {
    pub(crate) fn note_enqueue(&self) {
        self.enqueued.set(true);
    }
    pub(crate) fn complete(&mut self) -> Result<(), String> {
        // Called only after all device transfers have completed successfully.
        self.validity.repair(&self.ranges)?;
        self.completed = true;
        Ok(())
    }
}
impl Drop for RestoreMutation<'_> {
    fn drop(&mut self) {
        if self.enqueued.get() && !self.completed {
            self.validity.mark_invalid(std::mem::take(&mut self.ranges));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn r(identity: &str, start: u64, end: u64) -> Range {
        Range {
            identity: identity.into(),
            start,
            end,
        }
    }
    fn phase(validity: &RestoreValidity, ranges: Vec<Range>) -> RestoreMutation<'_> {
        RestoreMutation {
            validity,
            ranges,
            enqueued: Cell::new(false),
            completed: false,
        }
    }
    #[test]
    fn rejection_before_enqueue_keeps_good_state() {
        let state = RestoreValidity::default();
        drop(phase(&state, vec![r("state", 16, 32)]));
        assert!(state.require_ready().is_ok());
    }
    #[test]
    fn failed_restore_requires_all_affected_ranges_and_full_extent() {
        let state = RestoreValidity::default();
        {
            let p = phase(&state, vec![r("carry", 16, 32), r("rows", 48, 80)]);
            p.note_enqueue();
        }
        assert!(state.require_ready().is_err());
        state
            .repair(&[r("carry", 16, 32), r("rows", 48, 64)])
            .unwrap();
        assert!(state.require_ready().is_err());
        state.repair(&[r("rows", 48, 96)]).unwrap();
        assert!(state.require_ready().is_ok());
    }
    #[test]
    fn unwind_after_enqueue_preserves_invalidity() {
        let state = RestoreValidity::default();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let p = phase(&state, vec![r("carry", 16, 32)]);
            p.note_enqueue();
            panic!("callback");
        }))
        .is_err());
        assert!(state.require_ready().is_err());
    }
    #[test]
    fn different_owner_or_layout_cannot_clear_a_failed_restore() {
        let state = RestoreValidity::default();
        state.mark_invalid(vec![r("gpu/a/layout1", 16, 32)]);
        state
            .repair(&[
                r("gpu/b/layout1", 16, 32),
                r("gpu/a/layout2", 16, 32),
                r("gpu/a/layout1", 32, 48),
            ])
            .unwrap();
        assert!(state.require_ready().is_err());
        state.repair(&[r("gpu/a/layout1", 16, 32)]).unwrap();
        assert!(state.require_ready().is_ok());
    }
    #[test]
    fn successful_restore_does_not_clear_unrelated_external_state() {
        let state = RestoreValidity::default();
        state.mark_invalid(vec![r("stage", 16, 32)]);
        let mut p = phase(&state, vec![r("canonical", 48, 80)]);
        p.note_enqueue();
        p.complete().unwrap();
        drop(p);
        assert!(state.require_ready().is_err());
        state.repair(&[r("stage", 16, 32)]).unwrap();
        assert!(state.require_ready().is_ok());
    }
}
