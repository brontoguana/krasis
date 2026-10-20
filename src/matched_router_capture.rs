// Diagnostic-only loaded-buffer admission; no allocations or dispatch changes.
#[derive(Debug)]
pub(crate) struct RouterCaptureLayout {
    pub hidden: usize,
    pub experts: usize,
    pub topk: usize,
    pub gate_rows: usize,
    pub gate_cols: usize,
    pub gate_dtype: u8,
    pub scratch_elements: usize,
    pub hidden_elements: usize,
    pub hidden_ptr: u64,
    pub gate_ptr: u64,
    pub logits_ptr: u64,
    pub ids_ptr: u64,
    pub weights_ptr: u64,
    pub bias_ptr: u64,
    pub correction_ptr: u64,
}
#[derive(Debug)]
pub(crate) struct Span {
    pub name: &'static str,
    pub ptr: u64,
    pub bytes: usize,
}
impl RouterCaptureLayout {
    pub fn checked_spans(&self) -> Result<Vec<Span>, String> {
        if self.hidden == 0
            || self.experts == 0
            || self.topk == 0
            || self.topk > self.experts
            || self.hidden > i32::MAX as usize
            || self.experts > i32::MAX as usize
            || self.gate_dtype != 0
            || self.gate_rows != self.experts
            || self.gate_cols != self.hidden
            || self.hidden_elements < self.hidden
        {
            return Err(
                "router capture requires actual nonempty BF16 direct-hidden gate geometry".into(),
            );
        }
        let scratch = self
            .hidden
            .checked_add(self.experts)
            .ok_or("router scratch overflow")?;
        if self.scratch_elements < scratch {
            return Err("router logits exceed live scratch allocation".into());
        }
        let extent = |n: usize, size: usize| {
            n.checked_mul(size)
                .ok_or_else(|| "router extent overflow".to_string())
        };
        let mut spans = vec![
            Span {
                name: "normalized-input.bf16",
                ptr: self.hidden_ptr,
                bytes: extent(self.hidden, 2)?,
            },
            Span {
                name: "gate-weight.bf16",
                ptr: self.gate_ptr,
                bytes: extent(
                    self.gate_rows
                        .checked_mul(self.gate_cols)
                        .ok_or("router weight shape overflow")?,
                    2,
                )?,
            },
            Span {
                name: "gate-logits.f32",
                ptr: self.logits_ptr,
                bytes: extent(self.experts, 4)?,
            },
            Span {
                name: "route-ids.i32",
                ptr: self.ids_ptr,
                bytes: extent(self.topk, 4)?,
            },
            Span {
                name: "route-weights.f32",
                ptr: self.weights_ptr,
                bytes: extent(self.topk, 4)?,
            },
        ];
        for (name, ptr) in [
            ("gate-bias.f32", self.bias_ptr),
            ("score-correction.f32", self.correction_ptr),
        ] {
            if ptr != 0 {
                spans.push(Span {
                    name,
                    ptr,
                    bytes: extent(self.experts, 4)?,
                });
            }
        }
        for x in &spans {
            if x.ptr == 0
                || x.ptr
                    .checked_add(
                        u64::try_from(x.bytes).map_err(|_| "router extent does not fit pointer")?,
                    )
                    .is_none()
            {
                return Err(format!("invalid router capture span {}", x.name));
            }
        }
        for (i, x) in spans.iter().enumerate() {
            for y in &spans[..i] {
                if x.ptr < y.ptr + y.bytes as u64 && y.ptr < x.ptr + x.bytes as u64 {
                    return Err(format!("router capture aliases {} and {}", x.name, y.name));
                }
            }
        }
        Ok(spans)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn layout() -> RouterCaptureLayout {
        RouterCaptureLayout {
            hidden: 3,
            experts: 5,
            topk: 2,
            gate_rows: 5,
            gate_cols: 3,
            gate_dtype: 0,
            scratch_elements: 8,
            hidden_elements: 3,
            hidden_ptr: 100,
            gate_ptr: 200,
            logits_ptr: 300,
            ids_ptr: 400,
            weights_ptr: 500,
            bias_ptr: 0,
            correction_ptr: 0,
        }
    }
    #[test]
    fn measured_dimensions_and_optional_parameters() {
        let mut x = layout();
        let s = x.checked_spans().unwrap();
        assert_eq!(
            s.iter().map(|x| x.bytes).collect::<Vec<_>>(),
            vec![6, 30, 20, 8, 8]
        );
        x.bias_ptr = 600;
        x.correction_ptr = 700;
        assert_eq!(x.checked_spans().unwrap().len(), 7);
    }
    #[test]
    fn rejects_invalid_geometry_extent_and_alias_before_reads() {
        let mut x = layout();
        x.gate_dtype = 5;
        assert!(x.checked_spans().is_err());
        x = layout();
        x.gate_cols = 4;
        assert!(x.checked_spans().is_err());
        x = layout();
        x.scratch_elements = 7;
        assert!(x.checked_spans().is_err());
        x = layout();
        x.ids_ptr = 315;
        assert!(x.checked_spans().is_err());
        x = layout();
        x.gate_ptr = u64::MAX - 4;
        assert!(x.checked_spans().is_err());
        x = layout();
        x.hidden = usize::MAX;
        x.gate_cols = x.hidden;
        x.hidden_elements = x.hidden;
        x.scratch_elements = usize::MAX;
        assert!(x.checked_spans().is_err());
        x = layout();
        x.topk = 6;
        assert!(x.checked_spans().is_err());
    }
}
