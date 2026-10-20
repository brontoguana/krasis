//! Prepared common sparse MLA attention and WVC path. No allocation, selection,
//! synchronization, or interpreter calls in enqueue. Real owners remain borrowed.
use crate::{
    stable_mla_gemm::{Candidate, Element, Matrix, Prepared, Shape, Span},
    stable_mla_ops::{Codec, Gather, Ops},
};
use cudarc::driver::sys;
use sha2::{Digest, Sha256};
use std::sync::Arc;
fn mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or("common MLA path extent overflow".into())
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Geometry {
    pub heads: usize,
    pub ccd: usize,
    pub rope: usize,
    pub values: usize,
    pub slots: usize,
    pub codec: Codec,
}
impl Geometry {
    pub(crate) fn shapes(self, rows: usize) -> Result<[Shape; 3], String> {
        let width = self
            .ccd
            .checked_add(self.rope)
            .ok_or("common MLA width overflow")?;
        if self.ccd == 0
            || self.ccd % 16 != 0
            || self.rope % 16 != 0
            || self.values == 0
            || self.heads == 0
            || self.slots == 0
            || rows == 0
        {
            return Err("invalid common MLA path geometry".into());
        }
        let matrix = |r, c, leading, batch, stride| Matrix {
            rows: r,
            cols: c,
            leading,
            batch,
            stride,
        };
        let gathered = matrix(width, self.slots, width, rows, mul(width, self.slots)?);
        let query = matrix(width, self.heads, width, rows, mul(width, self.heads)?);
        let scores = matrix(
            self.slots,
            self.heads,
            self.slots,
            rows,
            mul(self.slots, self.heads)?,
        );
        let latent = matrix(
            self.ccd,
            self.heads,
            self.ccd,
            rows,
            mul(self.ccd, self.heads)?,
        );
        let value_cache = Matrix {
            rows: self.ccd,
            ..gathered
        };
        let weight = matrix(
            self.ccd,
            self.values,
            self.ccd,
            self.heads,
            mul(self.ccd, self.values)?,
        );
        let token_major = matrix(
            self.ccd,
            rows,
            mul(self.heads, self.ccd)?,
            self.heads,
            self.ccd,
        );
        let projected = matrix(
            self.values,
            rows,
            self.values,
            self.heads,
            mul(rows, self.values)?,
        );
        let shapes = [
            Shape {
                a: gathered,
                b: query,
                c: scores,
                transpose_a: true,
                transpose_b: false,
            },
            Shape {
                a: value_cache,
                b: scores,
                c: latent,
                transpose_a: false,
                transpose_b: false,
            },
            Shape {
                a: weight,
                b: token_major,
                c: projected,
                transpose_a: true,
                transpose_b: false,
            },
        ];
        for shape in &shapes {
            shape.validate()?
        }
        Ok(shapes)
    }
    pub(crate) fn work_elements(self, rows: usize) -> Result<[usize; 7], String> {
        let s = self.shapes(rows)?;
        Ok([
            s[0].a.elements(false)?,
            s[0].b.elements(false)?,
            s[0].c.elements(true)?,
            s[1].b.elements(false)?,
            s[1].c.elements(true)?,
            s[2].c.elements(true)?,
            mul(mul(rows, self.heads)?, self.values)?,
        ])
    }
}
pub(crate) struct Work<'a> {
    pub gathered: Span<'a>,
    pub query: Span<'a>,
    pub scores: Span<'a>,
    pub weights: Span<'a>,
    pub attention: Span<'a>,
    pub projected: Span<'a>,
    pub output: Span<'a>,
}
pub(crate) struct Inputs<'a> {
    pub q_abs: Span<'a>,
    pub q_rope: Option<Span<'a>>,
    pub ckv: Span<'a>,
    pub kpe: Option<Span<'a>>,
    pub selected: Span<'a>,
    pub positions: Span<'a>,
    pub wvc: Span<'a>,
}
pub(crate) struct Path {
    ops: Arc<Ops>,
    geometry: Geometry,
    rows: usize,
    score: Prepared,
    latent: Prepared,
    wvc: Prepared,
    pub(crate) family_identity: String,
}
impl Path {
    /// Cold/request preparation only. Each supplied algorithm must be selected by
    /// runtime common-family admission/measurement, never by a per-shape fallback.
    pub(crate) fn prepare(
        ops: Arc<Ops>,
        geometry: Geometry,
        rows: usize,
        algorithms: [&Candidate; 3],
        workspace_capacity: usize,
    ) -> Result<Self, String> {
        let shapes = geometry.shapes(rows)?;
        let family_identity = format!(
            "common_mla_f32_v1:{:x}",
            Sha256::digest(
                format!(
                    "{:?}:{}:{:?}",
                    geometry,
                    ops.identity,
                    algorithms.map(|a| a.family_identity())
                )
                .as_bytes()
            )
        );
        Ok(Self {
            score: algorithms[0].prepare(shapes[0], workspace_capacity)?,
            latent: algorithms[1].prepare(shapes[1], workspace_capacity)?,
            wvc: algorithms[2].prepare(shapes[2], workspace_capacity)?,
            ops,
            geometry,
            rows,
            family_identity,
        })
    }
    /// Caller holds all allocations, this prepared path, and its stream/graph owner
    /// through completion, including partial enqueue failure/unwind. An error must
    /// poison the enclosing current request; no alternate arithmetic is attempted.
    pub(crate) unsafe fn enqueue(
        &self,
        gather: Gather,
        input: &Inputs<'_>,
        work: &Work<'_>,
        workspace: Option<&Span<'_>>,
        scale: f32,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        self.enqueue_observed(gather,input,work,workspace,scale,stream,None)
    }
    pub(crate) unsafe fn enqueue_observed(
        &self,
        gather: Gather,
        input: &Inputs<'_>,
        work: &Work<'_>,
        workspace: Option<&Span<'_>>,
        scale: f32,
        stream: sys::CUstream,
        observer: Option<&crate::matched_mla_stage::Observer<'_>>,
    ) -> Result<(), String> {
        let g = self.geometry;
        if gather.codec != g.codec
            || gather.rows != self.rows
            || gather.heads != g.heads
            || gather.ccd != g.ccd
            || gather.rope != g.rope
            || gather.slots != g.slots
            || !scale.is_finite()
            || scale <= 0.0
        {
            return Err("common MLA prepared geometry/scale mismatch".into());
        }
        gather.validate()?;
        let spans = [
            &work.gathered,
            &work.query,
            &work.scores,
            &work.weights,
            &work.attention,
            &work.projected,
            &work.output,
        ];
        let elements = g.work_elements(self.rows)?;
        let mut regions = [(0, 0); 7];
        if input.wvc.element() != Element::F32 {
            return Err("common MLA WVC source must retain F32 precision".into());
        }
        for i in 0..7 {
            if spans[i].element() != if i == 6 { Element::Bf16 } else { Element::F32 } {
                return Err("common MLA workspace dtype mismatch".into());
            }
            regions[i] = spans[i].range(mul(elements[i], if i == 6 { 2 } else { 4 })?)?;
        }
        let readonly = [
            Some(&input.q_abs),
            input.q_rope.as_ref(),
            Some(&input.ckv),
            input.kpe.as_ref(),
            Some(&input.selected),
            Some(&input.positions),
            Some(&input.wvc),
        ];
        let overlaps =
            |a: (u64, u64), b: (u64, u64)| a.0 < a.1 && b.0 < b.1 && a.0 < b.1 && b.0 < a.1;
        for (i, &r) in regions.iter().enumerate() {
            for owner in readonly.iter().flatten() {
                if overlaps(r, owner.full_range()?) {
                    return Err("common MLA writable workspace aliases retained input".into());
                }
            }
            for (j, &other) in regions.iter().enumerate().skip(i + 1) {
                if (i, j) != (1, 5) && overlaps(r, other) {
                    return Err("common MLA simultaneous workspace planes alias".into());
                }
            }
            if let Some(scratch) = workspace {
                if overlaps(r, scratch.full_range()?) {
                    return Err("common MLA GEMM workspace aliases phase plane".into());
                }
            }
        }
        if let Some(scratch) = workspace {
            for owner in readonly.iter().flatten() {
                if overlaps(scratch.full_range()?, owner.full_range()?) {
                    return Err("common MLA GEMM workspace aliases retained input".into());
                }
            }
        }
        use crate::matched_mla_stage::Layout;
        let observe=|stage:&str,span:&Span<'_>,layout:Layout|->Result<(),String>{if let Some(callback)=observer{callback(stage,span,layout)?;}Ok(())};
        if observer.is_some(){
            observe("wvc_weights",&input.wvc,Layout::Row{rows:1,width:mul(mul(g.heads,g.ccd)?,g.values)?})?;
            let indices=input.selected.subview(mul(mul(gather.row_start,g.slots)?,4)?,mul(mul(self.rows,g.slots)?,4)?)?;
            let positions=input.positions.subview(mul(gather.row_start,4)?,mul(self.rows,4)?)?;
            observe("selected",&indices,Layout::Row{rows:self.rows,width:g.slots})?;
            observe("positions",&positions,Layout::Row{rows:self.rows,width:1})?;
        }
        self.ops.gather(
            gather,
            &work.gathered,
            &work.query,
            &work.scores,
            &input.q_abs,
            input.q_rope.as_ref(),
            &input.ckv,
            input.kpe.as_ref(),
            &input.selected,
            &input.positions,
            stream,
        )?;
        observe("gathered",&work.gathered,Layout::Row{rows:self.rows,width:mul(g.slots,g.ccd.checked_add(g.rope).ok_or("MLA observed width")?)?})?;
        observe("query",&work.query,Layout::Row{rows:self.rows,width:mul(g.heads,g.ccd.checked_add(g.rope).ok_or("MLA observed query width")?)?})?;
        observe("initial_scores",&work.scores,Layout::Row{rows:self.rows,width:mul(g.heads,g.slots)?})?;
        self.score.enqueue(
            &work.gathered,
            &work.query,
            &work.scores,
            &work.scores,
            workspace,
            scale,
            1.0,
            stream,
        )?;
        observe("scores",&work.scores,Layout::Row{rows:self.rows,width:mul(g.heads,g.slots)?})?;
        self.ops.softmax(
            &work.scores,
            &work.weights,
            self.rows,
            g.heads,
            g.slots,
            stream,
        )?;
        observe("weights",&work.weights,Layout::Row{rows:self.rows,width:mul(g.heads,g.slots)?})?;
        self.latent.enqueue(
            &work.gathered,
            &work.weights,
            &work.attention,
            &work.attention,
            workspace,
            1.0,
            0.0,
            stream,
        )?;
        observe("attention",&work.attention,Layout::Row{rows:self.rows,width:mul(g.heads,g.ccd)?})?;
        self.wvc.enqueue(
            &input.wvc,
            &work.attention,
            &work.projected,
            &work.projected,
            workspace,
            1.0,
            0.0,
            stream,
        )?;
        observe("projected",&work.projected,Layout::Head{rows:self.rows,heads:g.heads,width:g.values})?;
        self.ops.scatter(
            &work.projected,
            &work.output,
            self.rows,
            g.heads,
            g.values,
            self.rows,
            stream,
        )?;
        observe("output",&work.output,Layout::Row{rows:self.rows,width:mul(g.heads,g.values)?})
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn readonly_token_major_batches_and_rotary_value_slicing_are_physical() {
        let g = Geometry {
            heads: 3,
            ccd: 48,
            rope: 64,
            values: 80,
            slots: 69,
            codec: Codec::K6,
        };
        let s = g.shapes(67).unwrap();
        assert_eq!(s[1].a.leading, 112);
        assert_eq!(s[1].a.rows, 48);
        assert_eq!(s[2].b.leading, 144);
        assert_eq!(s[2].b.stride, 48);
        assert_eq!(s[2].b.elements(false).unwrap(), 3 * 67 * 48);
        assert!(s[2].b.elements(true).is_err());
        assert_eq!(g.work_elements(67).unwrap()[6], 67 * 3 * 80);
    }
    #[test]
    fn arbitrary_head_counts_and_value_widths_reject_overflow() {
        let g = Geometry {
            heads: 1,
            ccd: 16,
            rope: 0,
            values: 37,
            slots: 1,
            codec: Codec::K4,
        };
        assert!(g.shapes(1).is_ok());
        assert!(Geometry {
            heads: usize::MAX,
            ..g
        }
        .shapes(2)
        .is_err());
        assert!(Geometry { rope: 17, ..g }.shapes(2).is_err());
        assert!(g.shapes(0).is_err());
    }
}
