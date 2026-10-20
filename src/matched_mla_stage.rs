//! Explicit read-only stage observations. Ordinary enqueue has no observer.
use crate::stable_mla_gemm::{Element,Span};
use crate::matched_probe::Target;
use cudarc::driver::sys;
use serde_json::json;
#[derive(Clone,Copy,Debug)]
pub(crate) enum Layout { Row{rows:usize,width:usize}, Head{rows:usize,heads:usize,width:usize} }
impl Layout {
    pub(crate) fn ranges(self,row:usize,bytes:usize)->Result<Vec<(usize,usize)>,String>{
        if !matches!(bytes,2|4){return Err("MLA observed dtype width invalid".into());}
        let (rows,heads,width)=match self{Self::Row{rows,width}=>(rows,1,width),Self::Head{rows,heads,width}=>(rows,heads,width)};
        if rows==0||heads==0||width==0||row>=rows{return Err("MLA observed row geometry invalid".into());}
        let stride=width.checked_mul(bytes).ok_or("MLA observed row byte overflow")?;
        let extent=rows.checked_mul(heads).and_then(|n|n.checked_mul(stride)).ok_or("MLA observed extent overflow")?;
        if rows==1{return Ok(vec![(0,extent)]);}
        let mut out=Vec::new();out.try_reserve_exact(heads).map_err(|e|format!("MLA observed range allocation: {e}"))?;for h in 0..heads{let offset=h.checked_mul(rows).and_then(|n|n.checked_add(row)).and_then(|n|n.checked_mul(stride)).ok_or("MLA observed head offset overflow")?;if offset.checked_add(stride).map_or(true,|n|n>extent){return Err("MLA observed range outside source".into());}out.push((offset,stride));}Ok(out)
    }
}
pub(crate) type Observer<'a> = dyn Fn(&str,&Span<'_>,Layout)->Result<(),String>+'a;
pub(crate) fn dtype(element:Element)->Result<(&'static str,usize),String>{match element{Element::F32=>Ok(("float32",4)),Element::Bf16=>Ok(("bfloat16",2)),Element::I32=>Ok(("int32",4)),Element::Bytes=>Err("MLA stage requires an actual typed operand".into())}}
pub(crate) fn prefill(target:&Target,stage:&str,span:&Span<'_>,layout:Layout,row:usize,absolute_start:usize,maximum_bytes:usize,stream:sys::CUstream)->Result<(),String>{
    if stage.is_empty()||!stage.bytes().all(|c|c.is_ascii_alphanumeric()||c==b'_'){return Err("invalid MLA stage component".into());}
    let (dtype,size)=dtype(span.element())?;let ranges=layout.ranges(row,size)?;
    let total=ranges.iter().try_fold(0usize,|n,(_,b)|n.checked_add(*b).ok_or("MLA observed total overflow"))?;if maximum_bytes==0||total>maximum_bytes{return Err("MLA observed stage exceeds explicit byte bound".into());}
    let parts=ranges.iter().map(|&(offset,bytes)|span.subview(offset,bytes)).collect::<Result<Vec<_>,_>>()?;
    // Preserve head-major source strides; each retained part is borrowed until its fenced download completes.
    for (index,part) in parts.iter().enumerate(){target.device(&format!("{stage}-part{index}.bin"),part.pointer(),ranges[index].1,stream)?;}
    target.write(&format!("{stage}.json"),&serde_json::to_vec_pretty(&json!({"stage":stage,"dtype":dtype,"element_bytes":size,"source_layout":format!("{layout:?}"),"local_row":row,"absolute_row":absolute_start.checked_add(row).ok_or("MLA observed absolute row overflow")?,"ranges":ranges,"part_order":"head order; concatenate complete parts","diagnostic_gpu_allocation_bytes":0})).map_err(|e|e.to_string())?)
}
pub(crate) fn graph_layout(stage:&str,heads:usize,ccd:usize,rope:usize,values:usize,slots:usize)->Result<(usize,&'static str),String>{
    if heads==0||ccd==0||values==0||slots==0{return Err("invalid actual MLA observation geometry".into());}
    let width=ccd.checked_add(rope).ok_or("MLA observed cache width overflow")?;
    let mul=|a:usize,b:usize|a.checked_mul(b).ok_or_else(||"MLA graph observation extent overflow".to_string());
    match stage{
        "wvc_weights"=>Ok((mul(mul(heads,ccd)?,values)?,"float32")),
        "selected"=>Ok((slots,"int32")),"positions"=>Ok((1,"int32")),
        "gathered"=>Ok((mul(slots,width)?,"float32")),"query"=>Ok((mul(heads,width)?,"float32")),
        "initial_scores"|"scores"|"weights"=>Ok((mul(heads,slots)?,"float32")),
        "attention"=>Ok((mul(heads,ccd)?,"float32")),"projected"=>Ok((mul(heads,values)?,"float32")),
        "output"=>Ok((mul(heads,values)?,"bfloat16")),_=>Err("unknown actual MLA stage".into())
    }
}
#[cfg(test)] mod tests{use super::*;
#[test]fn mla_observed_rows_preserve_head_strides_and_reject_overflow(){assert_eq!(Layout::Row{rows:3,width:5}.ranges(1,4).unwrap(),vec![(20,20)]);assert_eq!(Layout::Head{rows:3,heads:2,width:5}.ranges(1,4).unwrap(),vec![(20,20),(80,20)]);assert_eq!(Layout::Head{rows:1,heads:2,width:5}.ranges(0,4).unwrap(),vec![(0,40)]);assert!(Layout::Row{rows:3,width:5}.ranges(3,4).is_err());assert!(Layout::Head{rows:usize::MAX,heads:2,width:5}.ranges(0,4).is_err());assert!(Layout::Row{rows:1,width:0}.ranges(0,4).is_err());assert!(Layout::Row{rows:1,width:1}.ranges(0,3).is_err());}
#[test]fn mla_observed_range_allocation_overflow_returns_error(){let heads=(isize::MAX as usize)/std::mem::size_of::<(usize,usize)>()+1;assert!(Layout::Head{rows:2,heads,width:1}.ranges(0,2).is_err());}
#[test]fn mla_observed_extents_follow_actual_geometry_and_dtype(){assert_eq!(graph_layout("gathered",3,48,16,80,69).unwrap(),(4416,"float32"));assert_eq!(graph_layout("projected",3,48,16,80,69).unwrap(),(240,"float32"));assert_eq!(graph_layout("output",3,48,16,80,69).unwrap(),(240,"bfloat16"));assert_eq!(graph_layout("selected",3,48,16,80,69).unwrap(),(69,"int32"));assert!(graph_layout("unknown",3,48,16,80,69).is_err());assert!(graph_layout("gathered",3,usize::MAX,16,80,69).is_err());assert!(dtype(Element::Bytes).is_err());}
}
