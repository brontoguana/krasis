//! CPU F64 mathematical reference for captured loaded BF16 dot products.
//! This is an operator comparison, not an original-model/Witness verdict.
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest,Sha256};
use std::{fs,path::PathBuf};
#[derive(Deserialize)] #[serde(deny_unknown_fields)]
struct Identity {path:PathBuf,bytes:usize,sha256:String}
impl Identity { fn read(&self)->Result<Vec<u8>,String>{let b=fs::read(&self.path).map_err(|e|e.to_string())?;if b.len()!=self.bytes||format!("{:x}",Sha256::digest(&b))!=self.sha256{return Err(format!("identity mismatch {}",self.path.display()));}Ok(b)} }
#[derive(Deserialize)] #[serde(deny_unknown_fields)]
struct Plan {schema:usize,layout:String,rows:usize,cols:usize,weight:Identity,input:Identity,prefill:Identity,decode:Identity}
fn bf(b:&[u8],i:usize)->Result<f64,String>{let v=f32::from_bits((u16::from_le_bytes([b[2*i],b[2*i+1]])as u32)<<16);if !v.is_finite(){return Err("nonfinite captured BF16".into());}Ok(v as f64)}
fn rne(v:f64)->Result<u16,String>{let f=v as f32;if !f.is_finite(){return Err("nonfinite reference output".into());}let b=f.to_bits();Ok((b.wrapping_add(0x7fff+((b>>16)&1))>>16)as u16)}
fn bytes(a:usize,b:usize)->Result<usize,String>{a.checked_mul(b).and_then(|n|n.checked_mul(2)).ok_or("extent overflow".into())}
fn run()->Result<(),String>{if std::env::var("KRASIS_DEV_SCRIPT").as_deref()!=Ok("1"){return Err("run through ./dev bf16-dot-reference".into());}let args:Vec<_>=std::env::args().collect();if args.len()!=3{return Err("expected plan and fresh output directory".into());}let raw=fs::read(&args[1]).map_err(|e|e.to_string())?;let p:Plan=serde_json::from_slice(&raw).map_err(|e|e.to_string())?;
if p.schema!=1||p.layout!="row_major_nk"||p.rows==0||p.cols==0{return Err("unsupported BF16 reference contract".into());}
let w=p.weight.read()?;let x=p.input.read()?;let pre=p.prefill.read()?;let dec=p.decode.read()?;
if w.len()!=bytes(p.rows,p.cols)?||x.len()!=bytes(1,p.cols)?||pre.len()!=bytes(1,p.rows)?||dec.len()!=pre.len(){return Err("captured extent mismatch".into());}
let inputs=(0..p.cols).map(|i|bf(&x,i)).collect::<Result<Vec<_>,_>>()?;
let mut output=Vec::with_capacity(pre.len());let mut pre_diff=0usize;let mut dec_diff=0usize;let mut pre_sse=0.0f64;let mut dec_sse=0.0f64;let mut pre_max=0.0f64;let mut dec_max=0.0f64;let mut norm=0.0f64;
for n in 0..p.rows{let mut dot=0.0f64;for k in 0..p.cols{dot=bf(&w,n*p.cols+k)?.mul_add(inputs[k],dot);}let bits=rne(dot)?;output.extend(bits.to_le_bytes());pre_diff+=usize::from(pre[2*n..2*n+2]!=bits.to_le_bytes());dec_diff+=usize::from(dec[2*n..2*n+2]!=bits.to_le_bytes());let pe=bf(&pre,n)?-dot;let de=bf(&dec,n)?-dot;pre_sse+=pe*pe;dec_sse+=de*de;pre_max=pre_max.max(pe.abs());dec_max=dec_max.max(de.abs());norm+=dot*dot;}
let out=PathBuf::from(&args[2]);fs::create_dir(&out).map_err(|e|e.to_string())?;fs::write(out.join("reference.bf16"),&output).map_err(|e|e.to_string())?;
let rel=|v:f64|if norm>0.0{Some((v/norm).sqrt())}else{None};let report=json!({"schema":1,"plan_sha256":format!("{:x}",Sha256::digest(&raw)),"rows":p.rows,"cols":p.cols,"prefill_different_from_f64_rne":pre_diff,"decode_different_from_f64_rne":dec_diff,"prefill_max_abs_to_f64":pre_max,"decode_max_abs_to_f64":dec_max,"prefill_relative_l2":rel(pre_sse),"decode_relative_l2":rel(dec_sse),"reference_sha256":format!("{:x}",Sha256::digest(&output)),"scope":"CPU F64 reduction of exact captured loaded BF16 coefficients/input, rounded F32 then BF16; not exact-real accumulator, serving, Witness or speed acceptance"});fs::write(out.join("REPORT.json"),serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;println!("{report}");Ok(())}
fn main(){if let Err(e)=run(){eprintln!("ERROR: {e}");std::process::exit(1);}}
