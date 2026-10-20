//! Checksum-bound decode and cached-continuation likelihood diagnostics. Use ./dev.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs,io::{Read,Write},net::{SocketAddr,TcpStream},path::{Path,PathBuf},time::Duration};
type Result<T> = std::result::Result<T,String>;
fn digest(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}
fn token_hash(ids:&[u32])->String {let mut h=Sha256::new();for id in ids {h.update(id.to_le_bytes());}format!("{:x}",h.finalize())}
fn ids(v:&Value)->Result<Vec<u32>> {serde_json::from_value(v.clone()).map_err(|e|format!("exact u32 token IDs required: {e}"))}
fn num(v:&Value,k:&str)->Result<usize> {usize::try_from(v[k].as_u64().ok_or_else(||format!("missing integer {k}"))?).map_err(|_|format!("integer overflow {k}"))}
fn text<'a>(v:&'a Value,k:&str)->Result<&'a str> {v[k].as_str().ok_or_else(||format!("missing string {k}"))}
fn read_json(p:&Path)->Result<Value> {serde_json::from_slice(&fs::read(p).map_err(|e|format!("{}: {e}",p.display()))?).map_err(|e|e.to_string())}
fn write_new(p:&Path,bytes:&[u8])->Result<()> {let mut f=fs::OpenOptions::new().write(true).create_new(true).open(p).map_err(|e|format!("{}: {e}",p.display()))?;f.write_all(bytes).and_then(|_|f.sync_all()).map_err(|e|e.to_string())?;fs::File::open(p.parent().ok_or("evidence parent missing")?).and_then(|d|d.sync_all()).map_err(|e|e.to_string())}
fn checked_json(p:&Path,sha:&str)->Result<Value> {let b=fs::read(p).map_err(|e|e.to_string())?;if sha.len()!=64 || !sha.bytes().all(|x|x.is_ascii_hexdigit()) || digest(&b)!=sha {return Err(format!("input checksum mismatch: {}",p.display()));}serde_json::from_slice(&b).map_err(|e|e.to_string())}
struct Entry {index:usize,document:String,full:Vec<u32>,seed:Vec<u32>,seed_forced:Vec<u32>,targets:Vec<u32>,boundary:usize}
fn prepare(input:&Value)->Result<Vec<Entry>> {
    if input["schema"]!=1 || input["diagnostic_only"]!=true || input["qualifies_candidate"]!=false {return Err("unsupported quality input contract".into());}
    let entries=input["entries"].as_array().ok_or("missing input entries")?;if entries.is_empty(){return Err("empty quality input set".into());}
    let mut out=Vec::new();for (i,e) in entries.iter().enumerate(){
        let index=num(e,"index")?;if index!=i{return Err("noncontiguous quality inputs".into());}
        let source=&e["source_request"];let path=Path::new(text(source,"path")?);if !path.is_absolute(){return Err("absolute frozen source request required".into());}
        let raw=fs::read(path).map_err(|e|e.to_string())?;if raw.len()!=num(source,"bytes")?||digest(&raw)!=text(source,"sha256")?{return Err("canonical source request changed".into());}
        let req:Value=serde_json::from_slice(&raw).map_err(|e|e.to_string())?;let source_ids=ids(&req["input_token_ids"])?;let source_targets=ids(&req["target_token_ids"])?;
        if source_ids.is_empty()||source_ids.len()!=source_targets.len(){return Err("canonical target count mismatch".into());}
        let mut all=source_ids;all.push(*source_targets.last().ok_or("missing canonical last target")?);
        if all[1..]!=source_targets||token_hash(&all)!=text(e,"all_token_ids_sha256")? {return Err("canonical next-token histories changed".into());}
        let full=ids(&e["full_prompt_ids"])?;let seed=ids(&e["seed_prompt_ids"])?;let seed_forced=ids(&e["seed_forced_output_ids"])?;let targets=ids(&e["authoritative_targets"])?;let boundary=num(e,"cached_consumed_boundary")?;
        if full.len()!=num(e,"logical_prompt_tokens")?||full.is_empty()||targets.len()<2||seed_forced.len()!=2||seed.len().checked_add(1)!=Some(boundary)
            || full.len().checked_sub(boundary)!=Some(num(e,"suffix_tokens")?)||all.len()!=full.len()+targets.len()
            || all.get(..full.len())!=Some(full.as_slice())||all.get(full.len()..)!=Some(targets.as_slice())||token_hash(&full)!=text(e,"prompt_token_ids_sha256")?
            || all.get(..seed.len())!=Some(seed.as_slice())||all.get(seed.len()..seed.len()+2)!=Some(seed_forced.as_slice()) {
            return Err("frozen full/seed/suffix/target history admission failed".into());
        }
        out.push(Entry{index,document:text(e,"document")?.into(),full,seed,seed_forced,targets,boundary});
    }Ok(out)
}
fn response_body(raw:&[u8])->Result<&[u8]> {
    let pos=raw.windows(4).position(|x|x==b"\r\n\r\n").ok_or("missing HTTP boundary")?;let header=std::str::from_utf8(&raw[..pos]).map_err(|e|e.to_string())?;let mut lines=header.split("\r\n");
    let status=lines.next().and_then(|s|s.split_whitespace().nth(1)).ok_or("invalid HTTP status")?;let mut length=None;
    for line in lines {let (k,v)=line.split_once(':').ok_or("invalid HTTP header")?;if k.eq_ignore_ascii_case("content-length"){if length.is_some(){return Err("duplicate Content-Length".into());}length=Some(v.trim().parse::<usize>().map_err(|_|"invalid Content-Length")?);}if k.eq_ignore_ascii_case("transfer-encoding"){return Err("unexpected transfer encoding".into());}}
    let body=&raw[pos+4..];if length!=Some(body.len()){return Err("truncated/ambiguous HTTP body".into());}if status!="200" {return Err(format!("HTTP {status}: {}",String::from_utf8_lossy(body)));}Ok(body)
}
fn persist_response(output:&Path,label:&str,request:&[u8],wire:&[u8],error:Option<(&str,&std::io::Error)>)->Result<Value> {
    // Persist received bytes and failure provenance before parsing or returning errors.
    write_new(&output.join(format!("{label}.response.http")),wire)?;
    let parsed=if error.is_none(){response_body(wire).and_then(|b|serde_json::from_slice(b).map_err(|e|e.to_string()))}else{Err("transport failed; partial response is not scoreable".into())};
    let failure=error.map(|(stage,e)|json!({"stage":stage,"kind":format!("{:?}",e.kind()),"raw_os_error":e.raw_os_error(),"message":e.to_string()}));
    write_new(&output.join(format!("{label}.transport.json")),&serde_json::to_vec_pretty(&json!({"schema":2,"request_sha256":digest(request),"response_sha256":digest(wire),"request_bytes":request.len(),"response_bytes":wire.len(),"transport_complete":error.is_none(),"transport_error":failure,"response_valid":parsed.is_ok(),"validation_error":parsed.as_ref().err(),"scoreable":parsed.is_ok(),"diagnostic_only":true,"qualifies_candidate":false})).map_err(|e|e.to_string())?)?;
    if let Some((stage,e))=error{return Err(format!("{stage}: {e}; retained {} response bytes; not scored",wire.len()));}parsed
}
fn receive<R:Read>(reader:&mut R,request:&[u8],output:&Path,label:&str)->Result<Value> {
    let mut wire=Vec::new();let result=reader.read_to_end(&mut wire);
    persist_response(output,label,request,&wire,result.as_ref().err().map(|e|("read",e)))
}
fn exchange(address:SocketAddr,method:&str,route:&str,body:&[u8],output:&Path,label:&str)->Result<Value> {
    let request=format!("{method} {route} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len());let mut raw=request.into_bytes();raw.extend_from_slice(body);write_new(&output.join(format!("{label}.request.http")),&raw)?;
    let timeout=Duration::from_secs(3600);let mut stream=match TcpStream::connect_timeout(&address,timeout){Ok(s)=>s,Err(e)=>return persist_response(output,label,&raw,&[],Some(("connect",&e)))};
    if let Err(e)=stream.set_read_timeout(Some(timeout)).and_then(|_|stream.set_write_timeout(Some(timeout))){return persist_response(output,label,&raw,&[],Some(("configure",&e)));}
    if let Err(e)=stream.write_all(&raw){return persist_response(output,label,&raw,&[],Some(("write",&e)));}
    receive(&mut stream,&raw,output,label)
}
fn post(address:SocketAddr,payload:&Value,output:&Path,label:&str)->Result<Value> {let body=serde_json::to_vec(payload).map_err(|e|e.to_string())?;exchange(address,"POST","/v1/chat/completions",&body,output,label)}
fn cache_stats(address:SocketAddr,output:&Path,label:&str)->Result<Value> {exchange(address,"GET","/v1/session-cache/stats",&[],output,label)}
fn validate_cache_hit(before:&Value,after:&Value,boundary:usize)->Result<Value> {
    if before["object"]!="krasis_session_cache_stats"||after["object"]!="krasis_session_cache_stats"||before["enabled"]!=true||after["enabled"]!=true||num(&before["resident"],"active_gpu_tokens")?!=boundary{return Err("real seed cache boundary unavailable".into());}
    let active_before=num(&before["hits"],"active_gpu")?;let ram_before=num(&before["hits"],"pageable_ram")?;let active_after=num(&after["hits"],"active_gpu")?;let ram_after=num(&after["hits"],"pageable_ram")?;
    let active=active_after.checked_sub(active_before).ok_or("cache counter decreased")?;let ram=ram_after.checked_sub(ram_before).ok_or("cache counter decreased")?;
    if active.checked_add(ram)!=Some(1)||num(&after["hits"],"total")?.checked_sub(num(&before["hits"],"total")?)!=Some(1){return Err("cached quality request did not record exactly one real cache hit".into());}
    Ok(json!({"actual_seed_consumed_tokens":boundary,"active_gpu_hits":active,"pageable_ram_hits":ram,"before":before,"after":after}))
}
fn payload(label:&str,input:&[u32],budget:usize,cache:bool)->Value {json!({"messages":[{"role":"user","content":"Explicit raw-token diagnostic; template text is not a model input."}],"max_tokens":budget,"temperature":0,"stream":false,"prefix_cache":cache,"chat_template_kwargs":{"enable_thinking":false},"debug_first_token_boundary":true,"debug_matched_teacher_forcing":label,"debug_matched_raw_input_token_ids":input})}
fn validate_response(value:&Value,label:&str,input:&[u32],forced:&[u32],budget:usize)->Result<Vec<u32>> {
    let debug=&value["krasis_debug"];if ids(&debug["input_token_ids"])?!=input || num(&value["usage"],"prompt_tokens")?!=input.len() || debug["input_source"]!="explicit_checksum_bound_raw_tokens" || debug["teacher_forcing"]["label"].as_str()!=Some(label) || ids(&debug["teacher_forcing"]["forced_inputs"])?!=forced {return Err("server did not execute admitted raw-token input".into());}
    let completion=ids(&debug["completion_token_ids"])?;if completion.len()!=num(&value["usage"],"completion_tokens")? || completion.len()!=budget || completion.get(..forced.len())!=Some(forced) {return Err("authoritative previous-token completion history changed or target output incomplete".into());}
    Ok(completion)
}
fn logits(p:&Path,vocab:usize)->Result<Vec<f32>> {let raw=fs::read(p).map_err(|e|format!("{}: {e}",p.display()))?;if vocab==0||raw.len()!=vocab.checked_mul(4).ok_or("vocabulary bytes overflow")?{return Err("incomplete full vocabulary logits".into());}let v:Vec<f32>=raw.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();if v.iter().any(|x|!x.is_finite()){return Err("nonfinite raw logits".into());}Ok(v)}
fn score(v:&[f32],target:u32)->Result<Value> {
    let t=target as usize;if v.is_empty()||t>=v.len()||v.iter().any(|x|!x.is_finite()){return Err("invalid score operands".into());}
    let max=v.iter().copied().fold(f32::NEG_INFINITY,f32::max) as f64;let sum=v.iter().map(|&x|(x as f64-max).exp()).sum::<f64>();let nll=max+sum.ln()-v[t] as f64;
    let rank=1+v.iter().filter(|&&x|x>v[t]).count();let greedy=v.iter().position(|&x|x as f64==max).ok_or("no maximum")?;
    if !nll.is_finite()||nll<0. {return Err("invalid normalized target likelihood".into());}Ok(json!({"target":target,"nll":nll,"probability":(-nll).exp(),"rank":rank,"raw_argmax_lowest_id":greedy,"vocab_size":v.len()}))
}
fn validate_ordinary_choice(values:&[f32],metadata:&Value,key:&str)->Result<usize> {
    let choice=num(metadata,key)?;if values.is_empty()||choice>=values.len()||values.iter().any(|x|!x.is_finite()){return Err("actual ordinary-choice proof missing or invalid".into());}
    let max=values.iter().copied().fold(f32::NEG_INFINITY,f32::max);if values[choice]!=max{return Err("actual ordinary choice disagrees with retained greedy logits".into());}Ok(choice)
}
fn collect(teacher:&Path,label:&str,input:&[u32],start:usize,targets:&[u32],result:&Value)->Result<Value> {
    if input.is_empty()||targets.len()<2||start>=input.len(){return Err("invalid target/history/cache geometry".into());}
    let completion=validate_response(result,label,input,&targets[..targets.len()-1],targets.len())?;
    let dir=teacher.join(label);let meta=read_json(&dir.join("first-logits.json"))?;let vocab=num(&meta,"vocab_size")?;
    if text(&meta,"input_ids_sha256")?!=token_hash(input)||num(&meta,"logical_tokens")?!=input.len()||num(&meta,"sequence_start")?!=start||meta["logits_source"]!="actual_prefill_engine_h_logits"||meta["qualifies_candidate"]!=false{return Err("actual first-logit/history/cache-boundary proof missing".into());}
    let first=read_json(&dir.join("first-selection.json"))?;if num(&first,"sequence_start")?!=start || num(&first,"logical_tokens")?!=input.len() || ids(&first["forced_inputs"])?!=targets[..targets.len()-1] || num(&first,"forced_first")?!=targets[0] as usize {return Err("teacher forcing proof mismatch".into());}
    let mut rows=Vec::new();let mut total=0.;for (i,&target) in targets.iter().enumerate(){let file=if i==0{dir.join("first-logits.f32")}else{dir.join(format!("step-{}-logits.f32",i-1))};let values=logits(&file,vocab)?;let mut row=score(&values,target)?;row["consumed_tokens"]=json!(input.len()+i);row["last_input_position"]=json!(input.len()+i-1);row["target_position"]=json!(input.len()+i);row["logits_sha256"]=json!(digest(&fs::read(&file).map_err(|e|e.to_string())?));row["logits_file"]=json!(file);
        if i>0 {let selected=read_json(&dir.join(format!("step-{}.json",i-1)))?;if num(&selected,"step")?!=i-1||num(&selected,"input_token")?!=targets[i-1] as usize||num(&selected,"logical_tokens")?!=input.len()+i||num(&selected,"actually_fed_next")?!=completion[i] as usize||selected["next_token_forced"].as_bool()!=Some(i+1<targets.len())||num(&selected,"ordinary_greedy_under_forced_prefix")?>=vocab{return Err("actual consumed authoritative history changed".into());}validate_ordinary_choice(&values,&selected,"ordinary_greedy_under_forced_prefix")?;row["ordinary_choice_under_forced_prefix"]=selected["ordinary_greedy_under_forced_prefix"].clone();row["actually_selected"]=selected["actually_fed_next"].clone();row["next_forced"]=selected["next_token_forced"].clone();}
        else {validate_ordinary_choice(&values,&first,"ordinary_greedy_first")?;row["ordinary_choice_under_forced_prefix"]=first["ordinary_greedy_first"].clone();row["actually_selected"]=first["forced_first"].clone();row["next_forced"]=json!(true);}
        total+=row["nll"].as_f64().ok_or("missing score")?;rows.push(row);
    }
    Ok(json!({"label":label,"validated_actual_sequence_start":start,"actual_suffix_tokens":input.len()-start,"targets":targets.len(),"sum_nll":total,"mean_nll":total/targets.len() as f64,"rows":rows,"completion_ids":result["krasis_debug"]["completion_token_ids"],"finish_reason":result["choices"][0]["finish_reason"],"qualifies_candidate":false}))
}
fn run(args:&[String])->Result<()> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref()!=Ok("1"){return Err("use ./dev heldout-decode".into());}if args.len()!=5{return Err("usage: ./dev heldout-decode <port> <inputs.json> <inputs-sha256> <teacher-directory> <fresh-output>".into());}
    let port=args[0].parse::<u16>().map_err(|_|"invalid port")?;if port==0{return Err("port must be nonzero".into());}let address=SocketAddr::from(([127,0,0,1],port));let input_path=Path::new(&args[1]);let teacher=Path::new(&args[3]);let output=PathBuf::from(&args[4]);if !input_path.is_absolute()||!teacher.is_absolute()||!output.is_absolute(){return Err("absolute frozen input/teacher/output paths required".into());}
    let input=checked_json(input_path,&args[2])?;let entries=prepare(&input)?;fs::create_dir(&output).map_err(|e|format!("fresh output required: {e}"))?;let mut documents=Vec::new();let mut full_nll=0.;let mut cached_nll=0.;let mut count=0;
    for e in entries {let full_label=format!("quality{}_full",e.index);let seed_label=format!("quality{}_seed",e.index);let cache_label=format!("quality{}_cached",e.index);let forced=&e.targets[..e.targets.len()-1];
        let full=post(address,&payload(&full_label,&e.full,e.targets.len(),false),&output,&full_label)?;validate_response(&full,&full_label,&e.full,forced,e.targets.len())?;let full_score=collect(teacher,&full_label,&e.full,0,&e.targets,&full)?;
        let seed=post(address,&payload(&seed_label,&e.seed,2,true),&output,&seed_label)?;let completion=validate_response(&seed,&seed_label,&e.seed,&e.seed_forced,2)?;if completion!=e.seed_forced{return Err("seed did not consume the exact declared boundary".into());}
        let before=cache_stats(address,&output,&format!("{cache_label}.before-stats"))?;
        let cached=post(address,&payload(&cache_label,&e.full,e.targets.len(),true),&output,&cache_label)?;validate_response(&cached,&cache_label,&e.full,forced,e.targets.len())?;let cached_score=collect(teacher,&cache_label,&e.full,e.boundary,&e.targets,&cached)?;let after=cache_stats(address,&output,&format!("{cache_label}.after-stats"))?;let reuse=validate_cache_hit(&before,&after,e.boundary)?;
        full_nll+=full_score["sum_nll"].as_f64().ok_or("full score missing")?;cached_nll+=cached_score["sum_nll"].as_f64().ok_or("cached score missing")?;count+=e.targets.len();let row=json!({"index":e.index,"document":e.document,"full":full_score,"cached":cached_score,"actual_cache_reuse":reuse});write_new(&output.join(format!("document-{}.json",e.index)),&serde_json::to_vec_pretty(&row).map_err(|e|e.to_string())?)?;documents.push(row);
    }
    let result=json!({"schema":1,"diagnostic_only":true,"qualifies_candidate":false,"inputs_sha256":args[2],"corpus_sha256":input["corpus_sha256"],"targets_per_contract":count,"full_mean_nll":full_nll/count as f64,"cached_mean_nll":cached_nll/count as f64,"documents":documents,"independent_BF16_Witness_comparison_still_required":true,"ordinary_consistency_not_established_by_forcing":true});write_new(&output.join("RESULT.json"),&serde_json::to_vec_pretty(&result).map_err(|e|e.to_string())?)?;println!("Targets per contract: {count}; full NLL {:.9}; cached NLL {:.9}; diagnostic only",full_nll/count as f64,cached_nll/count as f64);Ok(())
}
fn main(){if let Err(error)=run(&std::env::args().skip(1).collect::<Vec<_>>()){eprintln!("ERROR: {error}");std::process::exit(1);}}
#[cfg(test)] mod tests {
    use super::*;
    struct Temp(PathBuf);
    impl Temp {fn new()->Self {static NEXT:std::sync::atomic::AtomicUsize=std::sync::atomic::AtomicUsize::new(0);let p=std::env::temp_dir().join(format!("heldout-decode-{}-{}",std::process::id(),NEXT.fetch_add(1,std::sync::atomic::Ordering::Relaxed)));fs::create_dir(&p).unwrap();Self(p)}}
    impl Drop for Temp {fn drop(&mut self){fs::remove_dir_all(&self.0).unwrap();}}
    #[test] fn interrupted_read_keeps_bytes_and_cannot_score_even_complete_json() {
        struct Broken {bytes:std::io::Cursor<Vec<u8>>}
        impl Read for Broken {fn read(&mut self,b:&mut [u8])->std::io::Result<usize>{let n=self.bytes.read(b)?;if n==0{Err(std::io::Error::new(std::io::ErrorKind::ConnectionReset,"peer reset"))}else{Ok(n)}}}
        for bytes in [b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\n{\"pa".to_vec(),b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}".to_vec()] {
            let temp=Temp::new();let mut reader=Broken{bytes:std::io::Cursor::new(bytes.clone())};assert!(receive(&mut reader,b"request",&temp.0,"failed").is_err());assert_eq!(fs::read(temp.0.join("failed.response.http")).unwrap(),bytes);
            let meta=read_json(&temp.0.join("failed.transport.json")).unwrap();assert_eq!(meta["transport_complete"],false);assert_eq!(meta["scoreable"],false);assert_eq!(meta["response_valid"],false);assert_eq!(meta["transport_error"]["stage"],"read");assert_eq!(meta["transport_error"]["kind"],"ConnectionReset");assert_eq!(meta["response_sha256"],digest(&bytes));assert_eq!(meta["response_bytes"],bytes.len());assert!(!temp.0.join("RESULT.json").exists());
        }
    }
    #[test] fn clean_eof_truncation_is_retained_and_rejected() {let temp=Temp::new();let bytes=b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\n{}";assert!(receive(&mut &bytes[..],b"request",&temp.0,"short").is_err());let meta=read_json(&temp.0.join("short.transport.json")).unwrap();assert_eq!(meta["transport_complete"],true);assert_eq!(meta["scoreable"],false);assert!(meta["transport_error"].is_null());assert_eq!(fs::read(temp.0.join("short.response.http")).unwrap(),bytes);}
    #[test] fn complete_targets_and_selection_proofs_are_required() {
        let temp=Temp::new();let dir=temp.0.join("quality0_full");fs::create_dir(&dir).unwrap();let response=json!({"usage":{"prompt_tokens":2,"completion_tokens":2},"krasis_debug":{"input_source":"explicit_checksum_bound_raw_tokens","input_token_ids":[3,7],"completion_token_ids":[1,0],"teacher_forcing":{"label":"quality0_full","forced_inputs":[1]}}});
        let save=|name:&str,v:Value|fs::write(dir.join(name),serde_json::to_vec(&v).unwrap()).unwrap();
        save("first-logits.json",json!({"vocab_size":2,"input_ids_sha256":token_hash(&[3,7]),"logical_tokens":2,"sequence_start":0,"logits_source":"actual_prefill_engine_h_logits","qualifies_candidate":false}));save("first-selection.json",json!({"sequence_start":0,"logical_tokens":2,"forced_inputs":[1],"forced_first":1,"ordinary_greedy_first":1}));
        fs::write(dir.join("first-logits.f32"),[0f32,1f32].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>()).unwrap();
        assert!(collect(&temp.0,"quality0_full",&[3,7],0,&[1,0],&response).is_err());
        fs::write(dir.join("step-0-logits.f32"),[1f32,0f32].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>()).unwrap();save("step-0.json",json!({"step":0,"input_token":1,"logical_tokens":3,"actually_fed_next":0,"next_token_forced":false,"ordinary_greedy_under_forced_prefix":0}));
        assert_eq!(collect(&temp.0,"quality0_full",&[3,7],0,&[1,0],&response).unwrap()["targets"],2);
        let mut short=response.clone();short["usage"]["completion_tokens"]=json!(1);short["krasis_debug"]["completion_token_ids"]=json!([1]);assert!(collect(&temp.0,"quality0_full",&[3,7],0,&[1,0],&short).is_err());
        save("step-0.json",json!({"step":0,"input_token":0,"logical_tokens":3,"actually_fed_next":0,"next_token_forced":false,"ordinary_greedy_under_forced_prefix":0}));assert!(collect(&temp.0,"quality0_full",&[3,7],0,&[1,0],&response).is_err());
        fs::write(dir.join("step-0-logits.f32"),[0u8;4]).unwrap();assert!(collect(&temp.0,"quality0_full",&[3,7],0,&[1,0],&response).is_err());assert!(collect(&temp.0,"quality0_full",&[3,7],1,&[1,0],&response).is_err());
    }
    #[test] fn actual_greedy_choice_cannot_be_missing_out_of_range_or_inconsistent() {
        let v=[0f32,1.,1.];assert_eq!(validate_ordinary_choice(&v,&json!({"chosen":2}),"chosen").unwrap(),2);
        for invalid in [json!({}),json!({"chosen":0}),json!({"chosen":3}),json!({"chosen":1.5}),json!({"chosen":-1})]{assert!(validate_ordinary_choice(&v,&invalid,"chosen").is_err());}
        assert!(validate_ordinary_choice(&[f32::NAN],&json!({"chosen":0}),"chosen").is_err());
    }
    #[test] fn cache_evidence_requires_actual_hit_and_boundary() {
        let before=json!({"object":"krasis_session_cache_stats","enabled":true,"resident":{"active_gpu_tokens":7},"hits":{"active_gpu":2,"pageable_ram":1,"total":3}});let mut after=before.clone();after["hits"]["active_gpu"]=json!(3);after["hits"]["total"]=json!(4);
        assert!(validate_cache_hit(&before,&after,7).is_ok());assert!(validate_cache_hit(&before,&after,6).is_err());assert!(validate_cache_hit(&before,&before,7).is_err());after["hits"]["active_gpu"]=json!(2);after["hits"]["pageable_ram"]=json!(2);assert!(validate_cache_hit(&before,&after,7).is_ok());after["enabled"]=json!(false);assert!(validate_cache_hit(&before,&after,7).is_err());
    }
    #[test] fn normalized_scoring_rejects_corruption_and_retains_regression(){let a=score(&[0.,1.,2.],2).unwrap();let b=score(&[3.,4.,5.],2).unwrap();assert!((a["nll"].as_f64().unwrap()-b["nll"].as_f64().unwrap()).abs()<1e-12);assert_eq!(a["rank"],1);assert!((a["probability"].as_f64().unwrap()-0.6652409557748218).abs()<1e-12);assert!(score(&[3.,1.,0.],2).unwrap()["nll"].as_f64().unwrap()>a["nll"].as_f64().unwrap());assert!(score(&[f32::NAN],0).is_err());assert!(score(&[0.],1).is_err());assert!(score(&[],0).is_err());}
    #[test] fn transport_cannot_score_truncated_duplicate_or_failed_responses(){assert_eq!(response_body(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}").unwrap(),b"{}");for x in [b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\n{}".as_slice(),b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}",b"HTTP/1.1 500 Error\r\nContent-Length: 2\r\n\r\n{}",b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n{}"]{assert!(response_body(x).is_err());}}
    #[test] fn raw_response_requires_exact_inputs_and_forced_previous_tokens(){let v=json!({"usage":{"prompt_tokens":2,"completion_tokens":2},"krasis_debug":{"input_source":"explicit_checksum_bound_raw_tokens","input_token_ids":[3,7],"completion_token_ids":[4,5],"teacher_forcing":{"label":"quality0_full","forced_inputs":[4]}}});assert!(validate_response(&v,"quality0_full",&[3,7],&[4],2).is_ok());assert!(validate_response(&v,"quality0_full",&[7,3],&[4],2).is_err());assert!(validate_response(&v,"quality0_full",&[3,7],&[5],2).is_err());assert!(validate_response(&v,"quality0_full",&[3,7],&[4],1).is_err());assert!(ids(&json!([1.5])).is_err());assert!(ids(&json!([-1])).is_err());}
}
