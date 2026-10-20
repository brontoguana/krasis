#[path = "../reference_rounding.rs"]
mod reference_rounding;
use cudarc::driver::{CudaDevice, DevicePtr, LaunchAsync, LaunchConfig};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::{Path, PathBuf}};

use krasis::weights::marlin;
#[path = "matched_arithmetic/hqq_weight.rs"]
mod hqq_weight;
#[path = "matched_arithmetic/hqq_accuracy.rs"]
mod hqq_accuracy;

fn read(path: &Path, bytes: usize) -> Result<Vec<u8>, String> {
    let data = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if data.len() != bytes { return Err(format!("{}: {} bytes, expected {bytes}", path.display(), data.len())) }
    Ok(data)
}
fn u16s(data: &[u8]) -> Vec<u16> { data.chunks_exact(2).map(|x| u16::from_le_bytes([x[0],x[1]])).collect() }
fn f32s(data: &[u8]) -> Vec<f32> { data.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect() }
fn bfvalues(data: &[u16]) -> Vec<f64> { data.iter().map(|&x| bf16::from_bits(x).to_f64()).collect() }
fn save16(path: &Path, data: &[u16]) -> Result<(), String> { fs::write(path, data.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>()).map_err(|e|e.to_string()) }
fn save64(path: &Path, data: &[f64]) -> Result<(), String> { fs::write(path, data.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>()).map_err(|e|e.to_string()) }
fn number(v: &Value, name: &str) -> Result<usize, String> { v[name].as_u64().and_then(|n| usize::try_from(n).ok()).filter(|&n|n>0).ok_or(format!("invalid {name}")) }
fn metrics(actual: &[f64], reference: &[f64]) -> Result<Value, String> {
    if actual.len()!=reference.len() || reference.is_empty() { return Err("metric shape mismatch".into()) }
    let mut diff2=0.; let mut ref2=0.; let mut max=0.; let mut max_at=0;
    let mut mismatches=Vec::new();
    for (i,(&a,&b)) in actual.iter().zip(reference).enumerate() {
        if !a.is_finite() || !b.is_finite() { return Err(format!("nonfinite coordinate {i}")) }
        diff2+=(a-b)*(a-b); ref2+=b*b;
        if (a-b).abs()>max { max=(a-b).abs(); max_at=i; }
        if reference_rounding::bf16_from_f64_bits(a)!=reference_rounding::bf16_from_f64_bits(b) {
            mismatches.push(json!({"coordinate":i,"actual":a,"reference":b}));
        }
    }
    if ref2==0. { return Err("zero reference norm; relative L2 undefined".into()) }
    Ok(json!({"count":actual.len(),"max_abs":max,"max_abs_coordinate":max_at,
        "difference_l2":diff2.sqrt(),"reference_l2":ref2.sqrt(),"relative_l2":(diff2/ref2).sqrt(),
        "bf16_mismatch_count":mismatches.len(),"bf16_mismatches":mismatches}))
}

fn gemm(dev: &std::sync::Arc<CudaDevice>, input: &cudarc::driver::CudaSlice<u16>, weights: &cudarc::driver::CudaSlice<u16>,
    rows: usize, cols: usize, height: usize, contract: Option<&Value>) -> Result<Vec<u16>,String> {
    use cudarc::cublas::{result as r, sys as c};
    let out=dev.alloc_zeros::<u16>(rows*height).map_err(|e|e.to_string())?;
    let handle=r::create_handle().map_err(|e|e.to_string())?;
    let result=(|| unsafe {
        r::set_stream(handle,*dev.cu_stream() as c::cudaStream_t).map_err(|e|e.to_string())?;
        // BEGIN actual captured cuBLAS handle contract
        if let Some(contract) = contract {
            let check = |name:&str,status:c::cublasStatus_t| -> Result<(),String> {
                if status==c::cublasStatus_t::CUBLAS_STATUS_SUCCESS {Ok(())}else{Err(format!("actual Q cuBLAS replay {name}: {status:?}"))}
            };
            let mut version=0i32;
            check("version",c::lib().cublasGetVersion_v2(handle,&mut version))?;
            if contract["version"]!=version {return Err("cuBLAS version does not match capture".into())}
            let math=match contract["math_mode"].as_u64(){
                Some(0)=>c::cublasMath_t::CUBLAS_DEFAULT_MATH,
                Some(1)=>c::cublasMath_t::CUBLAS_TENSOR_OP_MATH,
                Some(2)=>c::cublasMath_t::CUBLAS_PEDANTIC_MATH,
                Some(3)=>c::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
                Some(16)=>c::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
                _=>return Err("unsupported captured cuBLAS math flags; no substitute contract".into()),
            };
            let atomics=match contract["atomics_mode"].as_u64(){
                Some(0)=>c::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED,
                Some(1)=>c::cublasAtomicsMode_t::CUBLAS_ATOMICS_ALLOWED,
                _=>return Err("invalid captured cuBLAS atomics mode".into()),
            };
            let sm=contract["sm_count_target"].as_i64().and_then(|n|i32::try_from(n).ok()).filter(|&n|n>=0).ok_or("invalid captured cuBLAS SM target")?;
            if std::env::var("CUBLAS_WORKSPACE_CONFIG").ok()!=contract["workspace_environment"].as_str().map(str::to_owned){return Err("workspace environment differs from captured contract".into())}
            check("math",c::lib().cublasSetMathMode(handle,math))?;
            check("atomics",c::lib().cublasSetAtomicsMode(handle,atomics))?;
            check("pointer",c::lib().cublasSetPointerMode_v2(handle,c::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST))?;
            check("SM target",c::lib().cublasSetSmCountTarget(handle,sm))?;
        }
        // END actual captured cuBLAS handle contract
        let alpha=1f32; let beta=0f32;
        r::gemm_ex(handle,c::cublasOperation_t::CUBLAS_OP_T,c::cublasOperation_t::CUBLAS_OP_N,
            rows as i32,height as i32,cols as i32,(&alpha as *const f32).cast(),
            (*weights.device_ptr() as *const std::ffi::c_void),c::cudaDataType::CUDA_R_16BF,cols as i32,
            (*input.device_ptr() as *const std::ffi::c_void),c::cudaDataType::CUDA_R_16BF,cols as i32,
            (&beta as *const f32).cast(),(*out.device_ptr() as *mut std::ffi::c_void),
            c::cudaDataType::CUDA_R_16BF,rows as i32,c::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            c::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT).map_err(|e|e.to_string())?;
        dev.dtoh_sync_copy(&out).map_err(|e|e.to_string())
    })();
    unsafe { r::destroy_handle(handle).map_err(|e|e.to_string())?; }
    result
}

fn q_probe(input_dir: &Path, out: &Path, gpu: usize, warps: usize) -> Result<Value,String> {
    let desc:Value=serde_json::from_slice(&fs::read(input_dir.join("descriptor.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let rows=number(&desc,"rows")?; let cols=number(&desc,"cols")?;
    let group=number(&desc,"group_size")?; let ps=number(&desc,"packed_row_stride_bytes")?;
    let ss=number(&desc,"scales_row_stride_bytes")?; let zs=number(&desc,"zeros_row_stride_bytes")?;
    let height=number(&desc,"height")?; let selected=desc["row"].as_u64().ok_or("missing row")? as usize;
    if desc["nbits"]!=6 || desc["materialize_bf16"]!=true || selected>=height || cols%4!=0 || group%4!=0
        || ![1,2,4,8,16,32].contains(&warps) || [rows,cols,group,ps,ss,zs,height].iter().any(|&n|n>i32::MAX as usize) {
        return Err("unsupported or invalid matched HQQ6 descriptor/launch".into());
    }
    let packed=read(&input_dir.join("packed.bin"),rows*ps)?;
    let scales=f32s(&read(&input_dir.join("scales.f32"),rows*ss)?);
    let zeros=f32s(&read(&input_dir.join("zeros.f32"),rows*zs)?);
    let x=u16s(&read(&input_dir.join("input.bf16"),cols*2)?);
    let captured=u16s(&read(&input_dir.join("production-prefill.bf16"),rows*2)?);
    let materialized=u16s(&read(&input_dir.join("materialized.bf16"),rows*cols*2)?);
    let mut w=vec![0f32;rows*cols]; let mut wb=vec![0u16;rows*cols];
    let mut ref32=vec![0f64;rows]; let mut refbf=vec![0f64;rows];
    let mut scalar=vec![0u16;rows]; let mut vec4=vec![0u16;rows]; let mut vec4bf=vec![0u16;rows];
    for row in 0..rows {
        for col in 0..cols {
            let index=row*ps+(col/4)*3;
            let bits=packed[index] as u32 | (packed[index+1] as u32)<<8 | (packed[index+2] as u32)<<16;
            let q=((bits >> ((col%4)*6)) & 63) as f32;
            let val=(q-zeros[row*(zs/4)+col/group])*scales[row*(ss/4)+col/group];
            if !val.is_finite() { return Err("nonfinite dequantized weight".into()) }
            w[row*cols+col]=val; wb[row*cols+col]=bf16::from_f32(val).to_bits();
            let xf=bf16::from_bits(x[col]).to_f64();
            ref32[row]+=val as f64*xf; refbf[row]+=bf16::from_bits(wb[row*cols+col]).to_f64()*xf;
        }
        for (quartet, rounded, dst) in [(false,false,&mut scalar),(true,false,&mut vec4),(true,true,&mut vec4bf)] {
            let mut lanes=[0f32;32];
            for lane in 0..32 {
                if quartet {
                    for quartet in (lane..cols/4).step_by(32) { for offset in 0..4 {
                        let col=quartet*4+offset;
                        let val=if rounded {bf16::from_bits(wb[row*cols+col]).to_f32()} else {w[row*cols+col]};
                        lanes[lane]=val.mul_add(bf16::from_bits(x[col]).to_f32(),lanes[lane]);
                    }}
                } else {
                    for col in (lane..cols).step_by(32) { lanes[lane]=w[row*cols+col].mul_add(bf16::from_bits(x[col]).to_f32(),lanes[lane]); }
                }
            }
            for offset in [16,8,4,2,1] { let old=lanes; for lane in 0..32-offset { lanes[lane]=old[lane]+old[lane+offset]; } }
            dst[row]=bf16::from_f32(lanes[0]).to_bits();
        }
    }
    save64(&out.join("reference-f32-dequant.f64"),&ref32)?;
    save64(&out.join("reference-bf16-rounded.f64"),&refbf)?;
    save16(&out.join("reference-scalar-fma.bf16"),&scalar)?;
    save16(&out.join("reference-vec4-fma.bf16"),&vec4)?;
    save16(&out.join("reference-vec4-bf16-weight-fma.bf16"),&vec4bf)?;
    if wb!=materialized { return Err(format!("loaded materialization disagrees with independent dequant at {} coordinates",wb.iter().zip(&materialized).filter(|(a,b)|a!=b).count())) }
    let dev=CudaDevice::new(gpu).map_err(|e|e.to_string())?;
    dev.load_ptx(cudarc::nvrtc::Ptx::from_src(include_str!(concat!(env!("OUT_DIR"),"/decode_kernels.ptx"))),
        "matched_decode",&["hqq6_decode_gemv_bf16","hqq6_decode_gemv_bf16_vec4"]).map_err(|e|e.to_string())?;
    let dp=dev.htod_copy(packed).map_err(|e|e.to_string())?;
    let ds=dev.htod_copy(scales).map_err(|e|e.to_string())?;
    let dz=dev.htod_copy(zeros).map_err(|e|e.to_string())?;
    let dx=dev.htod_copy(x.clone()).map_err(|e|e.to_string())?;
    let dw=dev.htod_copy(materialized).map_err(|e|e.to_string())?;
    let mut results=Vec::new();
    for (name,threads) in [("hqq6_decode_gemv_bf16",256usize),("hqq6_decode_gemv_bf16_vec4",warps*32)] {
        let dy=dev.alloc_zeros::<u16>(rows).map_err(|e|e.to_string())?;
        let func=dev.get_func("matched_decode",name).ok_or("decode kernel missing")?;
        unsafe { func.launch(LaunchConfig {grid_dim:(rows.div_ceil(threads/32) as u32,1,1),block_dim:(threads as u32,1,1),shared_mem_bytes:(cols*2) as u32},
            (*dp.device_ptr(),*ds.device_ptr(),*dz.device_ptr(),*dx.device_ptr(),*dy.device_ptr(),rows as i32,cols as i32,group as i32,ps as i32,ss as i32,zs as i32)).map_err(|e|e.to_string())?; }
        let data=dev.dtoh_sync_copy(&dy).map_err(|e|e.to_string())?;
        save16(&out.join(format!("{name}.bf16")),&data)?;
        let reference=if name.ends_with("vec4") {&vec4} else {&scalar};
        results.push(json!({"path":name,"warps":threads/32,"contract":"F32 HQQ dequant; F32 FMA lane sums; warp reduction; BF16 RNE output",
            "vs_f32_f64":metrics(&bfvalues(&data),&ref32)?,"vs_bf16_f64":metrics(&bfvalues(&data),&refbf)?,
            "vs_matching_order_cpu":metrics(&bfvalues(&data),&bfvalues(reference))?,"vs_original_prefill":metrics(&bfvalues(&data),&bfvalues(&captured))?}));
    }
    for m in [1,height] {
        let mut matrix=vec![0u16;m*cols]; let row=if m==1 {0} else {selected}; matrix[row*cols..(row+1)*cols].copy_from_slice(&x);
        let input=dev.htod_copy(matrix).map_err(|e|e.to_string())?;
        let full=gemm(&dev,&input,&dw,rows,cols,m,None)?; let y=&full[row*rows..(row+1)*rows];
        save16(&out.join(format!("materialized-cublas-height{m}.bf16")),y)?;
        results.push(json!({"path":"materialized_cublas","height":m,"other_rows":"zeros for isolated selected-row replay",
            "vs_f32_f64":metrics(&bfvalues(y),&ref32)?,"vs_bf16_f64":metrics(&bfvalues(y),&refbf)?,
            "vs_original_prefill":metrics(&bfvalues(y),&bfvalues(&captured))?}));
    }
    Ok(json!({"descriptor":desc,"gpu":gpu,"materialization_matches_independent":true,
        "weight_rounding_only":metrics(&refbf,&ref32)?,"f32_accumulation_order_only":metrics(&bfvalues(&vec4),&bfvalues(&scalar))?,
        "fixed_vec4_order_weight_rounding":metrics(&bfvalues(&vec4bf),&bfvalues(&vec4))?,"paths":results}))
}

fn shared_reference(input_dir: &Path, out: &Path) -> Result<Value,String> {
    let mut results=Vec::new();
    for (name,input_name,output_name) in [("w1","input.bf16","w1.bf16"),("w2","activation.bf16","w2.bf16")] {
        let d:Value=serde_json::from_slice(&fs::read(input_dir.join(format!("{name}-descriptor.json"))).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        let n=number(&d,"n")?; let k=number(&d,"k")?; let gs=number(&d,"group_size")?; let groups=number(&d,"groups")?;
        if d["bits"]!=8 || k%16!=0 || n%16!=0 || groups!=k.div_ceil(gs) { return Err("invalid shared INT8 geometry".into()) }
        let packed=read(&input_dir.join(format!("{name}-packed.bin")),n*k)?;
        let scales=u16s(&read(&input_dir.join(format!("{name}-scales.bf16")),groups*n*2)?);
        let weight=marlin::MarlinRepacked {packed:packed.chunks_exact(4).map(|x|u32::from_le_bytes(x.try_into().unwrap())).collect(),scales,k,n,group_size:gs};
        let w=marlin::dequantize_marlin_int8(&weight);
        let x=bfvalues(&u16s(&read(&input_dir.join(input_name),k*2)?));
        let actual=bfvalues(&u16s(&read(&input_dir.join(output_name),n*2)?));
        let mut f32_ref=vec![0.;n]; let mut bf_ref=vec![0.;n];
        let mut forward=vec![0u16;n]; let mut reverse=vec![0u16;n]; let mut abs_sum=vec![0.;n];
        for r in 0..n {
            let mut f=0f32;let mut rev=0f32;
            for c in 0..k {
                let wb=bf16::from_f32(w[r*k+c]).to_f32();
                f32_ref[r]+=w[r*k+c] as f64*x[c]; bf_ref[r]+=wb as f64*x[c]; abs_sum[r]+=(wb as f64*x[c]).abs();
                f=wb.mul_add(x[c] as f32,f);
                let rc=k-1-c;rev=bf16::from_f32(w[r*k+rc]).to_f32().mul_add(x[rc] as f32,rev);
            }
            forward[r]=bf16::from_f32(f).to_bits();reverse[r]=bf16::from_f32(rev).to_bits();
        }
        save16(&out.join(format!("{name}-forward-f32-fma.bf16")),&forward)?;
        save16(&out.join(format!("{name}-reverse-f32-fma.bf16")),&reverse)?;
        save64(&out.join(format!("{name}-absolute-product-sums.f64")),&abs_sum)?;
        save64(&out.join(format!("{name}-f32-dequant-reference.f64")),&f32_ref)?;
        save64(&out.join(format!("{name}-bf16-rounded-reference.f64")),&bf_ref)?;
        results.push(json!({"operation":name,"descriptor":d,"vs_f32_dequant":metrics(&actual,&f32_ref)?,"vs_bf16_weights":metrics(&actual,&bf_ref)?,"bf16_output_rounding_only":metrics(&bfvalues(&bf_ref.iter().map(|&v|reference_rounding::bf16_from_f64_bits(v)).collect::<Vec<_>>()),&bf_ref)?,"vs_forward_f32_fma":metrics(&actual,&bfvalues(&forward))?,"vs_reverse_f32_fma":metrics(&actual,&bfvalues(&reverse))?,"independent_order_variation":metrics(&bfvalues(&forward),&bfvalues(&reverse))?}));
    }
    let desc:Value=serde_json::from_slice(&fs::read(input_dir.join("w1-descriptor.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    if desc["activation"]!=0 { return Err("unsupported matched activation contract".into()) }
    let raw_clipped=desc["deepseek_v4_activation"].as_bool().ok_or("missing raw-clipping activation flag")?;
    let limit=desc["swiglu_limit"].as_f64().filter(|v|v.is_finite()).ok_or("invalid loaded activation limit")?;
    if (raw_clipped && limit<=0.) || (!raw_clipped && limit>0.) {
        return Err("reference supports vanilla or explicitly raw-projection-clipped SwiGLU; other limited equations require a separate reference".into());
    }
    let n=number(&desc,"n")?/2;
    let w1=bfvalues(&u16s(&read(&input_dir.join("w1.bf16"),n*4)?));
    let act=bfvalues(&u16s(&read(&input_dir.join("activation.bf16"),n*2)?));
    let expected:Vec<_>=(0..n).map(|i| {
        let gate=if raw_clipped {w1[i].min(limit)} else {w1[i]};
        let up=if raw_clipped {w1[n+i].clamp(-limit,limit)} else {w1[n+i]};
        gate/(1.+(-gate).exp())*up
    }).collect();
    save64(&out.join("activation-reference.f64"),&expected)?;
    results.push(json!({"operation":"activation","contract":if raw_clipped {"SiLU(min(raw_gate, limit)) * clamp(raw_up, -limit, limit); BF16 output"} else {"SiLU(gate) * up; BF16 output"},"limit":limit,"vs_reference":metrics(&act,&expected)?}));
    Ok(json!({"paths":results,"reference":"independent CPU inverse Marlin packing and F64 dot products; F32/BF16-dequant weights compared separately"}))
}

// BEGIN actual full Q production replay
fn full_q_dimensions(desc:&Value,contract:&Value,warps:usize)->Result<(usize,usize,usize,usize),String> {
    let rows=number(&desc,"rows")?;let cols=number(&desc,"cols")?;let height=number(&desc,"height")?;
    let selected=desc["row"].as_u64().and_then(|x|usize::try_from(x).ok()).ok_or("missing selected row")?;
    if desc["nbits"]!=6 || desc["materialize_bf16"]!=true || ![1,2,4,8,16,32].contains(&warps) || selected>=height||[rows,cols,height].iter().any(|&n|n>i32::MAX as usize)
        ||contract["m"]!=rows||contract["n"]!=height||contract["k"]!=cols
        ||contract["transpose_a"]!="T"||contract["transpose_b"]!="N"
        ||contract["lda"]!=cols||contract["ldb"]!=cols||contract["ldc"]!=rows
        ||contract["alpha"]!=1.0||contract["beta"]!=0.0||contract["compute"]!="CUBLAS_COMPUTE_32F"
        ||contract["algorithm"]!="CUBLAS_GEMM_DEFAULT"||contract["pointer_mode"]!=0
        ||contract["stream_matches_production"]!=true ||contract["stream_is_null"].as_bool().is_none()
        ||contract["input_dtype"]!="bf16" ||contract["weights_dtype"]!="bf16" ||contract["output_dtype"]!="bf16" {
        return Err("unsupported or malformed actual full Q GEMM contract".into());
    }
    Ok((rows,cols,height,selected))
}

fn q_full_probe(input_dir: &Path, out: &Path, gpu: usize, warps: usize) -> Result<Value,String> {
    let desc:Value=serde_json::from_slice(&fs::read(input_dir.join("descriptor.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let contract:Value=serde_json::from_slice(&fs::read(input_dir.join("cublas-contract.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let (rows,cols,height,selected)=full_q_dimensions(&desc,&contract,warps)?;
    let input_count=height.checked_mul(cols).ok_or("full Q input overflow")?;
    let output_count=height.checked_mul(rows).ok_or("full Q output overflow")?;
    let weight_count=rows.checked_mul(cols).ok_or("full Q weights overflow")?;
    let matrix=u16s(&read(&input_dir.join("full-input.bf16"),input_count.checked_mul(2).ok_or("input bytes overflow")?)?);
    let captured=u16s(&read(&input_dir.join("full-production-prefill.bf16"),output_count.checked_mul(2).ok_or("output bytes overflow")?)?);
    let weights=u16s(&read(&input_dir.join("materialized.bf16"),weight_count.checked_mul(2).ok_or("weight bytes overflow")?)?);
    let selected_input=u16s(&read(&input_dir.join("input.bf16"),cols.checked_mul(2).ok_or("selected input bytes overflow")?)?);
    let selected_output=u16s(&read(&input_dir.join("production-prefill.bf16"),rows.checked_mul(2).ok_or("selected output bytes overflow")?)?);
    if matrix[selected*cols..(selected+1)*cols]!=selected_input||captured[selected*rows..(selected+1)*rows]!=selected_output {
        return Err("selected row does not match actual full operands".into());
    }
    let device=match contract["stream_is_null"].as_bool(){Some(true)=>CudaDevice::new(gpu),Some(false)=>CudaDevice::new_with_stream(gpu),None=>return Err("missing actual stream kind".into())}.map_err(|e|e.to_string())?;
    let dx=device.htod_copy(matrix).map_err(|e|e.to_string())?;
    let dw=device.htod_copy(weights).map_err(|e|e.to_string())?;
    let result=gemm(&device,&dx,&dw,rows,cols,height,Some(&contract))?;
    save16(&out.join("full-replay-output.bf16"),&result)?;
    let mismatches=result.iter().zip(&captured).enumerate().filter_map(|(i,(x,y))|(x!=y).then_some(i)).collect::<Vec<_>>();
    fs::write(out.join("full-control.json"),serde_json::to_vec_pretty(&json!({"actual_input_rows":height,"exact":mismatches.is_empty(),"mismatch_count":mismatches.len(),"first_mismatches":mismatches.iter().take(64).collect::<Vec<_>>(),"cublas_contract":contract,"selected_row":selected})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    if !mismatches.is_empty(){return Err("full Q production replay control failed; arithmetic attribution blocked".into())}
    // The complete control precedes the separately labelled isolated references.
    let reference=q_probe(input_dir,out,gpu,warps)?;
    Ok(json!({"actual_full_operand_control_exact":true,"independent_selected_row_references":reference,"qualifies_candidate":false}))
}
// END actual full Q production replay

// BEGIN selected routed expert independent reference
type SelectedDotReferences=(Vec<f64>,Vec<f64>,Vec<f64>,Vec<u16>,Vec<u16>);
fn selected_dot_references(w:&[f32],x:&[f64],n:usize,k:usize)->Result<SelectedDotReferences,String>{
    if n==0||k==0||n.checked_mul(k)!=Some(w.len())||x.len()!=k||w.iter().any(|x|!x.is_finite())||x.iter().any(|x|!x.is_finite()){return Err("invalid/nonfinite independent dot operands".into())}
    let mut full=vec![0f64;n];let mut rounded=vec![0f64;n];let mut abs_sums=vec![0f64;n];let mut forward=vec![0u16;n];let mut reverse=vec![0u16;n];
    for r in 0..n {let mut f=0f32;let mut rev=0f32;for c in 0..k {
        let wf=w[r*k+c];let wb=bf16::from_f32(wf).to_f32();full[r]+=wf as f64*x[c];rounded[r]+=wb as f64*x[c];abs_sums[r]+=(wb as f64*x[c]).abs();f=wb.mul_add(x[c] as f32,f);
        let rc=k-1-c;rev=bf16::from_f32(w[r*k+rc]).to_f32().mul_add(x[rc] as f32,rev);
    }forward[r]=bf16::from_f32(f).to_bits();reverse[r]=bf16::from_f32(rev).to_bits();}
    Ok((full,rounded,abs_sums,forward,reverse))
}
#[cfg(test)]
mod selected_dot_tests {
    use super::*;
    #[test]fn independent_reference_separates_weight_rounding_and_cancellation_order(){
        let (full,rounded,_,forward,_)=selected_dot_references(&[1.00390625],&[2.],1,1).unwrap();
        assert_eq!(full,vec![2.0078125]);assert_eq!(rounded,vec![2.]);assert_eq!(bf16::from_bits(forward[0]).to_f32(),2.);
        let (full,rounded,_,forward,reverse)=selected_dot_references(&[1e8,1.,-1e8,1.],&[1.;4],1,4).unwrap();
        assert_eq!(full,vec![2.]);assert_eq!(rounded,vec![2.]);assert_eq!(bf16::from_bits(forward[0]).to_f32(),1.);assert_eq!(bf16::from_bits(reverse[0]).to_f32(),0.);
    }
    #[test]fn independent_reference_rejects_invalid_layout_and_nonfinite_operands(){
        assert!(selected_dot_references(&[1.],&[1.],2,1).is_err());assert!(selected_dot_references(&[f32::NAN],&[1.],1,1).is_err());assert!(selected_dot_references(&[1.],&[f64::INFINITY],1,1).is_err());
    }
}
// Independent selected-expert arithmetic reference. Not a serving implementation.
fn routed_reference(input_dir: &Path, out: &Path) -> Result<Value, String> {
    let routing:Value=serde_json::from_slice(&fs::read(input_dir.join("routing.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let topk=number(&routing,"topk")?;
    let ids_raw=read(&input_dir.join("route-ids.i32"),topk.checked_mul(4).ok_or("route bytes overflow")?)?;
    let ids:Vec<_>=ids_raw.chunks_exact(4).map(|x|i32::from_le_bytes(x.try_into().unwrap())).collect();
    let weights=f32s(&read(&input_dir.join("route-weights.f32"),topk*4)?);
    let experts=number(&routing,"experts")?;
    let mut unique=std::collections::HashSet::new();
    if ids.iter().any(|&i|i<0 || i as usize>=experts || !unique.insert(i)) || weights.iter().any(|w|!w.is_finite()||*w<0.) {return Err("invalid selected routing".into())}
    if routing["selected_experts"]!=json!(ids) || routing["device_diagnostic_allocation_bytes"]!=0 {return Err("routing metadata not actual zero-allocation capture".into())}
    let mut results=Vec::new();let mut peak_weight_elements=0usize;
    for (slot,&expert) in ids.iter().enumerate() {
        for (name,input_name,output_name) in [("w1","expert-input.bf16","w1-slots.bf16"),("w2","activation-slots.bf16","w2-slots.bf16")] {
            let d:Value=serde_json::from_slice(&fs::read(input_dir.join(format!("{name}-expert{expert}-descriptor.json"))).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let n=number(&d,"n")?;let k=number(&d,"k")?;let gs=number(&d,"group_size")?;let groups=number(&d,"groups")?;let bits=number(&d,"bits")?;
            if ![4,8].contains(&bits)||k%16!=0||n%64!=0||gs>k||k%gs!=0||groups!=k.div_ceil(gs)||d["expert"]!=expert||d["device_diagnostic_allocation_bytes"]!=0 {return Err("invalid selected expert geometry".into())}
            let count=k.checked_mul(n).ok_or("weight shape overflow")?;
            let packed_bytes=count.checked_mul(bits).ok_or("packed bytes overflow")?/8;
            let packed=read(&input_dir.join(format!("{name}-expert{expert}-packed.bin")),packed_bytes)?;
            let scales=u16s(&read(&input_dir.join(format!("{name}-expert{expert}-scales.bf16")),groups.checked_mul(n).and_then(|x|x.checked_mul(2)).ok_or("scale bytes overflow")?)?);
            let weight=marlin::MarlinRepacked{packed:packed.chunks_exact(4).map(|x|u32::from_le_bytes(x.try_into().unwrap())).collect(),scales,k,n,group_size:gs};
            let w=if bits==4 {marlin::dequantize_marlin(&weight)} else {marlin::dequantize_marlin_int8(&weight)};
            if w.len()!=count||w.iter().any(|x|!x.is_finite()){return Err("nonfinite or invalid independently unpacked weight".into())}
            peak_weight_elements=peak_weight_elements.max(count);
            let input=u16s(&read(&input_dir.join(input_name),if name=="w1" {k*2} else {topk*k*2})?);
            let x=bfvalues(if name=="w1" {&input} else {&input[slot*k..(slot+1)*k]});
            let outputs=u16s(&read(&input_dir.join(output_name),topk*n*2)?);
            let actual=bfvalues(&outputs[slot*n..(slot+1)*n]);
            let (full,rounded,abs_sums,forward,reverse)=selected_dot_references(&w,&x,n,k)?;
            let stem=format!("{name}-expert{expert}");
            save64(&out.join(format!("{stem}-f32-dequant-reference.f64")),&full)?;
            save64(&out.join(format!("{stem}-bf16-scaled-reference.f64")),&rounded)?;
            save64(&out.join(format!("{stem}-absolute-product-sums.f64")),&abs_sums)?;
            save16(&out.join(format!("{stem}-forward-fma.bf16")),&forward)?;
            save16(&out.join(format!("{stem}-reverse-fma.bf16")),&reverse)?;
            results.push(json!({"expert":expert,"slot":slot,"operation":name,"descriptor":d,"vs_f32_dequant_f64":metrics(&actual,&full)?,"vs_bf16_scaled_f64":metrics(&actual,&rounded)?,"vs_forward_fma":metrics(&actual,&bfvalues(&forward))?,"vs_reverse_fma":metrics(&actual,&bfvalues(&reverse))?,"independent_order_variation":metrics(&bfvalues(&forward),&bfvalues(&reverse))?}));
        }
    }
    let first:Value=serde_json::from_slice(&fs::read(input_dir.join(format!("w2-expert{}-descriptor.json",ids[0]))).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let width=number(&first,"n")?;
    let factor=first["routed_scaling_factor"].as_f64().filter(|x|x.is_finite()).ok_or("invalid routed scaling")? as f32;
    if !factor.is_finite(){return Err("nonfinite loaded routed scale".into())}
    let slots=bfvalues(&u16s(&read(&input_dir.join("w2-slots.bf16"),topk*width*2)?));
    let actual:Vec<_>=f32s(&read(&input_dir.join("routed-accumulator.f32"),width*4)?).into_iter().map(|x|x as f64).collect();
    let mut fused=vec![0f64;width];let mut separate=vec![0f64;width];let mut ideal=vec![0f64;width];
    for col in 0..width {let mut f=0f32;let mut s=0f32;for slot in 0..topk {let w=weights[slot]*factor;let x=slots[slot*width+col] as f32;if w!=0. {f=x.mul_add(w,f);s+=x*w;ideal[col]+=x as f64*w as f64;}}fused[col]=f as f64;separate[col]=s as f64;}
    save64(&out.join("scatter-fma-reference.f64"),&fused)?;save64(&out.join("scatter-separate-reference.f64"),&separate)?;save64(&out.join("scatter-f64-reference.f64"),&ideal)?;
    let scatter=json!({"contract":"Actual compact original token-row*topk+slot layout, F32(weight * loaded factor), sequential slot accumulation. FMA and separate multiply/add contracts measured independently.","exact_f32_fma":actual==fused,"exact_f32_separate":actual==separate,"vs_fma":metrics(&actual,&fused)?,"vs_separate":metrics(&actual,&separate)?,"vs_f64":metrics(&actual,&ideal)?});
    Ok(json!({"scatter":scatter,"diagnostic_only":true,"qualifies_candidate":false,"paths":results,"peak_single_unpacked_weight_elements":peak_weight_elements,"reference_contract":"Independent inverse Marlin packing; exact captured input BF16 upcast; compare F64 dots of full F32 reconstructed weights versus BF16-rounded reconstructed weights. Grouped Marlin scales fragments in BF16 before F32 MMA. Sequential forward/reverse F32 FMA are sensitivity references, not emulation of actual tensor-core or split-K order.","limits":"Captured activation is held fixed for W2; this isolates each operator and does not re-run the whole FFN or prove an implementation defect from tensor-core rounding alone."}))
}
// END selected routed expert independent reference

fn run() -> Result<(),String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref()!=Ok("1") { return Err("Run through ./dev matched-arithmetic".into()) }
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args == ["--help"] { println!("./dev matched-arithmetic <q|q-full|shared|routed|q-weight-build|q-weight|q-weight-accuracy> <capture> <fresh-output> <gpu-ordinal> <measured-decode-warps>"); return Ok(()) }
    if args.len()!=5 { return Err("usage: ./dev matched-arithmetic <q|q-full|shared|routed|q-weight-build|q-weight|q-weight-accuracy> <capture> <output> <gpu-ordinal> <measured-decode-warps>".into()) }
    let input=PathBuf::from(&args[1]); let out=PathBuf::from(&args[2]);
    fs::create_dir(&out).map_err(|e|format!("fresh output directory required: {e}"))?;
    let gpu=args[3].parse().map_err(|_|"invalid GPU ordinal")?;let warps=args[4].parse().map_err(|_|"invalid measured decode warps")?;
    let result=match args[0].as_str() {"q-weight-accuracy"=>hqq_accuracy::run(&input,&out)?,"q-weight-build"=>hqq_weight::build(&input,&out)?,"q-weight"=>hqq_weight::probe(&input,&out,gpu,warps)?,"q"=>q_probe(&input,&out,gpu,warps)?,"q-full"=>q_full_probe(&input,&out,gpu,warps)?,"shared"=>shared_reference(&input,&out)?,"routed"=>routed_reference(&input,&out)?,_=>return Err("invalid mode".into())};
    fs::write(out.join("REPORT.json"),serde_json::to_vec_pretty(&result).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let mut manifest=Vec::new();
    let identity_paths=if input.is_file(){vec![input.clone()]}else{fs::read_dir(&input).map_err(|e|e.to_string())?.map(|e|e.map(|x|x.path()).map_err(|e|e.to_string())).collect::<Result<Vec<_>,_>>()?};
    for path in identity_paths { if path.is_file() {let b=fs::read(&path).map_err(|e|e.to_string())?;manifest.push(json!({"path":path,"bytes":b.len(),"sha256":format!("{:x}",Sha256::digest(&b))}));} }
    fs::write(out.join("INPUT_IDENTITIES.json"),serde_json::to_vec_pretty(&manifest).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    println!("Matched arithmetic evidence saved: {} (diagnostic collection, not qualification)",out.display());
    Ok(())
}
fn main() { if let Err(e)=run() {eprintln!("{e}");std::process::exit(1)} }

#[cfg(test)]
mod tests {
    use super::*;
    // BEGIN bounded full Q contract regression
    #[test]
    fn full_q_contract_rejects_shape_dtype_and_arithmetic_changes() {
        let desc=json!({"rows":8,"cols":4,"height":3,"row":1,"nbits":6,"materialize_bf16":true});
        let contract=json!({"m":8,"n":3,"k":4,"lda":4,"ldb":4,"ldc":8,"transpose_a":"T","transpose_b":"N","alpha":1.0,"beta":0.0,"compute":"CUBLAS_COMPUTE_32F","algorithm":"CUBLAS_GEMM_DEFAULT","pointer_mode":0,"stream_matches_production":true,"stream_is_null":false,"input_dtype":"bf16","weights_dtype":"bf16","output_dtype":"bf16"});
        assert_eq!(full_q_dimensions(&desc,&contract,16).unwrap(),(8,4,3,1));
        for (key,value) in [("n",json!(2)),("beta",json!(1.0)),("compute",json!("other")),("pointer_mode",json!(1)),("input_dtype",json!("f32")),("stream_matches_production",json!(false)),("stream_is_null",Value::Null)] {
            let mut bad=contract.clone();bad[key]=value;assert!(full_q_dimensions(&desc,&bad,16).is_err(),"{key}");
        }
        for (key,value) in [("row",json!(3)),("nbits",json!(4)),("materialize_bf16",json!(false)),("rows",json!(0))] {
            let mut bad=desc.clone();bad[key]=value;assert!(full_q_dimensions(&bad,&contract,16).is_err(),"{key}");
        }
        assert!(full_q_dimensions(&desc,&contract,3).is_err());
    }
    // END bounded full Q contract regression
    #[test]
    fn relative_l2_uses_vector_difference_and_reference_norm() {
        let m=metrics(&[3.,4.], &[0.,5.]).unwrap();
        assert!((m["relative_l2"].as_f64().unwrap()-10f64.sqrt()/5.).abs()<1e-14);
        assert_eq!(m["max_abs_coordinate"],0);
        assert_eq!(m["bf16_mismatch_count"],2);
    }
    #[test]
    fn invalid_metrics_fail_explicitly() {
        for (a,b) in [(vec![],vec![]),(vec![1.],vec![0.]),(vec![f64::NAN],vec![1.]),(vec![1.],vec![1.,2.])] {
            assert!(metrics(&a,&b).is_err());
        }
    }
}
