// Bounded compatibility path. Query bands preserve the full reduction over keys;
// cached BF16 values are concatenated before grouped-query head expansion.
impl PrefillEngine {
    pub(crate) fn qwen35_reference_attention(&self, q:u64, k:u64, v:u64, ck:u64, cv:u64,
        out:u64, rows:usize, prefix:usize, heads:usize, kv_heads:usize, dim:usize, scale:f32) -> Result<(),String> {
        use cudarc::cublas::{sys as bs, result as br};
        if q == 0 || k == 0 || v == 0 || out == 0 || rows == 0 || heads == 0 || dim == 0 || kv_heads == 0 || heads % kv_heads != 0 || (prefix > 0 && (ck == 0 || cv == 0)) {
            return Err("invalid Qwen reference attention geometry/cache".into());
        }
        let keys=prefix.checked_add(rows).ok_or("attention extent overflow")?;
        // The supported bounded Qwen profile admits at most 4096 tokens.
        if keys > 4096 || heads > 256 || dim > 256 { return Err("bounded reference attention exceeds supported geometry".into()); }
        let band=rows.min(256).min((16*1024*1024 / (heads*keys*2)).max(1));
        let kv_count=heads*keys*dim;
        let q_count=heads*band*dim;
        let score_count=heads*band*keys;
        let bytes=(2*kv_count+2*q_count+score_count)*2;
        let (free,_)=cudarc::driver::result::mem_get_info().map_err(|e|e.to_string())?;
        // Joint model admission charges these buffers against the configured
        // ceiling. Physical free memory also enforces the reserve with ballast.
        if free < bytes+600*1024*1024 {
            return Err(format!("Qwen attention {} bytes would breach memory ceiling/reserve",bytes));
        }
        let packed_k=GpuBuf::<u16>::alloc_uninit(kv_count)?;
        let packed_v=GpuBuf::<u16>::alloc_uninit(kv_count)?;
        let packed_q=GpuBuf::<u16>::alloc_uninit(q_count)?;
        let packed_y=GpuBuf::<u16>::alloc_uninit(q_count)?;
        let scores=GpuBuf::<u16>::alloc_uninit(score_count)?;
        let func=|name|self.device.get_func("prefill_kernels",name).map(|f|extract_cu_func(&f)).ok_or_else(||format!("missing {name}"));
        let pack=func("qwen35_attention_pack")?;let unpack=func("qwen35_attention_unpack")?;let softmax=func("qwen35_attention_softmax")?;
        let pack_input=|src:u64,cache:u64,dst:u64,n:usize,sh:usize,p:usize,off:usize| -> Result<(),String> {
            let mut args=(src,cache,dst,n as i32,heads as i32,sh as i32,dim as i32,p as i32,off as i32);
            unsafe { launch(pack,(((n*heads*dim+255)/256) as u32,1,1),(256,1,1),0,self.stream,&mut[
                &mut args.0 as *mut _ as _,&mut args.1 as *mut _ as _,&mut args.2 as *mut _ as _,&mut args.3 as *mut _ as _,&mut args.4 as *mut _ as _,&mut args.5 as *mut _ as _,&mut args.6 as *mut _ as _,&mut args.7 as *mut _ as _,&mut args.8 as *mut _ as _]) }
        };
        let result=(|| -> Result<(),String> {
            pack_input(k,ck,packed_k.ptr,keys,kv_heads,prefix,0)?;
            pack_input(v,cv,packed_v.ptr,keys,kv_heads,prefix,0)?;
            unsafe { br::set_stream(self.cublas_handle,self.stream as _).map_err(|e|format!("attention stream {e:?}"))?; }
            let alpha=1f32;let beta=0f32;
            for offset in (0..rows).step_by(band) {
                let n=(rows-offset).min(band);
                pack_input(q,0,packed_q.ptr,n,heads,0,offset)?;
                unsafe {
                    br::gemm_strided_batched_ex(self.cublas_handle,bs::cublasOperation_t::CUBLAS_OP_T,bs::cublasOperation_t::CUBLAS_OP_N,
                        keys as i32,n as i32,dim as i32,&alpha as *const _ as _,packed_k.ptr as _,bs::cudaDataType::CUDA_R_16BF,dim as i32,(keys*dim) as i64,
                        packed_q.ptr as _,bs::cudaDataType::CUDA_R_16BF,dim as i32,(n*dim) as i64,&beta as *const _ as _,scores.ptr as _,bs::cudaDataType::CUDA_R_16BF,keys as i32,(n*keys) as i64,
                        heads as i32,bs::cublasComputeType_t::CUBLAS_COMPUTE_32F,bs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT).map_err(|e|format!("Qwen QK {e:?}"))?;
                    let mut a=(scores.ptr,n as i32,keys as i32,(prefix+offset) as i32,scale);
                    launch(softmax,((heads*n) as u32,1,1),(128,1,1),0,self.stream,&mut[&mut a.0 as *mut _ as _,&mut a.1 as *mut _ as _,&mut a.2 as *mut _ as _,&mut a.3 as *mut _ as _,&mut a.4 as *mut _ as _])?;
                    br::gemm_strided_batched_ex(self.cublas_handle,bs::cublasOperation_t::CUBLAS_OP_N,bs::cublasOperation_t::CUBLAS_OP_N,
                        dim as i32,n as i32,keys as i32,&alpha as *const _ as _,packed_v.ptr as _,bs::cudaDataType::CUDA_R_16BF,dim as i32,(keys*dim) as i64,
                        scores.ptr as _,bs::cudaDataType::CUDA_R_16BF,keys as i32,(n*keys) as i64,&beta as *const _ as _,packed_y.ptr as _,bs::cudaDataType::CUDA_R_16BF,dim as i32,(n*dim) as i64,
                        heads as i32,bs::cublasComputeType_t::CUBLAS_COMPUTE_32F,bs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT).map_err(|e|format!("Qwen PV {e:?}"))?;
                    let mut a=(packed_y.ptr,out,n as i32,heads as i32,dim as i32,offset as i32);
                    launch(unpack,(((n*heads*dim+255)/256) as u32,1,1),(256,1,1),0,self.stream,&mut[&mut a.0 as *mut _ as _,&mut a.1 as *mut _ as _,&mut a.2 as *mut _ as _,&mut a.3 as *mut _ as _,&mut a.4 as *mut _ as _,&mut a.5 as *mut _ as _])?;
                }
            }
            Ok(())
        })();
        // All readers finish before storage is freed, including partial failure.
        let completion=self.stream_sync();
        result.and(completion)
    }
}

impl PrefillEngine {
    fn qwen35_boundary(&self,label:&str,layer:usize,position:usize,pointer:u64,count:usize)->Result<(),String> {
        if self.kernels.qwen35_gdn_accurate.is_none() {return Ok(());}
        let Ok(directory)=std::env::var("KRASIS_QWEN_BOUNDARY_DIR") else {return Ok(())};
        if count > 8*1024*1024 {return Err("diagnostic boundary exceeds bounded capture".into());}
        self.stream_sync()?;
        let mut values=vec![0u16;count];
        unsafe {cuda_sys::lib().cuMemcpyDtoH_v2(values.as_mut_ptr() as _,pointer,count*2).result().map_err(|e|e.to_string())?;}
        std::fs::create_dir_all(&directory).map_err(|e|e.to_string())?;
        let bytes:Vec<u8>=values.iter().flat_map(|x|x.to_le_bytes()).collect();
        std::fs::write(std::path::Path::new(&directory).join(format!("layer{layer}-pos{position}-{label}.bf16")),bytes).map_err(|e|e.to_string())
    }
}
