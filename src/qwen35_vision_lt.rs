// cuBLASLt contraction contract used by BF16 Linear in the frozen backend.
// Descriptor ownership is explicit on success and every error path.
use cudarc::cublaslt::{result as lr, sys as ls};
struct VisionLtDescriptors {
    handle: ls::cublasLtHandle_t,
    desc: ls::cublasLtMatmulDesc_t,
    pref: ls::cublasLtMatmulPreference_t,
    layouts: Vec<ls::cublasLtMatrixLayout_t>,
}
impl Drop for VisionLtDescriptors {
    fn drop(&mut self) { unsafe {
        for x in &self.layouts { let _=lr::destroy_matrix_layout(*x); }
        if !self.pref.is_null() { let _=lr::destroy_matmul_pref(self.pref); }
        if !self.desc.is_null() { let _=lr::destroy_matmul_desc(self.desc); }
        let _=lr::destroy_handle(self.handle);
    }}
}
impl Qwen35Vision {
    fn linear_lt(&self,x:u64,w:u64,bias:u64,y:u64,m:usize,n:usize,k:usize) -> Result<(),String> {
        let run=|| -> Result<(),lr::CublasError> {
            let mut owner=VisionLtDescriptors{handle:lr::create_handle()?,desc:std::ptr::null_mut(),pref:std::ptr::null_mut(),layouts:Vec::new()};
            owner.desc=lr::create_matmul_desc(ls::cublasComputeType_t::CUBLAS_COMPUTE_32F,ls::cudaDataType::CUDA_R_32F)?;
            let trans=1i32;
            let ep=ls::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS;
            unsafe {
                lr::set_matmul_desc_attribute(owner.desc,ls::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,&trans as *const _ as _,std::mem::size_of_val(&trans))?;
                lr::set_matmul_desc_attribute(owner.desc,ls::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,&ep as *const _ as _,std::mem::size_of_val(&ep))?;
                lr::set_matmul_desc_attribute(owner.desc,ls::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias as *const _ as _,std::mem::size_of_val(&bias))?;
            }
            for (r,c,ld) in [(k,n,k),(k,m,k),(n,m,n)] { owner.layouts.push(lr::create_matrix_layout(ls::cudaDataType::CUDA_R_16BF,r as u64,c as u64,ld as i64)?); }
            owner.pref=lr::create_matmul_pref()?;
            let size=1024*1024usize;
            unsafe {
                lr::set_matmul_pref_attribute(owner.pref,ls::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&size as *const _ as _,std::mem::size_of_val(&size))?;
                let a=lr::get_matmul_algo_heuristic(owner.handle,owner.desc,owner.layouts[0],owner.layouts[1],owner.layouts[2],owner.layouts[2],owner.pref)?;
                let alpha=1f32;let beta=0f32;
                ls::lib().cublasLtMatmul(owner.handle,owner.desc,&alpha as *const _ as _,w as _,owner.layouts[0],x as _,owner.layouts[1],&beta as *const _ as _,y as _,owner.layouts[2],y as _,owner.layouts[2],&a.algo,*self.lt_workspace.device_ptr() as _,size,*self.device.cu_stream() as _).result()?;
            }
            Ok(())
        };
        run().map_err(|e|format!("vision compatible Linear {m}x{n}x{k}: {e}"))
    }
}
