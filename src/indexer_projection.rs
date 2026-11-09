//! Fixed 64-row BF16 projection tiles. Padding keeps reduction geometry independent
//! of logical chunk size; the engine owns tail storage until its stream completes.
use crate::gpu_prefill::GpuBuf;
use cudarc::{cublas::{result as blas, sys as b}, driver::{sys, CudaDevice, DevicePtr}};

pub(crate) const TILE: usize = 64;
pub(crate) const N: usize = 4096;
pub(crate) const K: usize = 1536;
pub(crate) const ELEMENTS: usize = TILE * (K + N);

pub(crate) fn enabled() -> Result<bool,String> {
    match std::env::var("KRASIS_INDEXER_QUERY_TILE") {
        Err(std::env::VarError::NotPresent) => Ok(true),
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1"|"true"|"yes"|"on" => Ok(true),
            "0"|"false"|"no"|"off" => Ok(false),
            _ => Err("KRASIS_INDEXER_QUERY_TILE expects a boolean".into()),
        },
        Err(e) => Err(e.to_string()),
    }
}

pub(crate) struct Tile {
    tail: GpuBuf<u16>,
}
impl Tile {
    pub(crate) fn create(device:&CudaDevice) -> Result<Option<Self>,String> {
        if !enabled()? { return Ok(None); }
        device.bind_to_thread().map_err(|e|e.to_string())?;
        if device.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e|e.to_string())? < 8 { return Ok(None); }
        Ok(Some(Self { tail:GpuBuf::alloc_zeroed(ELEMENTS)? }))
    }
    /// Call on the engine's serial compute stream with live, disjoint model
    /// input/weight/output owners. All tail copies are ordered on that stream.
    pub(crate) unsafe fn enqueue(&self, handle:b::cublasHandle_t, stream:sys::CUstream,
        input:u64,weight:u64,output:u64,rows:usize,n:usize,k:usize) -> Result<bool,String> {
        // Preserve the existing small-batch reduction family. At real dimensions,
        // cuBLAS itself changes reductions below 64 rows; reusing the larger
        // tile there changes BF16 boundaries. Keep that separate intervention
        // out of this exact candidate.
        if n!=N || k!=K || rows<TILE { return Ok(false); }
        let extent=|p:u64,count:Option<usize>| -> Result<u64,String> {
            if p==0 || p%2!=0 { return Err("indexer tile invalid BF16 pointer".into()); }
            p.checked_add(count.and_then(|v|v.checked_mul(2)).ok_or("indexer tile extent overflow")? as u64)
                .ok_or("indexer tile pointer overflow".into())
        };
        let ie=extent(input,rows.checked_mul(K))?;
        let we=extent(weight,Some(N*K))?;
        let oe=extent(output,rows.checked_mul(N))?;
        if (output<ie && input<oe) || (output<we && weight<oe) {
            return Err("indexer tile output aliases input/weight".into());
        }
        let mut math=std::mem::zeroed();
        b::lib().cublasGetMathMode(handle,&mut math).result().map_err(|e|format!("indexer tile math mode: {e:?}"))?;
        if math!=b::cublasMath_t::CUBLAS_DEFAULT_MATH { return Ok(false); }
        blas::set_stream(handle,stream.cast()).map_err(|e|format!("indexer tile stream: {e:?}"))?;
        let alpha=1f32;let beta=0f32;
        for start in (0..rows).step_by(TILE) {
            let live=(rows-start).min(TILE);
            let (a,c)=if live==TILE {(input+(start*K*2)as u64,output+(start*N*2)as u64)} else {
                let a=*self.tail.device_ptr();let c=a+(TILE*K*2)as u64;
                sys::lib().cuMemsetD8Async(a,0,TILE*K*2,stream).result().map_err(|e|format!("indexer tile padding: {e:?}"))?;
                sys::lib().cuMemcpyDtoDAsync_v2(a,input+(start*K*2)as u64,live*K*2,stream).result().map_err(|e|format!("indexer tile input: {e:?}"))?;
                (a,c)
            };
            blas::gemm_ex(handle,b::cublasOperation_t::CUBLAS_OP_T,b::cublasOperation_t::CUBLAS_OP_N,
                N as i32,TILE as i32,K as i32,(&alpha as *const f32).cast(),
                weight as *const _,b::cudaDataType::CUDA_R_16BF,K as i32,
                a as *const _,b::cudaDataType::CUDA_R_16BF,K as i32,(&beta as *const f32).cast(),
                c as *mut _,b::cudaDataType::CUDA_R_16BF,N as i32,
                b::cublasComputeType_t::CUBLAS_COMPUTE_32F,b::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT)
                .map_err(|e|format!("indexer tile GEMM: {e:?}"))?;
            if live<TILE {
                sys::lib().cuMemcpyDtoDAsync_v2(output+(start*N*2)as u64,c,live*N*2,stream).result()
                    .map_err(|e|format!("indexer tile output: {e:?}"))?;
            }
        }
        Ok(true)
    }
}
