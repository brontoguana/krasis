//! Compact cold-only payload; original resident and CPU sources stay untouched.
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, sys as cu};
use memmap2::{MmapMut, MmapOptions};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs::File, io::Read, sync::Arc};
use crate::weights::GpuCacheIdentity;

fn check(rc: cu::CUresult, what: &str) -> Result<(),String> {
    if rc==cu::CUresult::CUDA_SUCCESS {Ok(())} else {Err(format!("cold tier {what}: {rc:?}"))}
}
fn word(b:&[u8],p:usize)->Result<usize,String>{
    usize::try_from(u64::from_le_bytes(b[p..p+8].try_into().unwrap())).map_err(|_|"cold tier size overflow".into())
}
struct Segment { mapping:MmapMut, registered:bool }
impl Drop for Segment {fn drop(&mut self){if self.registered {unsafe {
    let _=cu::lib().cuMemHostUnregister(self.mapping.as_mut_ptr().cast());
}}}}
pub(crate) struct Runtime {
    device:Arc<CudaDevice>, segments:Vec<Segment>,
    locations:Vec<(usize,usize)>, ordinals:HashMap<usize,usize>,
    scratch:CudaSlice<u8>, function:cu::CUfunction,
    words:[u32;4], compact_bytes:usize,
}
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}
impl Runtime {
    pub(crate) fn new(device:&Arc<CudaDevice>, identity:&GpuCacheIdentity,
        sizes:[usize;4], ordinals:HashMap<usize,usize>, count:usize,
        function:cu::CUfunction)->Result<Self,String>{
        device.bind_to_thread().map_err(|e|e.to_string())?;
        let mut name=identity.path.as_os_str().to_os_string();name.push(".cold3");
        let mut file=File::open(std::path::PathBuf::from(name)).map_err(|e|format!("cold tier sidecar: {e}"))?;
        let mut header=[0u8;4096];file.read_exact(&mut header).map_err(|e|e.to_string())?;
        if &header[..8]!=b"KRCOLD3\0" || header[8..16]!=[1,0,0,0,3,0,0,0]
            || word(&header,16)? as u64!=identity.source_bytes || header[24..88]!=identity.header
            || Some(header[88..120].try_into().unwrap())!=identity.routed_expert_sha256 {
            return Err("cold tier source identity/format mismatch".into());
        }
        let compact_bytes=(sizes[0]+sizes[2])*3/4+sizes[1]+sizes[3];
        for (i,n) in sizes.into_iter().chain([count,compact_bytes]).enumerate(){
            if word(&header,152+i*8)?!=n {return Err("cold tier component geometry mismatch".into());}
        }
        if sizes.iter().any(|n|*n==0 || *n%4!=0) || ordinals.len()!=count
            || file.metadata().map_err(|e|e.to_string())?.len()!=4096+(count*compact_bytes) as u64 {
            return Err("cold tier payload geometry mismatch".into());
        }
        let mut segments=Vec::new();let mut locations=Vec::with_capacity(count);
        let mut root=Sha256::new();root.update(b"KRASIS_ROUTED_EXPERT_SHA256_V1");
        root.update((compact_bytes as u64).to_le_bytes());root.update((count as u64).to_le_bytes());
        // Expert sizes and the header are page-aligned on admitted geometry.
        // Bound each writable-private VMA below existing per-layer source pins.
        if compact_bytes%4096!=0 {return Err("cold tier expert extent is not page aligned".into());}
        for start in (0..count).step_by(64){
            let len=(count-start).min(64)*compact_bytes;
            let mapping=unsafe{MmapOptions::new().offset((4096+start*compact_bytes) as u64).len(len).map_copy(&file)}.map_err(|e|e.to_string())?;
            for i in 0..len/compact_bytes {
                root.update(Sha256::digest(&mapping[i*compact_bytes..(i+1)*compact_bytes]));
                locations.push((segments.len(),i*compact_bytes));
            }
            segments.push(Segment{mapping,registered:false});
        }
        let measured:[u8;32]=root.finalize().into();
        if measured.as_slice()!=&header[120..152] {return Err("cold tier payload hash mismatch".into());}
        for segment in &mut segments {unsafe{
            check(cu::lib().cuMemHostRegister_v2(segment.mapping.as_mut_ptr().cast(),segment.mapping.len(),0),"host registration")?;
            segment.registered=true;
        }}
        let (mut free,mut total)=(0usize,0usize);
        unsafe{check(cu::lib().cuMemGetInfo_v2(&mut free,&mut total),"reserve query")?;}
        if free<compact_bytes+600*1024*1024 {return Err("cold tier scratch would violate 600 MiB reserve".into());}
        let scratch=device.alloc_zeros::<u8>(compact_bytes).map_err(|e|e.to_string())?;
        let mut words=[0u32;4];for i in 0..4 {words[i]=u32::try_from(sizes[i]/4).map_err(|_|"cold tier word count overflow")?;}
        eprintln!("STREAMED COLD INT3 experts={count} raw_bytes={} compact_bytes={compact_bytes} resident=original cpu=original promotions=frozen",sizes.iter().sum::<usize>());
        Ok(Self{device:Arc::clone(device),segments,locations,ordinals,scratch,function,words,compact_bytes})
    }
    /// One ordered copy+expand pair. The existing copy-stream completion event
    /// makes the normal graph pointers ready; scratch is reused on that stream.
    pub(crate) fn enqueue(&self, source:usize, output:u64, stream:cu::CUstream)->Result<usize,String>{
        let ordinal=*self.ordinals.get(&source).ok_or("cold tier source absent")?;
        let (segment,offset)=self.locations[ordinal];
        let host=unsafe{self.segments[segment].mapping.as_ptr().add(offset)};
        let mut input=*self.scratch.device_ptr();let mut out=output;let mut words=self.words;
        unsafe{
            check(cu::lib().cuMemcpyHtoDAsync_v2(input,host.cast(),self.compact_bytes,stream),"payload copy")?;
            let total=words.iter().map(|v|*v as usize).sum::<usize>();
            let mut params=[&mut input as *mut _ as *mut std::ffi::c_void,&mut out as *mut _ as *mut std::ffi::c_void,
                &mut words[0] as *mut _ as *mut std::ffi::c_void,&mut words[1] as *mut _ as *mut std::ffi::c_void,
                &mut words[2] as *mut _ as *mut std::ffi::c_void,&mut words[3] as *mut _ as *mut std::ffi::c_void];
            check(cu::lib().cuLaunchKernel(self.function,total.div_ceil(256) as u32,1,1,256,1,1,0,stream,params.as_mut_ptr(),std::ptr::null_mut()),"expand launch")?;
        }
        Ok(self.compact_bytes)
    }
}
impl Drop for Runtime {fn drop(&mut self){
    // Finish consumers before host unregister/unmap or scratch destruction.
    if self.device.bind_to_thread().is_ok(){unsafe{let _=cu::lib().cuCtxSynchronize();}}
}}
