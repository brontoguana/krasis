//! Offline compact cold-expert sidecar from an existing local Marlin cache.
#[path = "../optimisations.rs"]
#[allow(dead_code)]
mod build_optimisations;
use memmap2::Mmap;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;

fn word(b: &[u8], p: usize) -> usize { u64::from_le_bytes(b[p..p+8].try_into().unwrap()) as usize }
fn root(bytes: usize, hashes: &[[u8;32]]) -> [u8;32] {
    let mut h=Sha256::new(); h.update(b"KRASIS_ROUTED_EXPERT_SHA256_V1");
    h.update((bytes as u64).to_le_bytes()); h.update((hashes.len() as u64).to_le_bytes());
    for x in hashes {h.update(x);} h.finalize().into()
}
fn pack(src: &[u8], out: &mut Vec<u8>) {
    for b in src.chunks_exact(4) {
        let x=u32::from_le_bytes(b.try_into().unwrap()); let mut y=0u32;
        for j in 0..8 {let q=(x>>(j*4))&15;
            // Round half-way values to the even three-bit code; clip +8.
            let c=((q>>1)+u32::from(q&3==3)).min(7);
            y|=c<<(j*3);
        }
        out.extend_from_slice(&y.to_le_bytes()[..3]);
    }
}
fn main() -> Result<(),Box<dyn std::error::Error>> {
    build_optimisations::print_startup();
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    if args.len()!=2 {return Err("usage: streamed-cold-build cache.bin output.cold3".into());}
    let source=File::open(&args[0])?; let mmap=unsafe{Mmap::map(&source)?};
    assert_eq!(&mmap[..8],b"KRAS\x07\0\0\0");
    let (h,m,e,l,g)=(word(&mmap,8),word(&mmap,16),word(&mmap,24),word(&mmap,32),word(&mmap,40));
    assert!(h>0 && m>0 && g>0 && h%g==0 && m%g==0 && h%64==0 && m%64==0);
    // This tier is intentionally limited to standard-width gated experts.
    assert!(h!=m,"padded expert widths need an explicit layout");
    let sizes=[h*m, h/g*2*m*2, m*h/2, m/g*h*2];
    let raw=sizes.iter().sum::<usize>(); let count=e*l;
    let compact=(sizes[0]+sizes[2])*3/4+sizes[1]+sizes[3];
    assert!(64+count*raw<=mmap.len());
    let path=PathBuf::from(&args[1]); assert!(!path.exists());
    let temporary=path.with_extension("cold3.partial");
    let mut file=OpenOptions::new().create_new(true).write(true).open(&temporary)?;
    file.write_all(&[0u8;4096])?;
    let mut raw_hashes=Vec::with_capacity(count); let mut compact_hashes=Vec::with_capacity(count);
    let pool=rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
    for start in (0..count).step_by(8) {
        let rows=pool.install(|| (start..(start+8).min(count)).into_par_iter().map(|i| {
            let src=&mmap[64+i*raw..64+(i+1)*raw]; let mut out=Vec::with_capacity(compact); let mut at=0;
            for (j,size) in sizes.iter().copied().enumerate(){
                if j%2==0 {pack(&src[at..at+size],&mut out);}else{out.extend_from_slice(&src[at..at+size]);} at+=size;
            }
            assert_eq!(out.len(),compact);
            let a:[u8;32]=Sha256::digest(src).into(); let b:[u8;32]=Sha256::digest(&out).into();
            (out,a,b)
        }).collect::<Vec<_>>());
        for (bytes,a,b) in rows {file.write_all(&bytes)?;raw_hashes.push(a);compact_hashes.push(b);}
        if start%288==0 {eprintln!("packed {}/{} experts",(start+8).min(count),count);}
    }
    let mut header=[0u8;4096]; header[..8].copy_from_slice(b"KRCOLD3\0");
    header[8..12].copy_from_slice(&1u32.to_le_bytes());header[12..16].copy_from_slice(&3u32.to_le_bytes());
    header[16..24].copy_from_slice(&(mmap.len() as u64).to_le_bytes());header[24..88].copy_from_slice(&mmap[..64]);
    header[88..120].copy_from_slice(&root(raw,&raw_hashes));header[120..152].copy_from_slice(&root(compact,&compact_hashes));
    for (i,n) in sizes.into_iter().chain([count,compact]).enumerate(){header[152+i*8..160+i*8].copy_from_slice(&(n as u64).to_le_bytes());}
    file.seek(SeekFrom::Start(0))?;file.write_all(&header)?;file.sync_all()?;drop(file);
    std::fs::rename(temporary,path)?;
    eprintln!("complete: {} experts, {} -> {} bytes/expert, {:.3}% saved",count,raw,compact,100.0*(1.0-compact as f64/raw as f64));
    Ok(())
}
