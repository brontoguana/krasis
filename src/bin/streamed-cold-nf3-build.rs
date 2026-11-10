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
fn learn(hist: &[u64;16]) -> (u32,[u8;32],u64,u64) {
    let mut best_cost=u64::MAX;let mut best=0u32;
    for mask in 0u32..65536 {
        if mask.count_ones()!=8 || mask&(1<<8)==0 {continue;}
        let levels=(0..16u32).filter(|q|mask&(1<<q)!=0).collect::<Vec<_>>();
        let mut cost=0u64;
        for q in 0..16u32 {
            let dist=levels.iter().map(|&c|c.abs_diff(q)).min().unwrap() as u64;
            cost+=hist[q as usize]*dist*dist;
        }
        if cost<best_cost {best_cost=cost;best=levels.iter().enumerate().fold(0,|v,(i,&q)|v|(q<<(i*4)));}
    }
    let mut map=[0u8;32];let mut linear_cost=0;
    for q in 0..16u32 {
        let base=q>>1;let linear=(base+u32::from(q&3==3)).min(7)*2;
        let d=linear.abs_diff(q) as u64;linear_cost+=hist[q as usize]*d*d;
        for parity in 0..2 {
            let mut distance=u32::MAX;let mut selected=0;
            for i in 0..8 {
                let c=(best>>(i*4))&15;let d=c.abs_diff(q);
                if d<distance || d==distance && parity==1 {distance=d;selected=i;}
            }
            map[q as usize+parity*16]=selected as u8;
        }
    }
    (best,map,best_cost,linear_cost)
}
fn pack(src: &[u8], out: &mut Vec<u8>, map:&[u8;32]) {
    for b in src.chunks_exact(4) {
        let x=u32::from_le_bytes(b.try_into().unwrap());let mut y=0u32;
        for j in 0..8 {let q=((x>>(j*4))&15)as usize;let c=map[q+(j&1)*16] as u32;y|=c<<(j*3);}
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
    let temporary=path.with_extension("coldnf3.partial");
    let mut file=OpenOptions::new().create_new(true).write(true).open(&temporary)?;
    file.write_all(&[0u8;4096])?;
    let mut raw_hashes=Vec::with_capacity(count); let mut compact_hashes=Vec::with_capacity(count);
    let pool=rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
    let histograms=pool.install(|| (0..count).into_par_iter().map(|i|{
        let src=&mmap[64+i*raw..64+(i+1)*raw];let mut hist=[0u64;16];
        for range in [0..sizes[0],sizes[0]+sizes[1]..sizes[0]+sizes[1]+sizes[2]] {
            for &byte in &src[range] {hist[(byte&15)as usize]+=1;hist[(byte>>4)as usize]+=1;}
        }
        hist
    }).collect::<Vec<_>>());
    let mut learned=Vec::new();
    for layer in 0..l {
        let mut hist=[0u64;16];
        for h in &histograms[layer*e..(layer+1)*e] {for i in 0..16 {hist[i]+=h[i];}}
        let result=learn(&hist);eprintln!("layer={layer} codebook=0x{:08x} nf3_squared_code_error={} linear_squared_code_error={}",result.0,result.2,result.3);
        learned.push(result);
    }
    for start in (0..count).step_by(8) {
        let rows=pool.install(|| (start..(start+8).min(count)).into_par_iter().map(|i| {
            let src=&mmap[64+i*raw..64+(i+1)*raw]; let mut out=Vec::with_capacity(compact); let mut at=0;
            for (j,size) in sizes.iter().copied().enumerate(){
                if j%2==0 {pack(&src[at..at+size],&mut out,&learned[i/e].1);}else{out.extend_from_slice(&src[at..at+size]);} at+=size;
            }
            assert_eq!(out.len(),compact);
            let a:[u8;32]=Sha256::digest(src).into(); let b:[u8;32]=Sha256::digest(&out).into();
            (out,a,b)
        }).collect::<Vec<_>>());
        for (bytes,a,b) in rows {file.write_all(&bytes)?;raw_hashes.push(a);compact_hashes.push(b);}
        if start%288==0 {eprintln!("packed {}/{} experts",(start+8).min(count),count);}
    }
    let mut header=[0u8;4096]; header[..8].copy_from_slice(b"KRCLNF3\0");
    header[8..12].copy_from_slice(&1u32.to_le_bytes());header[12..16].copy_from_slice(&3u32.to_le_bytes());
    header[16..24].copy_from_slice(&(mmap.len() as u64).to_le_bytes());header[24..88].copy_from_slice(&mmap[..64]);
    header[88..120].copy_from_slice(&root(raw,&raw_hashes));header[120..152].copy_from_slice(&root(compact,&compact_hashes));
    for (i,n) in sizes.into_iter().chain([count,compact]).enumerate(){header[152+i*8..160+i*8].copy_from_slice(&(n as u64).to_le_bytes());}
    assert!(l<=256);
    for (i,entry) in learned.iter().enumerate() {header[200+i*4..204+i*4].copy_from_slice(&entry.0.to_le_bytes());}
    let checksum:[u8;32]=Sha256::digest(&header[..4064]).into();header[4064..].copy_from_slice(&checksum);
    file.seek(SeekFrom::Start(0))?;file.write_all(&header)?;file.sync_all()?;drop(file);
    std::fs::rename(temporary,path)?;
    eprintln!("complete: {} experts, {} -> {} bytes/expert, {:.3}% saved",count,raw,compact,100.0*(1.0-compact as f64/raw as f64));
    Ok(())
}
