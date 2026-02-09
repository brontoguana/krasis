//! Safetensors file reading via mmap.
//!
//! Provides zero-copy access to weight tensors stored in safetensors format.
//! Files are mmap'd so the OS manages page-in/eviction automatically.

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// A memory-mapped safetensors file with lazy tensor access.
pub struct MmapSafetensors {
    _mmap: Mmap,
    // TODO: tensor name → (offset, shape, dtype) index
}

impl MmapSafetensors {
    /// Open and mmap a safetensors file.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(MmapSafetensors { _mmap: mmap })
    }

    /// Get a raw byte slice for a named tensor.
    pub fn tensor_data(&self, _name: &str) -> Option<&[u8]> {
        // TODO: parse header, look up tensor by name, return slice
        todo!("safetensors tensor lookup")
    }
}
