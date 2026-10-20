// Pure pre-DMA semantic admission for the complete fixed-block KDA carry.
// Other operational allocations/signatures require their own complete proof.
fn validate_kda_snapshot_semantics(
    allocations: &[crate::session_cache::SequenceStateAllocation],
    tokens: usize,
    blobs: &[&crate::session_cache::SequenceStateBlob],
) -> Result<(), String> {
    if allocations.len() != blobs.len() {
        return Err("KDA snapshot semantic inventory incomplete".into());
    }
    for (a, b) in allocations.iter().zip(blobs) {
        b.validate()?;
        if !crate::session_cache::snapshot_blob_can_restore(a, a.used_bytes(tokens), b) {
            return Err(format!("KDA snapshot semantic layout:{}", a.name));
        }
        let role = a
            .name
            .rsplit('.')
            .next()
            .ok_or("KDA snapshot semantic role missing")?;
        match role {
            "absolute" => {
                let expected = u64::try_from(tokens)
                    .map_err(|_| "KDA snapshot token extent overflow")?
                    .to_le_bytes();
                if b.bytes.as_slice() != expected {
                    return Err(format!("KDA snapshot semantic absolute:{}", a.name));
                }
            }
            "pending" => {
                let prefix = a
                    .name
                    .strip_suffix(".pending")
                    .ok_or("KDA snapshot pending prefix")?;
                let raw = allocations
                    .iter()
                    .find(|p| p.name == format!("{prefix}.raw0"))
                    .ok_or("KDA snapshot pending schema absent")?;
                let rows = *raw
                    .shape
                    .first()
                    .ok_or("KDA snapshot pending rows absent")?;
                if rows == 0 {
                    return Err("KDA snapshot pending rows zero".into());
                }
                let expected = i32::try_from(tokens % rows)
                    .map_err(|_| "KDA snapshot pending extent overflow")?
                    .to_le_bytes();
                if b.bytes.as_slice() != expected {
                    return Err(format!("KDA snapshot semantic pending:{}", a.name));
                }
            }
            "error" => {
                if b.bytes.as_slice() != 0i32.to_le_bytes() {
                    return Err(format!("KDA snapshot semantic error:{}", a.name));
                }
            }
            "complete" | "incoming" | "latest" | "raw0" | "raw1" | "raw2" | "raw3" | "raw4" => {
                if a.dtype != "float32" || a.element_size != 4 || b.bytes.len() % 4 != 0 {
                    return Err(format!("KDA snapshot semantic F32 extent:{}", a.name));
                }
                let active = if role.starts_with("raw") {
                    let rows = *a.shape.first().ok_or("KDA snapshot raw rows absent")?;
                    if rows == 0 {
                        return Err("KDA snapshot raw rows zero".into());
                    }
                    let stride = *a
                        .strides_bytes
                        .first()
                        .ok_or("KDA snapshot raw stride absent")?;
                    (tokens % rows)
                        .checked_mul(stride)
                        .ok_or("KDA snapshot active row extent overflow")?
                } else {
                    b.bytes.len()
                };
                if active > b.bytes.len() || active % 4 != 0 {
                    return Err("KDA snapshot active extent invalid".into());
                }
                for (i, v) in b.bytes.chunks_exact(4).enumerate() {
                    let bits =
                        u32::from_le_bytes(v.try_into().map_err(|_| "KDA snapshot F32 width")?);
                    if i * 4 < active {
                        if !f32::from_bits(bits).is_finite() {
                            return Err(format!("KDA snapshot active nonfinite:{}", a.name));
                        }
                    } else if bits != KDA_PENDING_INVALID_F32_BITS {
                        return Err(format!("KDA snapshot inactive canonical NaN:{}", a.name));
                    }
                }
            }
            _ => return Err(format!("KDA snapshot unknown semantic role:{}", a.name)),
        }
    }
    Ok(())
}
