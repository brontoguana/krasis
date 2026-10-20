//! Offline coefficient-preserving controls; never creates a production cache.
use super::*;
use std::io::Write;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CropPlan {
    source_plan: Identity,
    fixture_limit: usize,
    output_raw_limit_per_gpu: usize,
}
fn inverse(p: &[usize]) -> Vec<usize> {
    let mut out = vec![usize::MAX; p.len()];
    for (i, &v) in p.iter().enumerate() {
        assert_eq!(out[v], usize::MAX);
        out[v] = i;
    }
    out
}
fn packed_index(n: usize, k: usize, rows: usize, inv: &[usize]) -> usize {
    (k / 16) * rows * 16 + (n / 64) * 1024 + inv[(n % 64 / 16) * 256 + (k % 16) * 16 + n % 16]
}
fn scale_index(n: usize, k: usize, rows: usize, group: usize, inv: &[usize]) -> usize {
    (k / group) * rows + (n / inv.len()) * inv.len() + inv[n % inv.len()]
}
fn code(b: &[u8], slot: usize, bits: usize) -> u32 {
    let pack = 32 / bits;
    let word = u32::from_le_bytes(b[slot / pack * 4..slot / pack * 4 + 4].try_into().unwrap());
    (word >> ((slot % pack) * bits)) & ((1 << bits) - 1)
}
fn scalar(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
fn write_bytes(out: &Path, name: &str, bytes: &[u8]) -> Result<Identity, String> {
    let path = out.join(name);
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| e.to_string())?;
    f.write_all(bytes).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())?;
    Identity::of(&path)
}
pub fn execute(path: &Path, out: &Path) -> Result<Value, String> {
    let cp: CropPlan = read_json(path)?;
    let source: Plan =
        serde_json::from_slice(&cp.source_plan.read()?).map_err(|e| e.to_string())?;
    // Fixed test geometries, not runtime dispatch/budget values. Includes non-power-of-two K,
    // channelwise scales, group32/64, large layout tiles and generic unaligned groups.
    let geometries = [
        (64, 16, 16),
        (128, 32, 32),
        (192, 128, 128),
        (64, 48, 16),
        (128, 160, 32),
        (192, 256, 64),
        (1024, 2048, 128),
        (64, 16, 3),
    ];
    let sources: Vec<_> = ["shared-w1", "routed-w1-expert226"]
        .iter()
        .map(|name| {
            source
                .cases
                .iter()
                .find(|c| c.name == *name)
                .ok_or_else(|| format!("missing actual source {name}"))
        })
        .collect::<Result<_, _>>()?;
    let raw_bound = sources.len()
        * geometries
            .iter()
            .map(|(n, _, _)| n * 67 * 6 * 4)
            .sum::<usize>();
    let fixture_bound = sources
        .iter()
        .map(|src| {
            geometries
                .iter()
                .map(|&(n, k, g)| n * k * src.bits / 8 + k.div_ceil(g) * n * 2 + k * 2 + n * 16)
                .sum::<usize>()
        })
        .sum::<usize>();
    if raw_bound > cp.output_raw_limit_per_gpu || fixture_bound > cp.fixture_limit {
        return Err("prospective crop extent exceeds admission".into());
    }
    let (sg, sc) = layout_reference::generate_scale_perms();
    let mut cases = Vec::new();
    let mut proofs = Vec::new();
    let mut bytes = 0usize;
    for src in sources {
        if src.n % 64 != 0 || src.k % 16 != 0 || ![4, 8].contains(&src.bits) || src.group == 0 {
            return Err("invalid actual source geometry".into());
        }
        let p = src.packed.read()?;
        let s = u16s(&src.scales.read()?)?;
        let x = u16s(&src.input.read()?)?;
        if p.len() != src.n * src.k * src.bits / 8
            || s.len() != src.k.div_ceil(src.group) * src.n
            || src.input_offset % 2 != 0
            || x.len() < src.input_offset / 2 + src.k
        {
            return Err("actual source extents invalid".into());
        }
        let wp = if src.bits == 4 {
            layout_reference::generate_weight_perm_int4()
        } else {
            layout_reference::generate_weight_perm_int8()
        };
        let iw = inverse(&wp);
        let source_si = inverse(if src.group < src.k { &sg[..] } else { &sc[..] });
        for &(n, k, g) in &geometries {
            let old_cols: Vec<_> = (0..k)
                .map(|j| {
                    if g == 3 {
                        (j / g) * src.group + j % g
                    } else {
                        j
                    }
                })
                .collect();
            if n > src.n || old_cols.iter().any(|&j| j >= src.k) {
                return Err("crop outside source".into());
            }
            let si = inverse(if g < k { &sg[..] } else { &sc[..] });
            let mut words = vec![0u32; n * k / (32 / src.bits)];
            let mut scales = vec![0u16; k.div_ceil(g) * n];
            let input: Vec<_> = old_cols
                .iter()
                .map(|&j| x[src.input_offset / 2 + j])
                .collect();
            let mut refs = vec![0f64; n];
            let mut abs = vec![0f64; n];
            for row in 0..n {
                for col in 0..k {
                    let old = old_cols[col];
                    let q = code(&p, packed_index(row, old, src.n, &iw), src.bits);
                    let sb = s[scale_index(row, old, src.n, src.group, &source_si)];
                    let slot = packed_index(row, col, n, &iw);
                    let pack = 32 / src.bits;
                    words[slot / pack] |= q << ((slot % pack) * src.bits);
                    let scale_slot = scale_index(row, col, n, g, &si);
                    // Every proposed new scale group must preserve all actual source coefficients.
                    let first = (col / g) * g;
                    if sb != s[scale_index(row, old_cols[first], src.n, src.group, &source_si)] {
                        return Err("crop group spans different actual scales".into());
                    }
                    scales[scale_slot] = sb;
                    let weight = ((q as i32 - (1 << (src.bits - 1))) as f32) * scalar(sb);
                    let product = weight as f64 * scalar(input[col]) as f64;
                    if !product.is_finite() {
                        return Err("nonfinite actual coefficient".into());
                    }
                    refs[row] += product;
                    abs[row] += product.abs();
                }
            }
            let packed: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            // Decode the entire new representation independently before publishing any fixture.
            for row in 0..n {
                for col in 0..k {
                    if code(&packed, packed_index(row, col, n, &iw), src.bits)
                        != code(&p, packed_index(row, old_cols[col], src.n, &iw), src.bits)
                        || scales[scale_index(row, col, n, g, &si)]
                            != s[scale_index(row, old_cols[col], src.n, src.group, &source_si)]
                    {
                        return Err("new coefficient differs source".into());
                    }
                }
            }
            let name = format!("{}-n{n}-k{k}-g{g}", src.name);
            let input_id = write_bytes(
                out,
                &format!("{name}-input.bf16"),
                &input
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let packed_id = write_bytes(out, &format!("{name}-packed.bin"), &packed)?;
            let scales_id = write_bytes(
                out,
                &format!("{name}-scales.bf16"),
                &scales
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let reference_id = write_bytes(
                out,
                &format!("{name}-reference.f64"),
                &refs
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let abs_id = write_bytes(
                out,
                &format!("{name}-abs-products.f64"),
                &abs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            )?;
            bytes += input_id.bytes
                + packed_id.bytes
                + scales_id.bytes
                + reference_id.bytes
                + abs_id.bytes;
            proofs.push(json!({"name":name,"coefficientsPreserved":n*k,"source":src,"sourceColumns":old_cols,"absoluteProducts":abs_id}));
            cases.push(Case {
                name,
                n,
                k,
                bits: src.bits,
                group: g,
                input: input_id,
                input_offset: 0,
                packed: packed_id,
                scales: scales_id,
                reference: reference_id,
            });
        }
    }
    if bytes != fixture_bound {
        return Err("crop extent reconciliation failed".into());
    }
    save(
        &out.join("CASES.json"),
        &serde_json::to_value(&cases).map_err(|e| e.to_string())?,
    )?;
    Ok(
        json!({"passed":true,"sourcePlan":cp.source_plan,"fixtureBytes":bytes,"rawBoundPerGpu":raw_bound,"proofs":proofs,"scope":"Actual coefficient/input selection and native-codec repacking only. Independent F64 full logical F32 quant-weight dots; no new quantization, model prompts, runtime cache or performance representative claim."}),
    )
}
