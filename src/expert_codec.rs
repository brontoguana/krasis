//! Bit-exact, independently decodable rANS backing for Marlin expert payloads.
//!
//! The format is intentionally expert-contiguous: one host-to-device copy
//! carries an expert's task descriptors, lane offsets, states, and entropy
//! payload. A CUDA block decodes one output stripe; its lanes own independent
//! rANS states while writing coalesced bytes to the ordinary Marlin staging
//! layout. Packed weights use a 16-symbol nibble model. BF16 scale low/high
//! bytes use separate 256-symbol models so their very different statistics are
//! retained without changing a single output bit.

const RANS_SCALE_BITS: u32 = 12;
const RANS_TOTAL: u32 = 1 << RANS_SCALE_BITS;
const RANS_MASK: u32 = RANS_TOTAL - 1;
const RANS_L: u32 = 1 << 23;
pub const CODEC_LANES: usize = 128;
pub const CODEC_TABLES: usize = 3;
pub const CODEC_ALPHABET: usize = 256;
pub const CODEC_DECODE_SLOTS: usize = RANS_TOTAL as usize;
pub const MAX_EXPERT_CHUNKS: usize = 4;
pub const EXPERT_BLOB_MAGIC: u32 = u32::from_le_bytes(*b"KREC");
pub const EXPERT_BLOB_VERSION: u32 = 1;
const EXPERT_HEADER_WORDS: usize = 8;
const TASK_WORDS: usize = 4;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExpertChunkPlan {
    pub source_offset: usize,
    pub source_bytes: usize,
    pub task_start: usize,
    pub task_count: usize,
}

pub type CodecResult<T> = Result<T, String>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentKind {
    PackedNibbles,
    Bf16Scales,
}

#[derive(Clone, Copy, Debug)]
pub struct ExpertComponent<'a> {
    pub bytes: &'a [u8],
    pub kind: ComponentKind,
}

#[derive(Clone, Debug)]
pub struct CodecHistogram {
    counts: [[u64; CODEC_ALPHABET]; CODEC_TABLES],
}

impl Default for CodecHistogram {
    fn default() -> Self {
        Self {
            counts: [[0; CODEC_ALPHABET]; CODEC_TABLES],
        }
    }
}

impl CodecHistogram {
    pub fn observe(&mut self, component: ExpertComponent<'_>) {
        match component.kind {
            ComponentKind::PackedNibbles => {
                for &byte in component.bytes {
                    self.counts[0][usize::from(byte & 0x0f)] += 1;
                    self.counts[0][usize::from(byte >> 4)] += 1;
                }
            }
            ComponentKind::Bf16Scales => {
                for (index, &byte) in component.bytes.iter().enumerate() {
                    self.counts[1 + (index & 1)][usize::from(byte)] += 1;
                }
            }
        }
    }

    pub fn build_tables(&self) -> CodecResult<CodecTables> {
        Ok(CodecTables {
            tables: [
                RansTable::from_counts(&self.counts[0], 16)?,
                RansTable::from_counts(&self.counts[1], 256)?,
                RansTable::from_counts(&self.counts[2], 256)?,
            ],
        })
    }

    pub fn merge(&mut self, other: &Self) {
        for table in 0..CODEC_TABLES {
            for symbol in 0..CODEC_ALPHABET {
                self.counts[table][symbol] =
                    self.counts[table][symbol].saturating_add(other.counts[table][symbol]);
            }
        }
    }
}

#[derive(Clone, Debug)]
struct RansTable {
    frequencies: [u16; CODEC_ALPHABET],
    starts: [u16; CODEC_ALPHABET],
    decode_symbols: [u16; CODEC_DECODE_SLOTS],
    alphabet: usize,
}

impl RansTable {
    fn from_counts(counts: &[u64; CODEC_ALPHABET], alphabet: usize) -> CodecResult<Self> {
        if alphabet == 0 || alphabet > CODEC_ALPHABET {
            return Err(format!("invalid rANS alphabet size {alphabet}"));
        }
        let total_count = counts[..alphabet].iter().copied().sum::<u64>();
        if total_count == 0 {
            return Err("cannot build an rANS table from an empty histogram".to_string());
        }

        let mut frequencies = [0_u16; CODEC_ALPHABET];
        let mut remainders = Vec::with_capacity(alphabet);
        let mut frequency_sum = 0_u32;
        for symbol in 0..alphabet {
            let count = counts[symbol];
            if count == 0 {
                remainders.push((0_u128, symbol));
                continue;
            }
            let scaled = u128::from(count) * u128::from(RANS_TOTAL);
            let quotient = (scaled / u128::from(total_count)).max(1) as u32;
            let remainder = scaled % u128::from(total_count);
            frequencies[symbol] = u16::try_from(quotient)
                .map_err(|_| "normalized rANS frequency exceeds u16".to_string())?;
            frequency_sum += quotient;
            remainders.push((remainder, symbol));
        }

        if frequency_sum < RANS_TOTAL {
            remainders.sort_by(|left, right| right.cmp(left));
            let mut remaining = RANS_TOTAL - frequency_sum;
            let mut cursor = 0_usize;
            while remaining > 0 {
                let symbol = remainders[cursor % remainders.len()].1;
                if counts[symbol] > 0 {
                    frequencies[symbol] += 1;
                    remaining -= 1;
                }
                cursor += 1;
            }
        } else if frequency_sum > RANS_TOTAL {
            remainders.sort();
            let mut excess = frequency_sum - RANS_TOTAL;
            while excess > 0 {
                let mut changed = false;
                for &(_, symbol) in &remainders {
                    if frequencies[symbol] > 1 {
                        frequencies[symbol] -= 1;
                        excess -= 1;
                        changed = true;
                        if excess == 0 {
                            break;
                        }
                    }
                }
                if !changed {
                    return Err("rANS normalization cannot reduce frequencies to total".to_string());
                }
            }
        }

        Self::from_frequencies(frequencies, alphabet)
    }

    fn from_frequencies(frequencies: [u16; CODEC_ALPHABET], alphabet: usize) -> CodecResult<Self> {
        if alphabet == 0 || alphabet > CODEC_ALPHABET {
            return Err(format!("invalid rANS alphabet size {alphabet}"));
        }
        if frequencies[alphabet..]
            .iter()
            .any(|&frequency| frequency != 0)
        {
            return Err("rANS table has frequencies beyond its alphabet".to_string());
        }
        let frequency_sum = frequencies[..alphabet]
            .iter()
            .map(|&frequency| u32::from(frequency))
            .sum::<u32>();
        if frequency_sum != RANS_TOTAL {
            return Err(format!(
                "normalized rANS frequencies sum to {frequency_sum}, expected {RANS_TOTAL}"
            ));
        }

        let mut starts = [0_u16; CODEC_ALPHABET];
        let mut decode_symbols = [0_u16; CODEC_DECODE_SLOTS];
        let mut cursor = 0_usize;
        for symbol in 0..alphabet {
            starts[symbol] = u16::try_from(cursor)
                .map_err(|_| "rANS cumulative start exceeds u16".to_string())?;
            let frequency = usize::from(frequencies[symbol]);
            for slot in &mut decode_symbols[cursor..cursor + frequency] {
                *slot = symbol as u16;
            }
            cursor += frequency;
        }
        if cursor != CODEC_DECODE_SLOTS {
            return Err(format!(
                "normalized rANS table has {cursor} slots, expected {CODEC_DECODE_SLOTS}"
            ));
        }
        Ok(Self {
            frequencies,
            starts,
            decode_symbols,
            alphabet,
        })
    }

    fn encode(&self, symbols: &[u8]) -> CodecResult<Vec<u8>> {
        let mut state = RANS_L;
        let mut renormalized = Vec::new();
        for &symbol in symbols.iter().rev() {
            let symbol_index = usize::from(symbol);
            if symbol_index >= self.alphabet || self.frequencies[symbol_index] == 0 {
                return Err(format!(
                    "symbol {symbol_index} is absent from the rANS table"
                ));
            }
            let frequency = u32::from(self.frequencies[symbol_index]);
            let start = u32::from(self.starts[symbol_index]);
            let maximum = ((RANS_L >> RANS_SCALE_BITS) << 8) * frequency;
            while state >= maximum {
                renormalized.push(state as u8);
                state >>= 8;
            }
            state = ((state / frequency) << RANS_SCALE_BITS) + (state % frequency) + start;
        }
        let mut encoded = Vec::with_capacity(4 + renormalized.len());
        encoded.extend_from_slice(&state.to_le_bytes());
        encoded.extend(renormalized.into_iter().rev());
        Ok(encoded)
    }

    fn decode_symbol(&self, state: &mut u32, input: &[u8], cursor: &mut usize) -> CodecResult<u8> {
        let slot = (*state & RANS_MASK) as usize;
        let symbol = usize::from(self.decode_symbols[slot]);
        let frequency = u32::from(self.frequencies[symbol]);
        let start = u32::from(self.starts[symbol]);
        *state = frequency * (*state >> RANS_SCALE_BITS) + (slot as u32 - start);
        while *state < RANS_L {
            let byte = input
                .get(*cursor)
                .copied()
                .ok_or_else(|| "truncated rANS lane stream".to_string())?;
            *cursor += 1;
            *state = (*state << 8) | u32::from(byte);
        }
        Ok(symbol as u8)
    }
}

#[derive(Clone, Debug)]
pub struct CodecTables {
    tables: [RansTable; CODEC_TABLES],
}

impl CodecTables {
    pub const SERIALIZED_FREQUENCIES_BYTES: usize =
        CODEC_TABLES * CODEC_ALPHABET * std::mem::size_of::<u16>();

    pub fn serialized_frequencies(&self) -> Vec<u8> {
        self.tables
            .iter()
            .flat_map(|table| table.frequencies.iter())
            .flat_map(|frequency| frequency.to_le_bytes())
            .collect()
    }

    pub fn from_serialized_frequencies(bytes: &[u8]) -> CodecResult<Self> {
        if bytes.len() != Self::SERIALIZED_FREQUENCIES_BYTES {
            return Err(format!(
                "serialized codec table is {} bytes, expected {}",
                bytes.len(),
                Self::SERIALIZED_FREQUENCIES_BYTES,
            ));
        }
        let mut tables = Vec::with_capacity(CODEC_TABLES);
        for table_index in 0..CODEC_TABLES {
            let mut frequencies = [0_u16; CODEC_ALPHABET];
            for (symbol, frequency) in frequencies.iter_mut().enumerate() {
                let offset = (table_index * CODEC_ALPHABET + symbol) * 2;
                *frequency = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            }
            tables.push(RansTable::from_frequencies(
                frequencies,
                if table_index == 0 { 16 } else { 256 },
            )?);
        }
        Ok(Self {
            tables: tables.try_into().unwrap(),
        })
    }

    pub fn gpu_decode_symbols(&self) -> Vec<u16> {
        self.tables
            .iter()
            .flat_map(|table| table.decode_symbols.iter().copied())
            .collect()
    }

    pub fn gpu_frequencies(&self) -> Vec<u16> {
        self.tables
            .iter()
            .flat_map(|table| table.frequencies.iter().copied())
            .collect()
    }

    pub fn gpu_starts(&self) -> Vec<u16> {
        self.tables
            .iter()
            .flat_map(|table| table.starts.iter().copied())
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
struct Task {
    output_offset: u32,
    output_bytes: u32,
    lane_offsets_index: u32,
    mode: u32,
}

#[derive(Clone, Debug)]
pub struct EncodedExpert {
    pub blob: Vec<u8>,
    pub original_bytes: usize,
    pub task_count: usize,
}

impl EncodedExpert {
    pub fn ratio(&self) -> f64 {
        self.blob.len() as f64 / self.original_bytes as f64
    }

    pub fn decode_cpu(&self, tables: &CodecTables) -> CodecResult<Vec<u8>> {
        let header = parse_header(&self.blob)?;
        let mut output = vec![0_u8; header.original_bytes];
        for task_index in 0..header.task_count {
            let task = read_task(&self.blob, header.task_offset, task_index)?;
            for lane in 0..CODEC_LANES {
                let lane_offset_word = task
                    .lane_offsets_index
                    .checked_add(lane as u32)
                    .ok_or_else(|| "lane offset index overflow".to_string())?;
                let lane_offset = read_u32(
                    &self.blob,
                    header.lane_offsets_offset + usize::try_from(lane_offset_word).unwrap() * 4,
                )? as usize;
                let state_bytes = self
                    .blob
                    .get(lane_offset..lane_offset + 4)
                    .ok_or_else(|| "missing rANS lane state".to_string())?;
                let mut state = u32::from_le_bytes(state_bytes.try_into().unwrap());
                let mut input_cursor = lane_offset + 4;
                let output_start = usize::try_from(task.output_offset).unwrap();
                let output_end = output_start + usize::try_from(task.output_bytes).unwrap();
                let mut output_index = output_start + lane;
                while output_index < output_end {
                    let table_index = if task.mode == 0 { 0 } else { 1 + (lane & 1) };
                    let table = &tables.tables[table_index];
                    output[output_index] = if task.mode == 0 {
                        let low = table.decode_symbol(&mut state, &self.blob, &mut input_cursor)?;
                        let high =
                            table.decode_symbol(&mut state, &self.blob, &mut input_cursor)?;
                        low | (high << 4)
                    } else {
                        table.decode_symbol(&mut state, &self.blob, &mut input_cursor)?
                    };
                    output_index += CODEC_LANES;
                }
            }
        }
        Ok(output)
    }
}

pub fn encode_expert(
    components: &[ExpertComponent<'_>],
    tables: &CodecTables,
    lane_bytes: usize,
) -> CodecResult<EncodedExpert> {
    if components.is_empty() {
        return Err("expert codec requires at least one component".to_string());
    }
    if lane_bytes == 0 {
        return Err("lane_bytes must be non-zero".to_string());
    }
    let task_output_capacity = CODEC_LANES
        .checked_mul(lane_bytes)
        .ok_or_else(|| "expert codec task size overflow".to_string())?;
    let original_bytes = components.iter().try_fold(0_usize, |total, component| {
        total
            .checked_add(component.bytes.len())
            .ok_or_else(|| "expert source size overflow".to_string())
    })?;
    if original_bytes == 0 || original_bytes > u32::MAX as usize {
        return Err(format!("unsupported expert source size {original_bytes}"));
    }

    let mut tasks = Vec::<Task>::new();
    let mut lane_streams = Vec::<Vec<u8>>::new();
    let mut component_output_offset = 0_usize;
    for component in components {
        let mode = match component.kind {
            ComponentKind::PackedNibbles => 0,
            ComponentKind::Bf16Scales => 1,
        };
        let mut component_offset = 0_usize;
        while component_offset < component.bytes.len() {
            let task_bytes = task_output_capacity.min(component.bytes.len() - component_offset);
            let lane_offsets_index = u32::try_from(lane_streams.len())
                .map_err(|_| "too many expert codec lane streams".to_string())?;
            tasks.push(Task {
                output_offset: u32::try_from(component_output_offset + component_offset)
                    .map_err(|_| "expert task output offset exceeds u32".to_string())?,
                output_bytes: u32::try_from(task_bytes)
                    .map_err(|_| "expert task byte count exceeds u32".to_string())?,
                lane_offsets_index,
                mode,
            });
            for lane in 0..CODEC_LANES {
                let mut symbols = Vec::with_capacity(lane_bytes * if mode == 0 { 2 } else { 1 });
                let mut source_index = component_offset + lane;
                while source_index < component_offset + task_bytes {
                    let byte = component.bytes[source_index];
                    if mode == 0 {
                        symbols.push(byte & 0x0f);
                        symbols.push(byte >> 4);
                    } else {
                        symbols.push(byte);
                    }
                    source_index += CODEC_LANES;
                }
                let table_index = if mode == 0 { 0 } else { 1 + (lane & 1) };
                lane_streams.push(tables.tables[table_index].encode(&symbols)?);
            }
            component_offset += task_bytes;
        }
        component_output_offset += component.bytes.len();
    }

    let task_offset = EXPERT_HEADER_WORDS * 4;
    let lane_offsets_offset = task_offset + tasks.len() * TASK_WORDS * 4;
    let payload_offset = lane_offsets_offset + lane_streams.len() * 4;
    let total_bytes = lane_streams
        .iter()
        .try_fold(payload_offset, |total, stream| {
            total
                .checked_add(stream.len())
                .ok_or_else(|| "encoded expert size overflow".to_string())
        })?;
    if total_bytes > u32::MAX as usize {
        return Err("encoded expert exceeds u32 addressable format".to_string());
    }

    let mut blob = Vec::with_capacity(total_bytes);
    for word in [
        EXPERT_BLOB_MAGIC,
        EXPERT_BLOB_VERSION,
        original_bytes as u32,
        tasks.len() as u32,
        task_offset as u32,
        lane_offsets_offset as u32,
        payload_offset as u32,
        total_bytes as u32,
    ] {
        blob.extend_from_slice(&word.to_le_bytes());
    }
    for task in &tasks {
        for word in [
            task.output_offset,
            task.output_bytes,
            task.lane_offsets_index,
            task.mode,
        ] {
            blob.extend_from_slice(&word.to_le_bytes());
        }
    }
    let mut stream_offset = payload_offset;
    for stream in &lane_streams {
        blob.extend_from_slice(&(stream_offset as u32).to_le_bytes());
        stream_offset += stream.len();
    }
    for stream in lane_streams {
        blob.extend_from_slice(&stream);
    }
    if blob.len() != total_bytes {
        return Err(format!(
            "encoded expert length {} does not match planned {total_bytes}",
            blob.len()
        ));
    }
    Ok(EncodedExpert {
        blob,
        original_bytes,
        task_count: tasks.len(),
    })
}

/// Split an encoded expert into the same component runs as the canonical
/// Marlin payload. The first range includes the shared blob metadata; later
/// ranges contain only their contiguous lane streams. This lets the runtime
/// overlap H2D and decode within one expert while retaining at most the four
/// copies already used by the uncompressed component layout.
pub fn plan_expert_chunks(
    blob: &[u8],
    component_bytes: [usize; MAX_EXPERT_CHUNKS],
) -> CodecResult<[ExpertChunkPlan; MAX_EXPERT_CHUNKS]> {
    if component_bytes.iter().any(|&bytes| bytes == 0) {
        return Err("expert compression components must all be non-empty".to_string());
    }
    let header = parse_header(blob)?;
    let declared_bytes = component_bytes.iter().try_fold(0_usize, |total, &bytes| {
        total
            .checked_add(bytes)
            .ok_or_else(|| "expert compression component length overflow".to_string())
    })?;
    if declared_bytes != header.original_bytes {
        return Err(format!(
            "expert compression components total {declared_bytes} bytes, blob declares {}",
            header.original_bytes,
        ));
    }

    let mut plans = [ExpertChunkPlan::default(); MAX_EXPERT_CHUNKS];
    let mut task_cursor = 0_usize;
    let mut output_cursor = 0_usize;
    for (component_index, &bytes) in component_bytes.iter().enumerate() {
        let component_end = output_cursor
            .checked_add(bytes)
            .ok_or_else(|| "expert compression component boundary overflow".to_string())?;
        let task_start = task_cursor;
        while task_cursor < header.task_count {
            let task = read_task(blob, header.task_offset, task_cursor)?;
            let task_output = task.output_offset as usize;
            if task_output >= component_end {
                break;
            }
            if task_output != output_cursor {
                return Err(format!(
                    "expert compression task {task_cursor} starts at {task_output}, expected {output_cursor}",
                ));
            }
            let task_bytes = task.output_bytes as usize;
            if task_bytes == 0 {
                return Err(format!(
                    "expert compression task {task_cursor} has zero output bytes"
                ));
            }
            output_cursor = output_cursor
                .checked_add(task_bytes)
                .ok_or_else(|| "expert compression task output overflow".to_string())?;
            if output_cursor > component_end {
                return Err(format!(
                    "expert compression task {task_cursor} crosses component boundary {component_end}",
                ));
            }
            task_cursor += 1;
        }
        if output_cursor != component_end || task_cursor == task_start {
            return Err(format!(
                "expert compression component {component_index} ended at {output_cursor}, expected {component_end}",
            ));
        }
        plans[component_index].task_start = task_start;
        plans[component_index].task_count = task_cursor - task_start;
    }
    if task_cursor != header.task_count || output_cursor != header.original_bytes {
        return Err(format!(
            "expert compression chunk plan consumed {task_cursor}/{} tasks and {output_cursor}/{} bytes",
            header.task_count, header.original_bytes,
        ));
    }

    let task_payload_start = |task_index: usize| -> CodecResult<usize> {
        let task = read_task(blob, header.task_offset, task_index)?;
        let lane_word = task.lane_offsets_index as usize;
        let lane_offset_pos = header
            .lane_offsets_offset
            .checked_add(
                lane_word
                    .checked_mul(std::mem::size_of::<u32>())
                    .ok_or_else(|| "expert compression lane-offset overflow".to_string())?,
            )
            .ok_or_else(|| "expert compression lane-offset overflow".to_string())?;
        Ok(read_u32(blob, lane_offset_pos)? as usize)
    };
    for component_index in 0..MAX_EXPERT_CHUNKS {
        let source_offset = if component_index == 0 {
            0
        } else {
            task_payload_start(plans[component_index].task_start)?
        };
        let source_end = if component_index + 1 == MAX_EXPERT_CHUNKS {
            blob.len()
        } else {
            task_payload_start(plans[component_index + 1].task_start)?
        };
        if source_end <= source_offset || source_end > blob.len() {
            return Err(format!(
                "expert compression component {component_index} has invalid source range {source_offset}..{source_end} for {} bytes",
                blob.len(),
            ));
        }
        plans[component_index].source_offset = source_offset;
        plans[component_index].source_bytes = source_end - source_offset;
    }
    if plans[0].source_offset != 0
        || plans
            .windows(2)
            .any(|pair| pair[0].source_offset + pair[0].source_bytes != pair[1].source_offset)
        || plans[MAX_EXPERT_CHUNKS - 1].source_offset + plans[MAX_EXPERT_CHUNKS - 1].source_bytes
            != blob.len()
    {
        return Err("expert compression chunk source ranges are not contiguous".to_string());
    }
    Ok(plans)
}

struct ParsedHeader {
    original_bytes: usize,
    task_count: usize,
    task_offset: usize,
    lane_offsets_offset: usize,
}

fn parse_header(blob: &[u8]) -> CodecResult<ParsedHeader> {
    if blob.len() < EXPERT_HEADER_WORDS * 4 {
        return Err("encoded expert header is truncated".to_string());
    }
    if read_u32(blob, 0)? != EXPERT_BLOB_MAGIC {
        return Err("encoded expert magic mismatch".to_string());
    }
    if read_u32(blob, 4)? != EXPERT_BLOB_VERSION {
        return Err("encoded expert version mismatch".to_string());
    }
    let original_bytes = read_u32(blob, 8)? as usize;
    let task_count = read_u32(blob, 12)? as usize;
    let task_offset = read_u32(blob, 16)? as usize;
    let lane_offsets_offset = read_u32(blob, 20)? as usize;
    let total_bytes = read_u32(blob, 28)? as usize;
    if total_bytes != blob.len() {
        return Err(format!(
            "encoded expert declares {total_bytes} bytes but has {}",
            blob.len()
        ));
    }
    let task_end = task_offset
        .checked_add(task_count * TASK_WORDS * 4)
        .ok_or_else(|| "expert task range overflow".to_string())?;
    let lane_end = lane_offsets_offset
        .checked_add(task_count * CODEC_LANES * 4)
        .ok_or_else(|| "expert lane-offset range overflow".to_string())?;
    if task_offset < EXPERT_HEADER_WORDS * 4
        || lane_offsets_offset < task_end
        || lane_end > blob.len()
    {
        return Err("encoded expert metadata ranges are invalid".to_string());
    }
    Ok(ParsedHeader {
        original_bytes,
        task_count,
        task_offset,
        lane_offsets_offset,
    })
}

fn read_task(blob: &[u8], task_offset: usize, task_index: usize) -> CodecResult<Task> {
    let base = task_offset + task_index * TASK_WORDS * 4;
    Ok(Task {
        output_offset: read_u32(blob, base)?,
        output_bytes: read_u32(blob, base + 4)?,
        lane_offsets_index: read_u32(blob, base + 8)?,
        mode: read_u32(blob, base + 12)?,
    })
}

fn read_u32(bytes: &[u8], offset: usize) -> CodecResult<u32> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| format!("missing u32 at encoded expert offset {offset}"))?;
    Ok(u32::from_le_bytes(raw.try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rans_expert_round_trip_preserves_component_layout() {
        let packed_a: Vec<u8> = (0..19_331)
            .map(|i| ((i * 17 + i / 11) & 0xff) as u8)
            .collect();
        let scales_a: Vec<u8> = (0..1_286)
            .map(|i| {
                if i & 1 == 0 {
                    (i * 29) as u8
                } else {
                    0x3d + (i % 3) as u8
                }
            })
            .collect();
        let packed_b: Vec<u8> = (0..11_007)
            .map(|i| ((i * 7 + i / 5) & 0xff) as u8)
            .collect();
        let scales_b: Vec<u8> = (0..734)
            .map(|i| if i & 1 == 0 { (i * 13) as u8 } else { 0x3e })
            .collect();
        let components = [
            ExpertComponent {
                bytes: &packed_a,
                kind: ComponentKind::PackedNibbles,
            },
            ExpertComponent {
                bytes: &scales_a,
                kind: ComponentKind::Bf16Scales,
            },
            ExpertComponent {
                bytes: &packed_b,
                kind: ComponentKind::PackedNibbles,
            },
            ExpertComponent {
                bytes: &scales_b,
                kind: ComponentKind::Bf16Scales,
            },
        ];
        let mut histogram = CodecHistogram::default();
        for component in components {
            histogram.observe(component);
        }
        let tables = histogram.build_tables().unwrap();
        let encoded = encode_expert(&components, &tables, 37).unwrap();
        let decoded = encoded.decode_cpu(&tables).unwrap();
        let expected: Vec<u8> = components
            .iter()
            .flat_map(|component| component.bytes.iter().copied())
            .collect();
        assert_eq!(decoded, expected);

        let serialized = tables.serialized_frequencies();
        let restored_tables = CodecTables::from_serialized_frequencies(&serialized).unwrap();
        assert_eq!(
            encode_expert(&components, &restored_tables, 37)
                .unwrap()
                .decode_cpu(&restored_tables)
                .unwrap(),
            expected,
        );

        let plan = plan_expert_chunks(
            &encoded.blob,
            [
                packed_a.len(),
                scales_a.len(),
                packed_b.len(),
                scales_b.len(),
            ],
        )
        .unwrap();
        assert_eq!(plan[0].source_offset, 0);
        assert_eq!(
            plan.iter().map(|chunk| chunk.source_bytes).sum::<usize>(),
            encoded.blob.len(),
        );
        assert_eq!(
            plan.iter().map(|chunk| chunk.task_count).sum::<usize>(),
            encoded.task_count,
        );
        for pair in plan.windows(2) {
            assert_eq!(
                pair[0].source_offset + pair[0].source_bytes,
                pair[1].source_offset,
            );
            assert_eq!(pair[0].task_start + pair[0].task_count, pair[1].task_start,);
        }
    }
}
