// Cold capture binding identity. Input values change per token; owned storage,
// immutable preparation parameters and workspace geometry must stay captured.
// This identity does not establish producer mutual exclusion or GPU ordering.
fn model_workspace_identity(b: &ModelFamilyBinding<'_>) -> Result<serde_json::Value, String> {
    let producer = b.producer;
    let arena = b.owner.arena.as_ref();
    producer.raw.schema.validate()?;
    producer.prepared.schema.validate()?;
    arena.raw.schema.validate()?;
    let arrays = arena.arrays.iter().map(|(role, buf)| {
        Ok(serde_json::json!({"role":role,"dtype":"float32","ptr":pointer(buf),
            "elements":buf.len(),"bytes":buf.len().checked_mul(std::mem::size_of::<f32>()).ok_or("capture F32 span overflow")?}))
    }).collect::<Result<Vec<_>, String>>()?;
    let powers = arena.powers.iter().map(|(role, buf)| {
        Ok(serde_json::json!({"role":role,"dtype":"int32","ptr":pointer(buf),
            "elements":buf.len(),"bytes":buf.len().checked_mul(std::mem::size_of::<i32>()).ok_or("capture I32 span overflow")?}))
    }).collect::<Result<Vec<_>, String>>()?;
    Ok(serde_json::json!({
        "producer":{
            "arithmetic":producer.arithmetic_sha256,"device":producer.raw.device.ordinal(),
            "context":*producer.raw.device.cu_primary_ctx() as usize,
            "raw_stream":producer.raw.stream.handle as usize,"prepared_stream":producer.prepared.stream.handle as usize,
            "source_epoch":producer.source_epoch,"input_capacity":producer.input_capacity,"prepared_capacity":producer.capacity,
            "raw_schema":producer.raw.schema,"raw_planes":producer.raw.inventory(),
            "prepared_schema":producer.prepared.schema,"prepared_planes":producer.prepared.inventory(),
            "parameter_spans":producer.parameter_admissions,
            "a_log":{"ptr":pointer(&producer.a_log),"elements":producer.a_log.len(),"bytes":producer.a_log.len().checked_mul(std::mem::size_of::<f32>()).ok_or("capture A span overflow")?},
            "bias":{"ptr":pointer(&producer.bias),"elements":producer.bias.len(),"bytes":producer.bias.len().checked_mul(std::mem::size_of::<f32>()).ok_or("capture bias span overflow")?},
            "raw_admissions":producer.raw.admissions,"prepared_admissions":producer.prepared.admissions,
            "kernel_ptx":producer.kernels.ptx_sha256
        },
        "recurrence_arena":{
            "arithmetic":arena.arithmetic_sha256,"device":arena.device.ordinal(),
            "context":*arena.device.cu_primary_ctx() as usize,"stream":arena.stream.handle as usize,
            "heads":arena.heads,"dim":arena.dim,"capacity":arena.max_new_rows,"padded":arena.padded,
            "raw_schema":arena.raw.schema,"raw_planes":arena.raw.inventory(),"raw_admissions":arena.raw.admissions,
            "arrays":arrays,"powers":powers,"workspace_admissions":arena.native_admissions,
            "bf16_emitted":{"ptr":pointer(&arena.bf16_emitted),"elements":arena.bf16_emitted.len(),"bytes":arena.bf16_emitted.len().checked_mul(std::mem::size_of::<u16>()).ok_or("capture BF16 span overflow")?},
            "count":pointer(&arena.count),"raw_count":pointer(&arena.raw_count),"one":pointer(&arena.one),
            "kernel_roles":arena.functions.keys().copied().collect::<Vec<_>>()
        }
    }))
}
