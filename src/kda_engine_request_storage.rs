// Explicit diagnostic current-storage lifetime; no arithmetic selector.
pub(crate) struct KdaEngineRequestStorage {
    inner: std::sync::Mutex<KdaEngineRequestStorageState>,
    layers: Vec<usize>,
    pub(crate) reserve: usize,
    guard_prefill_writers: bool,
    shadow: Option<KdaActualPrefillShadowPlan>,
    serving_guards: Vec<PreparedLogitGuard>,
    serving_execution: std::sync::Mutex<Option<ServingExecutionDomain>>,
    serving_allocations: Vec<crate::session_cache::SequenceStateAllocation>,
    serving_logit_tickets: std::sync::Mutex<Vec<ServingLogitTicket>>,
}
struct KdaEngineRequestStorageState {
    // Domain timelines drain/drop while all lifetime family/arena owners live.
    resource_domain: ProducerResourceDomain,
    domain_generation: u64,
    live_sequence_epoch: u64,
    poisoned: bool,
    lifetime: KdaPrefillStorageLifetime,
    static_identity: serde_json::Value,
    events: Vec<serde_json::Value>,
    prefill_writer_phases: u64,
    shadow_records: Vec<serde_json::Value>,
    shadow_started: bool,
}
impl KdaEngineRequestStorage {
    pub(crate) fn new(
        lifetime: KdaPrefillStorageLifetime,
        layers: Vec<usize>,
        reserve: usize,
        guard_prefill_writers: bool,
        shadow: Option<KdaActualPrefillShadowPlan>,
    ) -> Result<Arc<Self>, String> {
        if lifetime.layers != layers || !lifetime.explicitly_retired()? {
            return Err("engine storage needs complete initially retired inventory".into());
        }
        if let Some(plan) = shadow.as_ref() {
            if !guard_prefill_writers || plan.rows == 0 {
                return Err(
                    "actual shadow requires complete writer phase and positive rows".into(),
                );
            }
            std::fs::create_dir(&plan.output)
                .map_err(|e| format!("actual shadow fresh output:{e}"))?;
        }
        let static_identity = request_storage_static_identity(&lifetime)?;
        let mut resource_domain =
            ProducerResourceDomain::build_members(current_resource_members(&lifetime)?, None)?;
        current_domain_admit_streams(&mut resource_domain, &current_lifetime_streams(&lifetime)?)?;
        Ok(Arc::new(Self {
            inner: std::sync::Mutex::new(KdaEngineRequestStorageState {
                resource_domain,
                domain_generation: 0,
                live_sequence_epoch: 0,
                poisoned: false,
                lifetime,
                static_identity,
                events: Vec::new(),
                prefill_writer_phases: 0,
                shadow_records: Vec::new(),
                shadow_started: false,
            }),
            layers,
            reserve,
            guard_prefill_writers,
            shadow,
            serving_guards: Vec::new(),
            serving_execution: std::sync::Mutex::new(None),
            serving_allocations: Vec::new(),
            serving_logit_tickets: std::sync::Mutex::new(Vec::new()),
        }))
    }
    pub(crate) fn layers(&self) -> &[usize] {
        &self.layers
    }
    pub(crate) fn retire_before_release(&self, reason: &str) -> Result<(), String> {
        let mut state = self
            .inner
            .try_lock()
            .map_err(|_| "engine request source lifecycle concurrently leased/poisoned")?;
        if state.poisoned {
            return Err("current engine resource storage poisoned".into());
        }
        if state.lifetime.explicitly_retired()? {
            return Ok(());
        }
        let before = state.lifetime.current_prefill_epochs()?;
        let expected = workspace_inventory_identity(state.lifetime.bound()?, &self.layers)?;
        let (after, retained_decode_wrappers) =
            current_storage_retire(&mut state, &expected, &before)?;
        if request_storage_static_identity(&state.lifetime)? != state.static_identity {
            current_storage_poison(&mut state);
            return Err(
                "engine retirement changed retained preparation/state/params/decode binding".into(),
            );
        }
        let inventory = state.lifetime.storage_identity()?;
        state.events.push(serde_json::json!({"transition":"retire","reason":reason,"epochs_before":before,"epochs_after":after,"inventory":inventory,"all_former_prefill_wrapper_leases_released":retained_decode_wrappers==0,"prefill_wrappers_intentionally_retained_by_decode":retained_decode_wrappers}));
        Ok(())
    }
    pub(crate) fn attach_after_prepare(
        &self,
        sources: Vec<RetainedPrefillKdaSource>,
        rows: usize,
    ) -> Result<(), String> {
        let mut state = self
            .inner
            .try_lock()
            .map_err(|_| "engine request source lifecycle concurrently leased/poisoned")?;
        if state.poisoned {
            return Err("current engine resource storage poisoned".into());
        }
        if !state.lifetime.explicitly_retired()? {
            return Err("engine preparation did not retire previous sources".into());
        }
        if rows == 0 || sources.len() != self.layers.len() {
            return Err("engine preparation current source inventory empty/incomplete".into());
        }
        for source in &sources {
            if source
                .inner
                .schema
                .planes
                .first()
                .and_then(|p| p.shape.first())
                .copied()
                != Some(rows)
            {
                return Err(
                    "engine preparation source row extent differs from actual scratch".into(),
                );
            }
        }
        let before = state.lifetime.current_prefill_epochs()?;
        let after = current_storage_attach(&mut state, sources, &before)?;
        if request_storage_static_identity(&state.lifetime)? != state.static_identity {
            current_storage_poison(&mut state);
            return Err(
                "engine attachment changed retained preparation/state/params/decode binding".into(),
            );
        }
        let inventory = state.lifetime.storage_identity()?;
        state.events.push(serde_json::json!({"transition":"attach","scratch_rows":rows,"epochs_before":before,"epochs_after":after,"inventory":inventory,"complete_current_actual_sources":true}));
        Ok(())
    }
    pub(crate) fn audit(
        &self,
    ) -> Result<
        (
            crate::session_cache::SequenceStateRegistry,
            serde_json::Value,
        ),
        String,
    > {
        let state = self
            .inner
            .try_lock()
            .map_err(|_| "engine request source lifecycle concurrently leased/poisoned")?;
        if state.poisoned {
            return Err("current engine resource storage poisoned".into());
        }
        if request_storage_static_identity(&state.lifetime)? != state.static_identity {
            return Err("engine request retained static identity changed".into());
        }
        Ok((
            state.lifetime.registry()?,
            serde_json::json!({"passed":true,"layers":self.layers,"phase":if state.lifetime.explicitly_retired()?{"retired"}else{"bound"},"resource_domain":current_domain_audit(&state.resource_domain)?,"domain_generation":state.domain_generation,"live_sequence_epoch":state.live_sequence_epoch,"current_identity":state.lifetime.storage_identity()?,"static_identity":state.static_identity,"events":state.events,"guard_prefill_writers":self.guard_prefill_writers,"prefill_writer_phases":state.prefill_writer_phases,"shadow_records":state.shadow_records,"selected_for_serving":false}),
        ))
    }
}
fn request_storage_static_identity(
    life: &KdaPrefillStorageLifetime,
) -> Result<serde_json::Value, String> {
    fn row(
        layer: usize,
        prefill: &PreparedOwner,
        decode: &PreparedOwner,
        producer: &KdaRawProducer,
        prepared: &NativeRawOwner,
        a_log: &GpuBuf<f32>,
        bias: &GpuBuf<f32>,
        arithmetic: &str,
        kernel: &str,
        lower: f32,
        scale: f32,
        epsilon: f32,
    ) -> Result<serde_json::Value, String> {
        let arena = model_workspace_identity(&ModelFamilyBinding {
            layer,
            owner: prefill,
            producer,
        })?["recurrence_arena"]
            .clone();
        Ok(
            serde_json::json!({"layer":layer,"prepared_schema":prepared.schema,"prepared":prepared.inventory(),"parameter_spans":[native_pointer_admission(&prepared.stream,pointer(a_log),a_log.len().checked_mul(4).ok_or("engine A bytes")?)?,native_pointer_admission(&prepared.stream,pointer(bias),bias.len().checked_mul(4).ok_or("engine bias bytes")?)?],"a_log":pointer(a_log),"bias":pointer(bias),"arithmetic":arithmetic,"kernel_ptx":kernel,"lower":lower.to_bits(),"scale":scale.to_bits(),"epsilon":epsilon.to_bits(),"recurrence_arena":arena,"decode":model_family_identity(&[ModelFamilyBinding{layer,owner:decode,producer}])?}),
        )
    }
    let mut rows = Vec::new();
    match &life.state {
        KdaPrefillStorageState::Bound(families) => {
            for (f, &layer) in families.iter().zip(&life.layers) {
                let p = &f.prefill_producer;
                rows.push(row(
                    layer,
                    &f.prefill,
                    &f.decode,
                    &f.decode_producer,
                    &p.prepared,
                    &p.a_log,
                    &p.bias,
                    &p.arithmetic_sha256,
                    &p.kernels.ptx_sha256,
                    p.lower,
                    p.scale,
                    p.epsilon,
                )?);
            }
        }
        KdaPrefillStorageState::Retired(families) => {
            for (f, &layer) in families.iter().zip(&life.layers) {
                let p = &f.preparation;
                rows.push(row(
                    layer,
                    &f.prefill,
                    &f.decode,
                    &f.decode_producer,
                    &p.prepared,
                    &p.a_log,
                    &p.bias,
                    &p.arithmetic_sha256,
                    &p.kernels.ptx_sha256,
                    p.lower,
                    p.scale,
                    p.epsilon,
                )?);
            }
        }
        KdaPrefillStorageState::Poisoned => {
            return Err("engine storage static identity poisoned/extracted".into())
        }
    }
    Ok(serde_json::json!({"families":rows,"registry":life.registry()?.inventory(0)}))
}
