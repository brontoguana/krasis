// Private model-owned resource domain prototype. It has no serving selector.
// Owned member snapshots retain source/parameter owners for the entire domain;
// shared GPU values remain mutable only through admitted producer work.
struct ProducerResourceRegistration {
    workspaces: Vec<ProducerWorkspaceSpans>,
    _device: Arc<CudaDevice>,
}
static PRODUCER_RESOURCE_DOMAINS: std::sync::OnceLock<
    std::sync::Mutex<Vec<std::sync::Weak<ProducerResourceRegistration>>>,
> = std::sync::OnceLock::new();
impl ProducerResourceRegistration {
    fn acquire(
        workspaces: &[ProducerWorkspaceSpans],
        device: &Arc<CudaDevice>,
        replacing: Option<&Arc<Self>>,
    ) -> Result<Arc<Self>, String> {
        // Topology checked every positive extent and endpoint beforehand.
        let mut registry = PRODUCER_RESOURCE_DOMAINS
            .get_or_init(|| std::sync::Mutex::new(Vec::new()))
            .lock()
            .map_err(|_| "producer resource domain registry poisoned")?;
        registry.retain(|entry| entry.strong_count() != 0);
        if let Some(old) = replacing {
            if !Arc::ptr_eq(device, &old._device)
                || Arc::strong_count(old) != 1
                || !registry
                    .iter()
                    .any(|entry| entry.as_ptr() == Arc::as_ptr(old))
            {
                return Err("resource handoff registration foreign/retained/absent".into());
            }
        }
        for existing in registry.iter().filter_map(std::sync::Weak::upgrade) {
            if replacing.is_some_and(|old| Arc::ptr_eq(old, &existing)) {
                continue;
            }
            for a in workspaces {
                for b in &existing.workspaces {
                    if a.context == b.context
                        && a.spans.iter().any(|&(p, n)| {
                            b.spans
                                .iter()
                                .any(|&(q, m)| p < q + m as u64 && q < p + n as u64)
                        })
                    {
                        return Err(
                            "producer allocations already belong to a live resource domain".into(),
                        );
                    }
                }
            }
        }
        let token = Arc::new(Self {
            workspaces: workspaces.to_vec(),
            _device: device.clone(),
        });
        registry.push(Arc::downgrade(&token));
        Ok(token)
    }
}
struct ProducerResourceGroup {
    timeline: std::sync::Mutex<StateTimeline>,
}
struct ProducerResourceDomain {
    // Drop/drain every timeline while all producer planes and parameters live.
    groups: Vec<ProducerResourceGroup>,
    producers: Vec<OwnedProducerResourceMember>,
    producer_groups: Vec<usize>,
    // Declaration order keeps registration alive through every event teardown.
    _registration: Arc<ProducerResourceRegistration>,
}
impl ProducerResourceDomain {
    fn checked(producers: &[&KdaRawProducer]) -> Result<Self, String> {
        Self::build_checked(producers, None)
    }
    fn build_checked(
        producers: &[&KdaRawProducer],
        replacing: Option<&Arc<ProducerResourceRegistration>>,
    ) -> Result<Self, String> {
        let first = producers.first().ok_or("producer domain inventory empty")?;
        let device = &first.raw.device;
        let mut unique: Vec<&KdaRawProducer> = Vec::new();
        for &producer in producers {
            if unique.iter().any(|p| std::ptr::eq(*p, producer)) {
                continue;
            }
            producer.raw.stream.validate()?;
            producer.prepared.stream.validate()?;
            if !Arc::ptr_eq(device, &producer.raw.device)
                || !Arc::ptr_eq(device, &producer.prepared.device)
                || producer.raw.stream.handle != producer.prepared.stream.handle
            {
                return Err("producer domain foreign device/context or preparation stream".into());
            }
            unique.push(producer);
        }
        Self::build_members(
            unique
                .iter()
                .map(|p| OwnedProducerResourceMember::from_producer(p))
                .collect(),
            replacing,
        )
    }
    fn build_members(
        members: Vec<OwnedProducerResourceMember>,
        replacing: Option<&Arc<ProducerResourceRegistration>>,
    ) -> Result<Self, String> {
        let first = members.first().ok_or("resource typed inventory empty")?;
        let device = first.device();
        let workspaces = members
            .iter()
            .map(|m| m.checked_workspace(device))
            .collect::<Result<Vec<_>, _>>()?;
        let topology = ProducerResourceTopology::checked(&workspaces)?;
        let registration = ProducerResourceRegistration::acquire(&workspaces, device, replacing)?;
        let mut groups = Vec::with_capacity(topology.groups.len());
        let mut producer_groups = vec![usize::MAX; members.len()];
        for (group_index, indices) in topology.groups.iter().enumerate() {
            let mut streams: Vec<Arc<NativeStream>> = Vec::new();
            for &index in indices {
                producer_groups[index] = group_index;
                let stream = members[index].stream();
                if !streams.iter().any(|s| s.handle == stream.handle) {
                    streams.push(stream.clone());
                }
            }
            let mut event = std::ptr::null_mut();
            check(
                unsafe {
                    sys::lib().cuEventCreate(
                        &mut event,
                        sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
                    )
                },
                "typed resource completion event create",
            )?;
            groups.push(ProducerResourceGroup {
                timeline: std::sync::Mutex::new(StateTimeline {
                    event,
                    device: device.clone(),
                    recorded: false,
                    poisoned: false,
                    phases: 0,
                    streams,
                }),
            });
        }
        if producer_groups.contains(&usize::MAX) {
            return Err("typed resource topology incomplete".into());
        }
        Ok(Self {
            groups,
            producers: members,
            producer_groups,
            _registration: registration,
        })
    }
}

struct ProducerResourceProof<'a> {
    storage: ModelBindingProof<'a>,
    domain: &'a ProducerResourceDomain,
    groups: Vec<usize>,
}
impl<'a> ProducerResourceProof<'a> {
    fn checked(
        domain: &'a ProducerResourceDomain,
        bindings: &'a [ModelFamilyBinding<'a>],
        expected: &serde_json::Value,
    ) -> Result<Self, String> {
        let storage = ModelBindingProof::checked(bindings, expected)?;
        let mut groups = std::collections::BTreeSet::new();
        for b in bindings {
            let index = domain
                .producers
                .iter()
                .position(|p| p.matches(b.producer))
                .ok_or("complete graph producer absent from model resource domain")?;
            groups.insert(domain.producer_groups[index]);
        }
        Ok(Self {
            storage,
            domain,
            groups: groups.into_iter().collect(),
        })
    }
}
struct ProducerResourceAdmission<'a> {
    arena: ModelArenaAdmission<'a>,
    resources: Vec<std::sync::MutexGuard<'a, StateTimeline>>,
    stream: &'a NativeStream,
}
impl<'a> ProducerResourceAdmission<'a> {
    fn checked(
        proof: &'a ProducerResourceProof<'a>,
        stream: &'a NativeStream,
    ) -> Result<Self, String> {
        stream.validate()?;
        let mut resources = Vec::with_capacity(proof.groups.len());
        for &index in &proof.groups {
            let resource = proof.domain.groups[index]
                .timeline
                .try_lock()
                .map_err(|_| "producer resource concurrently leased/poisoned")?;
            if !Arc::ptr_eq(&resource.device, &stream.device)
                || !resource.streams.iter().any(|s| s.handle == stream.handle)
            {
                return Err("producer resource foreign/unadmitted execution stream".into());
            }
            if resource.poisoned {
                return Err("producer resource poisoned".into());
            }
            resource
                .phases
                .checked_add(1)
                .ok_or("producer resource phase overflow")?;
            resources.push(resource);
        }
        let arena = ModelArenaAdmission::from_proof(&proof.storage, stream)?;
        Ok(Self {
            arena,
            resources,
            stream,
        })
    }
    fn validate_ready(&self) -> Result<(), String> {
        self.arena.families.validate_ready()?;
        for resource in &self.resources {
            if resource.poisoned {
                return Err("producer resource poisoned".into());
            }
            resource
                .phases
                .checked_add(1)
                .ok_or("producer resource phase overflow")?;
        }
        Ok(())
    }
    fn poison_all(&mut self) {
        for resource in &mut self.resources {
            resource.poisoned = true;
        }
        for family in &mut self.arena.families.slots {
            if let Some(timeline) = family.as_mut() {
                timeline.poisoned = true;
            }
        }
    }
    fn capture_only<T>(
        &mut self,
        work: impl FnOnce(&[EnqueueLease<'a>]) -> Result<T, String>,
    ) -> Result<T, String> {
        self.validate_ready()?;
        let before = self
            .resources
            .iter()
            .map(|r| (r.phases, r.recorded, r.poisoned))
            .collect::<Vec<_>>();
        let leases = &self.arena.arenas;
        let result = self.arena.families.capture_only(|| work(leases));
        for (resource, want) in self.resources.iter().zip(before) {
            if (resource.phases, resource.recorded, resource.poisoned) != want {
                return Err("capture changed committed producer resource timeline".into());
            }
        }
        result
    }
    fn execute<T>(
        &mut self,
        work: impl FnOnce(&[EnqueueLease<'a>]) -> Result<T, String>,
    ) -> Result<T, String> {
        self.validate_ready()?;
        for resource in &mut self.resources {
            resource.poisoned = true;
        }
        for resource in &self.resources {
            if resource.recorded {
                if let Err(error) = check(
                    unsafe { sys::lib().cuStreamWaitEvent(self.stream.handle, resource.event, 0) },
                    "producer resource preceding completion wait",
                ) {
                    self.poison_all();
                    return Err(error);
                }
            }
        }
        let result = self.arena.execute(work);
        let mut record_failure = None;
        for resource in &mut self.resources {
            match check(
                unsafe { sys::lib().cuEventRecord(resource.event, self.stream.handle) },
                "producer resource completion record",
            ) {
                Ok(()) => resource.recorded = true,
                Err(error) => {
                    if record_failure.is_none() {
                        record_failure = Some(error);
                    }
                }
            }
        }
        if let Some(error) = record_failure {
            self.poison_all();
            return Err(error);
        }
        match result {
            Ok(value) => {
                for resource in &mut self.resources {
                    resource.phases += 1;
                    resource.poisoned = false;
                }
                Ok(value)
            }
            Err(error) => {
                self.poison_all();
                Err(error)
            }
        }
    }
}
