// Actual captured work and fresh-thread controls; no serving selection.
fn resource_graph_capture<'a>(
    proof: &ProducerResourceProof<'_>,
    owner: &'a PreparedOwner,
    producer: &'a KdaRawProducer,
    source: &'a NativeRawOwner,
    destination: &'a NativeRawOwner,
) -> Result<(OwnerGraph<'a>, Value), String> {
    let stream = &owner.stream;
    let mut held = ProducerResourceAdmission::checked(proof, stream)?;
    let before = held
        .resources
        .iter()
        .map(|x| (x.phases, x.recorded, x.poisoned))
        .collect::<Vec<_>>();
    let mut graph = OwnerGraph {
        owner,
        graph: std::ptr::null_mut(),
        exec: std::ptr::null_mut(),
    };
    held.capture_only(|leases| {
        check(
            unsafe {
                sys::lib().cuStreamBeginCapture_v2(
                    stream.handle,
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            },
            "resource graph begin",
        )?;
        let run = (|| {
            resource_copy_input(stream, source, destination)?;
            producer.enqueue(0, 1, owner)?;
            owner.decode_capture_in(model_arena_lease(owner, leases)?, &producer.view(1)?)
        })();
        let end = check(
            unsafe { sys::lib().cuStreamEndCapture(stream.handle, &mut graph.graph) },
            "resource graph end",
        );
        run?;
        end?;
        check(
            unsafe { sys::lib().cuGraphInstantiateWithFlags(&mut graph.exec, graph.graph, 0) },
            "resource graph instantiate",
        )
    })?;
    let after = held
        .resources
        .iter()
        .map(|x| (x.phases, x.recorded, x.poisoned))
        .collect::<Vec<_>>();
    if before != after {
        return Err("real resource graph capture changed timeline".into());
    }
    drop(held);
    let mut n = 0;
    check(
        unsafe { sys::lib().cuGraphGetNodes(graph.graph, std::ptr::null_mut(), &mut n) },
        "resource actual graph count",
    )?;
    let mut nodes = vec![std::ptr::null_mut(); n];
    check(
        unsafe { sys::lib().cuGraphGetNodes(graph.graph, nodes.as_mut_ptr(), &mut n) },
        "resource actual graph nodes",
    )?;
    if n != nodes.len() {
        return Err("resource graph count changed".into());
    }
    let mut entries = Vec::new();
    let mut kernels = 0;
    let mut copies = 0;
    for node in nodes {
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        check(
            unsafe { sys::lib().cuGraphNodeGetType(node, &mut kind) },
            "resource node type",
        )?;
        match kind {
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => {
                let mut x = std::mem::MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::zeroed();
                check(
                    unsafe { sys::lib().cuGraphKernelNodeGetParams_v2(node, x.as_mut_ptr()) },
                    "resource kernel params",
                )?;
                let x = unsafe { x.assume_init() };
                let name = if x.func == producer.kernels.domain.cu_function() {
                    "kda_raw_domain"
                } else if x.func == producer.kernels.prepare.cu_function() {
                    "kda_raw_prepare"
                } else {
                    owner
                        .arena
                        .functions
                        .iter()
                        .find_map(|(name, f)| (f.cu_function() == x.func).then_some(*name))
                        .ok_or("resource foreign graph kernel")?
                };
                entries.push(json!({"type":"kernel","name":name,"grid":[x.gridDimX,x.gridDimY,x.gridDimZ],"block":[x.blockDimX,x.blockDimY,x.blockDimZ],"shared_bytes":x.sharedMemBytes}));
                kernels += 1;
            }
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => {
                let mut x = std::mem::MaybeUninit::<sys::CUDA_MEMCPY3D>::zeroed();
                check(
                    unsafe { sys::lib().cuGraphMemcpyNodeGetParams(node, x.as_mut_ptr()) },
                    "resource copy params",
                )?;
                let x = unsafe { x.assume_init() };
                if x.srcMemoryType != sys::CUmemorytype::CU_MEMORYTYPE_DEVICE
                    || x.dstMemoryType != sys::CUmemorytype::CU_MEMORYTYPE_DEVICE
                    || x.Height != 1
                    || x.Depth != 1
                {
                    return Err("resource graph nondevice copy".into());
                }
                let matched = source.inventory().iter().zip(destination.inventory()).any(
                    |((p, from), (q, to))| {
                        p == &q
                            && *from == x.srcDevice
                            && to == x.dstDevice
                            && p.storage_bytes == x.WidthInBytes
                    },
                );
                if !matched {
                    return Err("resource graph copy outside admitted inputs".into());
                }
                entries.push(json!({"type":"device-copy","source":x.srcDevice,"destination":x.dstDevice,"bytes":x.WidthInBytes}));
                copies += 1;
            }
            _ => return Err(format!("resource graph unexpected node {kind:?}")),
        }
    }
    if kernels == 0 || copies != source.schema.planes.len() {
        return Err("resource graph incomplete kernel/copy inventory".into());
    }
    Ok((
        graph,
        json!({"nodes":n,"kernels":kernels,"copies":copies,"entries":entries,"resource_before":before,"resource_after":after,"committed_timeline_unchanged":true}),
    ))
}
fn resource_partial_raw(
    stream: Arc<NativeStream>,
    schema: Schema,
    allocation: &Arc<GpuBuf<u16>>,
    start: usize,
    reserve: usize,
) -> Result<NativeRawOwner, String> {
    let mut offset = start;
    let mut buffers = Vec::new();
    for plane in &schema.planes {
        buffers.push(NativeBuffer::Bf16View(NativeBf16View::checked(
            NativeBf16Allocation::GpuBuf(allocation.clone()),
            offset,
            plane.elements,
            &stream,
        )?));
        offset = offset
            .checked_add(plane.elements)
            .ok_or("partial typed view overflow")?;
    }
    NativeRawOwner::from_owned_buffers(stream, schema, buffers, reserve)
}
fn resource_extended_controls(
    root: &Path,
    device: &Arc<CudaDevice>,
    domain: &ProducerResourceDomain,
    pa: &KdaRawProducer,
    pb: &KdaRawProducer,
    i0: &Arc<NativeRawOwner>,
    i1: &Arc<NativeRawOwner>,
    sa: &Arc<NativeStream>,
    sb: &Arc<NativeStream>,
    reserve: usize,
    transport: &str,
    math: &str,
    c: &[u8],
    d: &[u8],
    cs: &OwnerSnapshot,
    ds: &OwnerSnapshot,
) -> Result<Value, String> {
    let make = |stream: Arc<NativeStream>| {
        PreparedOwner::new_on_stream(
            device.clone(),
            stream,
            pa.heads,
            pa.dim,
            1,
            reserve,
            transport,
            math,
        )
    };
    let ga = make(sa.clone())?;
    let gb = make(sb.clone())?;
    ga.initialize_timeline(&[sa.clone(), sb.clone()])?;
    gb.initialize_timeline(&[sa.clone(), sb.clone()])?;
    let ba = [ModelFamilyBinding {
        layer: 0,
        owner: &ga,
        producer: pa,
    }];
    let bb = [ModelFamilyBinding {
        layer: 1,
        owner: &gb,
        producer: pb,
    }];
    let proofa = ProducerResourceProof::checked(domain, &ba, &model_family_identity(&ba)?)?;
    let proofb = ProducerResourceProof::checked(domain, &bb, &model_family_identity(&bb)?)?;
    let (graph_a, ia) = resource_graph_capture(&proofa, &ga, pa, i0, &pa.raw)?;
    let (graph_b, ib) = resource_graph_capture(&proofb, &gb, pb, i1, &pb.raw)?;
    {
        let held = ProducerResourceAdmission::checked(&proofa, sa)?;
        let (rejection, thread) = request_worker(device, "resource competing graph proof", || {
            match ProducerResourceAdmission::checked(&proofb, sb) {
                Err(e) if e == "producer resource concurrently leased/poisoned" => Ok(e),
                Err(e) => Err(format!("resource competing unexpected {e}")),
                Ok(_) => Err("resource competing thread admitted".into()),
            }
        })?;
        owner_save(
            root,
            "resource-competing-thread.json",
            &serde_json::to_vec(
                &json!({"error":rejection,"worker":thread,"callback_entered":false}),
            )
            .map_err(|e| e.to_string())?,
        )?;
        drop(held);
    }
    let (graph_a, ta) = request_worker(device, "resource graph a launch", move || {
        let mut held = ProducerResourceAdmission::checked(&proofa, sa)?;
        held.execute(|_| {
            check(
                unsafe { sys::lib().cuGraphLaunch(graph_a.exec, sa.handle) },
                "protected resource graph a launch",
            )
        })?;
        drop(held);
        Ok(graph_a)
    })?;
    let (graph_b, tb) = request_worker(device, "resource graph b launch", move || {
        let mut held = ProducerResourceAdmission::checked(&proofb, sb)?;
        held.execute(|_| {
            check(
                unsafe { sys::lib().cuGraphLaunch(graph_b.exec, sb.handle) },
                "protected resource graph b launch",
            )
        })?;
        drop(held);
        Ok(graph_b)
    })?;
    for (row, owner, expected, state) in [(0, &ga, c, cs), (1, &gb, d, ds)] {
        let output = resource_output(owner)?;
        let snapshot = resource_snapshot(owner)?;
        owner_save(root, &format!("graph-row{row}-output.f32"), &output)?;
        owner_save(
            root,
            &format!("graph-row{row}-state.json"),
            &serde_json::to_vec(&snapshot).map_err(|e| e.to_string())?,
        )?;
        owner_compare(&output, expected, "complete graph resource output")?;
        owner_compare(
            &serde_json::to_vec(&snapshot.blobs).map_err(|e| e.to_string())?,
            &serde_json::to_vec(&state.blobs).map_err(|e| e.to_string())?,
            "complete graph resource state",
        )?;
    }
    drop(graph_a);
    drop(graph_b);
    let schema = Schema::checked(RawContract::KdaBf16VectorGate, 1, pa.heads, pa.dim, pa.dim)?;
    let elements = schema
        .requested_bytes
        .checked_div(2)
        .ok_or("partial source elements")?;
    let shift = elements
        .checked_div(2)
        .filter(|x| *x > 0)
        .ok_or("partial source positive shift")?;
    let backing = Arc::new(native_zeroed::<u16>(
        sa,
        elements
            .checked_add(shift)
            .ok_or("partial backing extent")?,
    )?);
    let r0 = Arc::new(resource_partial_raw(
        sa.clone(),
        schema.clone(),
        &backing,
        0,
        reserve,
    )?);
    let r1 = Arc::new(resource_partial_raw(
        sb.clone(),
        schema,
        &backing,
        shift,
        reserve,
    )?);
    if r0.addresses() == r1.addresses() {
        return Err("partial sources equal".into());
    }
    let producer = |raw: Arc<NativeRawOwner>| {
        KdaRawProducer::from_live_raw_owner(
            raw,
            pa.heads,
            pa.dim,
            1,
            reserve,
            pa.lower,
            pa.scale,
            pa.epsilon,
            pa.kernels.clone(),
            &resource_parameter_bytes(&pa.a_log)?,
            &resource_parameter_bytes(&pa.bias)?,
        )
    };
    let p0 = producer(r0.clone())?;
    let p1 = producer(r1.clone())?;
    let partial = ProducerResourceDomain::checked(&[&p0, &p1])?;
    if partial.groups.len() != 1 {
        return Err("partial actual spans not merged".into());
    }
    let o0 = make(sa.clone())?;
    let o1 = make(sb.clone())?;
    o0.initialize_timeline(&[sa.clone(), sb.clone()])?;
    o1.initialize_timeline(&[sa.clone(), sb.clone()])?;
    let b0 = [ModelFamilyBinding {
        layer: 0,
        owner: &o0,
        producer: &p0,
    }];
    let b1 = [ModelFamilyBinding {
        layer: 1,
        owner: &o1,
        producer: &p1,
    }];
    let q0 = ProducerResourceProof::checked(&partial, &b0, &model_family_identity(&b0)?)?;
    let q1 = ProducerResourceProof::checked(&partial, &b1, &model_family_identity(&b1)?)?;
    {
        let mut held = ProducerResourceAdmission::checked(&q0, sa)?;
        if ProducerResourceAdmission::checked(&q1, sb).is_ok() {
            return Err("partial competing proof admitted".into());
        }
        held.execute(|leases| {
            resource_copy_input(sa, i0, &r0)?;
            resource_enqueue(&o0, &p0, leases)
        })?;
    }
    {
        let mut held = ProducerResourceAdmission::checked(&q1, sb)?;
        held.execute(|leases| {
            resource_copy_input(sb, i1, &r1)?;
            resource_enqueue(&o1, &p1, leases)
        })?;
    }
    for (row, owner, expected, state) in [(0, &o0, c, cs), (1, &o1, d, ds)] {
        let output = resource_output(owner)?;
        let snapshot = resource_snapshot(owner)?;
        owner_save(root, &format!("partial-row{row}-output.f32"), &output)?;
        owner_save(
            root,
            &format!("partial-row{row}-state.json"),
            &serde_json::to_vec(&snapshot).map_err(|e| e.to_string())?,
        )?;
        owner_compare(&output, expected, "complete partial resource output")?;
        owner_compare(
            &serde_json::to_vec(&snapshot.blobs).map_err(|e| e.to_string())?,
            &serde_json::to_vec(&state.blobs).map_err(|e| e.to_string())?,
            "complete partial resource state",
        )?;
    }
    let result = json!({"passed":true,"graph_a":ia,"graph_b":ib,"thread_a":ta,"thread_b":tb,"partial_source_shift_elements":shift,"partial_source_elements":elements,"partial_source_addresses":[r0.addresses(),r1.addresses()],"partial_source_groups":partial.groups.len(),"actual_graph_capture_replay":true,"partial_overlap_native":true,"cross_thread_domain":true,"all_complete_outputs_and_states_equal":true,"adopted":false});
    owner_save(
        root,
        "resource-extended.json",
        &serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
    )?;
    Ok(result)
}

fn resource_parameter_bytes(x: &GpuBuf<f32>) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0u8; x.len().checked_mul(4).ok_or("resource parameter size")?];
    check(
        unsafe { sys::lib().cuMemcpyDtoH_v2(bytes.as_mut_ptr().cast(), pointer(x), bytes.len()) },
        "resource actual immutable parameter copy",
    )?;
    Ok(bytes)
}
