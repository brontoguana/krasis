struct ModelFamilyEmission { float: GpuBuf<f32>, bf16: GpuBuf<u16> }
struct ModelFamilyGraph<'a> {
    inner: OwnerGraph<'a>,
    bindings: &'a [ModelFamilyBinding<'a>],
    emissions: &'a [ModelFamilyEmission],
    identity: serde_json::Value,
}
impl Drop for ModelFamilyGraph<'_> {
    fn drop(&mut self) {
        self.inner.owner.device.bind_to_thread().expect("complete graph teardown context bind");
        for b in self.bindings {
            let slot=b.owner.state.timeline.lock().expect("complete graph teardown family lock");
            if let Some(t)=slot.as_ref(){
                if t.recorded {check(unsafe{sys::lib().cuEventSynchronize(t.event)},"complete graph final family event").expect("complete graph final family completion");}
                for stream in &t.streams {stream.drain().expect("complete graph final retained stream drain");}
            }
        }
    }
}
fn capture_model_families<'a>(bindings:&'a[ModelFamilyBinding<'a>],emissions:&'a[ModelFamilyEmission],stream:&NativeStream)->Result<ModelFamilyGraph<'a>,String>{
    if emissions.len()!=bindings.len(){return Err("complete graph emission family count".into());}
    let identity=model_family_identity(bindings)?;
    let mut graph=OwnerGraph{owner:bindings.first().ok_or("complete graph empty")?.owner,graph:std::ptr::null_mut(),exec:std::ptr::null_mut()};
    let mut admission=ModelFamilyAdmission::checked(bindings,stream,&identity)?;
    admission.capture_only(||{
        check(unsafe{sys::lib().cuStreamBeginCapture_v2(stream.handle,sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)},"complete graph capture begin")?;
        let run=(|| -> Result<(),String> {
            for (b,out) in bindings.iter().zip(emissions){
                if b.owner.stream.handle!=stream.handle {return Err("capture family owner stream".into());}
                let lease=b.owner.arena.lease()?;b.owner.validate_lease(&lease)?;
                b.producer.enqueue(0,1,b.owner)?;
                b.owner.decode_capture_in(&lease,&b.producer.view(1)?)?;
                let hd=b.owner.heads.checked_mul(b.owner.dim).ok_or("complete graph output geometry")?;
                if out.float.len()!=hd||out.bf16.len()!=hd{return Err("complete graph output storage extent".into());}
                timeline_copy(stream,pointer(&out.float),b.owner.p("emitted"),hd.checked_mul(4).ok_or("emission F32 extent")?)?;
                timeline_copy(stream,pointer(&out.bf16),pointer(&b.owner.arena.bf16_emitted),hd.checked_mul(2).ok_or("emission BF16 extent")?)?;
            }
            Ok(())
        })();
        let end=check(unsafe{sys::lib().cuStreamEndCapture(stream.handle,&mut graph.graph)},"complete graph capture end");
        run?;end?;
        check(unsafe{sys::lib().cuGraphInstantiateWithFlags(&mut graph.exec,graph.graph,0)},"complete graph instantiate")
    })?;
    Ok(ModelFamilyGraph{inner:graph,bindings,emissions,identity})
}
fn model_graph_inventory(graph:&ModelFamilyGraph<'_>)->Result<Value,String>{
    let mut n=0;check(unsafe{sys::lib().cuGraphGetNodes(graph.inner.graph,std::ptr::null_mut(),&mut n)},"complete graph node count")?;
    let mut nodes=vec![std::ptr::null_mut();n];check(unsafe{sys::lib().cuGraphGetNodes(graph.inner.graph,nodes.as_mut_ptr(),&mut n)},"complete graph nodes")?;
    if n!=nodes.len(){return Err("complete graph node count changed".into());}
    let mut inventory=Vec::new();
    for &node in &nodes {
        let mut kind=sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        check(unsafe{sys::lib().cuGraphNodeGetType(node,&mut kind)},"complete graph node type")?;
        match kind {
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL=>{
                let mut storage=std::mem::MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::zeroed();check(unsafe{sys::lib().cuGraphKernelNodeGetParams_v2(node,storage.as_mut_ptr())},"complete graph actual kernel")?;let a=unsafe{storage.assume_init()};
                let name=graph.bindings.iter().find_map(|b|{
                    if a.func==b.producer.kernels.domain.cu_function(){Some("kda_raw_domain")}
                    else if a.func==b.producer.kernels.prepare.cu_function(){Some("kda_raw_prepare")}
                    else {b.owner.arena.functions.iter().find_map(|(name,f)|(f.cu_function()==a.func).then_some(*name))}
                }).ok_or("complete graph unknown actual function")?;
                inventory.push(json!({"node":node as usize,"type":"kernel","name":name,"function":a.func as usize,"grid":[a.gridDimX,a.gridDimY,a.gridDimZ],"block":[a.blockDimX,a.blockDimY,a.blockDimZ],"shared_bytes":a.sharedMemBytes}));
            }
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY=>{
                let mut storage=std::mem::MaybeUninit::<sys::CUDA_MEMCPY3D>::zeroed();check(unsafe{sys::lib().cuGraphMemcpyNodeGetParams(node,storage.as_mut_ptr())},"complete graph actual copy")?;let a=unsafe{storage.assume_init()};
                if a.srcMemoryType!=sys::CUmemorytype::CU_MEMORYTYPE_DEVICE||a.dstMemoryType!=sys::CUmemorytype::CU_MEMORYTYPE_DEVICE||a.Height!=1||a.Depth!=1{return Err("complete graph non-device emission copy".into());}
                let admitted=graph.bindings.iter().zip(graph.emissions).any(|(b,out)|{
                    (a.srcDevice==b.owner.p("emitted")&&a.dstDevice==pointer(&out.float)&&a.WidthInBytes==out.float.len()*4)
                    ||(a.srcDevice==pointer(&b.owner.arena.bf16_emitted)&&a.dstDevice==pointer(&out.bf16)&&a.WidthInBytes==out.bf16.len()*2)
                });if !admitted{return Err("complete graph foreign output copy".into());}
                inventory.push(json!({"node":node as usize,"type":"device-copy","source":a.srcDevice,"destination":a.dstDevice,"bytes":a.WidthInBytes}));
            }
            _=>return Err(format!("complete graph unexpected node:{kind:?}")),
        }
    }
    let mut e=0;check(unsafe{sys::lib().cuGraphGetEdges(graph.inner.graph,std::ptr::null_mut(),std::ptr::null_mut(),&mut e)},"complete graph edge count")?;
    let(mut from,mut to)=(vec![std::ptr::null_mut();e],vec![std::ptr::null_mut();e]);check(unsafe{sys::lib().cuGraphGetEdges(graph.inner.graph,from.as_mut_ptr(),to.as_mut_ptr(),&mut e)},"complete graph edges")?;
    if e!=from.len()||e!=to.len()||e+1!=n||from.iter().chain(&to).any(|x|!nodes.contains(x)){return Err("complete graph explicit stream chain".into());}
    Ok(json!({"complete":true,"nodes":inventory,"edges":from.iter().zip(to).map(|(f,t)|json!([*f as usize,t as usize])).collect::<Vec<_>>(),"family_identity":graph.identity,"capture_host_timeline_unchanged":true}))
}
