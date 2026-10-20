// Shared fresh/workspace-only allocator; persistent carry is never allocated here.
impl PreparedArena {
    fn allocate_storage(
        device: Arc<CudaDevice>,
        stream: Arc<NativeStream>,
        heads: usize,
        dim: usize,
        max_new_rows: usize,
        reserve: usize,
        functions: BTreeMap<&'static str, CudaFunction>,
        arithmetic_sha256: String,
    ) -> Result<Arc<Self>, String> {
        stream.validate()?;
        if !Arc::ptr_eq(&device, &stream.device) {
            return Err("arena storage foreign context".into());
        }
        let PreparedLayout {
            raw_schema,
            pending_schema: _,
            hd,
            padded,
            cells,
            groups,
            layouts,
            power_layouts,
            allocation_bytes,
            persistent_bytes,
        } = PreparedLayout::checked(heads, dim, max_new_rows)?;
        let requested_bytes = allocation_bytes
            .checked_sub(persistent_bytes)
            .ok_or("arena-only allocation accounting")?;
        if memory()?[0]
            < requested_bytes
                .checked_add(reserve)
                .ok_or("arena reserve overflow")?
        {
            return Err("measured arena storage admission".into());
        }
        for (attribute, extent) in [
            (
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                cells.div_ceil(256).max(heads),
            ),
            (
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                groups.max(dim.div_ceil(16)),
            ),
            (
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
                dim,
            ),
        ] {
            if extent > device.attribute(attribute).map_err(|e| e.to_string())? as usize {
                return Err("arena storage actual grid limit".into());
            }
        }
        for name in TRANSPORT.iter().chain(MATH.iter()) {
            if !functions.contains_key(name) {
                return Err("arena storage incomplete loaded kernel contract".into());
            }
        }
        let raw = NativeRawOwner::allocate(stream.clone(), raw_schema, reserve)?;
        let mut arrays = BTreeMap::new();
        for (name, n) in layouts
            .into_iter()
            .filter(|(name, _)| !matches!(*name, "incoming" | "complete" | "latest"))
        {
            arrays.insert(name, native_zeroed::<f32>(&stream, n)?);
        }
        let mut powers = BTreeMap::new();
        for (name, n) in power_layouts {
            powers.insert(name, native_zeroed::<i32>(&stream, n)?);
        }
        let bf16_emitted = native_zeroed::<u16>(&stream, max_new_rows * hd)?;
        let count = native_zeroed::<i32>(&stream, 1)?;
        let raw_count = native_zeroed::<i32>(&stream, 1)?;
        let one = native_zeroed::<i32>(&stream, 1)?;
        check(
            unsafe { sys::lib().cuMemsetD32Async(pointer(&one), 1, 1, stream.handle) },
            "native one constant",
        )?;
        let mut arena_admissions = Vec::new();
        for x in arrays.values() {
            arena_admissions.push(native_pointer_admission(&stream, pointer(x), x.len() * 4)?);
        }
        for x in powers.values() {
            arena_admissions.push(native_pointer_admission(&stream, pointer(x), x.len() * 4)?);
        }
        for x in [&count, &raw_count, &one] {
            arena_admissions.push(native_pointer_admission(&stream, pointer(x), 4)?);
        }
        arena_admissions.push(native_pointer_admission(
            &stream,
            pointer(&bf16_emitted),
            bf16_emitted.len() * 2,
        )?);
        let arena = Arc::new(PreparedArena {
            device: device.clone(),
            heads,
            dim,
            max_new_rows,
            padded,
            functions,
            raw,
            arrays,
            powers,
            bf16_emitted,
            count,
            raw_count,
            one,
            arithmetic_sha256: arithmetic_sha256.clone(),
            requested_bytes,
            enqueue: std::sync::Mutex::new(()),
            native_admissions: arena_admissions,
            stream: stream.clone(),
        });
        stream.drain()?;
        if memory()?[0] < reserve {
            return Err("observed arena storage allocation floor".into());
        }
        Ok(arena)
    }
    fn allocate_like(
        source: &Self,
        max_new_rows: usize,
        reserve: usize,
    ) -> Result<Arc<Self>, String> {
        Self::allocate_storage(
            source.device.clone(),
            source.stream.clone(),
            source.heads,
            source.dim,
            max_new_rows,
            reserve,
            source.functions.clone(),
            source.arithmetic_sha256.clone(),
        )
    }
}
