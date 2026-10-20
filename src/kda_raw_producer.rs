// Private diagnostic raw producer. No production dispatch is selected.
struct KdaRawKernels {
    device: Arc<CudaDevice>,
    domain: CudaFunction,
    prepare: CudaFunction,
    ptx_sha256: String,
}
impl KdaRawKernels {
    fn load(device: Arc<CudaDevice>, ptx: &str) -> Result<Arc<Self>, String> {
        device.bind_to_thread().map_err(|e| e.to_string())?;
        device
            .load_ptx(
                ptx.into(),
                "kda_native_raw_producer",
                &["kda_raw_domain", "kda_raw_prepare"],
            )
            .map_err(|e| e.to_string())?;
        let domain = device
            .get_func("kda_native_raw_producer", "kda_raw_domain")
            .ok_or("raw KDA domain symbol")?;
        let prepare = device
            .get_func("kda_native_raw_producer", "kda_raw_prepare")
            .ok_or("raw KDA prepare symbol")?;
        Ok(Arc::new(Self {
            device,
            domain,
            prepare,
            ptx_sha256: format!("{:x}", Sha256::digest(ptx.as_bytes())),
        }))
    }
}
struct KdaRawProducer {
    raw: Arc<NativeRawOwner>,
    prepared: Arc<NativeRawOwner>,
    a_log: Arc<GpuBuf<f32>>,
    bias: Arc<GpuBuf<f32>>,
    kernels: Arc<KdaRawKernels>,
    heads: usize,
    dim: usize,
    capacity: usize,
    input_capacity: usize,
    source_epoch: u64,
    lower: f32,
    scale: f32,
    epsilon: f32,
    parameter_admissions: Vec<NativePointerAdmission>,
    requested_bytes: usize,
    allocated_bytes: usize,
    arithmetic_sha256: String,
}
struct KdaRawPreparedView<'a> {
    owner: &'a KdaRawProducer,
    schema: Schema,
}
impl PreparedInput for KdaRawPreparedView<'_> {
    fn schema(&self) -> &Schema {
        &self.schema
    }
    fn device(&self) -> &Arc<CudaDevice> {
        &self.owner.prepared.device
    }
    fn stream(&self) -> sys::CUstream {
        self.owner.prepared.stream.handle
    }
    fn addresses(&self) -> Vec<u64> {
        self.owner.prepared.addresses()
    }
}
impl KdaRawProducer {
    fn allocation_blueprint(
        heads: usize,
        dim: usize,
        capacity: usize,
    ) -> Result<(Schema, Schema, usize), String> {
        let raw_schema =
            Schema::checked(RawContract::KdaBf16VectorGate, capacity, heads, dim, dim)?;
        let prepared_schema = Schema::checked(
            RawContract::KdaPreparedF32VectorGate,
            capacity,
            heads,
            dim,
            dim,
        )?;
        let a_bytes = heads.checked_mul(4).ok_or("raw A bytes")?;
        let bias_bytes = heads
            .checked_mul(dim)
            .and_then(|x| x.checked_mul(4))
            .ok_or("raw bias bytes")?;
        let requested_bytes = raw_schema
            .requested_bytes
            .checked_add(prepared_schema.requested_bytes)
            .and_then(|x| x.checked_add(a_bytes))
            .and_then(|x| x.checked_add(bias_bytes))
            .ok_or("raw KDA total bytes")?;
        Ok((raw_schema, prepared_schema, requested_bytes))
    }

    fn new(
        stream: Arc<NativeStream>,
        heads: usize,
        dim: usize,
        capacity: usize,
        reserve: usize,
        lower: f32,
        scale: f32,
        epsilon: f32,
        kernels: Arc<KdaRawKernels>,
        a_log: &[u8],
        bias: &[u8],
    ) -> Result<Self, String> {
        Self::construct(
            stream, heads, dim, capacity, reserve, lower, scale, epsilon, kernels, a_log, bias,
            None,
        )
    }
    fn from_raw_owner(
        raw: Arc<NativeRawOwner>,
        heads: usize,
        dim: usize,
        capacity: usize,
        reserve: usize,
        lower: f32,
        scale: f32,
        epsilon: f32,
        kernels: Arc<KdaRawKernels>,
        a_log: &[u8],
        bias: &[u8],
    ) -> Result<Self, String> {
        Self::construct(
            raw.stream.clone(),
            heads,
            dim,
            capacity,
            reserve,
            lower,
            scale,
            epsilon,
            kernels,
            a_log,
            bias,
            Some(raw),
        )
    }
    fn from_live_raw_owner(
        raw: Arc<NativeRawOwner>,
        heads: usize,
        dim: usize,
        capacity: usize,
        reserve: usize,
        lower: f32,
        scale: f32,
        epsilon: f32,
        kernels: Arc<KdaRawKernels>,
        a_log: &[u8],
        bias: &[u8],
    ) -> Result<Self, String> {
        raw.schema.validate()?;
        let input_capacity = *raw
            .schema
            .planes
            .first()
            .and_then(|p| p.shape.first())
            .ok_or("live raw input row extent")?;
        Self::construct_extents(
            raw.stream.clone(),
            heads,
            dim,
            capacity,
            reserve,
            lower,
            scale,
            epsilon,
            kernels,
            a_log,
            bias,
            Some(raw),
            input_capacity,
        )
    }
    fn construct(
        stream: Arc<NativeStream>,
        heads: usize,
        dim: usize,
        capacity: usize,
        reserve: usize,
        lower: f32,
        scale: f32,
        epsilon: f32,
        kernels: Arc<KdaRawKernels>,
        a_log: &[u8],
        bias: &[u8],
        retained_raw: Option<Arc<NativeRawOwner>>,
    ) -> Result<Self, String> {
        Self::construct_extents(
            stream,
            heads,
            dim,
            capacity,
            reserve,
            lower,
            scale,
            epsilon,
            kernels,
            a_log,
            bias,
            retained_raw,
            capacity,
        )
    }
    fn construct_extents(
        stream: Arc<NativeStream>,
        heads: usize,
        dim: usize,
        capacity: usize,
        reserve: usize,
        lower: f32,
        scale: f32,
        epsilon: f32,
        kernels: Arc<KdaRawKernels>,
        a_log: &[u8],
        bias: &[u8],
        retained_raw: Option<Arc<NativeRawOwner>>,
        input_capacity: usize,
    ) -> Result<Self, String> {
        stream.validate()?;
        if !Arc::ptr_eq(&stream.device, &kernels.device) {
            return Err("raw KDA kernel/context mismatch".into());
        }
        let (output_raw_schema, prepared_schema, output_requested_bytes) =
            Self::allocation_blueprint(heads, dim, capacity)?;
        let raw_schema = Schema::checked(
            RawContract::KdaBf16VectorGate,
            input_capacity,
            heads,
            dim,
            dim,
        )?;
        if retained_raw.is_none() && input_capacity != capacity {
            return Err("nonretained producer requires original equal extents".into());
        }
        let requested_bytes = output_requested_bytes
            .checked_sub(output_raw_schema.requested_bytes)
            .and_then(|x| x.checked_add(raw_schema.requested_bytes))
            .ok_or("live source logical payload extent")?;
        if !lower.is_finite()
            || lower >= 0.
            || !scale.is_finite()
            || scale <= 0.
            || !epsilon.is_finite()
            || epsilon <= 0.
        {
            return Err("raw KDA preparation semantics".into());
        }
        let hd = heads.checked_mul(dim).ok_or("raw KDA HD overflow")?;
        if a_log.len() != heads.checked_mul(4).ok_or("raw A bytes")?
            || bias.len() != hd.checked_mul(4).ok_or("raw bias bytes")?
        {
            return Err("raw KDA parameter extents".into());
        }
        let decode = |b: &[u8]| -> Result<Vec<f32>, String> {
            let v = b
                .chunks_exact(4)
                .map(|x| f32::from_bits(u32::from_le_bytes(x.try_into().unwrap())))
                .collect::<Vec<_>>();
            if v.iter().any(|x| !x.is_finite()) {
                Err("raw KDA nonfinite parameter".into())
            } else {
                Ok(v)
            }
        };
        let a_values = decode(a_log)?;
        let bias_values = decode(bias)?;
        // Admission is completed before allocating any producer-owned workspace.
        if let Some(raw) = &retained_raw {
            raw.schema.validate()?;
            raw.stream.validate()?;
            if !Arc::ptr_eq(&raw.device, &stream.device)
                || raw.stream.handle != stream.handle
                || raw.schema != raw_schema
                || raw.buffers.len() != raw_schema.planes.len()
                || raw.admissions.len() != raw_schema.planes.len()
            {
                return Err("retained raw source schema/context/stream".into());
            }
            for ((plane, buffer), admitted) in raw_schema
                .planes
                .iter()
                .zip(&raw.buffers)
                .zip(&raw.admissions)
            {
                if buffer.dtype() != plane.dtype
                    || buffer.elements() != plane.elements
                    || admitted.ptr != buffer.address()
                    || admitted.requested_bytes != plane.storage_bytes
                {
                    return Err("retained raw source type/span".into());
                }
                native_pointer_admission(&stream, buffer.address(), plane.storage_bytes)?;
            }
        }
        let allocated_bytes = if retained_raw.is_some() {
            requested_bytes
                .checked_sub(raw_schema.requested_bytes)
                .ok_or("retained raw allocation accounting")?
        } else {
            requested_bytes
        };
        if memory()?[0]
            < allocated_bytes
                .checked_add(reserve)
                .ok_or("raw KDA reserve bytes")?
        {
            return Err("raw KDA measured allocation admission".into());
        }
        for (attribute, needed) in [
            (
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                capacity.max(
                    capacity
                        .checked_mul(hd)
                        .ok_or("raw domain capacity overflow")?
                        .div_ceil(256),
                ),
            ),
            (
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                heads,
            ),
            (
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                dim.next_power_of_two().max(256),
            ),
        ] {
            let limit = stream
                .device
                .attribute(attribute)
                .map_err(|e| e.to_string())?;
            if limit <= 0 || needed > limit as usize {
                return Err("raw KDA actual device geometry".into());
            }
        }
        let raw = match retained_raw {
            Some(raw) => raw,
            None => Arc::new(NativeRawOwner::allocate(
                stream.clone(),
                raw_schema,
                reserve,
            )?),
        };
        let prepared = Arc::new(NativeRawOwner::allocate(stream.clone(), prepared_schema, reserve)?);
        let a_log = Arc::new(native_zeroed::<f32>(&stream, heads)?);
        let bias = Arc::new(native_zeroed::<f32>(&stream, hd)?);
        let parameter_admissions = vec![
            native_pointer_admission(
                &stream,
                pointer(&a_log),
                a_values.len() * std::mem::size_of::<f32>(),
            )?,
            native_pointer_admission(
                &stream,
                pointer(&bias),
                bias_values.len() * std::mem::size_of::<f32>(),
            )?,
        ];
        for (destination, values) in [
            (pointer(&a_log), a_values.as_slice()),
            (pointer(&bias), bias_values.as_slice()),
        ] {
            let bytes = values
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or("raw parameter upload bytes")?;
            check(
                unsafe {
                    sys::lib().cuMemcpyHtoDAsync_v2(
                        destination,
                        values.as_ptr().cast(),
                        bytes,
                        stream.handle,
                    )
                },
                "raw admitted native parameter upload",
            )?;
        }
        stream.drain()?;
        if memory()?[0] < reserve {
            return Err("raw KDA observed allocation floor".into());
        }
        let mut hasher = Sha256::new();
        hasher.update(kernels.ptx_sha256.as_bytes());
        hasher.update(lower.to_le_bytes());
        hasher.update(scale.to_le_bytes());
        hasher.update(epsilon.to_le_bytes());
        hasher.update((heads as u64).to_le_bytes());
        hasher.update((dim as u64).to_le_bytes());
        hasher.update(
            &a_values
                .iter()
                .flat_map(|x| x.to_bits().to_le_bytes())
                .collect::<Vec<_>>(),
        );
        hasher.update(
            &bias_values
                .iter()
                .flat_map(|x| x.to_bits().to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let arithmetic_sha256 = format!("{:x}", hasher.finalize());
        Ok(Self {
            raw,
            prepared,
            a_log,
            bias,
            kernels,
            heads,
            dim,
            capacity,
            input_capacity,
            lower,
            scale,
            epsilon,
            parameter_admissions,
            requested_bytes,
            allocated_bytes,
            source_epoch: 0,
            arithmetic_sha256,
        })
    }
    fn view(&self, rows: usize) -> Result<KdaRawPreparedView<'_>, String> {
        if rows == 0 || rows > self.capacity {
            return Err("raw KDA prepared view extent".into());
        }
        Ok(KdaRawPreparedView {
            owner: self,
            schema: Schema::checked(
                RawContract::KdaPreparedF32VectorGate,
                rows,
                self.heads,
                self.dim,
                self.dim,
            )?,
        })
    }
    fn enqueue(&self, first: usize, rows: usize, consumer: &PreparedOwner) -> Result<(), String> {
        if rows == 0
            || rows > self.capacity
            || first
                .checked_add(rows)
                .is_none_or(|x| x > self.input_capacity)
            || self.heads != consumer.heads
            || self.dim != consumer.dim
            || !Arc::ptr_eq(&self.raw.device, &consumer.device)
            || self.raw.stream.handle != consumer.stream.handle
        {
            return Err("raw KDA producer/consumer geometry/context/stream".into());
        }
        consumer.stream.validate()?;
        let mut args = typed_pointers(&*self.raw);
        args.extend([
            Parameter::Ptr(pointer(&self.a_log)),
            Parameter::Ptr(pointer(&self.bias)),
            Parameter::Int(first as i32),
            Parameter::Int(rows as i32),
            Parameter::Int(self.heads as i32),
            Parameter::Int(self.dim as i32),
            Parameter::Ptr(pointer(&consumer.error)),
        ]);
        let cells = rows
            .checked_mul(self.heads)
            .and_then(|x| x.checked_mul(self.dim))
            .ok_or("raw KDA grid extent")?;
        invoke(
            &consumer.device,
            consumer.stream.handle,
            &self.kernels.domain,
            (
                u32::try_from(cells.div_ceil(256)).map_err(|_| "raw domain grid")?,
                1,
                1,
            ),
            256,
            0,
            &mut args,
        )?;
        let mut args = typed_pointers(&*self.raw);
        args.extend([
            Parameter::Ptr(pointer(&self.a_log)),
            Parameter::Ptr(pointer(&self.bias)),
        ]);
        args.extend(typed_pointers(self.prepared.as_ref()));
        args.extend([
            Parameter::Int(first as i32),
            Parameter::Int(rows as i32),
            Parameter::Int(self.heads as i32),
            Parameter::Int(self.dim as i32),
            Parameter::Float(self.lower),
            Parameter::Float(self.scale),
            Parameter::Float(self.epsilon),
            Parameter::Ptr(pointer(&consumer.error)),
        ]);
        let threads = self.dim.next_power_of_two().max(32);
        invoke(
            &consumer.device,
            consumer.stream.handle,
            &self.kernels.prepare,
            (
                u32::try_from(rows).map_err(|_| "raw rows grid")?,
                u32::try_from(self.heads).map_err(|_| "raw heads grid")?,
                1,
            ),
            threads as u32,
            (threads * 4) as u32,
            &mut args,
        )
    }
}
