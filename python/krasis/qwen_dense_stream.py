"""Startup-only host loading for the Rust/CUDA bounded dense execution path."""
import hashlib
import json
import tempfile
import logging
import os
from pathlib import Path
import torch
from krasis.layer import TransformerLayer
log = logging.getLogger(__name__)


def enabled(cfg):
    """Default only the requested 27B geometry to streaming; keep other models' policy."""
    if getattr(cfg, 'model_type', '') != 'qwen3_5_text':
        return False
    value = os.environ.get('KRASIS_QWEN_DENSE_STREAM')
    if value is None:
        return (getattr(cfg, 'hidden_size', 0), getattr(cfg, 'num_hidden_layers', 0),
                getattr(cfg, 'intermediate_size', 0)) == (5120,64,17408)
    if value.lower() not in ('0','1','false','true','off','on'):
        raise ValueError('KRASIS_QWEN_DENSE_STREAM must be 0/1, false/true or off/on')
    return value.lower() in ('1','true','on')


def source_bf16_bytes(cfg):
    h=cfg.hidden_size
    full=cfg.num_full_attention_layers
    linear=cfg.num_hidden_layers-full
    gqa=h*(2*cfg.num_attention_heads*cfg.head_dim + 2*cfg.num_key_value_heads*cfg.head_dim + cfg.num_attention_heads*cfg.head_dim)
    gdn=h*(2*cfg.linear_num_key_heads*cfg.linear_key_head_dim + 3*cfg.linear_num_value_heads*cfg.linear_value_head_dim + 2*cfg.linear_num_value_heads)
    return 2*(2*cfg.vocab_size*h + 3*cfg.num_hidden_layers*h*cfg.intermediate_size + full*gqa + linear*gdn)


def available_host_bytes():
    """Available physical RAM, bounded by the current Linux memory cgroup."""
    available=[]
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):available.append(int(line.split()[1])*1024)
    except (OSError,ValueError):pass
    if os.name == 'nt':
        import ctypes
        class Status(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
        status=Status();status.length=ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):available.append(status.avail_phys)
    root=Path('/sys/fs/cgroup')
    paths={root}
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            _,controllers,relative=line.split(':',2)
            if controllers=='':
                node=(root/relative.lstrip('/')).resolve()
                while node.is_relative_to(root):
                    paths.add(node)
                    if node==root:break
                    node=node.parent
    except (OSError,ValueError):pass
    for node in paths:
        try:
            cap=(node/'memory.max').read_text().strip()
            if cap!='max':available.append(max(0,int(cap)-int((node/'memory.current').read_text())))
        except (OSError,ValueError):pass
    if not available:raise RuntimeError('Cannot determine available host RAM for bounded Qwen loading')
    return min(available)


def host_memory_plan(cfg, packed=False):
    source=source_bf16_bytes(cfg)
    headroom=(12<<30) if source>(1<<30) else (64<<20)
    required=(source//4+6*(1<<30)) if packed else 2*source+headroom
    available=available_host_bytes()
    if available<required:
        profile='INT4' if packed else 'BF16 reference'
        recommended=32 if packed else 128
        raise RuntimeError(f'Bounded {profile} Qwen needs about {required/(1<<30):.1f} GiB available host RAM for weights and setup; available={available/(1<<30):.1f} GiB. The 27B profile recommends {recommended} GiB system RAM.')
    return dict(source_bf16_bytes=source,estimated_peak_bytes=required,available_bytes=available)


def aligned_bytes(tensor):
    return (tensor.numel() * tensor.element_size() + 255) & ~255


class PackedWeight:
    """Startup descriptor. Shape-only tensor retains no source weight storage."""
    @staticmethod
    def marker(shape, path, size):
        value = torch.empty(shape, dtype=torch.bfloat16, device='meta')
        value._packed_file = str(path)
        value._packed_bytes = size
        return value


class PackedCache:
    VERSION = 'qwen-dense-marlin-symmetric-g128-v1'
    def __init__(self, model, loader):
        from krasis.checkpoint_identity import checkpoint_identity
        from krasis import GpuDecodeStore
        identity = dict(checkpoint=checkpoint_identity(model.cfg.model_path),
                        format=self.VERSION, config=json.loads(Path(model.cfg.model_path,'config.json').read_text()))
        self.identity = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.root = Path(os.environ.get('KRASIS_QWEN_INT4_CACHE',str(Path(model.cfg.model_path)/'.krasis')))/self.VERSION/self.identity
        self.root.mkdir(parents=True, exist_ok=True)
        self.device=model.all_devices[0]
        self.store = GpuDecodeStore(torch.device(self.device).index or 0)
        self.entries = {}
        self.cfg=model.cfg

    def convert(self, value, name, embedding=False):
        if value.device.type!="cpu" or value.dtype!=torch.bfloat16:
            raise ValueError("INT4 conversion requires host BF16 source tensors")
        if name.endswith('in_proj_qkvz'):
            cfg=self.cfg;nk=cfg.linear_num_key_heads;dk=cfg.linear_key_head_dim
            band=(cfg.linear_num_value_heads//nk)*cfg.linear_value_head_dim
            pieces=value.reshape(nk,2*dk+2*band,value.shape[1])
            canonical=torch.cat((pieces[:,:dk].reshape(-1,value.shape[1]),pieces[:,dk:2*dk].reshape(-1,value.shape[1]),pieces[:,2*dk:2*dk+band].reshape(-1,value.shape[1])))
            z=pieces[:,2*dk+band:].reshape(-1,value.shape[1]).contiguous()
            first=self.convert(canonical,name+'-qkv');second=self.convert(z,name+'-z')
            result=PackedWeight.marker(value.shape,'',first._packed_bytes+second._packed_bytes)
            result._packed_parts=(first,second,nk,dk,band)
            return result
        rows, cols = value.shape
        if not embedding and (rows % 64 or cols % 128):
            # Only the explicitly bounded gate exception may remain BF16.
            if not name.endswith('in_proj_ba') or rows>128 or cols>self.cfg.hidden_size:
                raise ValueError(f'Unsupported streamed INT4 projection shape {name}: {rows}x{cols}')
            return value.to(self.device)
        path = self.root/(name+'.bin')
        meta = self.root/(name+'.json')
        expected = rows*cols*2 if embedding else ((rows*cols//2+255)&~255)+rows*(cols//128)*2
        contract = dict(identity=self.identity, shape=[rows,cols], bytes=expected,
                        layout='bf16-mapped-embedding' if embedding else self.VERSION)
        valid = False
        if path.is_file() and meta.is_file():
            try:
                record=json.loads(meta.read_text())
                valid=all(record.get(k)==v for k,v in contract.items()) and path.stat().st_size==expected
                if valid:
                    valid=record.get('sha256')==self.digest(path)
            except (OSError,ValueError):pass
        if not valid:
            # Atomic publication; an interrupted file never becomes a cache hit.
            fd, tmp = tempfile.mkstemp(dir=self.root, prefix=name+'.', suffix='.partial')
            os.close(fd)
            try:
                if embedding:
                    with open(tmp,'wb') as f:
                        f.write(value.contiguous().view(torch.uint8).numpy().data)
                        f.flush();os.fsync(f.fileno())
                else:
                    source=value.contiguous()
                    self.store.pack_dense_int4_file(source.data_ptr(),rows,cols,tmp)
                if Path(tmp).stat().st_size!=expected:raise RuntimeError('incomplete packed weight')
                contract['sha256']=self.digest(Path(tmp))
                os.replace(tmp,path)
                meta_fd,metadata_tmp=tempfile.mkstemp(dir=self.root,prefix=name+'.manifest.',suffix='.partial')
                try:
                    with os.fdopen(meta_fd,'w') as f:
                        json.dump(contract,f,sort_keys=True);f.flush();os.fsync(f.fileno())
                    os.replace(metadata_tmp,meta)
                finally:
                    Path(metadata_tmp).unlink(missing_ok=True)
            finally:
                Path(tmp).unlink(missing_ok=True)
        self.entries[name]=dict(path=str(path),bytes=expected,shape=[rows,cols])
        if embedding:
            result=torch.from_file(str(path),shared=False,size=rows*cols,dtype=torch.bfloat16).reshape(rows,cols)
            result._packed_file=str(path)
            result._packed_bytes=expected
            return result
        return PackedWeight.marker(value.shape,path,expected)

    @staticmethod
    def digest(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            while block:=f.read(8<<20):h.update(block)
        return h.hexdigest()


def register_weight(store, group, weight):
    if hasattr(weight,'_packed_parts'):
        first,second,nk,dk,band=weight._packed_parts
        return store.register_streamed_qwen_split(group,first._packed_file,second._packed_file,first.shape[0],second.shape[0],weight.shape[1],nk,dk,band)
    if hasattr(weight,'_packed_file'):
        return store.register_streamed_int4_file(group,weight._packed_file,*weight.shape)
    if weight.is_cuda:
        return store.register_weight(weight.data_ptr(),*weight.shape,0)
    return store.register_streamed_weight(group,weight.data_ptr(),*weight.shape)


def load_host_backed(model, loader):
    if len(model.all_devices) != 1 or model.stream_attention:
        raise ValueError("bounded dense Qwen requires one GPU and stream_attention=False")
    q = model.quant_cfg
    modes = tuple(getattr(q,n) for n in ('attention','dense_mlp','lm_head'))
    if modes not in (('bf16',)*3, ('int4',)*3):
        raise ValueError('bounded Qwen requires uniform INT4 or explicit reference BF16')
    packed = modes == ('int4',)*3
    cache = None
    if model.cfg.max_position_embeddings > 4096:
        raise ValueError('bounded Qwen requires --max-context-tokens <=4096 (image tokens included)')
    # Fail before reading checkpoint payload if the requested joint plan cannot fit.
    memory_plan(model.cfg, model.cfg.vocab_size * model.cfg.hidden_size * (33 if packed else 128)//64,
        model.cfg.max_position_embeddings, int(os.environ.get('KRASIS_QWEN_GPU_LIMIT_MIB', '8192')))
    model._dense_host_plan = host_memory_plan(model.cfg, packed=packed)
    log.info("Bounded Qwen host memory plan: %s", model._dense_host_plan)
    device = model.all_devices[0]
    limit_mib=min(int(os.environ.get('KRASIS_QWEN_GPU_LIMIT_MIB','8192')),torch.cuda.get_device_properties(device).total_memory>>20)
    if packed:
        initial_plan=packed_device_plan(model.cfg,limit_mib)
        memory_plan(model.cfg,initial_plan["slab_bytes"],model.cfg.max_position_embeddings,limit_mib,initial_plan["resident_bytes"])
    cache=PackedCache(model,loader) if packed else None
    cpu = torch.device('cpu')
    model.embedding = loader.load_embedding(cpu)
    if packed:
        model.embedding=cache.convert(model.embedding,"embedding",embedding=True)
        loader._handles.clear()
    model.final_norm = loader.load_final_norm(device)
    original_head_quant=loader.quant_cfg.lm_head
    loader.quant_cfg.lm_head="bf16"
    try:model.lm_head_data=loader.load_lm_head(cpu)
    finally:loader.quant_cfg.lm_head=original_head_quant
    if packed:
        model.lm_head_data=cache.convert(model.lm_head_data,"head")
        loader._handles.clear()
    model.layers = []
    model._layer_split = [(0, model.cfg.num_hidden_layers)]
    model._deepseek_v4_hc_head = None
    peak = getattr(model.lm_head_data,"_packed_bytes",aligned_bytes(model.lm_head_data)) if packed else max(aligned_bytes(model.embedding), aligned_bytes(model.lm_head_data))
    source_bytes = model.embedding.nbytes + getattr(model.lm_head_data,"_packed_bytes",model.lm_head_data.nbytes)
    for i in range(model.cfg.num_hidden_layers):
        w = loader.load_layer(i, cpu, attn_device=cpu)
        # Only the large projection matrices use the reusable slab. Small
        # norms, convolution parameters and state coefficients stay resident.
        large = []
        def prepare(value, name=""):
            if isinstance(value, torch.Tensor):
                if value.ndim == 2 and value.numel() >= model.cfg.hidden_size * 16:
                    value = cache.convert(value,f"layer{i}-{name}") if packed else value.contiguous()
                    if value.device.type != "cuda":large.append(value)
                    return value
                return value.to(device)
            if isinstance(value, dict):
                for key in list(value):value[key]=prepare(value[key],name+"-"+key)
                return value
            return value
        w = prepare(w)
        if packed:loader._handles.clear()
        peak = max(peak, sum(getattr(t,"_packed_bytes",aligned_bytes(t)) for t in large))
        source_bytes += sum(getattr(t,"_packed_bytes",t.nbytes) for t in large)
        model.layers.append(TransformerLayer(model.cfg, i, w, device,
            krasis_engine=None, gpu_prefill_manager=None,
            gpu_prefill_threshold=model.gpu_prefill_threshold))
        log.info('Host-backed dense layer %d/%d; %d matrix bytes', i+1,
                 model.cfg.num_hidden_layers, sum(getattr(t,'_packed_bytes',t.nbytes) for t in large))
    if packed:
        model._dense_packed_manifest=cache.entries
        del cache
        loader._handles.clear()
    # Source and pinned native owners coexist in explicit BF16 reference setup. The implementation
    # never stages the complete model on GPU, even during loading or warmup.
    device_plan=packed_device_plan(model.cfg,limit_mib) if packed else dict(slab_bytes=peak,resident_bytes=0)
    peak=device_plan["slab_bytes"]
    model._dense_stream_slab_bytes = peak
    model._dense_stream_host_bytes = source_bytes
    model._dense_memory_plan = memory_plan(model.cfg, peak,
        model.cfg.max_position_embeddings,limit_mib,
        resident_bytes=device_plan["resident_bytes"])
    log.info('Joint bounded Qwen memory plan: %s', model._dense_memory_plan)
    log.info('Bounded Qwen format=%s authoritative_host_bytes=%d slab_bytes=%d',
             'INT4' if packed else 'BF16 reference',source_bytes,peak)


def packed_device_plan(cfg,limit_mib):
    """Protect the resident head before sizing the reusable cold-layer slab.

    Registration is deterministic: head, then layer projections and MLPs.
    A protected head never needs the cold slab. Reinvest its unused slab bytes
    into resident weights, keeping the original combined device allocation.
    Explicit resident budgets retain their exact meaning.
    """
    h=cfg.hidden_size
    mlp=3*h*cfg.intermediate_size
    full=h*(3*cfg.num_attention_heads*cfg.head_dim+2*cfg.num_key_value_heads*cfg.head_dim)
    linear=h*(2*cfg.linear_num_key_heads*cfg.linear_key_head_dim+3*cfg.linear_num_value_heads*cfg.linear_value_head_dim)
    layer=max(([full] if cfg.num_full_attention_layers else [])+([linear] if cfg.num_hidden_layers>cfg.num_full_attention_layers else []))+mlp
    cold=layer*33//64
    head=cfg.vocab_size*h*33//64
    resident=resident_budget_bytes(limit_mib)
    if resident<head:return dict(slab_bytes=max(head,cold),resident_bytes=resident)
    if 'KRASIS_QWEN_RESIDENT_MIB' not in os.environ:
        resident+=max(0,head-cold)
    return dict(slab_bytes=cold,resident_bytes=resident)


def tensorcore_decode_enabled():
    value=os.environ.get('KRASIS_QWEN_INT4_TENSORCORE_DECODE','1').lower()
    if value not in ('0','1','false','true','off','on'):
        raise ValueError('KRASIS_QWEN_INT4_TENSORCORE_DECODE must be boolean')
    return value in ('1','true','on')


def resident_budget_bytes(limit_mib=None):
    limit=limit_mib if limit_mib is not None else int(os.environ.get('KRASIS_QWEN_GPU_LIMIT_MIB','8192'))
    mib=int(os.environ.get('KRASIS_QWEN_RESIDENT_MIB',str(max(0,min(limit,8192)-4096))))
    if mib<0:raise ValueError('resident weight budget must be nonnegative')
    return mib<<20


def memory_plan(cfg, slab_bytes, context_tokens, limit_mib=8192, resident_bytes=0):
    """Conservative joint peak plan; these allowances are not allocations."""
    if not 1 <= context_tokens <= 4096:
        raise ValueError('bounded Qwen context must be in 1..4096, including image tokens')
    full = cfg.num_full_attention_layers
    kv = full * cfg.num_key_value_heads * cfg.head_dim * 4 * context_tokens
    gdn = (cfg.num_hidden_layers-full) * cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim * 4
    # Rust and Torch state views, 256-row prefill, graph workspaces, native
    # vision temporaries and CUDA libraries are charged together with reserve.
    heads, dim = cfg.num_attention_heads, cfg.head_dim
    band = min(256, context_tokens, max(1, (16<<20)//(heads*context_tokens*2)))
    attention = (2*heads*context_tokens*dim + 2*heads*band*dim + heads*band*context_tokens)*2
    items = dict(weight_slab=slab_bytes, resident_weights=resident_bytes, kv=kv, recurrent_state=gdn*2,
        reference_attention=attention, recurrence_workspace=32<<20, vision_linear_workspace=1<<20,
        prefill_scratch=512<<20, vision_peak=256<<20,
        runtime_and_graphs=1024<<20, reserve=600<<20)
    peak = sum(items.values())
    if peak > limit_mib << 20:
        raise ValueError(f'bounded Qwen plan requires {peak/(1<<20):.0f} MiB including reserve; limit={limit_mib} MiB')
    return dict(allowance_bytes=items, estimated_peak_bytes=peak,
                device_limit_mib=limit_mib, context_tokens=context_tokens,
                max_image_patches=1024, prefill_chunk_tokens=256)
