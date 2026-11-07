"""Startup-only host loading for the Rust/CUDA bounded dense execution path."""
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


def host_memory_plan(cfg):
    source=source_bf16_bytes(cfg)
    headroom=(12<<30) if source>(1<<30) else (64<<20)
    required=2*source+headroom
    available=available_host_bytes()
    if available<required:
        raise RuntimeError(f'Bounded BF16 Qwen needs about {required/(1<<30):.1f} GiB available host RAM for source, pinned weights and setup; available={available/(1<<30):.1f} GiB. The 27B profile recommends 128 GiB system RAM.')
    return dict(source_bf16_bytes=source,estimated_peak_bytes=required,available_bytes=available)


def aligned_bytes(tensor):
    return (tensor.numel() * tensor.element_size() + 255) & ~255


def load_host_backed(model, loader):
    if len(model.all_devices) != 1 or model.stream_attention:
        raise ValueError("bounded dense Qwen requires one GPU and stream_attention=False")
    q = model.quant_cfg
    if any(getattr(q, n) != 'bf16' for n in ('attention', 'dense_mlp', 'lm_head')):
        raise ValueError("bounded dense Qwen currently streams source BF16; select BF16 attention/MLP/head")
    if model.cfg.max_position_embeddings > 4096:
        raise ValueError('bounded Qwen requires --max-context-tokens <=4096 (image tokens included)')
    # Fail before reading checkpoint payload if the requested joint plan cannot fit.
    memory_plan(model.cfg, model.cfg.vocab_size * model.cfg.hidden_size * 2,
        model.cfg.max_position_embeddings, int(os.environ.get('KRASIS_QWEN_GPU_LIMIT_MIB', '8192')))
    model._dense_host_plan = host_memory_plan(model.cfg)
    log.info("Bounded Qwen host memory plan: %s", model._dense_host_plan)
    device = model.all_devices[0]
    cpu = torch.device('cpu')
    model.embedding = loader.load_embedding(cpu)
    model.final_norm = loader.load_final_norm(device)
    model.lm_head_data = loader.load_lm_head(cpu)
    model.layers = []
    model._layer_split = [(0, model.cfg.num_hidden_layers)]
    model._deepseek_v4_hc_head = None
    peak = max(aligned_bytes(model.embedding), aligned_bytes(model.lm_head_data))
    source_bytes = model.embedding.nbytes + model.lm_head_data.nbytes
    for i in range(model.cfg.num_hidden_layers):
        w = loader.load_layer(i, cpu, attn_device=cpu)
        # Only the large projection matrices use the reusable slab. Small
        # norms, convolution parameters and state coefficients stay resident.
        large = []
        def prepare(value):
            if isinstance(value, torch.Tensor):
                if value.ndim == 2 and value.numel() >= model.cfg.hidden_size * 16:
                    value = value.contiguous()
                    large.append(value)
                    return value
                return value.to(device)
            if isinstance(value, dict): return {k: prepare(v) for k,v in value.items()}
            return value
        w = prepare(w)
        peak = max(peak, sum(aligned_bytes(t) for t in large))
        source_bytes += sum(t.nbytes for t in large)
        model.layers.append(TransformerLayer(model.cfg, i, w, device,
            krasis_engine=None, gpu_prefill_manager=None,
            gpu_prefill_threshold=model.gpu_prefill_threshold))
        log.info('Host-backed dense layer %d/%d; %d matrix bytes', i+1,
                 model.cfg.num_hidden_layers, sum(t.nbytes for t in large))
    # Source and pinned native owners coexist during setup. The implementation
    # never stages the complete model on GPU, even during loading or warmup.
    model._dense_stream_slab_bytes = peak
    model._dense_stream_host_bytes = source_bytes
    model._dense_memory_plan = memory_plan(model.cfg, peak,
        model.cfg.max_position_embeddings, int(os.environ.get('KRASIS_QWEN_GPU_LIMIT_MIB', '8192')))
    log.info('Joint bounded Qwen memory plan: %s', model._dense_memory_plan)
    log.info('Bounded BF16 source=%d slab=%d; setup host RAM may reach 2x source',
             source_bytes, peak)


def memory_plan(cfg, slab_bytes, context_tokens, limit_mib=8192):
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
    items = dict(weight_slab=slab_bytes, kv=kv, recurrent_state=gdn*2,
        reference_attention=attention, vision_linear_workspace=1<<20,
        prefill_scratch=512<<20, vision_peak=256<<20,
        runtime_and_graphs=1024<<20, reserve=600<<20)
    peak = sum(items.values())
    if peak > limit_mib << 20:
        raise ValueError(f'bounded Qwen plan requires {peak/(1<<20):.0f} MiB including reserve; limit={limit_mib} MiB')
    return dict(allowance_bytes=items, estimated_peak_bytes=peak,
                device_limit_mib=limit_mib, context_tokens=context_tokens,
                max_image_patches=1024, prefill_chunk_tokens=256)
