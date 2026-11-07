"""Startup-only packing and ownership for the bounded Gemma language profile."""
import gc
import json
import logging
import os
import tempfile
from pathlib import Path
import torch
from krasis.qwen_dense_stream import PackedWeight, PackedCache
from krasis.layer import TransformerLayer
log = logging.getLogger(__name__)
DEFAULT_CONTEXT = 32768


def enabled(cfg):
    if not getattr(cfg, 'gemma4_text', False):
        return False
    setting = os.environ.get('KRASIS_GEMMA_8GB', 'auto').lower()
    if setting not in ('auto','0','1'):
        raise ValueError('KRASIS_GEMMA_8GB must be auto, 0 or 1')
    if setting != 'auto':
        return setting == '1'
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory <= 8 * (1<<30)


class Cache(PackedCache):
    VERSION = 'gemma4-dense-marlin-symmetric-v1'

    def convert(self, value, name, embedding=False):
        if embedding:
            return super().convert(value, name, embedding=True)
        rows, cols = value.shape
        group = 128 if cols % 128 == 0 else 64
        if value.device.type != 'cpu' or value.dtype != torch.bfloat16 or rows % 64 or cols % group:
            raise ValueError(f'Unsupported Gemma INT4 matrix {name}: {value.shape}')
        path = self.root / (name + '.bin')
        meta = path.with_suffix('.json')
        size = ((rows * cols // 2 + 255) & ~255) + rows * (cols // group) * 2
        contract = dict(identity=self.identity, shape=[rows, cols], bytes=size, group_size=group, layout=self.VERSION)
        valid = False
        if path.is_file() and meta.is_file():
            try:
                record = json.loads(meta.read_text())
                valid = all(record.get(k) == v for k, v in contract.items()) and path.stat().st_size == size and record.get('sha256') == self.digest(path)
            except (OSError, ValueError):
                pass
        if not valid:
            fd, temp = tempfile.mkstemp(dir=self.root, prefix=name+'.', suffix='.partial')
            os.close(fd)
            try:
                source = value.contiguous()
                self.store.pack_dense_int4_file(source.data_ptr(), rows, cols, temp, group)
                if Path(temp).stat().st_size != size:
                    raise RuntimeError('Incomplete Gemma packed weight')
                contract['sha256'] = self.digest(Path(temp))
                os.replace(temp, path)
                fd, temp_meta = tempfile.mkstemp(dir=self.root, prefix=name+'.', suffix='.partial')
                try:
                    with os.fdopen(fd, 'w') as f:
                        json.dump(contract, f, sort_keys=True); f.flush(); os.fsync(f.fileno())
                    os.replace(temp_meta, meta)
                finally:
                    Path(temp_meta).unlink(missing_ok=True)
            finally:
                Path(temp).unlink(missing_ok=True)
        self.entries[name] = dict(path=str(path), bytes=size, shape=[rows, cols], group_size=group)
        result = PackedWeight.marker(value.shape, path, size)
        result._packed_group_size = group
        return result


def load(model, loader):
    cfg, q = model.cfg, model.quant_cfg
    if len(model.all_devices) != 1 or model.stream_attention or (cfg.hidden_size, cfg.num_hidden_layers) != (2816, 30):
        raise ValueError('Bounded Gemma requires the 26B-A4B checkpoint on one GPU')
    if (q.attention, q.dense_mlp, q.lm_head, q.kv_cache_format, q.ring_window_kv) != ('hqq4', 'int4', 'int4', 'k6v6', True):
        raise ValueError('Bounded Gemma requires HQQ4 attention, INT4 dense/head and k6v6 ring KV')
    if (q.gpu_expert_bits, q.expert_group_size, q.step_vision_quant, q.step_vision_group_size) != (4, 128, 'int4', 128):
        raise ValueError('Bounded Gemma requires INT4 g128 experts and native INT4 g128 vision')
    if os.environ.get('KRASIS_KV_STAGE_EXACT', '').lower() in ('0', 'false', 'no', 'off'):
        raise ValueError('Bounded Gemma requires exact prefill KV staging')
    # Qualified direct-cache paths keep sliding storage bounded and avoid a
    # prompt-sized concatenation arena. Explicit diagnostic opt-outs remain.
    os.environ.setdefault('KRASIS_GEMMA_FULL_HISTORY_WMMA', '1')
    os.environ.setdefault('KRASIS_GEMMA_BOUNDED_STAGE', '1')
    os.environ.setdefault('KRASIS_GEMMA_FULL_HISTORY_GEMM', '1')
    os.environ.setdefault('KRASIS_GEMMA_WINDOW_GEMM', '1')
    device = model.all_devices[0]
    cpu = torch.device('cpu')
    cache = Cache(model, loader)
    model._layer_split = [(0, cfg.num_hidden_layers)]
    # Embedding remains host BF16: native gather transfers only requested rows.
    # Its tied output projection has its own authoritative INT4 representation.
    model.embedding = cache.convert(loader.load_embedding(cpu), 'embedding', embedding=True)
    loader._handles.clear()
    model.final_norm = loader.load_final_norm(device)
    old_head, old_dense = q.lm_head, q.dense_mlp
    q.lm_head = q.dense_mlp = 'bf16'
    try:
        model.lm_head_data = cache.convert(loader.load_lm_head(cpu), 'head')
        loader._handles.clear()
        model._prepare_hqq_attention_cache()
        model._build_hqq_attention_cache_from_safetensors(loader, device)
        model.layers = []
        for i in range(cfg.num_hidden_layers):
            weights = loader.load_layer(i, cpu, attn_device=cpu)
            dense = weights['dense_mlp']
            for key in list(dense):
                dense[key] = cache.convert(dense[key], f'layer{i}-{key}')
            def move_small(value):
                if isinstance(value, dict):
                    return {k: move_small(v) for k,v in value.items()}
                if isinstance(value, torch.Tensor) and value.ndim < 2:
                    return value.to(device)
                return value
            weights = move_small(weights)
            # Router is small but needs native resident device storage.
            weights['gate'] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in weights['gate'].items()}
            model.layers.append(TransformerLayer(cfg, i, weights, device, krasis_engine=None, gpu_prefill_manager=None, gpu_prefill_threshold=model.gpu_prefill_threshold))
            loader._handles.clear()
        model._validate_hqq_attention_cache()
        model._load_hqq_attention_runtime_state()
    finally:
        q.lm_head, q.dense_mlp = old_head, old_dense
    model._deepseek_v4_hc_head = None
    model._dense_packed_manifest = cache.entries
    # All dense/head matrices stay resident; only routed experts stream.
    resident = sum((e['bytes'] + 255) & ~255 for k,e in cache.entries.items() if k != 'embedding')
    model._gemma_dense_resident_bytes = resident
    del cache
    loader._handles.clear()
    gc.collect()
    model._active_device = str(device)
    model._device_state[str(device)] = dict(layers=model.layers, embedding=model.embedding, final_norm=model.final_norm, lm_head_data=model.lm_head_data)
    log.info('Bounded Gemma dense/head resident bytes=%d, embedding host bytes=%d', resident, model.embedding.nbytes)
