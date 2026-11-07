"""Gemma image preprocessing and startup-only packed cache conversion."""
import gc
import hashlib
import json
import logging
import time
import os
import tempfile
from pathlib import Path
import torch
from safetensors import safe_open
from krasis.gemma4_vision import Gemma4ImagePreprocessor
from krasis.checkpoint_identity import checkpoint_identity
from krasis.step_vision_int4 import _quantize_int4_weight


def load(model):
    from krasis.krasis import Gemma4Vision
    root = Path(model.cfg.model_path)
    raw = json.loads((root/'config.json').read_text())
    v = raw['vision_config']
    expected = dict(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27, num_attention_heads=16, num_key_value_heads=16, head_dim=72, patch_size=16, pooling_kernel_size=3, position_embedding_size=10240, standardize=True, use_clipped_linears=False)
    if any(v.get(k) != val for k,val in expected.items()) or v.get('hidden_activation') != 'gelu_pytorch_tanh':
        raise ValueError('Native Gemma vision geometry is not supported')
    version = 'gemma4-vision-simple-int4-g128-v1'
    identity = hashlib.sha256(json.dumps(dict(checkpoint=checkpoint_identity(str(root)), format=version, config=v), sort_keys=True).encode()).hexdigest()
    cache = root/'.krasis'/version/identity
    cache.mkdir(parents=True, exist_ok=True)
    wm = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    entries = []
    manifest = {}
    for full, shard in sorted(wm.items()):
        if full.startswith('model.vision_tower.'):
            name = full.removeprefix('model.vision_tower.')
        elif full.startswith('model.embed_vision.'):
            name = 'embed_vision.'+full.removeprefix('model.embed_vision.')
        else:
            continue
        group = int(name.split('.')[2])+1 if name.startswith('encoder.layers.') else (28 if name.startswith('embed_vision.') or name.startswith('std_') else 0)
        packed = name.endswith('.weight') and ('proj' in name) and not name.endswith('norm.weight')
        path = cache/(name+'.bin')
        meta = path.with_suffix('.json')
        valid = False
        if path.is_file() and meta.is_file():
            try:
                record = json.loads(meta.read_text())
                valid = record['identity'] == identity and record['packed'] == packed and record['bytes'] == path.stat().st_size and record['sha256'] == digest(path)
            except (KeyError, ValueError, OSError):
                pass
        if not valid:
            with safe_open(str(root/shard), framework='pt', device='cpu') as f:
                value = f.get_tensor(full).to(torch.bfloat16).contiguous()
                shape = list(value.shape)
                rows = shape[0] if len(shape)>1 else 1
                cols = value.numel()//rows
                if packed:
                    p,s,_,_ = _quantize_int4_weight(value,128)
                    payload = p.numpy().tobytes()+s.view(torch.uint8).numpy().tobytes()
                else:
                    payload = value.view(torch.uint8).numpy().tobytes()
                del value
            fd,tmp = tempfile.mkstemp(dir=cache,prefix=name+'.',suffix='.partial')
            try:
                with os.fdopen(fd,'wb') as f:
                    f.write(payload);f.flush();os.fsync(f.fileno())
                record=dict(identity=identity,shape=shape,rows=rows,cols=cols,packed=packed,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
                os.replace(tmp,path)
                fd,mtmp=tempfile.mkstemp(dir=cache,prefix=name+'.',suffix='.partial')
                try:
                    with os.fdopen(fd,'w') as f:
                        json.dump(record,f,sort_keys=True);f.flush();os.fsync(f.fileno())
                    os.replace(mtmp,meta)
                finally:
                    Path(mtmp).unlink(missing_ok=True)
            finally:
                Path(tmp).unlink(missing_ok=True)
            del payload
            if packed:del p,s
            gc.collect()
        entries.append((name,str(path),record['rows'],record['cols'],packed,group))
        manifest[name]=record
    # The native encoder consumes all execution weights and controls scheduling.
    model._gemma_native_vision = Gemma4Vision(entries,model.all_devices[0].index or 0,2520)
    model._gemma_native_vision_manifest = manifest
    model._gemma_vision_raw_config=raw
    p=json.loads((root/'processor_config.json').read_text())
    image=p.get('image_processor')or{}
    model._gemma_vision_processor=Gemma4ImagePreprocessor(patch_size=16,pooling_kernel_size=3,max_soft_tokens=int(image.get('max_soft_tokens',280)),rescale_factor=float(image.get('rescale_factor',1/255)))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8<<20):h.update(block)
    return h.hexdigest()


def prepare(model,messages_json,rendered_prompt):
    images=model._extract_openai_images(messages_json)
    if len(images)!=1:
        raise ValueError('Bounded Gemma supports exactly one image per request')
    device=model.all_devices[0]
    # Admission precedes even the encoder's persistent slab allocation.
    estimated=len(model.tokenizer.encode(rendered_prompt,add_special_tokens=False))+280
    store=getattr(model,'_gpu_decode_store',None)
    required=(600+512)*(1<<20)+estimated*(model.cfg.hidden_size*2+4)
    if store is not None:
        store.py_hcs_evict_for_prefill(estimated,required)
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info(device)[0]<required:
        from krasis.model import KrasisVisionVramError
        raise KrasisVisionVramError('Insufficient VRAM for bounded Gemma image staging and the 600 MiB reserve; reduce context or expert residency.')
    if getattr(model,'_gemma_native_vision',None) is None:
        load(model)
    batch=model._gemma_vision_processor(images)
    count=int(batch['num_soft_tokens_per_image'][0])
    n=count*9
    pos=batch['image_position_ids'][0,:n]
    gh=int(pos[:,1].max())+1;gw=int(pos[:,0].max())+1
    ids=model.tokenizer.encode(rendered_prompt,add_special_tokens=False)
    ids,blocks=model._gemma_expand_image_token_ids(ids,[count])
    if len(ids)>model.cfg.max_position_embeddings:
        raise ValueError('Image and text exceed bounded Gemma context limit')
    device=model.all_devices[0]
    # Reclaim routed expert residency before the transient encoder/embedding
    # phase. The 512 MiB allowance bounds its slab, working tensors and library
    # workspace; the request embedding remains live through prefill.
    store=getattr(model,'_gpu_decode_store',None)
    required=(600+512)*(1<<20)+len(ids)*(model.cfg.hidden_size*2+4)
    if store is not None:
        store.py_hcs_evict_for_prefill(len(ids),required)
    torch.cuda.empty_cache()
    free,_=torch.cuda.mem_get_info(device)
    if free<required:
        from krasis.model import KrasisVisionVramError
        raise KrasisVisionVramError(f'Bounded Gemma image staging requires {required>>20} MiB free including the 600 MiB reserve; available {free>>20} MiB. Reduce context or expert residency.')
    pixels=batch['pixel_values'][0,:n].to(device=device,dtype=torch.bfloat16).contiguous()
    features=torch.empty((count,model.cfg.hidden_size),device=device,dtype=torch.bfloat16)
    inputs=torch.empty((len(ids),model.cfg.hidden_size),device=device,dtype=torch.bfloat16)
    block_tensor=torch.tensor(blocks,device=device,dtype=torch.int32)
    torch.cuda.synchronize(device)
    native=model._gemma_native_vision
    started=time.perf_counter()
    native.encode(pixels.data_ptr(),gh,gw,features.data_ptr())
    encoded=time.perf_counter()
    native.assemble(getattr(model, "_gemma_embedding_host_ptr", model.embedding.data_ptr()),model.cfg.vocab_size,model.cfg.hidden_size,ids,int(model._gemma_vision_raw_config['image_token_id']),features.data_ptr(),count,inputs.data_ptr(),float(model.cfg.embedding_scale))
    logging.getLogger(__name__).info('Native Gemma image grid=%dx%d soft_tokens=%d prompt_tokens=%d encode_ms=%.3f assemble_ms=%.3f free_mib=%d',gh,gw,count,len(ids),(encoded-started)*1000,(time.perf_counter()-encoded)*1000,torch.cuda.mem_get_info(device)[0]>>20)
    model._last_multimodal_prefill_tensors=(inputs,block_tensor)
    return dict(token_ids=ids,prompt_tokens=len(ids),hidden_size=model.cfg.hidden_size,inputs_embeds_ptr=inputs.data_ptr(),vision_block_ids_ptr=block_tensor.data_ptr(),vision_visible_left_ptr=0,vision_visible_right_ptr=0,vision_max_tokens=count,image_count=1,image_tokens=count,mrope_cos_ptr=0,mrope_sin_ptr=0,mrope_half_dim=0,rope_delta=0)
