"""Qwen3.5 image preprocessing and native Rust/CUDA encoder handoff.

This module performs processor/position bookkeeping only. The vision network
and its per-block weight staging execute in Qwen35Vision, never Transformers.
"""
import json
from types import SimpleNamespace
from pathlib import Path
import torch


def has_image_assets(path):
    root = Path(path)
    try:
        cfg = json.loads((root/'config.json').read_text())
        v = cfg.get('vision_config', {})
        wm = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        return (cfg.get('model_type') == 'qwen3_5' and
                v.get('model_type') in ('qwen3_5', 'qwen3_5_vision') and
                isinstance(cfg.get('image_token_id'), int) and
                (root/'preprocessor_config.json').is_file() and
                all(k in wm for k in ('model.visual.patch_embed.proj.weight',
                    'model.visual.merger.linear_fc2.weight', 'model.visual.pos_embed.weight')))
    except (OSError, ValueError, KeyError):
        return False


def prepare(model, messages_json, rendered_prompt):
    from krasis.krasis import Qwen35Vision, assemble_qwen35_image
    from transformers import AutoTokenizer, Qwen2VLImageProcessorFast, Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    images = model._extract_openai_images(messages_json)
    if len(images) != 1:
        raise ValueError('bounded Qwen3.5 vision supports exactly one image per request')
    device = model.all_devices[0]
    if model._qwen_vision_processor is None:
        root = model.cfg.model_path
        # Explicit processor classes avoid AutoConfig choosing a Qwen3-VL
        # encoder for Qwen3.5. All tokenizer/processor assets are local-only.
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        image_processor = Qwen2VLImageProcessorFast.from_pretrained(root, local_files_only=True)
        model._qwen_vision_processor = Qwen3VLProcessor(
            image_processor=image_processor, tokenizer=tokenizer,
            video_processor=Qwen3VLVideoProcessor())
        raw = json.loads((Path(root)/'config.json').read_text())
        model._qwen_vision_config = SimpleNamespace(image_token_id=raw['image_token_id'],
            vision_config=SimpleNamespace(**raw['vision_config']))
    processor = model._qwen_vision_processor
    batch = processor(text=[rendered_prompt], images=images, return_tensors='pt',
        return_mm_token_type_ids=True, images_kwargs={'size': {'shortest_edge':65536, 'longest_edge':262144}})
    grid = batch['image_grid_thw']
    t,h,w = map(int, grid[0].tolist())
    if t != 1 or h*w > 1024:
        raise ValueError('bounded Qwen3.5 image exceeds 1024 patches (512x512 pixel area)')
    ids = batch['input_ids'][0]
    if ids.numel() > model.cfg.max_position_embeddings:
        raise ValueError('image and text exceed bounded context limit')
    if model._qwen_vision_model is None:
        model._qwen_vision_model = Qwen35Vision(model.cfg.model_path,
            device.index if device.index is not None else 0, 1024)
    native = model._qwen_vision_model
    pixels = batch['pixel_values'].to(device=device,dtype=torch.bfloat16).contiguous()
    features = torch.empty((h*w//4, model.cfg.hidden_size),device=device,dtype=torch.bfloat16)
    torch.cuda.synchronize(device)
    native.encode(pixels.data_ptr(),h,w,features.data_ptr())
    # Gather only requested text rows from the host source; a full embedding
    # table is never put on GPU. The encoder result replaces image-pad rows.
    inputs = torch.empty((len(ids), model.cfg.hidden_size), device=device, dtype=torch.bfloat16)
    mask = ids == model._qwen_vision_config.image_token_id
    if int(mask.sum()) != len(features):
        raise ValueError('Qwen3.5 image-token/feature count mismatch')
    assemble_qwen35_image(device.index if device.index is not None else 0,
        model.embedding.data_ptr(), model.cfg.vocab_size, model.cfg.hidden_size,
        ids.tolist(), model._qwen_vision_config.image_token_id,
        features.data_ptr(), len(features), inputs.data_ptr())
    types = batch.get('mm_token_type_ids')
    if types is None:
        types = (ids == model._qwen_vision_config.image_token_id).long()[None]
    positions, delta = model._qwen_vl_position_ids_and_delta(ids,types[0],grid)
    cos,sin = model._qwen_vl_mrope_cos_sin(positions)
    cos,sin = cos.to(device),sin.to(device)
    torch.cuda.synchronize(device)
    model._last_multimodal_prefill_tensors = (inputs,cos,sin)
    return dict(token_ids=ids.tolist(),prompt_tokens=ids.numel(),hidden_size=model.cfg.hidden_size,
        inputs_embeds_ptr=inputs.data_ptr(),mrope_cos_ptr=cos.data_ptr(),mrope_sin_ptr=sin.data_ptr(),
        mrope_half_dim=cos.shape[-1],rope_delta=delta,
        vision_block_ids_ptr=0,vision_visible_left_ptr=0,vision_visible_right_ptr=0,
        vision_max_tokens=0,image_count=1,image_tokens=int(mask.sum()))
