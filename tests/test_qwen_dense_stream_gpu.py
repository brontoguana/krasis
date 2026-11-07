"""Explicit GPU checks for bounded dense ownership and image placement.

Uses a nonzero GDN+GQA checkpoint, plus complete operator/logit/state comparisons.
Real 27B and independent Transformers evidence lives in the campaign controllers.
"""
import gc
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer,models
from krasis.config import QuantConfig
from krasis.model import KrasisModel

@unittest.skipUnless(os.environ.get('KRASIS_RUN_GPU_TESTS')=='1','explicit GPU qualification')
class BoundedDenseGpuTests(unittest.TestCase):
    def test_image_placement_copies_only_selected_host_rows(self):
        from krasis.krasis import assemble_qwen35_image
        embedding=torch.arange(16*128,dtype=torch.float32).reshape(16,128).bfloat16()
        features=torch.randn(2,128,device='cuda',dtype=torch.bfloat16)
        ids=[1,3,2,3,4]
        out=torch.empty(5,128,device='cuda',dtype=torch.bfloat16)
        torch.cuda.synchronize()
        assemble_qwen35_image(0,embedding.data_ptr(),16,128,ids,3,features.data_ptr(),2,out.data_ptr())
        expected=embedding[ids].cuda();expected[torch.tensor(ids,device='cuda')==3]=features
        torch.testing.assert_close(out,expected,rtol=0,atol=0)
        for bad_ids,count in [([1,16],0),(ids,1)]:
            with self.assertRaisesRegex(ValueError,'geometry/token mismatch'):
                assemble_qwen35_image(0,embedding.data_ptr(),16,128,bad_ids,3,features.data_ptr(),count,out.data_ptr())

    def test_nonzero_gdn_gqa_streaming_matches_resident_arithmetic(self):
        torch.manual_seed(235682)
        h,inter,vocab,nk,nv,dim=128,256,64,1,16,128
        cfg=json.loads((Path(__file__).parent/'fixtures/qwen35-27b-config.json').read_text())
        cfg['text_config'].update(hidden_size=h,intermediate_size=inter,vocab_size=vocab,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=64,
            layer_types=['linear_attention','full_attention'],full_attention_interval=2,
            linear_num_key_heads=nk,linear_num_value_heads=nv,linear_key_head_dim=dim,
            linear_value_head_dim=dim,max_position_embeddings=512,
            rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':.25},eos_token_id=63)
        weights={};prefix='model.language_model'
        rnd=lambda *shape:(torch.randn(*shape)*.02).bfloat16()
        weights[prefix+'.embed_tokens.weight']=rnd(vocab,h)
        weights[prefix+'.norm.weight']=torch.zeros(h,dtype=torch.bfloat16)
        weights['lm_head.weight']=rnd(vocab,h)
        for i in range(2):
            p=f'{prefix}.layers.{i}'
            for n in ('input_layernorm','post_attention_layernorm'):weights[f'{p}.{n}.weight']=torch.zeros(h,dtype=torch.bfloat16)
            for n,r,c in [('gate_proj',inter,h),('up_proj',inter,h),('down_proj',h,inter)]:weights[f'{p}.mlp.{n}.weight']=rnd(r,c)
            if i==0:
                for n,r,c in [('in_proj_qkv',2*nk*dim+nv*dim,h),('in_proj_z',nv*dim,h),('in_proj_b',nv,h),('in_proj_a',nv,h),('out_proj',h,nv*dim)]:weights[f'{p}.linear_attn.{n}.weight']=rnd(r,c)
                weights[f'{p}.linear_attn.conv1d.weight']=rnd(2*nk*dim+nv*dim,1,4)
                weights[f'{p}.linear_attn.A_log']=torch.full((nv,),-2.,dtype=torch.float32)
                weights[f'{p}.linear_attn.dt_bias']=torch.zeros(nv,dtype=torch.bfloat16)
                weights[f'{p}.linear_attn.norm.weight']=torch.ones(dim,dtype=torch.bfloat16)
            else:
                for n,r,c in [('q_proj',256,h),('k_proj',64,h),('v_proj',64,h),('o_proj',h,128)]:weights[f'{p}.self_attn.{n}.weight']=rnd(r,c)
                for n in ('q_norm','k_norm'):weights[f'{p}.self_attn.{n}.weight']=torch.zeros(64,dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/'config.json').write_text(json.dumps(cfg))
            save_file(weights,str(p/'model.safetensors'))
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'model.safetensors' for k in weights}}))
            Tokenizer(models.WordLevel({f't{i}':i for i in range(vocab)},unk_token='t63')).save(str(p/'tokenizer.json'))
            # Installed FLA sidecar serves the real H=48 model. This H=16
            # ownership fixture explicitly uses the supported native fallback.
            def run(stream):
                with mock.patch.dict(os.environ,{'KRASIS_QWEN_DENSE_STREAM':str(stream),'KRASIS_PREFILL_DIAG_MAX_CHUNK_TOKENS':'256','KRASIS_NO_FLA':'1'}):
                    model=KrasisModel(root,pp_partition=[2],devices=['cuda:0'],kv_dtype=torch.bfloat16,
                        quant_cfg=QuantConfig(attention='bf16',dense_mlp='bf16',lm_head='bf16',shared_expert='bf16',kv_cache_format='bf16'),kv_cache_mb=4,max_context_tokens=512)
                    with mock.patch('krasis.model.Tokenizer',return_value=SimpleNamespace(eos_token_id=63)):
                        model.load(gpu_only=True)
                    store=model.setup_gpu_decode_store();store.allocate_prefill_engine(512)
                    ids=[i%62+1 for i in range(257)]
                    first,count,overflow=store.rust_prefill_tokens(ids,0.,disable_pinning=True)
                    self.assertEqual(count,257);self.assertFalse(overflow)
                    pre=torch.tensor(store.download_prefill_logits_f32())
                    store.debug_gpu_decode_step(first,count)
                    logits=torch.tensor(store.download_logits_f32())
                    la=model.layers[0].attention
                    state=(la._rust_conv_state.cpu(),la._rust_recur_state.cpu(),model.kv_caches[0].k_cache.cpu(),model.kv_caches[0].v_cache.cpu())
                    self.assertTrue(all(bool(t.count_nonzero()) for t in state))
                    self.assertTrue(bool(logits.isfinite().all()))
                    model.server_cleanup()
                    del store,model;gc.collect();torch.cuda.empty_cache()
                    return pre,logits,state
            resident=run(0);repeat=run(0);streamed=run(1)
            for a,b in zip(resident[:2],repeat[:2]):torch.testing.assert_close(a,b,rtol=0,atol=0)
            for a,b in zip(resident[2],repeat[2]):torch.testing.assert_close(a,b,rtol=0,atol=0)
            for a,b in zip(resident[:2],streamed[:2]):torch.testing.assert_close(a,b,rtol=0,atol=0)
            for a,b in zip(resident[2],streamed[2]):torch.testing.assert_close(a,b,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
