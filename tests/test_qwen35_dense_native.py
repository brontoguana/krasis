"""GPU regression: dense-only native prefill without an expert engine.

Run explicitly with KRASIS_RUN_GPU_TESTS=1 through ./dev python. The tiny
checkpoint has zero attention weights so its independent SwiGLU/logit oracle
can isolate dense registration, normalization and prefill scheduling.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models

from krasis.config import QuantConfig
from krasis.model import KrasisModel

@unittest.skipUnless(os.environ.get('KRASIS_RUN_GPU_TESTS') == '1', 'explicit GPU regression')
class DenseNativeTests(unittest.TestCase):
    def test_dense_prefill_matches_independent_swiglu_oracle(self):
        torch.manual_seed(1234)
        hidden, inter, vocab = 128, 256, 32
        with tempfile.TemporaryDirectory() as root:
            config = json.loads((Path(__file__).parent / 'fixtures/qwen35-27b-config.json').read_text())
            config['text_config'].update(hidden_size=hidden, intermediate_size=inter,
                vocab_size=vocab, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, head_dim=64, max_position_embeddings=32,
                layer_types=['full_attention'], full_attention_interval=1,
                linear_num_key_heads=1, linear_num_value_heads=3,
                eos_token_id=31, rope_parameters={'rope_type':'default',
                    'rope_theta':10000, 'partial_rotary_factor':0.0})
            Path(root, 'config.json').write_text(json.dumps(config))
            vocab_map = {f't{i}': i for i in range(vocab - 1)}
            vocab_map['[UNK]'] = vocab - 1
            Tokenizer(models.WordLevel(vocab_map, unk_token='[UNK]')).save(str(Path(root, 'tokenizer.json')))
            prefix = 'model.language_model'
            weights = {
                f'{prefix}.embed_tokens.weight': torch.randn(vocab, hidden).bfloat16(),
                f'{prefix}.norm.weight': torch.zeros(hidden).bfloat16(),
                'lm_head.weight': (torch.randn(vocab, hidden) * .1).bfloat16(),
            }
            layer = f'{prefix}.layers.0'
            for name in ('input_layernorm','post_attention_layernorm'):
                weights[f'{layer}.{name}.weight'] = torch.zeros(hidden).bfloat16()
            for name, shape in [('q_proj',(256,hidden)),('k_proj',(64,hidden)),
                                ('v_proj',(64,hidden)),('o_proj',(hidden,128))]:
                weights[f'{layer}.self_attn.{name}.weight'] = torch.zeros(shape).bfloat16()
            for name in ('q_norm','k_norm'):
                weights[f'{layer}.self_attn.{name}.weight'] = torch.zeros(64).bfloat16()
            for name, shape in [('gate_proj',(inter,hidden)),('up_proj',(inter,hidden)),
                                ('down_proj',(hidden,inter))]:
                weights[f'{layer}.mlp.{name}.weight'] = (torch.randn(shape) * .02).bfloat16()
            save_file(weights, str(Path(root, 'model.safetensors')))
            Path(root, 'model.safetensors.index.json').write_text(json.dumps({
                'weight_map': {key: 'model.safetensors' for key in weights}}))
            quant = QuantConfig(attention='bf16', dense_mlp='bf16', lm_head='bf16',
                                shared_expert='bf16', kv_cache_format='bf16')
            model = KrasisModel(root, pp_partition=[1], devices=['cuda:0'],
                kv_dtype=torch.bfloat16, quant_cfg=quant, kv_cache_mb=4,
                max_context_tokens=32)
            with mock.patch('krasis.model.Tokenizer', return_value=SimpleNamespace(eos_token_id=31)):
                model.load(gpu_only=True)
            self.assertIsNone(model.krasis_engine)
            store = model.setup_gpu_decode_store()
            # This failed before the native per-layer optional metadata fix.
            store.allocate_prefill_engine(32)
            actual, count, overflow = store.rust_prefill_tokens([1,2,3], 0.0, disable_pinning=True)
            self.assertEqual(count, 3)
            self.assertFalse(overflow)
            # Separate CPU reference using the upstream dense equation; the
            # zero attention output leaves the embedding residual unchanged.
            def reference_logits(token):
                x = weights[f'{prefix}.embed_tokens.weight'][token].float()
                norm = lambda v: (v * torch.rsqrt(v.square().mean() + 1e-6)).bfloat16().float()
                y = norm(x)
                gate = (weights[f'{layer}.mlp.gate_proj.weight'].float() @ y).bfloat16().float()
                up = (weights[f'{layer}.mlp.up_proj.weight'].float() @ y).bfloat16().float()
                act = (torch.nn.functional.silu(gate) * up).bfloat16().float()
                down = (weights[f'{layer}.mlp.down_proj.weight'].float() @ act).bfloat16().float()
                x = (x + down).bfloat16().float()
                return weights['lm_head.weight'].float() @ norm(x)
            self.assertEqual(actual, reference_logits(3).argmax().item())
            tokens = store.gpu_generate_stream_probe(
                tokenizer_path=str(Path(root, 'tokenizer.json')), first_token=actual,
                start_position=3, max_tokens=4, temperature=0.0,
                top_k=1, top_p=1.0, stop_ids=[])
            self.assertEqual(len(tokens), 4)
            self.assertEqual(tokens[0], actual)
            for previous, current in zip(tokens, tokens[1:]):
                self.assertEqual(current, reference_logits(previous).argmax().item())
            # Decode uses the same registered dense weights and explicit layer
            # scheduling. Compare the complete logit vector, not just prose.
            store.debug_gpu_decode_step(4, 3)
            observed = torch.tensor(store.download_logits_f32())
            torch.testing.assert_close(observed, reference_logits(4), rtol=.01, atol=.03)
            model.server_cleanup()
            torch.cuda.synchronize()

if __name__ == '__main__':
    unittest.main()
