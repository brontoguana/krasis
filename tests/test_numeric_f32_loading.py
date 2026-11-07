"""CPU checks for GLM numeric parameter precision, separate from matrix loads."""
import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from krasis.weight_loader import WeightLoader

class NumericF32LoadingTests(unittest.TestCase):
    def loader(self, values):
        x = WeightLoader.__new__(WeightLoader)
        x.cfg = SimpleNamespace(is_deepseek_v4=False, source_fp8_block_size=(2, 2))
        x._read_tensor = values.__getitem__
        x._weight_map = dict.fromkeys(values, 'fixture')
        return x

    def test_source_f32_preserved_and_legacy_control(self):
        w = torch.tensor([1.0001, -.003781, 9.035], dtype=torch.float32)
        x = self.loader({'p': w})
        with patch.dict(os.environ, {'KRASIS_GLM53_PRESERVE_NUMERIC_F32': '1'}):
            self.assertTrue(torch.equal(x._load_numeric_f32('p', 'cpu'), w))
            self.assertEqual(x._load_bf16('p', 'cpu').dtype, torch.bfloat16)
        with patch.dict(os.environ, {'KRASIS_GLM53_PRESERVE_NUMERIC_F32': '0'}):
            self.assertTrue(torch.equal(x._load_numeric_f32('p', 'cpu'), w.bfloat16().float()))

    def test_bf16_and_scaled_fp8_unchanged(self):
        for weight in (torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16),
                       torch.tensor([[1., 2.], [3., 4.]]).to(torch.float8_e4m3fn)):
            x = self.loader({'p': weight, 'p_scale_inv': torch.tensor([[.513]])})
            expected = x._read_and_dequant('p').float()
            with patch.dict(os.environ, {'KRASIS_GLM53_PRESERVE_NUMERIC_F32': '1'}):
                self.assertTrue(torch.equal(x._load_numeric_f32('p', 'cpu'), expected))

if __name__ == '__main__':
    unittest.main()
