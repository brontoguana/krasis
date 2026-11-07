"""Actual CUDA half-split RoPE kernels must read each pair before either write."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
NVCC = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"

class RotaryOwnership(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or not Path(NVCC).is_file():
            raise unittest.SkipTest("CUDA and nvcc required")
        cls.temp = tempfile.TemporaryDirectory()
        source = "#include <cuda_bf16.h>\n#include <cuda_runtime.h>\n__device__ float bf16_to_float(__nv_bfloat16 x){return __bfloat162float(x); }\n__device__ __nv_bfloat16 float_to_bf16(float x){return __float2bfloat16(x); }\n"
        for label, file, name in [("prefill", "prefill_kernels.cu", "rope_batched_half_split_kernel"), ("decode", "decode_kernels.cu", "apply_rope_half_split"), ("graph", "decode_kernels.cu", "apply_rope_half_split_g")]:
            text = (ROOT / "src/cuda" / file).read_text()
            start = text.index('extern "C" __global__ void ' + name + '(')
            end = text.index("\n}\n", start) + 3
            source += text[start:end]
            if label == "prefill":
                launch = f"{name}<<<rows,256,0,stream>>>((__nv_bfloat16*)q,(__nv_bfloat16*)k,pos,c,s,nq,nk,hd,half);"
            else:
                position = "*pos" if label == "decode" else "pos"
                launch = f"{name}<<<((nq+nk)*hd+255)/256,256,0,stream>>>((float*)q,(float*)k,c,s,{position},nq,nk,hd,half);"
            source += f'extern "C" int run_{label}(void*q,void*k,const int*pos,const float*c,const float*s,int rows,int nq,int nk,int hd,int half,cudaStream_t stream){{{launch}return int(cudaGetLastError());}}\n'
        path = Path(cls.temp.name)
        (path / "test.cu").write_text(source)
        subprocess.run([NVCC, "-shared", "-Xcompiler", "-fPIC", "-allow-unsupported-compiler", "-O3", "-arch=sm_80", "--use_fast_math", str(path / "test.cu"), "-o", str(path / "test.so")], check=True, capture_output=True)
        cls.library = ctypes.CDLL(str(path / "test.so"))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_quarter_turn_and_untouched_tail(self):
        # A quarter turn has an exact independent result in both BF16 and FP32.
        # Full and partial rotary widths exercise cross-warp/block ownership.
        for label in ["prefill", "decode", "graph"]:
            function = getattr(self.library, "run_" + label)
            function.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
            function.restype = ctypes.c_int
            for hd, half in [(256, 128), (512, 256), (512, 64)]:
                with self.subTest(label=label, head_dim=hd, half=half):
                    rows = 37 if label == "prefill" else 1
                    dtype = torch.bfloat16 if label == "prefill" else torch.float32
                    generator = torch.Generator().manual_seed(671)
                    originals = [torch.randint(-100, 100, (rows, heads, hd), generator=generator).to(dtype) for heads in [16, 8]]
                    expected = []
                    for x in originals:
                        y = x.clone()
                        y[..., :half] = -x[..., half:2*half]
                        y[..., half:2*half] = x[..., :half]
                        expected.append(y)
                    c = torch.zeros((rows, half), device="cuda")
                    s = torch.ones_like(c)
                    positions = torch.arange(rows, device="cuda", dtype=torch.int32)
                    host_position = ctypes.c_int(0)
                    for repeat in range(8):
                        q, k = [x.cuda() for x in originals]
                        torch.cuda.synchronize()
                        pos = ctypes.addressof(host_position) if label == "decode" else positions.data_ptr()
                        self.assertEqual(function(q.data_ptr(), k.data_ptr(), pos, c.data_ptr(), s.data_ptr(), rows, 16, 8, hd, half, torch.cuda.current_stream().cuda_stream), 0)
                        torch.cuda.synchronize()
                        for actual, want in zip([q, k], expected):
                            self.assertTrue(torch.equal(actual.cpu(), want), f"{label}, repeat {repeat}")

if __name__ == "__main__":
    unittest.main()
