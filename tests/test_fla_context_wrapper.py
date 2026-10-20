"""Cold compiler/ABI checks; no GPU initialization or model execution."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    raise SystemExit("Run through ./dev fla-wrapper-test")

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "src/cuda/fla/compile_kernels.py"
tree = ast.parse(source.read_text(), filename=str(source))
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"generate_context_owned_wrappers", "generate_c_wrapper"}]
namespace = {"os": os}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)

spec = importlib.util.spec_from_file_location("krasis_sidecar_build", ROOT / "scripts/build_sidecars.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
NVCC = module.find_nvcc()


class ContextWrapperTests(unittest.TestCase):
    def test_loaded_identity_tracks_payload_and_contract_not_build_directory(self):
        # These are host-only generation controls, never fake numerical evidence.
        kernels = [{"name": "unit", "safe_name": "unit_h32", "cubin": b"unit",
                    "meta": {"num_warps": 4, "shared": 0, "name": "unit_kernel"},
                    "args": [("input", "CUdeviceptr")]}]
        def generate(root, payload=b"unit", warps=4):
            root.mkdir()
            kernels[0]["cubin"] = payload
            kernels[0]["meta"]["num_warps"] = warps
            (root / "unit_h32.cubin").write_bytes(payload)
            identity = namespace["generate_c_wrapper"](kernels, str(root / "wrapper.cu"), str(root),
                                                        portable_embedded_cubins=False)
            self.assertRegex(identity, r"^[0-9a-f]{64}$")
            self.assertIn('return "' + identity + '";', (root / "wrapper.cu").read_text())
            return identity
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = generate(root / "first")
            self.assertEqual(identity, generate(root / "different-build-directory"))
            self.assertNotEqual(identity, generate(root / "changed-cubin", payload=b"changed"))
            self.assertNotEqual(identity, generate(root / "changed-host-launch", warps=8))

    def test_generated_cpp_compiles_both_embedding_formats(self):
        # Cubin fixture bytes are never loaded or executed. Native CUDA driver
        # types compile both complete legacy and explicit-context host ABIs.
        kernels = [{"name": "unit", "safe_name": "unit_h32", "cubin": b"unit",
                    "meta": {"num_warps": 4, "shared": 65536, "name": "unit_kernel"},
                    "args": [("input", "CUdeviceptr"), ("output", "CUdeviceptr"), ("T", "int32_t")]}]
        for portable in (False, True):
            with self.subTest(portable=portable), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "unit_h32.cubin").write_bytes(b"unit")
                out = root / "wrapper.cu"
                namespace["generate_c_wrapper"](kernels, str(out), str(root),
                                                portable_embedded_cubins=portable)
                text = out.read_text()
                self.assertIn("krasis_fla_unit_h32(", text)
                self.assertIn("krasis_fla_context_unit_h32(\n    void* opaque,\n", text)
                self.assertIn("cuLaunchKernel(ctx->func_unit_h32,", text)
                self.assertIn("cuFuncSetAttribute(ctx->func_unit_h32,", text)
                self.assertIn("current != ctx->cuda_context", text)
                launch = text.split("krasis_fla_context_unit_h32(", 1)[1]
                self.assertNotIn("cuCtxGetCurrent", launch)
                self.assertNotIn("mutex", launch)
                self.assertNotIn("calloc", launch)
                subprocess.run([NVCC, "-std=c++17", "-c", str(out), "-o", str(root / "wrapper.o")],
                               check=True, capture_output=True, text=True)

    def test_create_cleanup_returns_incomplete_owner(self):
        # Host API-contract injection only. Driver functions are explicit local
        # stubs; no CUDA driver/context/device is initialized by this executable.
        kernels = [{"safe_name": name, "meta": {"num_warps": 4, "shared": 0,
                    "name": name}, "args": []} for name in ("first", "second")]
        generated = "\n".join(namespace["generate_context_owned_wrappers"](kernels))
        fixture = r'''
#include <cuda.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#define KRASIS_FLA_EXPORT extern "C"
static unsigned char cubin_first[] = {0}, cubin_second[] = {0};
static int loaded=0, fail_load=0, fail_cleanup=0;
extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext* out){ *out=reinterpret_cast<CUcontext>(1); return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuModuleLoadData(CUmodule* out,const void*){if(fail_load && loaded==1)return CUDA_ERROR_OUT_OF_MEMORY; ++loaded; *out=reinterpret_cast<CUmodule>((uintptr_t)loaded); return CUDA_SUCCESS;}
extern "C" CUresult CUDAAPI cuModuleGetFunction(CUfunction* out,CUmodule,const char*){*out=reinterpret_cast<CUfunction>(1);return CUDA_SUCCESS;}
extern "C" CUresult CUDAAPI cuFuncSetAttribute(CUfunction,CUfunction_attribute,int){return CUDA_SUCCESS;}
extern "C" CUresult CUDAAPI cuModuleUnload(CUmodule){if(fail_cleanup)return CUDA_ERROR_UNKNOWN;--loaded;return CUDA_SUCCESS;}
extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,CUstream,void**,void**){return CUDA_SUCCESS;}
'''
        fixture += generated + r'''
int main(){
 void* owner=nullptr;
 CUresult code=krasis_fla_context_create(&owner);
 if(code!=CUDA_SUCCESS||!owner||loaded!=2)return 11;
 if(krasis_fla_context_destroy(owner)!=CUDA_SUCCESS||loaded!=0)return 12;
 owner=nullptr;fail_load=1;
 code=krasis_fla_context_create(&owner);
 if(code!=CUDA_ERROR_OUT_OF_MEMORY||owner||loaded!=0)return 13;
 fail_cleanup=1;
 code=krasis_fla_context_create(&owner);
 printf("{\"gpu_initialized\":false,\"abi\":%u,\"create_error\":%d,\"incomplete_owner_returned\":%s,\"live_stub_modules\":%d}\n",krasis_fla_context_abi_version(),(int)code,owner?"true":"false",loaded);
 if(!owner||loaded!=1)return 14;
 // Only this explicit CPU stub becomes recoverable. Production Rust must
 // terminate on incomplete CUDA cleanup, never retry and continue serving.
 fail_cleanup=0;
 if(krasis_fla_context_destroy(owner)!=CUDA_SUCCESS||loaded!=0)return 15;
 return krasis_fla_context_abi_version()==2?0:16;
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "context_failure.cu"
            source.write_text(fixture)
            executable = root / "context_failure"
            compiled = subprocess.run([NVCC, "-std=c++17", str(source), "-o", str(executable)],
                                      check=True, capture_output=True, text=True)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            print("FLA host contract injection:", result.stdout.strip(), flush=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_launch_metadata_is_an_error(self):
        for field in ("num_warps", "shared"):
            with self.subTest(field=field):
                metadata = {"num_warps": 4, "shared": 0, "name": "unit_kernel"}
                del metadata[field]
                with self.assertRaises(KeyError):
                    namespace["generate_context_owned_wrappers"]([
                        {"safe_name": "unit", "meta": metadata, "args": []}])


if __name__ == "__main__":
    unittest.main()
