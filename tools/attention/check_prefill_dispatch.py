"""Run the real native qualification guard's fallback boundary tests without a GPU."""
import argparse
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('source', type=Path, help='src/gpu_prefill.rs from the source under review')
a = p.parse_args()
text = a.source.read_text()
start = text.index('fn prefill_mla_f32_geometry(')
brace = text.index('{', start)
depth = 1
end = brace + 1
while depth:
    depth += (text[end] == '{') - (text[end] == '}')
    end += 1
function = text[start:end]
tests = r'''
#[test]
fn full_request_qualified_geometry_is_admitted() {
    assert!(prefill_mla_f32_geometry(7,64,256,0,512,2051,0.0625));
}
#[test]
fn other_cache_formats_and_heads_fall_back() {
    for kv in [0,4,5,6,8] { assert!(!prefill_mla_f32_geometry(kv,64,256,0,512,2051,0.0625)); }
    for heads in [0,1,32,63,65,128] { assert!(!prefill_mla_f32_geometry(7,heads,256,0,512,2051,0.0625)); }
}
#[test]
fn other_query_rotary_and_latent_dimensions_fall_back() {
    for nope in [0,128,255,257,512] { assert!(!prefill_mla_f32_geometry(7,64,nope,0,512,2051,0.0625)); }
    for rope in [1,2,64] { assert!(!prefill_mla_f32_geometry(7,64,256,rope,512,2051,0.0625)); }
    for width in [0,256,511,513,1024] { assert!(!prefill_mla_f32_geometry(7,64,256,0,width,2051,0.0625)); }
}
#[test]
fn operator_only_selection_counts_do_not_enable_runtime_dispatch() {
    for slots in [0,1,17,129,2050,2052,4096,4097] {
        assert!(!prefill_mla_f32_geometry(7,64,256,0,512,slots,0.0625));
    }
}
#[test]
fn other_or_nonfinite_scales_fall_back() {
    for scale in [0.0,-0.0625,0.03125,0.125,f32::NAN,f32::INFINITY] {
        assert!(!prefill_mla_f32_geometry(7,64,256,0,512,2051,scale));
    }
}
'''
with tempfile.TemporaryDirectory(prefix='krasis-prefill-dispatch-') as tmp:
    source = Path(tmp) / 'check.rs'
    binary = Path(tmp) / 'check'
    source.write_text(function + '\n' + tests)
    subprocess.run(['rustc', '--edition=2021', '--test', str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
