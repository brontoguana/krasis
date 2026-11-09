"""Exercise the actual build override validator without CUDA compilation."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class OptimisationBuildTest(unittest.TestCase):
    def test_overrides_and_cached_header(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            # Compile the actual generator, substituting only its heavy main.
            source = (ROOT / 'build.rs').read_text()
            source = source.replace('fn main() {', 'fn unused_build_main() {', 1)
            source = source.replace('#[path = "src/optimisations.rs"]',
                '#[path = ' + json.dumps(str(ROOT / 'src/optimisations.rs')) + ']')
            source += '\nfn main() { generate_optimisations(); }\n'
            (d / 'generator.rs').write_text(source)
            subprocess.run([str(Path.home()/'.cargo/bin/rustc'), '--edition=2021', '-Awarnings',
                            str(d/'generator.rs'), '-o', str(d/'generator')], check=True)
            env = dict(os.environ, OUT_DIR=str(d), KRASIS_OPT_DISABLE='', KRASIS_OPT_ENABLE='')
            def run(disable='', enable=''):
                return subprocess.run([str(d/'generator')], env={**env,
                    'KRASIS_OPT_DISABLE': disable, 'KRASIS_OPT_ENABLE': enable}, text=True, capture_output=True)
            first = run(); self.assertEqual(first.returncode, 0, first.stderr)
            header = d/'optimisations.h'; before = header.stat().st_mtime_ns
            self.assertEqual(run().returncode, 0)
            self.assertEqual(header.stat().st_mtime_ns, before)
            changed = run('KDA_RELAY,INDEX_RADIX_TOPK', 'VALUE_REASSOCIATION')
            self.assertEqual(changed.returncode, 0, changed.stderr)
            self.assertIn('#define KRASIS_OPT_KDA_RELAY 0', header.read_text())
            manifest = json.loads(next(line.split('=', 2)[2] for line in changed.stdout.splitlines()
                                      if line.startswith('cargo:rustc-env=KRASIS_OPT_MANIFEST=')))
            self.assertTrue(next(x for x in manifest if x['name']=='VALUE_REASSOCIATION')['enabled'])
            self.assertNotEqual(run('PREFILL_F32', 'VALUE_REASSOCIATION').returncode, 0)
            self.assertNotEqual(run('typo').returncode, 0)
            self.assertNotEqual(run('EXACT_ATTENTION', 'EXACT_ATTENTION').returncode, 0)
            # Build an actual module from the emitted flags, not a hand-written clone.
            env.update({line.split('=',2)[1]:line.split('=',2)[2]
                        for line in changed.stdout.splitlines() if line.startswith('cargo:rustc-env=')})
            (d/'module.rs').write_text('#[path = '+json.dumps(str(ROOT/'src/optimisations.rs'))+']\nmod opts;\nfn main(){assert!(!opts::KDA_RELAY);assert!(opts::VALUE_REASSOCIATION);println!("{}",opts::manifest());}\n')
            subprocess.run([str(Path.home()/'.cargo/bin/rustc'), '--edition=2021', '-Awarnings',
                            str(d/'module.rs'), '-o', str(d/'module')], env=env, check=True)
            subprocess.run([str(d/'module')], check=True, capture_output=True)

if __name__ == '__main__': unittest.main()
