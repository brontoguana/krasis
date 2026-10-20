"""Cold config setup only; production format construction/admission is Rust."""
import json
import os
from pathlib import Path
import sys
if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    raise SystemExit("Use ./dev scalar-gdn-observer <this-script> <absolute-plan> <fresh-output>")
if len(sys.argv) != 3 or not all(Path(arg).is_absolute() for arg in sys.argv[1:]):
    raise SystemExit("Absolute frozen plan and fresh host output are required")
plan_path, output = map(Path, sys.argv[1:])
if output.exists():
    raise RuntimeError("Cold HQQ preflight requires a fresh output")
plan = json.loads(plan_path.read_text())
from krasis.config import ModelConfig
import krasis.krasis as native
expected = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]).resolve() / "krasis"
if Path(native.__file__).resolve().parent != expected:
    raise RuntimeError("Cold preflight imported a different actual runtime")
config = ModelConfig.from_model_path(str(Path(plan["model_config"]["path"]).parent))
# Exact production model.py registration policy, derived from parsed config.
default = bool(config.is_deepseek_v4 or config.is_glm5_next)
result = json.loads(native.hqq_prefill_observation_preflight(str(plan_path), str(output), default))
if not result["passed"] or result["gpu_initialized"] or result["model_loaded"]:
    raise RuntimeError("Cold actual execution-format admission did not complete")
print(json.dumps(result))
