"""Release only native-owned GLM source tensors; preserve live gate aliases."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from krasis.model import KrasisModel

class SourceTensor:
    def __init__(self, device="cuda"):
        self.device = SimpleNamespace(type=device)
    def numel(self): return 128
    def element_size(self): return 2
    def cpu(self): return SourceTensor("cpu")

class SharedSourceOwnershipTests(unittest.TestCase):
    def model(self):
        m = KrasisModel.__new__(KrasisModel)
        m.cfg = SimpleNamespace(is_glm5_next=True)
        m.all_devices = [0]
        gate = SourceTensor()
        m.layers = [SimpleNamespace(shared_expert={"gate_up_proj": (SourceTensor(), SourceTensor()),
            "down_proj": SourceTensor(), "shared_expert_gate": gate}, shared_expert_gate=gate)
            for _ in range(2)]
        return m
    def invoke(self, m, layers=(0,), env=None):
        with patch("krasis.model.torch.Tensor", SourceTensor), patch("krasis.model.torch.cuda.empty_cache"), patch("krasis.model.torch.cuda.mem_get_info", return_value=(1,2)), patch("krasis.model.torch.cuda.memory_allocated", return_value=0), patch("krasis.model.torch.cuda.memory_reserved", return_value=0), patch.dict("os.environ", env or {}, clear=True):
            return m.release_owned_shared_expert_sources(SimpleNamespace(shared_expert_owned_layers=lambda: layers))
    def test_certified_sources_move_but_gate_and_unowned_layer_stay(self):
        m = self.model(); gate = m.layers[0].shared_expert_gate
        self.assertEqual(self.invoke(m), 3*256)
        self.assertEqual(m.layers[0].shared_expert["down_proj"].device.type, "cpu")
        self.assertIs(m.layers[0].shared_expert["shared_expert_gate"], gate)
        self.assertEqual(gate.device.type, "cuda")
        self.assertEqual(m.layers[1].shared_expert["down_proj"].device.type, "cuda")
        self.assertEqual(self.invoke(m), 0)
    def test_captured_torch_graph_keeps_its_weight_sources(self):
        m = self.model(); m.layers[0]._se_graph = object()
        self.assertEqual(self.invoke(m, layers=(0,1)), 3*256)
        self.assertEqual(m.layers[0].shared_expert["down_proj"].device.type, "cuda")
        self.assertEqual(m.layers[1].shared_expert["down_proj"].device.type, "cpu")

    def test_old_native_without_certificate_preserves_sources(self):
        m = self.model()
        self.assertEqual(m.release_owned_shared_expert_sources(SimpleNamespace()), 0)
        self.assertEqual(m.layers[0].shared_expert["down_proj"].device.type, "cuda")

    def test_unsupported_and_optout_preserve_storage(self):
        m = self.model()
        self.assertEqual(self.invoke(m, env={"KRASIS_RECLAIM_SHARED_SOURCES":"0"}), 0)
        self.assertEqual(self.invoke(m, layers=()), 0)
        m.all_devices = [0,1]
        self.assertEqual(self.invoke(m), 0)
        m.all_devices = [0]; m.cfg.is_glm5_next = False
        self.assertEqual(self.invoke(m), 0)

if __name__ == "__main__": unittest.main()
