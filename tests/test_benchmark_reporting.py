import unittest

from krasis.benchmark import _format_gpu_summary


class BenchmarkGpuSummaryTests(unittest.TestCase):
    def test_no_selected_gpus(self):
        self.assertEqual(_format_gpu_summary([]), "none")

    def test_homogeneous_gpus_are_compacted(self):
        gpus = [{"name": "RTX 5090"}, {"name": "RTX 5090"}]
        self.assertEqual(_format_gpu_summary(gpus), "2x RTX 5090")

    def test_heterogeneous_gpus_remain_visible_in_order(self):
        gpus = [{"name": "RTX PRO 6000"}, {"name": "RTX A4500"}]
        self.assertEqual(
            _format_gpu_summary(gpus),
            "RTX PRO 6000 + RTX A4500",
        )


if __name__ == "__main__":
    unittest.main()
