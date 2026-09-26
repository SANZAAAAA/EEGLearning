"""快速单元测试：不依赖真实 EEG 数据，几秒内跑完。

    python -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from eegnet.config import Config, parse_int_list
from eegnet.data import normalize
from eegnet.engine import compute_metrics
from eegnet.model import EEGNet, build_model


class TestConfig(unittest.TestCase):
    def test_parse_range_and_list(self):
        self.assertEqual(parse_int_list("1-3,7"), [1, 2, 3, 7])

    def test_parse_dedupes_and_sorts(self):
        self.assertEqual(parse_int_list("3,1,2,3"), [1, 2, 3])

    def test_parse_rejects_empty(self):
        with self.assertRaises(ValueError):
            parse_int_list("  ")


class TestNormalize(unittest.TestCase):
    def test_per_trial_gives_zero_mean_unit_std(self):
        x = np.random.RandomState(0).randn(4, 3, 50).astype(np.float32) * 5 + 2
        out, stats = normalize(x, "per_trial")
        self.assertIsNone(stats)
        np.testing.assert_allclose(out.mean(axis=(1, 2)), 0.0, atol=1e-5)
        np.testing.assert_allclose(out.std(axis=(1, 2)), 1.0, atol=1e-3)

    def test_global_uses_training_statistics(self):
        """用训练集统计量归一化测试集，不给测试集"重新居中"（防止信息泄漏）。"""
        rng = np.random.RandomState(0)
        train = rng.randn(6, 2, 30).astype(np.float32)
        shifted = (train[:3] + 10.0).astype(np.float32)      # 明显偏移的"测试集"

        _, stats = normalize(train, "global")
        out_global, _ = normalize(shifted, "global", stats)
        out_per_trial, _ = normalize(shifted, "per_trial")

        self.assertGreater(float(out_global.mean()), 5.0)     # 保留了偏移
        self.assertLess(abs(float(out_per_trial.mean())), 1e-4)  # 被重新居中

    def test_none_is_identity(self):
        x = np.random.RandomState(1).randn(2, 2, 10).astype(np.float32)
        out, _ = normalize(x, "none")
        np.testing.assert_array_equal(out, x)


class TestEEGNet(unittest.TestCase):
    def test_forward_shape_3d_input(self):
        model = EEGNet(n_channels=64, n_times=641, n_classes=2, kernel_length=80)
        self.assertEqual(tuple(model(torch.zeros(8, 64, 641)).shape), (8, 2))

    def test_forward_shape_4d_input_and_multiclass(self):
        model = EEGNet(n_channels=8, n_times=128, n_classes=4, kernel_length=16)
        self.assertEqual(tuple(model(torch.zeros(2, 1, 8, 128)).shape), (2, 4))

    def test_parameter_budget(self):
        """默认配置下应保持轻量（防止无意间把网络改大）。"""
        self.assertLess(EEGNet(64, 641, 2, kernel_length=80).n_parameters, 10_000)

    def test_kernel_length_derived_from_seconds(self):
        cfg = Config(kernel_seconds=0.5)
        model = build_model(cfg, n_channels=64, n_times=641, n_classes=2, sfreq=160.0)
        self.assertEqual(model.kernel_length, 80)          # 0.5 s * 160 Hz
        self.assertEqual(model.kernel_length % 2, 0)       # 必须是偶数

    def test_backward_pass_produces_gradients(self):
        model = EEGNet(8, 128, 2, kernel_length=16)
        logits = model(torch.randn(4, 8, 128))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1])).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))


class TestMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        prob = np.eye(2)[y]
        metrics = compute_metrics(y, prob, ["T1", "T2"])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["kappa"], 1.0)
        self.assertEqual(metrics["auc"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[3, 0], [0, 3]])

    def test_chance_level_predictions(self):
        y = np.array([0, 1, 0, 1])
        prob = np.array([[0.5, 0.5]] * 4)
        metrics = compute_metrics(y, prob, ["T1", "T2"])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["auc"], 0.5)

    def test_per_class_report_present(self):
        y = np.array([0, 0, 1, 1])
        prob = np.eye(2)[y]
        metrics = compute_metrics(y, prob, ["T1", "T2"])
        self.assertEqual(set(metrics["per_class"]), {"T1", "T2"})
        self.assertEqual(metrics["per_class"]["T1"]["support"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
