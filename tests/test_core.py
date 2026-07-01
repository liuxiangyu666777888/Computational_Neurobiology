from __future__ import annotations

import unittest

import numpy as np
import torch

from src.dataset import valid_slice_indices
from src.model import ResidualUNet2D, straight_through_clamp


class DatasetTests(unittest.TestCase):
    def test_valid_slice_indices_skips_blank_slices(self) -> None:
        clean = np.zeros((8, 8, 4), dtype=np.float32)
        clean[:, :, 2] = 1.0

        indices = valid_slice_indices(clean, min_foreground_ratio=0.5)

        self.assertEqual(indices, [2])

    def test_valid_slice_indices_falls_back_when_all_blank(self) -> None:
        clean = np.zeros((4, 4, 3), dtype=np.float32)

        indices = valid_slice_indices(clean, min_foreground_ratio=0.5)

        self.assertEqual(indices, [0, 1, 2])


class ModelTests(unittest.TestCase):
    def test_forward_shape_and_range(self) -> None:
        model = ResidualUNet2D(base_channels=4)
        x = torch.rand(2, 1, 32, 32)

        y = model(x)

        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)

    def test_straight_through_clamp_keeps_gradient(self) -> None:
        x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)

        y = straight_through_clamp(x).sum()
        y.backward()

        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))


if __name__ == "__main__":
    unittest.main()
