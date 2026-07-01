from __future__ import annotations

import unittest

import numpy as np
import torch

from src.dataset import Patch3DDenoiseDataset, valid_slice_indices
from src.model import ResidualUNet2D, ResidualUNet3D, straight_through_clamp


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

    def test_3d_patch_output_uses_depth_height_width_order(self) -> None:
        volume = np.random.rand(12, 20, 8).astype(np.float32)

        patch = Patch3DDenoiseDataset._crop_or_pad(volume, starts=(0, 0, 0), patch_size=(10, 18, 6))
        tensor = torch.from_numpy(np.transpose(patch, (2, 0, 1))).unsqueeze(0).float()

        self.assertEqual(tuple(tensor.shape), (1, 6, 10, 18))


class ModelTests(unittest.TestCase):
    def test_forward_shape_and_range(self) -> None:
        model = ResidualUNet2D(base_channels=4)
        x = torch.rand(2, 1, 32, 32)

        y = model(x)

        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)

    def test_25d_forward_outputs_center_slice_shape(self) -> None:
        model = ResidualUNet2D(in_channels=3, out_channels=1, base_channels=4)
        x = torch.rand(2, 3, 32, 32)

        y = model(x)

        self.assertEqual(tuple(y.shape), (2, 1, 32, 32))
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)

    def test_3d_forward_shape_and_range(self) -> None:
        model = ResidualUNet3D(base_channels=2)
        x = torch.rand(1, 1, 16, 32, 32)

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
