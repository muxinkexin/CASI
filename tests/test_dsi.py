import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsi_utils import dd_mask, nd_loss, gds_loss, refresh_due
from layers import get_smooth_loss
from evaluation_utils import align_depth, depth_metrics
from dsi_trainer import load_strict

spec = importlib.util.spec_from_file_location("compare_dsi", ROOT / "scripts/compare_dsi_results.py")
comparator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparator)


class DSIMathTests(unittest.TestCase):
    def test_dd_shape_and_strict_threshold(self):
        for batch in (1, 2):
            a = torch.full((batch, 1, 3, 4), 0.25)
            self.assertEqual(dd_mask(a, a).shape, a.shape)
            self.assertTrue(dd_mask(a, a).all())
            self.assertFalse(dd_mask(a, a + 0.5, beta=1, mu=0.5).any())
        with self.assertRaises(ValueError):
            dd_mask(torch.ones(2, 3, 4), torch.ones(2, 3, 4))
        with self.assertRaises(FloatingPointError):
            dd_mask(torch.full((1, 1, 2, 2), float("nan")), torch.ones(1, 1, 2, 2))

    def test_nd_no_cross_sample_broadcast(self):
        r = torch.tensor([0.2, 0.8]).reshape(2, 1, 1, 1).requires_grad_()
        sigma = torch.tensor([0.3, 0.7]).reshape(2, 1, 1, 1).requires_grad_()
        mask = torch.tensor([1.0, 0.0]).reshape(2, 1, 1, 1)
        expected = (torch.log1p(sigma[0]) + r[0].square() / (2 * sigma[0].square())) / 2
        self.assertTrue(torch.allclose(nd_loss(r, sigma, mask), expected.squeeze()))
        nd_loss(r, sigma, mask).backward()
        self.assertEqual(r.grad[1].item(), 0)
        self.assertEqual(sigma.grad[1].item(), 0)
        with self.assertRaises(ValueError):
            nd_loss(r[:, 0], sigma, mask)

    def test_nd_zero_sigma_and_all_masked(self):
        for mask_value in (0.0, 1.0):
            r = torch.ones((1, 1, 2, 2), requires_grad=True)
            sigma = torch.zeros_like(r, requires_grad=True)
            loss = nd_loss(r, sigma, torch.full_like(r, mask_value))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(r.grad).all())
            self.assertTrue(torch.isfinite(sigma.grad).all())
            if mask_value == 0:
                self.assertEqual(loss.item(), 0)

    def test_gds_matches_smoothness_for_reliable_pixels(self):
        torch.manual_seed(7)
        disp = torch.rand(2, 1, 8, 12) + 0.1
        image = torch.rand(2, 3, 8, 12)
        norm = disp / (disp.mean((2, 3), keepdim=True) + 1e-7)
        self.assertTrue(torch.allclose(gds_loss(disp, image, torch.ones_like(disp)), get_smooth_loss(norm, image)))

    def test_gds_only_amplifies_vertical_derivative(self):
        image = torch.zeros(1, 3, 4, 5)
        disp_y = torch.arange(1, 5, dtype=torch.float32).reshape(1, 1, 4, 1).expand(1, 1, 4, 5)
        disp_x = torch.arange(1, 6, dtype=torch.float32).reshape(1, 1, 1, 5).expand(1, 1, 4, 5)
        for disp, ratio in ((disp_y, 100), (disp_x, 1)):
            plain = gds_loss(disp, image, torch.ones_like(disp))
            strong = gds_loss(disp, image, torch.zeros_like(disp))
            self.assertTrue(torch.allclose(strong / plain, torch.tensor(float(ratio))))

    def test_refresh_sequence(self):
        self.assertEqual([i for i in range(1, 31) if refresh_due(i)], [1, 6, 11, 16, 21, 26])
        self.assertFalse(any(refresh_due(i, every=0) for i in range(31)))

    def test_m_and_rm_alignment(self):
        gt = np.array([1., 10., 100.])
        pred = np.array([2., 5., 200.])
        self.assertEqual(align_depth(gt, pred, "M")[1], 2.)
        self.assertEqual(align_depth(gt, pred, "RM")[1], 0.5)
        aligned, _ = align_depth(gt, gt * 3, "M")
        np.testing.assert_allclose(depth_metrics(gt, aligned), [0, 0, 0, 0, 1, 1, 1])

    def test_strict_checkpoint_loading(self):
        module = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pth"
            torch.save(module.state_dict(), path)
            load_strict(module, path)
            torch.save({"weight": module.weight}, path)
            with self.assertRaises(RuntimeError):
                load_strict(module, path)

    def test_comparison_gates_and_protocol_rejection(self):
        base = {"protocol": {"alignment": "M", "post_process": False}, "model": "lite-mono",
                "input_size": [192, 640], "num_images": 3, "checkpoint_selection": "final",
                "parameter_count": 10, "metrics": {"abs_rel": .107, "sq_rel": .765, "rmse": 4.561,
                "rmse_log": .183, "a1": .886, "a2": .963, "a3": .983}}
        base["per_image"] = [{"sample": str(i), "metrics": {"abs_rel": .107}} for i in range(3)]
        dsi = copy.deepcopy(base)
        dsi["metrics"].update(abs_rel=.103, sq_rel=.735, rmse=4.471, a1=.892)
        for row in dsi["per_image"]:
            row["metrics"]["abs_rel"] = .103
        result = comparator.compare(base, dsi)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["dynamic_region_status"], "NOT_ASSESSED")
        self.assertEqual(comparator.compare(base, dsi, require_dynamic=True)["status"], "FAIL")
        dsi["metrics"]["a1"] = .8
        self.assertEqual(comparator.compare(base, dsi)["status"], "FAIL")
        dsi["protocol"]["alignment"] = "RM"
        with self.assertRaises(ValueError):
            comparator.compare(base, dsi)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
