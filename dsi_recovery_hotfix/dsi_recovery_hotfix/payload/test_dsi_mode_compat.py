"""Regression checks for mode switching; includes injected non-bool train input."""
import copy
import unittest

import numpy as np
import torch
from torch import nn

from dsi_mode_compat import set_mode_tree, freeze_module


class NonBoolTrainWrapper(nn.Linear):
    def train(self, mode=True):
        # Simulate a wrapper passing numpy.bool_ into PyTorch's guarded method.
        return super().train(np.bool_(mode))


def fixture(broken=False):
    last = NonBoolTrainWrapper(4, 2) if broken else nn.Linear(4, 2)
    return nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(.3), last)


class ModeCompatibilityTests(unittest.TestCase):
    def test_reproduce_same_exception_then_switch_without_recursive_calls(self):
        model = fixture(broken=True)
        with self.assertRaisesRegex(ValueError, 'training mode is expected to be boolean'):
            model.eval()
        freeze_module(model)
        for child in model.modules():
            self.assertIs(child.training, False)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model(torch.randn(2, 4))).all())

    def test_eval_matches_standard_modules_and_preserves_bn(self):
        torch.manual_seed(42)
        reference = fixture().eval()
        model = copy.deepcopy(reference)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        freeze_module(model)
        x = torch.randn(3, 4, requires_grad=True)
        self.assertTrue(torch.equal(reference(x), model(x)))
        model(x).sum().backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(torch.equal(before[key], value) for key, value in model.state_dict().items()))

    def test_student_training_mode_is_native_bool(self):
        student = fixture()
        set_mode_tree(student, False)
        set_mode_tree(student, True)
        self.assertTrue(all(child.training is True for child in student.modules()))
        before = student[1].num_batches_tracked.item()
        student(torch.randn(3, 4)).sum().backward()
        self.assertEqual(student[1].num_batches_tracked.item(), before + 1)
        self.assertTrue(any(p.grad is not None for p in student.parameters()))

    def test_reject_non_bool_public_input(self):
        for invalid in (np.bool_(False), 'False', 0, None):
            with self.assertRaises(TypeError):
                set_mode_tree(fixture(), invalid)

    def test_refresh_sixth_epoch_and_next_training_mode(self):
        student, early, late = fixture(), fixture(broken=True), fixture(broken=True)
        freeze_module(early)
        freeze_module(late)
        updates = []
        for completed in range(1, 8):
            set_mode_tree(student, True)
            set_mode_tree(early, False)
            set_mode_tree(late, False)
            if (completed - 1) % 5 == 0:
                early.load_state_dict(student.state_dict())
                freeze_module(early)
                updates.append(completed)
        self.assertEqual(updates, [1, 6])
        self.assertTrue(all(child.training is False for model in (early, late) for child in model.modules()))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
