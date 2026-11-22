"""Actual CPU forward/backward, phase transition, frozen-state and resume smoke.

Uses temporary synthetic KITTI-layout PNGs, not KITTI training data or results.
Run from the repository root: python tests/smoke_dsi_pipeline.py
"""
import copy
import gc
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from options import LiteMonoOptions
from dsi_trainer import DSITrainer


def main():
    torch.set_num_threads(2)
    split = ROOT / "splits" / "dsi_smoke_fixture"
    if split.exists():
        raise RuntimeError("Refusing to overwrite an existing smoke fixture")
    split.mkdir()
    try:
        with tempfile.TemporaryDirectory(prefix="dsi-smoke-", dir=ROOT.parent) as directory:
            temp = Path(directory)
            image_dir = temp / "data" / "2011_09_26" / "fixture_sync" / "image_02" / "data"
            image_dir.mkdir(parents=True)
            rng = np.random.default_rng(17)
            for index in range(3):
                Image.fromarray(rng.integers(0, 256, size=(64, 128, 3), dtype=np.uint8)).save(image_dir / ("{:010d}.png".format(index)))
            for kind in ("train", "val"):
                (split / (kind + "_files.txt")).write_text("2011_09_26/fixture_sync 1 l\n" * 2)
            args = ["train.py", "--dsi_stage", "both", "--no_cuda", "--weights_init", "scratch",
                    "--data_path", str(temp / "data"), "--log_dir", str(temp / "logs"), "--model_name", "smoke",
                    "--png", "--model", "lite-mono-tiny", "--height", "192", "--width", "640",
                    "--batch_size", "2", "--num_epochs", "3", "--dsi_early_epoch", "0",
                    "--dsi_step2_epochs", "2", "--dsi_refresh_every", "1", "--num_workers", "0",
                    "--save_frequency", "1", "--drop_path", "0", "--log_frequency", "9999"]
            with patch.object(sys, "argv", args):
                opt = LiteMonoOptions().parse()
            opt.split = split.name
            trainer = DSITrainer(copy.deepcopy(opt))
            trainer.train()
            folder = temp / "logs" / "smoke" / "models" / "weights_4"
            assert (folder / "encoder.pth").is_file()
            assert trainer.phase == "inhibit" and trainer.refresh_history == [1, 2]
            for name, module in trainer.models_pose.items():
                raw = torch.load(folder.parent / "weights_2" / (name + ".pth"), weights_only=True)
                assert all(torch.equal(raw[key], value.cpu()) for key, value in module.state_dict().items())
                assert not module.training and all(not p.requires_grad and p.grad is None for p in module.parameters())
            for reference in trainer.reference.values():
                for module in reference.values():
                    assert not module.training and all(not p.requires_grad and p.grad is None for p in module.parameters())
            saved_lr = trainer.model_optimizer.param_groups[0]["lr"]
            expected = {key: value.cpu().clone() for key, value in trainer.models["encoder"].state_dict().items()}
            del trainer
            gc.collect()
            # Resume after the first inhibit epoch and actually train the next one.
            opt.dsi_resume = str(folder.parent / "weights_3")
            resumed = DSITrainer(copy.deepcopy(opt))
            assert resumed.resume_epoch == 1 and resumed.phase == "inhibit"
            assert resumed.refresh_history == [1]
            resumed.train()
            assert abs(resumed.model_optimizer.param_groups[0]["lr"] - saved_lr) < 1e-12
            assert all(torch.equal(expected[key], value.cpu()) for key, value in resumed.models["encoder"].state_dict().items())
            result = {"status": "PASS", "data": "synthetic PNG fixture, not a benchmark",
                      "raw_epochs": 3, "inhibit_epochs": 2, "forward_backward": True,
                      "frozen_pose_and_references": True, "epoch_boundary_resume": True,
                      "refresh_history": [1, 2], "restored_lr": saved_lr}
            print("SMOKE_RESULT=" + json.dumps(result))
    finally:
        shutil.rmtree(split)


if __name__ == "__main__":
    main()
