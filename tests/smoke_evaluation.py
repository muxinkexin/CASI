"""End-to-end evaluator checks: external disparity, PNG checkpoint and tail batch."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from options import LiteMonoOptions
from evaluate_depth import evaluate
import networks


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="dsi-eval-smoke-", dir=ROOT.parent) as directory:
        temp = Path(directory)
        split = temp / "splits" / "eigen"
        split.mkdir(parents=True)
        lines = ["2011_09_26/fixture_sync {} l".format(i) for i in range(3)]
        (split / "test_files.txt").write_text("\n".join(lines) + "\n")
        gt = np.stack([np.full((32, 64), 5 + i, dtype=np.float32) for i in range(3)])
        np.savez_compressed(split / "gt_depths.npz", data=gt)
        np.save(temp / "disp.npy", 1 / (gt * 2))
        region = np.zeros_like(gt, dtype=np.uint8)
        region[:, :, 20:40] = 1
        np.savez_compressed(temp / "regions.npz", data=region)
        args = ["evaluate_depth.py", "--no_cuda", "--split_path", str(temp / "splits"),
                "--ext_disp_to_eval", str(temp / "disp.npy"), "--eval_region_masks", str(temp / "regions.npz"),
                "--eval_metrics_json", str(temp / "external.json")]
        with patch.object(sys, "argv", args):
            opt = LiteMonoOptions().parse()
        for rm in (False, True):
            opt.RM_scaling = rm
            result = evaluate(opt)
            assert result["num_images"] == 3 and result["metrics"]["abs_rel"] < 1e-6
            assert result["regions"]["dynamic"]["num_images"] == 3
            assert result["protocol"]["alignment"] == ("RM" if rm else "M")
        bad = copy.deepcopy(opt)
        bad.disable_median_scaling = True
        try:
            evaluate(bad)
            raise AssertionError("Conflicting alignment options were accepted")
        except ValueError:
            pass
        opt.no_eval = True
        assert evaluate(opt) is None
        weights = temp / "weights"
        weights.mkdir()
        encoder = networks.LiteMono(model="lite-mono-tiny", height=192, width=640)
        decoder = networks.DepthDecoder(encoder.num_ch_enc, [0, 1, 2])
        state = encoder.state_dict()
        state.update(height=192, width=640, use_stereo=False)
        torch.save(state, weights / "encoder.pth")
        torch.save(decoder.state_dict(), weights / "depth.pth")
        image_dir = temp / "data" / "2011_09_26" / "fixture_sync" / "image_02" / "data"
        image_dir.mkdir(parents=True)
        for i in range(3):
            Image.fromarray(np.full((32, 64, 3), 80 + i * 20, dtype=np.uint8)).save(image_dir / ("{:010d}.png".format(i)))
        opt.ext_disp_to_eval = None
        opt.load_weights_folder = str(weights)
        opt.model = "lite-mono-tiny"
        opt.data_path = str(temp / "data")
        opt.png = True
        opt.no_eval = False
        opt.RM_scaling = False
        opt.num_workers = 0
        opt.eval_batch_size = 2
        opt.eval_metrics_json = str(temp / "checkpoint.json")
        result = evaluate(opt)
        assert result["num_images"] == 3 and result["parameter_count"] > 0
        assert result["input_size"] == [192, 640]
        print("EVAL_SMOKE_RESULT=" + json.dumps({"status": "PASS", "external_M_RM": True,
              "png_checkpoint_inference": True, "batch_sizes": [2, 1], "ROI_global_alignment": True,
              "note": "synthetic data; metrics are not a benchmark result"}))


if __name__ == "__main__":
    main()
