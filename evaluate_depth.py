from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from layers import disp_to_depth
from utils import readlines
from options import LiteMonoOptions
import datasets
import networks
import time
import json
from pathlib import Path
from evaluation_utils import (METRIC_NAMES, file_digest, eigen_mask, depth_metrics, align_depth, metric_summary)
from thop import clever_format
from thop import profile


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def profile_once(encoder, decoder, x):
    x_e = x[0, :, :, :].unsqueeze(0)
    x_d = encoder(x_e)
    flops_e, params_e = profile(encoder, inputs=(x_e, ), verbose=False)
    flops_d, params_d = profile(decoder, inputs=(x_d, ), verbose=False)

    flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")

    return flops, params, flops_e, params_e, flops_d, params_d


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    split_root = Path(opt.split_path).expanduser() if opt.split_path else Path(splits_dir)
    filenames = readlines(str(split_root / opt.eval_split / "test_files.txt"))
    profile_info = None
    parameter_count = None
    device = torch.device("cpu" if opt.no_cuda else "cuda")
    if opt.RM_scaling and opt.disable_median_scaling:
        raise ValueError("--RM_scaling conflicts with --disable_median_scaling")
    if opt.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path, map_location='cpu', weights_only=True)
        decoder_dict = torch.load(decoder_path, map_location='cpu', weights_only=True)

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False, img_ext=".png" if opt.png else ".jpg")
        dataloader = DataLoader(dataset, opt.eval_batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.LiteMono(model=opt.model,
                                    height=encoder_dict['height'],
                                    width=encoder_dict['width'])
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
        model_dict = encoder.state_dict()
        depth_model_dict = depth_decoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

        encoder.to(device)
        encoder.eval()
        depth_decoder.to(device)
        depth_decoder.eval()

        parameter_count = sum(p.numel() for model in (encoder, depth_decoder) for p in model.parameters())
        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].to(device)

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                if opt.eval_profile and profile_info is None:
                    profile_info = profile_once(encoder, depth_decoder, input_color)
                t1 = time_sync()
                output = depth_decoder(encoder(input_color))
                t2 = time_sync()

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if getattr(opt, "eval_eigen_to_benchmark", False):
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    output_dir = Path(opt.eval_out_dir).expanduser() if opt.eval_out_dir else Path(opt.load_weights_folder or ".")
    if opt.save_pred_disps:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / ("disps_" + opt.eval_split + "_split.npy"), pred_disps)
    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        return None

    gt_path = split_root / opt.eval_split / "gt_depths.npz"
    gt_depths = np.load(gt_path, fix_imports=True, encoding="latin1", allow_pickle=True)["data"]
    if len(pred_disps) != len(gt_depths) or len(pred_disps) != len(filenames):
        raise ValueError("Prediction/GT/test-list counts differ: {}/{}/{}".format(len(pred_disps), len(gt_depths), len(filenames)))
    if len(pred_disps) == 0:
        raise ValueError("Empty evaluation split")
    region_masks = None
    if opt.eval_region_masks:
        region_masks = np.load(opt.eval_region_masks, allow_pickle=True)["data"]
        if len(region_masks) != len(filenames):
            raise ValueError("Region mask count differs from test list")
    protocol = "none" if opt.disable_median_scaling else ("RM" if opt.RM_scaling else "M")
    print("-> Evaluating KITTI Eigen: cap=80m, alignment=" + protocol)
    errors, ratios, per_image = [], [], []
    region_rows = {"dynamic": [], "static": []}
    region_pixels = {"dynamic": 0, "static": 0}
    for i, gt_depth in enumerate(gt_depths):
        gt_depth = np.asarray(gt_depth, dtype=np.float64)
        if gt_depth.ndim != 2:
            raise ValueError("GT must be a two-dimensional depth image")
        gt_height, gt_width = gt_depth.shape
        pred_disp = cv2.resize(pred_disps[i], (gt_width, gt_height))
        if not np.isfinite(pred_disp).all() or np.any(pred_disp <= 0):
            raise ValueError("Nonfinite/nonpositive predicted disparity at sample " + str(i))
        mask = eigen_mask(gt_depth, MIN_DEPTH, MAX_DEPTH)
        if not mask.any():
            raise ValueError("No valid GT pixels after Eigen crop at sample " + str(i))
        pred_depth = (1.0 / pred_disp.astype(np.float64)) * opt.pred_depth_scale_factor
        _, ratio = align_depth(gt_depth[mask], pred_depth[mask], protocol)
        ratios.append(ratio)
        pred_depth = np.clip(pred_depth * ratio, MIN_DEPTH, MAX_DEPTH)
        row = depth_metrics(gt_depth[mask], pred_depth[mask])
        errors.append(row)
        record = {"index": i, "sample": filenames[i], "valid_pixels": int(mask.sum()),
                  "scale": ratio, "metrics": dict(zip(METRIC_NAMES, row.tolist()))}
        if region_masks is not None:
            dynamic = np.asarray(region_masks[i])
            if dynamic.shape != gt_depth.shape or not np.isin(dynamic, [0, 1]).all():
                raise ValueError("Each region mask must be binary and match the original GT coordinates")
            record["regions"] = {}
            for label, region in (("dynamic", dynamic.astype(bool)), ("static", ~dynamic.astype(bool))):
                selected = mask & region
                region_pixels[label] += int(selected.sum())
                if selected.any():
                    values = depth_metrics(gt_depth[selected], pred_depth[selected])
                    region_rows[label].append(values)
                    record["regions"][label] = dict(zip(METRIC_NAMES, values.tolist()))
        per_image.append(record)
    result = {
        "schema_version": 1, "checkpoint": str(Path(opt.load_weights_folder).resolve()) if opt.load_weights_folder else None,
        "external_disparity": str(Path(opt.ext_disp_to_eval).resolve()) if opt.ext_disp_to_eval else None,
        "checkpoint_selection": opt.eval_checkpoint_tag,
        "protocol": {"dataset": "KITTI", "eval_split": opt.eval_split,
                     "crop": "Eigen/Garg constants as original Lite-Mono", "min_depth": MIN_DEPTH, "max_depth": MAX_DEPTH,
                     "alignment": protocol, "post_process": opt.post_process,
                     "pred_depth_scale_factor": opt.pred_depth_scale_factor,
                     "min_depth_prediction": opt.min_depth, "max_depth_prediction": opt.max_depth,
                     "test_files_sha256": file_digest(split_root / opt.eval_split / "test_files.txt"),
                     "gt_sha256": file_digest(gt_path),
                     "region_masks_sha256": file_digest(opt.eval_region_masks) if opt.eval_region_masks else None,
                     "metric_aggregation": "mean of per-image metrics; ROI uses global-image alignment"},
        "model": opt.model, "parameter_count": parameter_count,
        "input_size": [int(encoder_dict["height"]), int(encoder_dict["width"])] if opt.ext_disp_to_eval is None else list(pred_disps.shape[-2:]),
        "num_images": len(errors), "metrics": metric_summary(errors),
        "regions": {label: {"num_images": len(rows), "valid_pixels": region_pixels[label], "metrics": metric_summary(rows)}
                    for label, rows in region_rows.items()} if region_masks is not None else None,
        "per_image": per_image}
    if opt.ext_disp_to_eval is None:
        result["checkpoint_sha256"] = {name: file_digest(Path(opt.load_weights_folder) / (name + ".pth")) for name in ("encoder", "depth")}
    print("\n  " + ("{:>8} | " * 7).format(*METRIC_NAMES))
    print(("&{: 8.3f}  " * 7).format(*[result["metrics"][name] for name in METRIC_NAMES]) + "\\\\")
    if profile_info:
        print("MACs/parameters (THOP):", profile_info)
    if opt.eval_metrics_json:
        path = Path(opt.eval_metrics_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print("Metrics JSON:", path)
    print("-> Done!")
    return result


if __name__ == "__main__":
    options = LiteMonoOptions()
    evaluate(options.parse())
