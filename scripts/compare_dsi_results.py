#!/usr/bin/env python3
"""Check comparable evaluation JSONs and report prespecified experiment gates."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def compare(baseline, dsi, min_improvement=0.02, require_dynamic=False):
    for key in ("protocol", "model", "input_size", "num_images", "checkpoint_selection"):
        if baseline[key] != dsi[key]:
            raise ValueError("Incomparable evaluation: " + key)
    if [x["sample"] for x in baseline["per_image"]] != [x["sample"] for x in dsi["per_image"]]:
        raise ValueError("Sample order differs")
    if not np.isfinite(list(baseline["metrics"].values()) + list(dsi["metrics"].values())).all():
        raise ValueError("Nonfinite evaluation metric")
    b, d = baseline["metrics"], dsi["metrics"]
    if b["abs_rel"] <= 0:
        raise ValueError("Baseline Abs Rel must be positive")
    improvement = (b["abs_rel"] - d["abs_rel"]) / b["abs_rel"]
    gates = {
        "abs_rel_relative_improvement": improvement >= min_improvement,
        "sq_rel_regression_at_most_1pct": d["sq_rel"] <= b["sq_rel"] * 1.01,
        "rmse_regression_at_most_1pct": d["rmse"] <= b["rmse"] * 1.01,
        "a1_drop_at_most_0_002": d["a1"] >= b["a1"] - 0.002}
    parameter_check = "NOT_ASSESSED"
    if baseline.get("parameter_count") is not None and dsi.get("parameter_count") is not None:
        gates["same_inference_parameter_count"] = baseline["parameter_count"] == dsi["parameter_count"]
        parameter_check = "PASS" if gates["same_inference_parameter_count"] else "FAIL"
    dynamic = "NOT_ASSESSED"
    if baseline.get("regions") and dsi.get("regions"):
        bm, dm = baseline["regions"]["dynamic"]["metrics"], dsi["regions"]["dynamic"]["metrics"]
        if bm is not None and dm is not None and bm["abs_rel"] > 0:
            dynamic = "PASS" if dm["abs_rel"] <= bm["abs_rel"] * 0.95 else "FAIL"
    if require_dynamic:
        gates["dynamic_abs_rel_improves_at_least_5pct"] = dynamic == "PASS"
    # Paired image bootstrap only describes test-image uncertainty for this pair
    # of checkpoints. It is not a substitute for repeating training seeds.
    delta = np.asarray([di["metrics"]["abs_rel"] - bi["metrics"]["abs_rel"]
                        for bi, di in zip(baseline["per_image"], dsi["per_image"])])
    if len(delta) == 0:
        raise ValueError("No per-image metrics")
    rng = np.random.default_rng(42)
    samples = np.asarray([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(2000)])
    ci = np.quantile(samples, [0.025, 0.975]).tolist()
    paper = "NOT_APPLICABLE"
    if baseline["protocol"]["alignment"] == "M" and baseline["input_size"] == [192, 640] and baseline["model"] == "lite-mono" and not baseline["protocol"]["post_process"]:
        paper = "MET_NUMERIC_REFERENCE" if d["abs_rel"] <= 0.103 else "NOT_MET_NUMERIC_REFERENCE"
    return {"status": "PASS" if all(gates.values()) else "FAIL", "gates": gates,
            "abs_rel_relative_improvement": improvement, "required_improvement": min_improvement,
            "dynamic_region_status": dynamic, "inference_parameter_status": parameter_check,
            "paired_image_abs_rel_delta_95pct_ci": ci, "paired_image_ci_below_zero": ci[1] < 0,
            "paper_0_103_numeric_reference": paper,
            "training_seed_reproducibility": "NOT_ASSESSED: repeat training with independent seeds",
            "note": "Paper numeric reference does not certify exact paper reproduction; compare local controlled baselines."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--dsi", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_abs_rel_improvement", type=float, default=0.02)
    parser.add_argument("--require_dynamic", action="store_true")
    opt = parser.parse_args()
    if not 0 <= opt.min_abs_rel_improvement < 1:
        parser.error("min_abs_rel_improvement must be in [0,1)")
    try:
        result = compare(json.loads(Path(opt.baseline).read_text()), json.loads(Path(opt.dsi).read_text()),
                         opt.min_abs_rel_improvement, opt.require_dynamic)
    except (ValueError, KeyError) as exc:
        result = {"status": "INVALID_COMPARISON", "reason": str(exc)}
    output = Path(opt.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
