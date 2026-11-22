#!/usr/bin/env python3
"""Read-only environment, split and input-file check; no data are modified."""
import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from options import LiteMonoOptions


def main():
    opt = LiteMonoOptions().parse()
    for name in ("torch", "torchvision", "timm", "tensorboardX", "cv2", "skimage", "thop",
                 "linear_warmup_cosine_annealing_warm_restarts_weight_decay"):
        importlib.import_module(name)
        print("IMPORT OK:", name)
    import torch
    if not opt.no_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --no_cuda only for CPU smoke checks")
    if opt.mypretrain and not Path(opt.mypretrain).expanduser().is_file():
        raise FileNotFoundError(opt.mypretrain)
    root = Path(__file__).resolve().parents[1] / "splits" / opt.split
    data = Path(opt.data_path).expanduser()
    extension = ".png" if opt.png else ".jpg"
    for split in ("train", "val"):
        lines = (root / (split + "_files.txt")).read_text().splitlines()
        if len(lines) < opt.batch_size:
            raise ValueError(split + " contains fewer samples than batch_size")
        # Enumerate every temporal-neighbor path, not only the target image.
        missing = []
        for line in lines:
            folder, index, side = line.split()
            camera = {"l": 2, "r": 3, "2": 2, "3": 3}[side]
            for offset in opt.frame_ids:
                image = data / folder / ("image_0" + str(camera)) / "data" / ("{:010d}".format(int(index) + offset) + extension)
                if not image.is_file():
                    missing.append(str(image))
                    if len(missing) >= 10:
                        break
            if len(missing) >= 10:
                break
        if missing:
            raise FileNotFoundError("Missing temporal input files (first examples):\n" + "\n".join(missing))
        print("INPUTS OK:", split, len(lines), "samples;", extension)
    print("READY: input paths and imports checked. Full GPU training is a separate step.")


if __name__ == "__main__":
    main()
