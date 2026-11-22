"""Lite-Mono + DSI reference implementation, opt-in via --dsi_stage.

DSI's public repository does not release the complete training loop. This loop
implements its paper equations and documents all engineering choices in docs/.
It intentionally has no FlexDepth adaptive mu, SDD, or proposed CASI components.
"""
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import networks
from dsi_mode_compat import set_mode_tree, freeze_module  # DSI_MODE_COMPAT_V1
from networks.variance_decoder import VarDecoder
from trainer import Trainer
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
from dsi_utils import dd_mask, nd_loss, gds_loss, resize_mask, refresh_due, sha256_file


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def load_strict(module, path, height=None, width=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if height is not None:
        for name, expected in (("height", height), ("width", width)):
            if name in state and state[name] != expected:
                raise ValueError("Checkpoint {}={} does not match {}".format(name, state[name], expected))
    # Only Lite-Mono encoder metadata may be removed; never silently discard weights.
    state = {k: v for k, v in state.items() if k not in ("height", "width", "use_stereo")}
    module.load_state_dict(state, strict=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DSITrainer(Trainer):
    def __init__(self, options):
        self.validate_options(options)
        seed_all(options.seed)
        self.phase = "raw"
        self.phase_epoch = 0
        self.reference = {}
        self.reference_provenance = {}
        self.refresh_history = []
        options.disable_automasking = True
        # Resume checkpoints already contain trained models and initial_depth.
        # Avoid reloading ImageNet files/downloads before restoring those states.
        initial_options = (options.weights_init, options.mypretrain)
        if options.dsi_resume:
            options.weights_init, options.mypretrain = "scratch", None
        try:
            super().__init__(options)
        finally:
            options.weights_init, options.mypretrain = initial_options
        if len(self.train_loader) == 0 or len(self.val_loader) == 0:
            raise ValueError("Train and validation splits must each contain at least batch_size samples")
        self.initial_depth = {name: cpu_state(self.models[name]) for name in ("encoder", "depth")}
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.resume_epoch = 0
        self.manifest_path = Path(self.log_path) / "dsi_run.json"
        split_root = Path(__file__).parent / "splits" / self.opt.split
        self.split_hashes = {name: sha256_file(split_root / (name + "_files.txt")) for name in ("train", "val")}
        self.config_keys = (
            "model", "height", "width", "scales", "frame_ids", "batch_size", "dataset", "split", "png",
            "weights_init", "num_layers", "drop_path", "weight_decay", "num_epochs", "lr", "min_depth", "max_depth",
            "disparity_smoothness", "no_ssim", "seed", "dsi_step2_epochs", "dsi_early_epoch", "dsi_beta",
            "dsi_mu", "dsi_gds_weight", "dsi_sigma_min", "dsi_refresh_start", "dsi_refresh_every",
            "dsi_lr", "dsi_min_lr", "dsi_cycle", "dsi_gamma")
        if self.opt.dsi_resume:
            self._restore_checkpoint(Path(self.opt.dsi_resume).expanduser())
        elif self.opt.dsi_stage == "inhibit":
            self._prepare_inhibit()
        self._write_manifest()
        self.save_opts()

    @staticmethod
    def validate_options(opt):
        if (opt.height, opt.width) not in ((192, 640), (320, 1024)):
            raise ValueError("Original Lite-Mono defines dilation only for 192x640 and 320x1024")
        if opt.pose_model_type != "separate_resnet" or opt.pose_model_input != "pairs":
            raise ValueError("DSI baseline supports separate_resnet pose with pairs input")
        if opt.num_layers != 18:
            raise ValueError("This DSI baseline fixes PoseNet and Var encoders to ResNet-18")
        if opt.frame_ids != [0, -1, 1] or opt.use_stereo or opt.predictive_mask or opt.avg_reprojection:
            raise ValueError("DSI requires --frame_ids 0 -1 1, monocular training, minimum reprojection")
        if opt.dataset not in ("kitti", "kitti_odom"):
            raise ValueError("This repository only implements KITTI loaders; Cityscapes is not silently substituted")
        if sorted(opt.scales) != [0, 1, 2] or opt.v1_multiscale:
            raise ValueError("DSI baseline uses --scales 0 1 2 with full-resolution reprojection")
        if opt.load_weights_folder:
            raise ValueError("Use --dsi_resume for resume, or --dsi_early_weights/--dsi_late_weights for stage 2")
        if opt.num_epochs <= 0 or opt.dsi_step2_epochs <= 0 or opt.batch_size <= 0 or opt.save_frequency <= 0:
            raise ValueError("Epochs, batch_size and save_frequency must be positive")
        if not 0 <= opt.dsi_early_epoch < opt.num_epochs or not 0 < opt.dsi_beta <= 1 or opt.dsi_mu <= 0:
            raise ValueError("Invalid early checkpoint index or DD-mask parameters")
        if opt.dsi_gds_weight < 1 or opt.dsi_sigma_min <= 0 or opt.dsi_cycle <= 0:
            raise ValueError("Invalid GDS, sigma floor, or LR cycle")
        if opt.dsi_refresh_start < 1 or opt.dsi_refresh_every < 0 or opt.log_frequency <= 0:
            raise ValueError("Invalid refresh or log interval")
        if not 0 < opt.dsi_min_lr <= opt.dsi_lr or not 0 < opt.dsi_gamma <= 1:
            raise ValueError("Invalid stage-2 LR settings")
        if len(opt.lr) != 6:
            raise ValueError("--lr must contain the original six Lite-Mono scheduler values")
        if opt.dsi_stage == "inhibit" and not opt.dsi_resume:
            if not opt.dsi_early_weights or not opt.dsi_late_weights:
                raise ValueError("Stage 2 needs both --dsi_early_weights and --dsi_late_weights")

    def _new_reference(self):
        # DepthDecoder retains its last output tensors; deepcopy after a training
        # forward would try to copy non-leaf autograd tensors. Build fresh modules.
        encoder = networks.LiteMono(model=self.opt.model, drop_path_rate=self.opt.drop_path,
                                    height=self.opt.height, width=self.opt.width).to(self.device)
        decoder = networks.DepthDecoder(encoder.num_ch_enc, self.opt.scales).to(self.device)
        return {"encoder": freeze_module(encoder),
                "depth": freeze_module(decoder)}

    def _prepare_inhibit(self, resume_state=None, checkpoint_dir=None):
        self.phase = "inhibit"
        self.phase_epoch = 0
        self.opt.disable_automasking = False
        for name in ("encoder", "depth"):
            self.models[name].load_state_dict(self.initial_depth[name], strict=True)
        self.reference = {"early": self._new_reference(), "late": self._new_reference()}
        if resume_state is None:
            root = Path(self.log_path) / "models"
            folders = {
                "early": Path(self.opt.dsi_early_weights).expanduser() if self.opt.dsi_early_weights else root / ("weights_" + str(self.opt.dsi_early_epoch)),
                "late": Path(self.opt.dsi_late_weights).expanduser() if self.opt.dsi_late_weights else root / ("weights_" + str(self.opt.num_epochs - 1))}
            if folders["early"].resolve() == folders["late"].resolve():
                raise ValueError("Early and late references must be distinct checkpoints")
            for role, folder in folders.items():
                for name, model in self.reference[role].items():
                    path = folder / (name + ".pth")
                    load_strict(model, path, self.opt.height, self.opt.width)
                    self.reference_provenance[role + "/" + name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
            pose_folder = folders["late"]
        else:
            for role in self.reference:
                for name in self.reference[role]:
                    self.reference[role][name].load_state_dict(resume_state["reference"][role][name], strict=True)
            self.reference_provenance = resume_state["reference_provenance"]
            pose_folder = checkpoint_dir
        for name, model in self.models_pose.items():
            load_strict(model, pose_folder / (name + ".pth"))
            freeze_module(model)
            model.zero_grad(set_to_none=True)
        self.models["var_encoder"] = networks.ResnetEncoder(18, resume_state is None and self.opt.weights_init == "pretrained", num_input_images=3).to(self.device)
        self.models["var"] = VarDecoder(self.models["var_encoder"].num_ch_enc, self.opt.scales).to(self.device)
        self.parameters_to_train = [p for model in self.models.values() for p in model.parameters() if p.requires_grad]
        self.model_optimizer = torch.optim.AdamW(self.parameters_to_train, lr=self.opt.dsi_lr, weight_decay=self.opt.weight_decay)
        self.model_lr_scheduler = ChainedScheduler(
            self.model_optimizer, T_0=self.opt.dsi_cycle, T_mul=1,
            eta_min=self.opt.dsi_min_lr, max_lr=self.opt.dsi_lr,
            warmup_steps=0, gamma=self.opt.dsi_gamma, last_epoch=-1)
        print("DSI stage 2: reset DepthNet, freeze Pose/late reference; fixed mu={}, beta={}".format(self.opt.dsi_mu, self.opt.dsi_beta))

    def set_train(self):
        for model in self.models.values():
            set_mode_tree(model, True)
        for model in self.models_pose.values():
            set_mode_tree(model, self.phase == "raw")
        for reference in self.reference.values():
            for model in reference.values():
                set_mode_tree(model, False)

    def set_eval(self):
        for model in self.models.values():
            set_mode_tree(model, False)
        for model in self.models_pose.values():
            set_mode_tree(model, False)
        for reference in self.reference.values():
            for model in reference.values():
                set_mode_tree(model, False)

    def process_batch(self, inputs):
        if self.phase == "raw":
            return super().process_batch(inputs)
        for key, value in inputs.items():
            inputs[key] = value.to(self.device)
        target_aug = inputs[("color_aug", 0, 0)]
        features = self.models["encoder"](target_aug)
        outputs = self.models["depth"](features)
        with torch.no_grad():
            outputs.update(self.predict_poses(inputs, features))
            refs = {role: modules["depth"](modules["encoder"](target_aug)) for role, modules in self.reference.items()}
        var_input = torch.cat([inputs[("color_aug", frame, 0)] for frame in self.opt.frame_ids], dim=1)
        outputs.update(self.models["var"](self.models["var_encoder"](var_input)))
        for scale in self.opt.scales:
            outputs[("dsi_mask", scale)] = dd_mask(refs["early"][("disp", scale)], refs["late"][("disp", scale)], self.opt.dsi_beta, self.opt.dsi_mu)
        self.generate_images_pred(inputs, outputs)
        return outputs, self.compute_losses(inputs, outputs)

    def compute_losses(self, inputs, outputs):
        if self.phase == "raw":
            return super().compute_losses(inputs, outputs)
        losses, total = {}, 0
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            target = inputs[("color", 0, 0)]
            reprojection = torch.cat([self.compute_reprojection_loss(outputs[("color", frame, scale)], target) for frame in self.opt.frame_ids[1:]], dim=1)
            residual = reprojection.min(dim=1, keepdim=True).values
            identity = torch.cat([self.compute_reprojection_loss(inputs[("color", frame, 0)], target) for frame in self.opt.frame_ids[1:]], dim=1)
            identity = identity + torch.randn_like(identity) * 1e-5
            automask = (residual < identity.min(dim=1, keepdim=True).values).to(disp.dtype).detach()
            mask = outputs[("dsi_mask", scale)]
            full_mask = resize_mask(mask, target.shape[-2:])
            sigma = F.interpolate(outputs[("var", scale)], size=target.shape[-2:], mode="bilinear", align_corners=False)
            photometric = nd_loss(residual, sigma, full_mask * automask, self.opt.dsi_sigma_min)
            smoothness = gds_loss(disp, inputs[("color", 0, scale)], mask, self.opt.dsi_gds_weight)
            loss = photometric + self.opt.disparity_smoothness * smoothness / (2 ** scale)
            outputs["identity_selection/{}".format(scale)] = automask[:, 0]
            losses["loss/{}".format(scale)] = loss
            losses["dsi/keep_{}".format(scale)] = mask.mean().detach()
            losses["dsi/active_{}".format(scale)] = (full_mask * automask).mean().detach()
            losses["dsi/nd_{}".format(scale)] = photometric.detach()
            losses["dsi/gds_{}".format(scale)] = smoothness.detach()
            losses["dsi/sigma_{}".format(scale)] = sigma.mean().detach()
            total = total + loss
        losses["loss"] = total / self.num_scales
        return losses

    def train(self):
        phases = ["raw", "inhibit"] if self.opt.dsi_stage == "both" else [self.opt.dsi_stage]
        if self.phase == "inhibit":
            phases = ["inhibit"]
        for phase in phases:
            if phase != self.phase:
                self._prepare_inhibit()
                self.resume_epoch = 0
            count = self.opt.num_epochs if phase == "raw" else self.opt.dsi_step2_epochs
            remaining = count - self.resume_epoch
            self.num_total_steps = self.step + len(self.train_loader) * remaining
            for self.phase_epoch in range(self.resume_epoch, count):
                self.epoch = self.phase_epoch + (0 if phase == "raw" else self.opt.num_epochs)
                self.run_epoch()
                completed = self.phase_epoch + 1
                if phase == "inhibit" and refresh_due(completed, self.opt.dsi_refresh_start, self.opt.dsi_refresh_every):
                    for name, model in self.reference["early"].items():
                        model.load_state_dict(self.models[name].state_dict(), strict=True)
                        freeze_module(model)
                    self.refresh_history.append(completed)
                    print("DSI refreshed early reference after inhibit epoch", completed)
                force = phase == "raw" and self.phase_epoch == self.opt.dsi_early_epoch
                if force or completed == count or completed % self.opt.save_frequency == 0:
                    self.save_model()
            self.resume_epoch = 0
        for writer in self.writers.values():
            writer.close()

    def run_epoch(self):
        self.set_train()
        # Preserve original Lite-Mono's beginning-of-epoch scheduler timing.
        self.model_lr_scheduler.step()
        if self.phase == "raw":
            self.model_pose_lr_scheduler.step()
        print("DSI phase={} phase_epoch={} checkpoint_index={}".format(self.phase, self.phase_epoch, self.epoch))
        for batch_idx, inputs in enumerate(self.train_loader):
            started = time.time()
            outputs, losses = self.process_batch(inputs)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Nonfinite DSI loss; stop before corrupting weights")
            self.model_optimizer.zero_grad(set_to_none=True)
            if self.phase == "raw":
                self.model_pose_optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters_to_train):
                raise FloatingPointError("Nonfinite Depth/Var gradient")
            self.model_optimizer.step()
            if self.phase == "raw":
                self.model_pose_optimizer.step()
            if batch_idx % self.opt.log_frequency == 0:
                self.log_time(batch_idx, time.time() - started, losses["loss"].detach().cpu())
                self.log("train", inputs, outputs, losses)
                # Validation iterator position is not part of a training resume.
                # Keep its random tie-breaking/shuffle from advancing train RNG.
                python_rng, numpy_rng = random.getstate(), np.random.get_state()
                devices = [torch.cuda.current_device()] if self.device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    self.val()
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
            self.step += 1

    def log(self, mode, inputs, outputs, losses):
        super().log(mode, inputs, outputs, losses)
        if self.phase == "inhibit":
            for j in range(min(4, self.opt.batch_size)):
                self.writers[mode].add_image("dsi_mask/{}".format(j), outputs[("dsi_mask", 0)][j].detach(), self.step)
                self.writers[mode].add_image("dsi_sigma/{}".format(j), outputs[("var", 0)][j].detach(), self.step)

    def _write_manifest(self):
        data = {"implementation": "Lite-Mono+DSI paper-based baseline", "phase": self.phase,
                "base_commit": "128f3b041ae87bf6573a414ed4c12cff55bc8b9d",
                "options": vars(self.opt), "split_sha256": self.split_hashes,
                "reference_provenance": self.reference_provenance, "refresh_completed_epochs": self.refresh_history,
                "torch_version": str(torch.__version__)}
        self.manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save_model(self):
        super().save_model()
        folder = Path(self.log_path) / "models" / ("weights_" + str(self.epoch))
        np_rng = np.random.get_state()
        state = {"format": 1, "phase": self.phase, "completed": self.phase_epoch + 1, "step": self.step,
                 "initial_depth": self.initial_depth, "reference_provenance": self.reference_provenance,
                 "reference": {role: {name: cpu_state(model) for name, model in modules.items()} for role, modules in self.reference.items()},
                 "refresh_history": self.refresh_history, "config": {key: getattr(self.opt, key) for key in self.config_keys},
                 "split_hashes": self.split_hashes,
                 "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                 "numpy_rng": (np_rng[0], np_rng[1].tolist(), np_rng[2], np_rng[3], np_rng[4]),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        temporary = folder / "dsi_state.pth.tmp"
        torch.save(state, temporary)
        os.replace(temporary, folder / "dsi_state.pth")
        self._write_manifest()

    def _restore_checkpoint(self, folder):
        state = torch.load(folder / "dsi_state.pth", map_location="cpu", weights_only=True)
        for key in self.config_keys:
            if state["config"][key] != getattr(self.opt, key):
                raise ValueError("Resume config changed: " + key)
        if state["split_hashes"] != self.split_hashes:
            raise ValueError("Resume split contents changed")
        if state["phase"] == "raw" and self.opt.dsi_stage == "inhibit":
            raise ValueError("For stage transition use reference folders, not --dsi_resume from raw")
        if state["phase"] == "inhibit" and self.opt.dsi_stage == "raw":
            raise ValueError("Cannot resume inhibit checkpoint as raw")
        self.initial_depth = state["initial_depth"]
        if state["phase"] == "inhibit":
            self._prepare_inhibit(state, folder)
        for name, model in self.models.items():
            load_strict(model, folder / (name + ".pth"), self.opt.height, self.opt.width)
        for name, model in self.models_pose.items():
            load_strict(model, folder / (name + ".pth"))
        self.model_optimizer.load_state_dict(torch.load(folder / "adam.pth", map_location=self.device, weights_only=True))
        if self.phase == "raw":
            self.model_pose_optimizer.load_state_dict(torch.load(folder / "adam_pose.pth", map_location=self.device, weights_only=True))
        # The third-party ChainedScheduler stores nested Python scheduler objects.
        # Replay its deterministic epoch steps instead of pickling unsafe objects.
        self.resume_epoch = state["completed"]
        for _ in range(self.resume_epoch):
            self.model_lr_scheduler.step()
            if self.phase == "raw":
                self.model_pose_lr_scheduler.step()
        self.step = state["step"]
        self.refresh_history = state["refresh_history"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        rng = state["numpy_rng"]
        np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), rng[2], rng[3], rng[4]))
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print("Resumed DSI phase={}, completed={}".format(self.phase, self.resume_epoch))
