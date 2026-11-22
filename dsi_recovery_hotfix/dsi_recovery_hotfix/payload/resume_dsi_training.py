#!/usr/bin/env python3
"""Validate an existing inhibit checkpoint and continue its original DSI run."""
import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_finite(value, torch, label):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all().item():
            raise ValueError('Nonfinite tensor: ' + label)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            verify_finite(child, torch, label + '/' + str(key))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            verify_finite(child, torch, label + '/' + str(index))
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite value: ' + label)


def inspect_checkpoint(folder, opt_path, repo, expected=None):
    import torch
    state_path = folder / 'dsi_state.pth'
    if not state_path.is_file():
        raise FileNotFoundError('Missing completed checkpoint state: ' + str(state_path))
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    if not isinstance(state, Mapping) or state.get('format') != 1 or state.get('phase') != 'inhibit':
        raise ValueError('Expected a format=1 inhibit checkpoint; refusing to restart raw training')
    config = state['config']
    completed = state['completed']
    raw_epochs, total = config['num_epochs'], config['dsi_step2_epochs']
    if type(completed) is not int or not 0 < completed < total:
        raise ValueError('Checkpoint must be unfinished inhibit training')
    if expected is not None and completed != expected:
        raise ValueError('Expected {} completed inhibit epochs, found {}'.format(expected, completed))
    index = raw_epochs + completed - 1
    if folder.name != 'weights_' + str(index):
        raise ValueError('Folder index and checkpoint metadata disagree')
    saved = json.loads(opt_path.read_text(encoding='utf-8'))
    for key, value in config.items():
        if key not in saved or saved[key] != value:
            raise ValueError('opt.json and checkpoint config disagree: ' + key)
    for role in ('early', 'late'):
        for name in ('encoder', 'depth'):
            if not state['reference'][role][name]:
                raise ValueError('Missing frozen reference: ' + role + '/' + name)
    if not state['initial_depth']['encoder'] or not state['initial_depth']['depth']:
        raise ValueError('Missing original initialization state')
    if not 0 <= state['step']:
        raise ValueError('Invalid global training step')
    verify_finite(state, torch, 'dsi_state')
    required = ('encoder.pth', 'depth.pth', 'var_encoder.pth', 'var.pth',
                'pose_encoder.pth', 'pose.pth', 'adam.pth')
    saved_learning_rates = None
    for name in required:
        path = folder / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError('Incomplete checkpoint: ' + str(path))
        value = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(value, Mapping) or not value:
            raise ValueError('Invalid state dictionary: ' + name)
        verify_finite(value, torch, name)
        if name == 'adam.pth':
            saved_learning_rates = [group['lr'] for group in value['param_groups']]
        del value
        print('CHECKPOINT FILE OK:', name, flush=True)
    split_root = repo / 'splits' / config['split']
    for kind in ('train', 'val'):
        if file_hash(split_root / (kind + '_files.txt')) != state['split_hashes'][kind]:
            raise ValueError('Split contents changed: ' + kind)
    data_path = Path(saved['data_path']).expanduser()
    if not data_path.is_dir():
        raise FileNotFoundError('Original data_path is unavailable: ' + str(data_path))
    if not saved.get('no_cuda', False) and not torch.cuda.is_available():
        raise RuntimeError('Original run uses CUDA; CUDA is unavailable in this environment')
    run_dir = folder.parent.parent
    refresh_first, refresh_every = config['dsi_refresh_start'], config['dsi_refresh_every']
    expected_history = [i for i in range(1, completed + 1)
                        if refresh_every > 0 and i >= refresh_first and (i - refresh_first) % refresh_every == 0]
    if state['refresh_history'] != expected_history:
        raise ValueError('Reference refresh history is inconsistent with the saved schedule')
    summary = {
        'status': 'READY', 'checkpoint': str(folder), 'phase': 'inhibit',
        'completed_inhibit_epochs': completed, 'next_phase_epoch': completed,
        'next_checkpoint_index': raw_epochs + completed,
        'remaining_inhibit_epochs': total - completed, 'final_checkpoint_index': raw_epochs + total - 1,
        'global_step': state['step'], 'refresh_history': state['refresh_history'],
        'saved_learning_rates': saved_learning_rates,
        'png': config['png'], 'data_path': str(data_path), 'run_dir': str(run_dir),
        'torch_version': str(torch.__version__), 'native_bool_type': str(type(False)),
        'module_train_implementation': torch.nn.Module.train.__module__,
        'dsi_state_sha256': file_hash(state_path), 'optimizer_sha256': file_hash(folder / 'adam.pth')}
    return state, saved, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--opt-json', default=None)
    parser.add_argument('--expect-completed', type=int, default=None)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    folder = Path(args.checkpoint).expanduser().resolve()
    opt_path = Path(args.opt_json).expanduser().resolve() if args.opt_json else folder.parent / 'opt.json'
    state, saved, summary = inspect_checkpoint(folder, opt_path, repo, args.expect_completed)
    from options import LiteMonoOptions
    options = LiteMonoOptions().parser.parse_args([])
    known = vars(options)
    for key, value in saved.items():
        if key in known:
            setattr(options, key, value)
    options.dsi_stage = 'both'
    options.dsi_resume = str(folder)
    options.load_weights_folder = None
    run_dir = folder.parent.parent
    options.log_dir, options.model_name = str(run_dir.parent), run_dir.name
    from dsi_trainer import DSITrainer
    DSITrainer.validate_options(options)
    print('RECOVERY PLAN=' + json.dumps(summary, indent=2), flush=True)
    if args.check_only:
        print('CHECK ONLY PASSED: no files changed; training has not started.', flush=True)
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = run_dir / ('opt_before_resume_' + stamp + '.json')
    shutil.copy2(opt_path, backup)
    (run_dir / ('recovery_plan_' + stamp + '.json')).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    # Avoid keeping a second CPU copy of references/initial state during init.
    del state
    print('RESUMING EXISTING TRAINING; metadata backup:', backup, flush=True)
    trainer = DSITrainer(options)
    trainer.train()


if __name__ == '__main__':
    # Same library environment cleanup as the original train.py, before torch.
    if 'LD_LIBRARY_PATH' in os.environ or 'LD_PRELOAD' in os.environ:
        env = os.environ.copy()
        env.pop('LD_LIBRARY_PATH', None)
        env.pop('LD_PRELOAD', None)
        os.execve(sys.executable, [sys.executable] + sys.argv, env)
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'litemono-matplotlib'))
    main()
