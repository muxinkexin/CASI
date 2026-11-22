#!/usr/bin/env python3
"""Patch DSI mode transitions and install checkpoint-driven recovery helpers."""
import argparse
import ast
from datetime import datetime
from pathlib import Path
import shutil
import tempfile

MARKER = '# DSI_MODE_COMPAT_V1'


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Expected exactly one patch target: ' + before[:90])
    return text.replace(before, after, 1)


def replace_method(text, name, replacement):
    tree = ast.parse(text)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DSITrainer')
    matches = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError('Cannot locate DSITrainer.' + name)
    node = matches[0]
    lines = text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno - 1]) + replacement + ''.join(lines[node.end_lineno:])


def patch_source(source):
    if MARKER in source:
        return source
    source = replace_once(source, 'import networks\n',
                          'import networks\nfrom dsi_mode_compat import set_mode_tree, freeze_module  ' + MARKER + '\n')
    source = replace_once(source, '        super().__init__(options)\n', '''        # Resume checkpoints already contain trained models and initial_depth.
        # Avoid reloading ImageNet files/downloads before restoring those states.
        initial_options = (options.weights_init, options.mypretrain)
        if options.dsi_resume:
            options.weights_init, options.mypretrain = "scratch", None
        try:
            super().__init__(options)
        finally:
            options.weights_init, options.mypretrain = initial_options
''')
    source = replace_once(source, 'encoder.eval().requires_grad_(False)', 'freeze_module(encoder)')
    source = replace_once(source, 'decoder.eval().requires_grad_(False)', 'freeze_module(decoder)')
    source = replace_once(source, 'model.requires_grad_(False).eval()', 'freeze_module(model)')
    source = replace_once(source, 'model.eval().requires_grad_(False)', 'freeze_module(model)')
    source = replace_once(source,
        'networks.ResnetEncoder(18, self.opt.weights_init == "pretrained", num_input_images=3)',
        'networks.ResnetEncoder(18, resume_state is None and self.opt.weights_init == "pretrained", num_input_images=3)')
    source = replace_method(source, 'set_train', '''    def set_train(self):
        for model in self.models.values():
            set_mode_tree(model, True)
        for model in self.models_pose.values():
            set_mode_tree(model, self.phase == "raw")
        for reference in self.reference.values():
            for model in reference.values():
                set_mode_tree(model, False)
''')
    source = replace_method(source, 'set_eval', '''    def set_eval(self):
        for model in self.models.values():
            set_mode_tree(model, False)
        for model in self.models_pose.values():
            set_mode_tree(model, False)
        for reference in self.reference.values():
            for model in reference.values():
                set_mode_tree(model, False)
''')
    compile(source, 'dsi_trainer.py', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(args.repo).expanduser().resolve()
    target = root / 'dsi_trainer.py'
    if not target.is_file() or not (root / 'train.py').is_file():
        parser.error('--repo must be the existing Lite-CASI/Mono repository root')
    source = patch_source(target.read_text(encoding='utf-8'))
    payload = Path(__file__).resolve().parent / 'payload'
    updates = {target: source.encode('utf-8')}
    for name in ('dsi_mode_compat.py', 'resume_dsi_training.py', 'test_dsi_mode_compat.py'):
        content = (payload / name).read_bytes()
        compile(content, name, 'exec')
        updates[root / name] = content
    changes = {p: data for p, data in updates.items() if not p.is_file() or p.read_bytes() != data}
    for path in changes:
        print('PATCH:', path)
    if args.dry_run or not changes:
        print('Dry run passed.' if args.dry_run else 'Already patched.')
        return
    backup = Path(tempfile.mkdtemp(prefix='dsi_recovery_backup_', dir=root.parent))
    prior = {path: path.read_bytes() if path.is_file() else None for path in changes}
    try:
        for path, data in changes.items():
            if prior[path] is not None:
                shutil.copy2(path, backup / path.name)
            path.write_bytes(data)
    except Exception:
        for path, data in prior.items():
            if data is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(data)
        raise
    print('PATCH COMPLETE; original code backup:', backup)
    print('No checkpoint, optimizer or training log has been modified.')


if __name__ == '__main__':
    main()
