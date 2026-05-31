# Copyright (c) OpenMMLab. All rights reserved.
"""Diagnostic 1 (P_fg stability) -- capture stage.

Runs the trained BEVDilation checkpoint over the nuScenes val split and dumps,
for every frame, the pre-sigmoid foreground logits ``z_fg`` produced by
``SVDB.fg_pred`` (the module reached at
``model.pts_middle_encoder.SVDB.fg_pred``). In ``simple_test`` this tensor is
computed but discarded, so we capture it with a forward hook rather than editing
the detector.

This script MUST run on the offline server: it needs the full mmdet3d stack, a
GPU, the trained checkpoint and the val data. It cannot run in a bare CI/dev
environment. Each per-frame dump is ~130 KB (180x180 float16 + small metadata),
so the full val split (6019 frames) is roughly ~800 MB on disk.

The companion ``diag1_analyze.py`` consumes these dumps (numpy-only, no GPU).

Example::

    python tools/diagnostics/diag1_capture.py \
        configs/bevdilation/bevdilation.py work_dirs/.../latest.pth \
        --out-dir diag1_dumps
"""
import argparse
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor

# `compat_cfg` / `setup_multi_processes` moved between mmdet and mmdet3d across
# versions -- mirror the fallback used in tools/test.py so this works here.
try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
try:
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Diagnostic 1 capture: dump per-frame foreground logits '
                    'z_fg from SVDB.fg_pred over the val split.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out-dir',
        default='diag1_dumps',
        help='directory to write per-frame .npz dumps')
    parser.add_argument(
        '--max-frames',
        type=int,
        default=-1,
        help='cap the number of frames captured (smoke test); -1 means all')
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- config (matches tools/test.py:153-166) ----
    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.gpu_ids = [0]

    # ---- dataset / dataloader: single-gpu, non-distributed, samples_per_gpu=1,
    # no shuffle, test_mode=True (matches tools/test.py:184-212) ----
    distributed = False
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # ---- model + checkpoint (matches tools/test.py:220-242; the 4D/DAL
    # special-cases there are no-ops for BEVDilation and are omitted) ----
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    model.eval()

    # ---- forward hook capturing pre-sigmoid z_fg from SVDB.fg_pred ----
    holder = {'z': None}

    def _hook(_module, _inp, out):
        holder['z'] = out.detach().float().cpu().numpy()

    fg_pred = model.module.pts_middle_encoder.SVDB.fg_pred
    handle = fg_pred.register_forward_hook(_hook)

    # join captured frames back to the raw infos by token
    info_by_token = {info['token']: info for info in dataset.data_infos}

    mmcv.mkdir_or_exist(args.out_dir)
    n_total = len(data_loader)
    if args.max_frames is not None and args.max_frames >= 0:
        n_total = min(n_total, args.max_frames)
    prog = mmcv.ProgressBar(n_total)

    try:
        for i, data in enumerate(data_loader):
            if args.max_frames is not None and args.max_frames >= 0 \
                    and i >= args.max_frames:
                break

            holder['z'] = None
            with torch.no_grad():
                model(return_loss=False, rescale=True, **data)

            # fail loud rather than dump an empty/garbage frame
            if holder['z'] is None:
                raise RuntimeError(
                    f'SVDB.fg_pred forward hook did not fire on frame {i}; '
                    'aborting instead of writing an empty dump.')

            z_fg = np.squeeze(holder['z'])
            assert z_fg.shape == (180, 180), \
                f'unexpected z_fg shape {z_fg.shape} on frame {i}'
            z_fg = z_fg.astype(np.float16)

            token = data['img_metas'][0].data[0][0]['sample_idx']
            info = info_by_token[token]

            out_path = osp.join(args.out_dir, f'{i:05d}_{token}.npz')
            np.savez_compressed(
                out_path,
                z_fg=z_fg,
                token=token,
                scene_token=info['scene_token'],
                timestamp=np.float64(info['timestamp']),
                ego2global_translation=np.asarray(
                    info['ego2global_translation'], dtype=np.float64),
                ego2global_rotation=np.asarray(
                    info['ego2global_rotation'], dtype=np.float64),
                lidar2ego_translation=np.asarray(
                    info['lidar2ego_translation'], dtype=np.float64),
                lidar2ego_rotation=np.asarray(
                    info['lidar2ego_rotation'], dtype=np.float64),
                gt_boxes=np.asarray(info['gt_boxes'], dtype=np.float32),
                gt_names=np.asarray(info['gt_names']),
                gt_velocity=np.asarray(info['gt_velocity'], dtype=np.float32),
                num_lidar_pts=np.asarray(info['num_lidar_pts'], dtype=np.int32),
                valid_flag=np.asarray(info['valid_flag'], dtype=bool),
            )
            prog.update()
    finally:
        handle.remove()

    print(f'\nWrote z_fg dumps for {n_total} frames to {args.out_dir}')


if __name__ == '__main__':
    main()
