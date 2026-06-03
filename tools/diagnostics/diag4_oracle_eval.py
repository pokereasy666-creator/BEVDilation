# Copyright (c) OpenMMLab. All rights reserved.
"""Diagnostic 4 Oracle A -- perfect-foreground-mask detection ceiling.

Runs the standard nuScenes val evaluation with the SVDB foreground mask replaced
by the ground-truth footprint (oracle). The resulting NDS/mAP is the upper bound
on what Temporal Foreground Propagation (TFP) could deliver -- the go/no-go gate.

The oracle is gated by ``model.oracle_fg`` (set here). With ``--oracle off`` the
flag is disabled and the run is a control that must reproduce the baseline
(74.6/72.0), proving the GT-collection + plumbing did not perturb anything.

The oracle mask is built inside the model by SVDB.obtain_bev_mask_gt from the
pipeline-transformed GT boxes -- the SAME call used to build the training target
-- so it is frame-correct by construction. A one-time [oracle-align] IoU check
inside Voxel_Generation.forward fails loud if the mask is misaligned.

MUST run on the offline server (needs the full mmdet3d stack, a GPU, the trained
checkpoint and val data). Cannot run in a bare CI/dev environment.

Examples::

    # oracle ceiling
    python tools/diagnostics/diag4_oracle_eval.py \
        configs/bevdilation/bevdilation_oracleA.py work_dirs/.../latest.pth
    # control (should reproduce baseline)
    python tools/diagnostics/diag4_oracle_eval.py \
        configs/bevdilation/bevdilation_oracleA.py work_dirs/.../latest.pth --oracle off
"""
import argparse
import os.path as osp
import sys

# Put the repo root on sys.path so model backbones can `from ops_dcnv3 import ...`
# (ops_dcnv3/ lives at the repo root). This file is tools/diagnostics/<this>.
sys.path.insert(
    0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmcv.runner import load_checkpoint, wrap_fp16_model  # noqa: E402

from mmdet3d.apis import single_gpu_test  # noqa: E402
from mmdet3d.datasets import build_dataloader, build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from mmdet.datasets import replace_ImageToTensor  # noqa: E402

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
        description='Diagnostic 4 Oracle A: GT-foreground-mask detection ceiling.')
    parser.add_argument('config', help='oracle test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--oracle',
        choices=['on', 'off'],
        default='on',
        help='on: inject the GT foreground mask at SVDB (ceiling); '
             'off: disable the flag (control, should reproduce baseline)')
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- config (matches tools/test.py) ----
    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.gpu_ids = [0]

    # ---- dataset / dataloader: single-gpu, non-distributed (matches tools/test.py) ----
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

    # ---- model + checkpoint (matches tools/test.py) ----
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

    # ---- the only behavioral switch ----
    model.oracle_fg = (args.oracle == 'on')
    print(f'[diag4] oracle_fg = {model.oracle_fg} (--oracle {args.oracle})')

    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    outputs = single_gpu_test(model, data_loader)

    # ---- evaluate exactly as tools/test.py --eval bbox ----
    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule']:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric='bbox'))
    metrics = dataset.evaluate(outputs, **eval_kwargs)
    print(metrics)

    nds = next((v for k, v in metrics.items() if k.endswith('NDS')), None)
    mapv = next((v for k, v in metrics.items() if k.endswith('mAP')), None)
    print(f'\n[diag4] oracle={args.oracle}  NDS={nds}  mAP={mapv}')


if __name__ == '__main__':
    main()
