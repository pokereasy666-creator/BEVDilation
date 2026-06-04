# Copyright (c) OpenMMLab. All rights reserved.
"""Diagnostic 4 -- SVDB foreground-mask oracle / no-dilation eval.

Runs the standard nuScenes val evaluation while substituting the SVDB foreground
mask at inference. Three modes (--mode, or the legacy --oracle on/off):
  * baseline    : no substitution (control; must reproduce baseline 74.6/72.0).
  * gt          : Oracle A -- inject the perfect GT footprint. NDS/mAP is the upper
                  bound on what Temporal Foreground Propagation (TFP) could deliver.
  * no_dilation : Proxy 4 -- inject an all-empty mask so expand_indices dilates zero
                  cells. The model runs on its original sparse LiDAR voxels only
                  (Mamba refinement, dense backbone, head all still run). The mAP
                  drop from baseline is dilation's total inference-time contribution
                  -- the upper bound on what any dilation-improvement method (the
                  persistence component) could ever recover.

gt mode builds the mask inside the model via SVDB.obtain_bev_mask_gt from the
pipeline-transformed GT boxes (the SAME call used for the training target), so it is
frame-correct by construction; a one-time [oracle-align] IoU check fails loud if it
is misaligned. no_dilation mode prints a one-time [no-dilation] count asserting the
mask is empty. Both checks live in Voxel_Generation.forward.

MUST run on the offline server (needs the full mmdet3d stack, a GPU, the trained
checkpoint and val data). Cannot run in a bare CI/dev environment.

Examples::

    # Oracle A ceiling (gt mask)
    python tools/diagnostics/diag4_oracle_eval.py \
        configs/bevdilation/bevdilation_oracleA.py work_dirs/.../latest.pth --mode gt
    # Proxy 4 (dilation disabled)
    python tools/diagnostics/diag4_oracle_eval.py \
        configs/bevdilation/bevdilation_oracleA.py work_dirs/.../latest.pth --mode no_dilation
    # control (should reproduce baseline) -- legacy --oracle off still works
    python tools/diagnostics/diag4_oracle_eval.py \
        configs/bevdilation/bevdilation_oracleA.py work_dirs/.../latest.pth --mode baseline
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
        help='legacy switch: on == "--mode gt" (Oracle A), off == "--mode baseline". '
             'Ignored when --mode is given.')
    parser.add_argument(
        '--mode',
        choices=['baseline', 'gt', 'no_dilation'],
        default=None,
        help='baseline: control, no injection (reproduces baseline); '
             'gt: inject the perfect GT foreground mask (Oracle A ceiling); '
             'no_dilation: inject an all-empty mask so dilation is disabled (Proxy 4). '
             'Overrides --oracle when set.')
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

    # ---- resolve mode (--mode wins; else derive from the legacy --oracle) ----
    if args.mode is not None:
        mode = args.mode
    else:
        mode = 'gt' if args.oracle == 'on' else 'baseline'
    # map mode -> (oracle_fg, oracle_mode); set on the model before MMDataParallel
    mode_map = {
        'baseline': (False, None),
        'gt': (True, 'gt'),
        'no_dilation': (True, 'no_dilation'),
    }
    model.oracle_fg, model.oracle_mode = mode_map[mode]
    print(f'[diag4] mode={mode} oracle_fg={model.oracle_fg} '
          f'oracle_mode={model.oracle_mode}')

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
    print(f'\n[diag4] mode={mode}  NDS={nds}  mAP={mapv}')


if __name__ == '__main__':
    main()
