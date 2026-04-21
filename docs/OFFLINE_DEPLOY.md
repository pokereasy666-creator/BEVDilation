# Offline deployment to an Ampere GPU server (e.g. NVIDIA A30)

This guide describes how to build the BEVDilation environment on an **online
builder host**, package it, and deploy it to an **offline target server** with
an NVIDIA Ampere GPU (A30, A100, A6000, 3090, etc.). Instructions default to
the A30 (compute capability `sm_80`); notes call out where to change
`TORCH_CUDA_ARCH_LIST` for other GPUs.

## Prerequisites

### Builder host (online)
- Ubuntu 20.04 (glibc 2.31) — must match the target OS major version.
- Miniconda or Anaconda.
- CUDA 11.7 toolkit installed at `/usr/local/cuda-11.7` (provides `nvcc`).
- Internet access to PyPI, `download.pytorch.org`, and `download.openmmlab.com`.
- A GPU is **not** required: `nvcc` with `TORCH_CUDA_ARCH_LIST` cross-compiles
  for the target. Set `FORCE_CUDA=1` if building on a GPU-less host.

### Target host (offline)
- Ubuntu 20.04 (same glibc as builder).
- NVIDIA driver ≥ 515 (R525+ recommended) — required for the CUDA 11.7
  runtime shipped inside the torch wheel.
- NVIDIA A30 (or any Ampere card matching the `TORCH_CUDA_ARCH_LIST` used at
  build time).
- CUDA toolkit is **not** required on the target.

## Pinned version matrix

| Component | Version |
|---|---|
| Python | 3.8 |
| PyTorch / torchvision | 1.13.1+cu117 / 0.14.1+cu117 |
| mmcv-full | 1.6.0 (cu117 / torch1.13 wheel) |
| mmdet / mmsegmentation | 2.25.1 / 0.25.0 |
| causal-conv1d / mamba-ssm | 1.1.0 / 1.1.2 |
| spconv | spconv-cu117 |
| `TORCH_CUDA_ARCH_LIST` | `"8.0"` for A30/A100; `"8.6"` for A6000/3090 |

## Phase 1 — build on the online host

```shell
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.0"        # A30 / A100
export FORCE_CUDA=1                      # set only when builder has no GPU
nvcc --version                           # expect release 11.7

conda create -n bevdilation python=3.8 -y
conda activate bevdilation
pip install --upgrade pip wheel setuptools

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

pip install mmcv-full==1.6.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmdet==2.25.1 mmsegmentation==0.25.0

pip install causal-conv1d==1.1.0
pip install mamba-ssm==1.1.2
pip install spconv-cu117

# Non-editable install so bev_pool_v2_ext ends up in site-packages and is
# captured by conda-pack. Do NOT use `pip install -e .` for packaging.
cd /path/to/BEVDilation
pip install -v .

cd /path/to/BEVDilation/ops_dcnv3
sh ./make.sh                             # honours TORCH_CUDA_ARCH_LIST

python -c "import torch, mmcv, mmdet, mmseg, mamba_ssm, causal_conv1d, spconv; \
           from mmdet3d.ops.bev_pool_v2 import bev_pool_v2; import DCNv3; print('ok')"
```

## Phase 2 — package

### Option A (recommended): `conda-pack`

```shell
conda deactivate
conda install -n base -c conda-forge conda-pack -y
conda pack -n bevdilation -o bevdilation-a30-cu117.tar.gz --ignore-missing-files

# Archive the repo for its configs, tools, and data prep scripts:
tar --exclude='.git' -czf BEVDilation-src.tar.gz -C /path/to BEVDilation

sha256sum bevdilation-a30-cu117.tar.gz BEVDilation-src.tar.gz \
    > bevdilation-a30-cu117.sha256
```

Ship the two tarballs plus the checksum file to the target.

Constraints:
- Target OS must match the builder (Ubuntu 20.04 → 20.04).
- `conda-pack` bundles the Python interpreter, so the target does not need
  conda itself; a conda install on the target just makes activation nicer.

### Option B: pip wheelhouse

Only choose this if the target already has a working Python 3.8 + CUDA torch
install and you're adding BEVDilation's deps alongside it.

```shell
mkdir -p wheels
pip download --platform manylinux2014_x86_64 --python-version 38 \
    --only-binary=:all: --dest ./wheels \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
pip download --dest ./wheels \
    mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip download --dest ./wheels \
    mmdet==2.25.1 mmsegmentation==0.25.0 \
    causal-conv1d==1.1.0 mamba-ssm==1.1.2 spconv-cu117 \
    numba==0.53.0 nuscenes-devkit lyft_dataset_sdk plyfile scikit-image \
    tensorboard 'trimesh>=2.35.39,<2.35.40' 'networkx>=2.2,<2.3'

# Pre-build the two in-tree CUDA extensions into wheels so the target doesn't
# need nvcc:
cd /path/to/BEVDilation && pip wheel . -w ./wheels
cd /path/to/BEVDilation/ops_dcnv3 && pip wheel . -w ../wheels

tar -czf bevdilation-a30-wheels.tar.gz wheels BEVDilation
```

On the target: `pip install --no-index --find-links ./wheels <wheel>`.

## Phase 3 — deploy on the offline target

Assumes Option A and that the tarballs have been copied to
`/opt/bevdilation/` on the target.

```shell
# 1. Unpack the environment
mkdir -p $HOME/miniconda3/envs/bevdilation
tar -xzf /opt/bevdilation/bevdilation-a30-cu117.tar.gz \
    -C $HOME/miniconda3/envs/bevdilation
source $HOME/miniconda3/envs/bevdilation/bin/activate
conda-unpack        # rewrites shebangs / embedded absolute paths

# 2. Unpack the source repo
mkdir -p /opt/bevdilation
tar -xzf /opt/bevdilation/BEVDilation-src.tar.gz -C /opt/bevdilation
cd /opt/bevdilation/BEVDilation

# 3. Verify driver and imports
nvidia-smi          # driver >= 515; device name must contain the target GPU
python -c "import torch; \
           print(torch.__version__, torch.cuda.is_available(), \
                 torch.cuda.get_device_name(0))"
python -c "from mmdet3d.ops.bev_pool_v2 import bev_pool_v2; \
           import DCNv3, mmcv, mmdet, mmseg, mamba_ssm, causal_conv1d, spconv; \
           print('ok')"
```

## Phase 4 — dataset and Hilbert template

Dataset preparation follows `README.md`. If the target is fully offline, run
nuScenes preprocessing on the builder (with a copy of the dataset) and rsync
the generated `bevdetv3-*.pkl` / `gt_database` files into
`data/nuscenes/` on the target.

Generate the Hilbert curve template on the target:

```shell
cd /opt/bevdilation/BEVDilation
mkdir -p data/hilbert
python ./tools/create_hilbert_curve_template.py
```

## End-to-end verification

```shell
# 1-GPU smoke training (A30 has 24 GB — you may need to reduce
# samples_per_gpu or enable fp16 in configs/bevdilation/bevdilation.py)
./tools/dist_train.sh configs/bevdilation/bevdilation.py 1

# Evaluation with a paper checkpoint copied to the target:
./tools/dist_test.sh configs/bevdilation/bevdilation.py <ckpt> 1 --eval mAP
```

## Troubleshooting

- **`ImportError: cannot import name 'bev_pool_v2_ext'`** — The extension was
  not compiled, or the wrong CUDA arch was baked in. Rebuild with
  `TORCH_CUDA_ARCH_LIST` matching the target (`"8.0"` for A30/A100).
- **`CUDA error: no kernel image is available for execution on the device`**
  — same root cause as above; the compiled kernel does not match the target's
  compute capability.
- **`conda-unpack: command not found`** — You are inside the unpacked env;
  `conda-unpack` lives at `$ENV/bin/conda-unpack`. Activate the env first,
  then run it.
- **Driver too old** — `nvidia-smi` reports driver < 515. Upgrade the driver
  on the target; you do not need to install the CUDA toolkit.
- **Builder host has no GPU** — Set `FORCE_CUDA=1` before running
  `pip install -v .` and `ops_dcnv3/make.sh`.
- **Out of memory on A30 (24 GB)** — Halve `samples_per_gpu` in the config or
  enable fp16. This is a config tweak, not an install step.
