**a. Create a conda virtual environment and activate it.**
```shell
conda create -n bevdilation python=3.8 -y
conda activate bevdilation
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
Tested combination: PyTorch 1.13.1 + torchvision 0.14.1 built against CUDA 11.7
(works with NVIDIA Ampere GPUs such as A30, A6000, 3090 and driver >= 515).
```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
```

**c. Install mmcv-full.** Use the prebuilt wheel that matches the torch + CUDA
combo above:
```shell
pip install mmcv-full==1.6.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0

```
**e. Install mmdet and mmseg.**
```shell
pip install causal-conv1d==1.1.0
pip install mamba-ssm==1.1.2
```

**f. Clone BEVDilation.**
```
git clone https://github.com/gwenzhang/BEVDilation.git
```

**g. Install BEVdilation**
```shell
cd /path/to/BEVDilation
pip install -v -e .
cd /path/to/BEVDilation/ops_dcnv3
sh ./make.sh
```

**h. Install spconv**
```shell
pip install spconv-cuxxx # select the corresponding cuda, e.g., spconv-cu117 for the pinned PyTorch 1.13.1+cu117 above
```

**i. (optional) Build for a specific GPU.** When compiling the in-tree CUDA
extensions (`bev_pool_v2_ext` via `pip install -v -e .` and `DCNv3` via
`ops_dcnv3/make.sh`), set `TORCH_CUDA_ARCH_LIST` to the compute capability of
your target GPU so the kernels are compiled for it. Examples:
```shell
export TORCH_CUDA_ARCH_LIST="8.0"       # A30 / A100
export TORCH_CUDA_ARCH_LIST="8.6"       # A6000 / 3090
export TORCH_CUDA_ARCH_LIST="8.0;8.6"   # multi-target build
export FORCE_CUDA=1                      # only needed when building on a host with no GPU
```
For offline deployment to an Ampere-only server, see
[docs/OFFLINE_DEPLOY.md](OFFLINE_DEPLOY.md).