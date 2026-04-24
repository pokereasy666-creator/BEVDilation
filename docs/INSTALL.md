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

**c2. Swap `opencv-python` for `opencv-python-headless`.** `mmcv-full` pulls
`opencv-python`, whose recent wheels bundle a hashed Qt5 library under
`cv2/.libs/` that fails to load on headless servers (symptom:
`ImportError: libQt5Core-195a14c9.so.5.15.18: cannot open shared object
file`). Replace it with the headless build — `cv2` is the same module, minus
the Qt/GTK dependencies. Note: pip treats the two packages as distinct, so
this must run AFTER `mmcv-full` is installed (pre-installing headless will
not stop mmcv from pulling the GUI wheel).
```shell
pip uninstall -y opencv-python opencv-python-headless
pip install opencv-python-headless==4.5.5.64
```
On an offline server, transfer the wheel and install via
`pip install --no-index --find-links /path/to/wheels opencv-python-headless==4.5.5.64`.

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