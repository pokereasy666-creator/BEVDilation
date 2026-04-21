#!/usr/bin/env bash
# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Default to Ampere (A30/A100) if the caller didn't pin an arch list.
# Override by exporting TORCH_CUDA_ARCH_LIST before running this script,
# e.g. `TORCH_CUDA_ARCH_LIST="8.0;8.6" sh ./make.sh`.
: "${TORCH_CUDA_ARCH_LIST:=8.0}"
export TORCH_CUDA_ARCH_LIST

python setup.py build install
