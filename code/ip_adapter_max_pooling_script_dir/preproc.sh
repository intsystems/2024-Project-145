#!/bin/bash

pip install diffusers
pip install git+https://github.com/tencent-ailab/IP-Adapter.git
cd IP-Adapter
git lfs install
git clone https://huggingface.co/h94/IP-Adapter
mv IP-Adapter/models models
mv IP-Adapter/sdxl_models sdxl_models
pip install einops
pip install accelerate