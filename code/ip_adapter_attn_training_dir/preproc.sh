#!/bin/bash

pip install git+https://github.com/tencent-ailab/IP-Adapter.git
cd IP-Adapter
git lfs install
git clone https://huggingface.co/h94/IP-Adapter
mv IP-Adapter/models models
mv IP-Adapter/sdxl_models sdxl_models
rm -r sdxl_models
pip install einops
pip install accelerate
pip install diffusers

python3 train_ip_adapter.py \
    --images_number 10 \
    --save_steps 10 \
    --num_train_epochs 10 \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --dir_data_json_files "dataset_persons_json" \
    --data_root_path "dataset_persons_images" \
    --image_encoder_path "models/image_encoder"