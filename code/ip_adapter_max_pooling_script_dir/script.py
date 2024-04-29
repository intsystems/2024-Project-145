import subprocess
import torch
import argparse
import os
from PIL import Image

class ImageProcessor:
    # def __init__(self):
    #     self.commands = [
    #         "pip install diffusers",
    #         "pip install git+https://github.com/tencent-ailab/IP-Adapter.git",
    #         "cd IP-Adapter",
    #         "git lfs install",
    #         "git clone https://huggingface.co/h94/IP-Adapter",
    #         "mv IP-Adapter/models models",
    #         "mv IP-Adapter/sdxl_models sdxl_models",
    #         "pip install einops",
    #         "pip install accelerate"
    #     ]

    # def run_commands(self):
    #     for cmd in self.commands:
    #         subprocess.run(cmd, shell=True)

    def initialize_ip_model(self):
        from diffusers import StableDiffusionPipeline, \
            StableDiffusionImg2ImgPipeline, \
            StableDiffusionInpaintPipelineLegacy, \
            DDIMScheduler, AutoencoderKL
        from ip_adapter.ip_adapter import IPAdapter

        base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
        vae_model_path = "stabilityai/sd-vae-ft-mse"
        image_encoder_path = "models/image_encoder/"
        ip_ckpt = "models/ip-adapter_sd15.bin"
        device = "cuda"

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(vae_model_path).to(
            dtype=torch.float16)

        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )

        self.ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

    def generate_images(self, output_dir, num_images, images_directory_path,
                        directory_name, gender):
        images = self.ip_model.generate(images_directory_path,
                                        num_samples=num_images,
                                        num_inference_steps=50, seed=42,
                                        prompt=f"photo of a {gender}", scale=0.6)

        parent_dir = output_dir
        new_dir = directory_name
        os.makedirs(os.path.join(parent_dir, new_dir))

        new_directory_path = output_dir + new_dir
        for i, img in enumerate(images):
            img_path = os.path.join(new_directory_path, f'image_{i + 1}.jpg')
            img.save(img_path)
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str,
                        help="Path to save the generated images")
    parser.add_argument("--num_images", type=int,
                        help="Number of images to generate", default=50)
    args = parser.parse_args()

    ip_processor = ImageProcessor()
    # ip_processor.run_commands()
    ip_processor.initialize_ip_model()

    top_dirs_path = "dataset_persons"
    sub_directories = [sub_dir.path for sub_dir in os.scandir(top_dirs_path) if
                       sub_dir.is_dir()]

    for directory in sub_directories:
        directory_name = os.path.basename(directory)
        if directory_name[0] == ".":
            continue
        parts = directory_name.split("_")
        name = "_".join(parts[:-1])
        gender = parts[-1]
        ip_processor.generate_images(args.save_path, args.num_images,
                                      directory, directory_name, gender)