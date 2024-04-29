import argparse
import shutil
import json
import subprocess
import os

class DreamBooth():

    # @staticmethod
    # def preprocessing():
    #     packages = [
    #         "accelerate>=0.25.0",
    #         "autocrop>=1.3.0",
    #         "awscli>=1.31.12",
    #         "boto3>=1.33.12",
    #         "bitsandbytes>=0.41.0",
    #         "deepspeed>=0.12.6",
    #         "diffusers==0.25.0",
    #         "matplotlib>=3.8.0",
    #         "peft==0.7.1",
    #         "pillow>=10.0.1",
    #         "prodigyopt>=1.0",
    #         "safetensors>=0.4.1",
    #         "sagemaker>=2.199.0",
    #         "tensorboard>=2.15.1",
    #         "torch==2.0.1",
    #         "torchvision>=0.15.2",
    #         "tqdm>=4.66.1",
    #         "transformers==4.36.2",
    #         "wandb>=0.16.1",
    #         "xformers>=0.0.19"
    #     ]

    #     for package in packages:
    #         subprocess.run(['pip', 'install', package], check=True)
    #     subprocess.run("pip install -q py_dreambooth", shell=True)
    

    def generation(self, OUTPUT_DIR_FOR_IMAGES, NUM_IMAGES, gender, name):
        import torch
        from py_dreambooth.dataset import LocalDataset
        from py_dreambooth.model import SdDreamboothModel
        from py_dreambooth.predictor import LocalPredictor
        from py_dreambooth.trainer import LocalTrainer
        from py_dreambooth.utils.image_helpers import display_images
        from py_dreambooth.utils.prompt_helpers import make_prompt

        DATA_DIR = "dataset_persons/" + name + "_" + gender
        OUTPUT_DIR_MODEL = "models"
        os.makedirs(OUTPUT_DIR_MODEL, exist_ok=True)
        dataset = LocalDataset(DATA_DIR)
        dataset = dataset.preprocess_images(detect_face=True)
        SUBJECT_NAME = "zwx"
        CLASS_NAME = gender

        model = SdDreamboothModel(
            subject_name=SUBJECT_NAME,
            class_name=CLASS_NAME,
            max_train_steps=1000,
        )

        trainer = LocalTrainer(output_dir=OUTPUT_DIR_MODEL)
        predictor = trainer.fit(model, dataset)

        PROMPT = ("photo of zwx " + gender)
        #prompt = PROMPT + ", ultra realistic, depth, beautiful lighting, hyperrealistic, octane, epic composition, masterpiece, depth of field,breathtaking, 8k resolution, establishing shot, artistic, octane render"
        #print(f"The prompt is as follows:\n{prompt}")

        images = predictor.predict(
            PROMPT,
            height=768,
            width=512,
            num_images_per_prompt=NUM_IMAGES,
        )

        new_folder_path = os.path.join(OUTPUT_DIR_FOR_IMAGES, name)
        new_folder_path += f"_{gender}"
        os.makedirs(new_folder_path, exist_ok=True)

        for idx, img in enumerate(images):
            img.save(
                os.path.join(new_folder_path, f"generated_image_{idx}.png"))
            
        shutil.rmtree("models")
        shutil.rmtree("dataset_persons/" + name + "_" + gender + "_preproc")
        torch.cuda.empty_cache()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str,
                        help="Path to save the generated images")
    parser.add_argument("--num_images", type=int,
                        help="Number of images to generate", default=50)
    args = parser.parse_args()

    model = DreamBooth()
    # DreamBooth().preprocessing()

    top_dirs_path = "dataset_persons"
    sub_directories = [sub_dir.path for sub_dir in
                       os.scandir(top_dirs_path) if sub_dir.is_dir()]

    for directory in sub_directories:
        directory_name = os.path.basename(directory)
        parts = directory_name.split("_")
        name = "_".join(parts[:-1])
        gender = parts[-1]
        #print(name, gender)
        if name[0] != '.' and gender != "preproc":
            model.generation(args.save_path, args.num_images, gender, name)