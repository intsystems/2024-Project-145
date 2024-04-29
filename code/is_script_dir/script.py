import torch
import os
import cv2
import numpy as np
import argparse

def is_metrics(torch_images):
    _ = torch.manual_seed(123)
    from torchmetrics.image.inception import InceptionScore
    inception = InceptionScore()
    inception.update(torch_images)
    file.write(str(inception.compute()[0].item()) + " " + str(inception.compute()[1].item()) + "\n")

def images_to_tensor(image_dir):
        images = []
        for filename in os.listdir(image_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img = cv2.imread(os.path.join(image_dir, filename))
                img_resized = cv2.resize(img, (299, 299))
                images.append(img_resized)

        np_images = np.array(images)
        torch_images = torch.from_numpy(np_images)
        torch_images = torch_images.permute(0, 3, 1, 2)

        is_metrics(torch_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_path", type=str,
                        help="Path to generated images")
    args = parser.parse_args()
    #os.system("pip install torch-fidelity")

    top_dirs_path = args.generated_path
    sub_directories = [sub_dir.path for sub_dir in os.scandir(top_dirs_path) if
                       sub_dir.is_dir()]
    with open("output.txt", "w") as file:
        for directory in sub_directories:
            images_to_tensor(directory)
