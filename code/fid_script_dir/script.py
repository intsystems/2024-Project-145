import subprocess
import os
import argparse

def get_subdirectories(path):
    subdirectories = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            subdirectories.append(item_path)
    return subdirectories

def fid(dir1, dir2):
    subdirectories1 = get_subdirectories(dir1)
    subdirectories2 = get_subdirectories(dir2)

    with open("output.txt", "w") as file:
        for subdirectory1, subdirectory2 in zip(subdirectories1, subdirectories2):
            REAL_PATH = subdirectory1 + "/"
            GENERATED_PATH = subdirectory2 + "/"
            output = subprocess.check_output((f'python3 -m pytorch_fid {REAL_PATH} {GENERATED_PATH}'),shell=True)
            lines = output.decode("utf-8").strip()
            last_line = lines.split("\n")[-1]
            file.write(last_line.split(':')[1].strip() + "\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", type=str,
                        help="Path to real images")
    parser.add_argument("--generated_path", type=str,
                        help="Path to generated images")
    args = parser.parse_args()

    # os.system("pip install opencv-python")
    # os.system("pip install torch")
    # os.system("pip install torchvision")
    # os.system("pip install python3")
    # os.system("pip install pillow")
    # os.system("pip install numpy")
    # os.system("pip install scipy")
    # os.system("pip install pytorch-fid")


    fid(args.real_path, args.generated_path)


