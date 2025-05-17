import os
import shutil
import kagglehub

# Download latest version
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

print("Path to dataset files:", path)
# os.makedirs("data", exist_ok=True)
shutil.move(path, "./data")
os.system("")