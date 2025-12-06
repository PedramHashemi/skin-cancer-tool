"""Making Embeddings using Pretrained Model.

Make features from the images using the pretrained model.
save the files in with npy format in the folder data/{train, test, valid}_embedding
"""

import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.warning(device)



mean= [np.float32(0.5706329), np.float32(0.5461266), np.float32(0.76312)]
stdev= [np.float32(0.16920634), np.float32(0.151464), np.float32(0.14013906)]
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=stdev),
])


resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()

for param in resnet.parameters():
    param.requires_grad = False

resnet = resnet.to(device)
resnet.eval()

logger.warning("Starting to create embeddings.")


for _ in ["train", "test", "valid"]:
    for dir in os.listdir(f"data/{_}"):
        os.makedirs(f"data/{_}_embedding/{dir}", exist_ok=True)
        for file in tqdm(os.listdir(os.path.join("data", _, dir))):
            if file.endswith(".jpg"):
                img = np.array(Image.open(os.path.join("data", _, dir, file)))
                img_transformed = transforms(img.astype(np.float32) / 255.)
                img_embedding = resnet(img_transformed.unsqueeze(0).to(device))
                np.save(
                    os.path.join("data", _+"_embedding", dir, file.split('.')[0] + ".npy"),
                    img_embedding.cpu().detach().numpy()
                )