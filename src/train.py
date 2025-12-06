"""Train a model for skin cancer classification using transfer learning."""
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms.v2 import Normalize, Compose
from models import TailModel
import logging
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights
from processor import Processor
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# [ ]: Logging models
# [ ]: Mlflow for tracking experiments
# [ ]: Final script for training the model


batch_size = 16
shuffle=True
train_data_path = "/home/bigbang/workshop/projects/skin-cancer-tool/data/train"
test_data_path = "/home/bigbang/workshop/projects/skin-cancer-tool/data/test"
valid_data_path = "/home/bigbang/workshop/projects/skin-cancer-tool/data/valid"

mean=[0.5706329, 0.5461266, 0.76312]
std=[0.16920634, 0.151464, 0.14013906]


logger.warning("Creating Train Data Loader.")
train_transform = Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    Normalize(
        mean=mean,
        std=std
    ),

])
train_data = torchvision.datasets.ImageFolder(
    root=train_data_path,
    transform=train_transform
)
train_data_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=shuffle
)

logger.info("Creating Valid Data Loader.")
val_transform = Compose([
    transforms.ToTensor(),
    Normalize(
        mean=mean,
        std=std
    ),
])
val_data = torchvision.datasets.ImageFolder(
    root=valid_data_path,
    transform=val_transform
)
val_data_loader = DataLoader(
    dataset=val_data,
    batch_size=batch_size
)

model = TailModel(num_classes=7, dropout=0.5)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=0.001
)

proc_transfer = Processor(
    model=model,
    loss_fn=loss,
    optimizer=optimizer,
)
proc_transfer.set_loaders(
    train_loader=train_data_loader,
    val_loader=val_data_loader
)
proc_transfer.to("cuda:0")
proc_transfer.train(n_epochs=10)
proc_transfer.set_tensorboard('skin_cancer', folder='runs')
fig = proc_transfer.plot_losses()
fig.savefig(f'images/loss_plot_{datetime.datetime()}.png')
