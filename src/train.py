"""Train a model for skin cancer classification using transfer learning."""

import os
import json
import logging
import logging.config
import datetime
import mlflow
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from models import TailModel
from processor import Processor
from dotenv import load_dotenv

load_dotenv()

# Setting up logging
with open('config/logging.json', 'r') as f:
    log_config = json.load(f)

logging.config.dictConfig(log_config)
logger = logging.getLogger(__name__)

# [ ]: Mlflow for tracking experiments
EXPERIMENT_NAME = "skin_cancer_classification"
logger.info(f"Setting up MLflow experiment: {EXPERIMENT_NAME}")
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
logger.info(f"MLflow Experiment ID: {experiment.experiment_id}")
# mlflow.pytorch.autolog()


BATCH_SIZE = 16
SHUFFLE=True
LEARNING_RATE = 0.001
DROPOUT = 0.5
N_EPOCHS = 2
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MEAN=[0.5706329, 0.5461266, 0.76312]
STD=[0.16920634, 0.151464, 0.14013906]


logger.info("Creating Train Data Loader.")
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(
        mean=MEAN,
        std=STD
    ),

])
train_data = torchvision.datasets.ImageFolder(
    root=os.environ.get("TRAIN_DATA_PATH"),
    transform=train_transform
)
train_data_loader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE
)

logger.info("Creating Valid Data Loader.")
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN,
        std=STD
    ),
])
val_data = torchvision.datasets.ImageFolder(
    root=os.environ.get("VALID_DATA_PATH"),
    transform=val_transform
)
val_data_loader = DataLoader(
    dataset=val_data,
    batch_size=BATCH_SIZE
)

logger.info("Instantiating the model.")
with mlflow.start_run() as run:
    model = TailModel(num_classes=7, dropout=DROPOUT)

    logger.info("Setting up loss function.")
    loss = torch.nn.CrossEntropyLoss()

    logger.info("Instantiating the optimizer.")
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE
    )

    logger.info("Setting up the training processor.")
    proc_transfer = Processor(
        model=model,
        loss_fn=loss,
        optimizer=optimizer,
    )

    logger.info("Adding data loaders.")
    proc_transfer.set_loaders(
        train_loader=train_data_loader,
        val_loader=val_data_loader
    )

    logger.info("Forcing the cuda.")
    proc_transfer.to(DEVICE)

    logger.info("Starting up the tesnorboard the training process.")
    proc_transfer.set_tensorboard('skin_cancer', folder='runs')
    proc_transfer.train(n_epochs=N_EPOCHS)


    logger.info("Plotting the losses.")
    fig = proc_transfer.plot_losses()
    # fig.savefig(f'./images/loss_plot_{datetime.datetime.today()}.png')
    mlflow.log_params(
        {
            "batch_size": BATCH_SIZE,
            "optimizer": optimizer.__class__.__name__,
            "loss_function": loss.__class__.__name__,
            "learning_rate": LEARNING_RATE,
            "dropout": DROPOUT,
            "device": DEVICE
        }
    )
    mlflow.log_figure(fig, "loss_plot.png")
    mlflow.pytorch.log_model(
        proc_transfer.model,
        artifact_path="model",
        input_example=torch.randn(1, 3, 244, 244).numpy()
    )
