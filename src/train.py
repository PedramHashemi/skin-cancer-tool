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

with open("config/pipeline.json", "r") as f:
    pipeline_config = json.load(f)

# [x]: Mlflow for tracking experiments
EXPERIMENT_NAME = "skin_cancer_classification"
logger.info(f"Setting up MLflow experiment: {EXPERIMENT_NAME}")
# mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
logger.info(f"MLflow Experiment ID: {experiment.experiment_id}")
# mlflow.pytorch.autolog()



logger.info("Creating Train Data Loader.")
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(
        mean=pipeline_config["MEAN"],
        std=pipeline_config["STD"]
    ),

])
train_data = torchvision.datasets.ImageFolder(
    root=os.environ.get("TRAIN_DATA_PATH"),
    transform=train_transform
)
train_data_loader = DataLoader(
    dataset=train_data,
    batch_size=pipeline_config["BATCH_SIZE"],
    shuffle=pipeline_config["SHUFFLE"]
)

logger.info("Creating Valid Data Loader.")
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=pipeline_config["MEAN"],
        std=pipeline_config["STD"]
    ),
])
val_data = torchvision.datasets.ImageFolder(
    root=os.environ.get("VALID_DATA_PATH"),
    transform=val_transform
)
val_data_loader = DataLoader(
    dataset=val_data,
    batch_size=pipeline_config["BATCH_SIZE"],
)

logger.info("Instantiating the model.")
with mlflow.start_run() as run:
    model = TailModel(num_classes=7, dropout=pipeline_config["DROPOUT"])

    logger.info("Setting up loss function.")
    loss = torch.nn.CrossEntropyLoss()

    logger.info("Instantiating the optimizer.")
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=pipeline_config["LEARNING_RATE"]
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
    proc_transfer.to(pipeline_config["DEVICE"])

    # logger.info("Starting up the tesnorboard the training process.")
    # proc_transfer.set_tensorboard('skin_cancer', folder='runs')
    proc_transfer.train(n_epochs=pipeline_config["N_EPOCHS"])
    model_location = f"models/skin_cancer_model_resnet18_{datetime.datetime.today().strftime('%Y%m%d_%H%M%S')}.pth"
    proc_transfer.save_checkpoint(model_location)

    logger.info("Plotting the losses.")
    fig = proc_transfer.plot_losses()
    # mlflow.log_artifact(model_location)
    mlflow.log_params(
        {
            "optimizer": optimizer.__class__.__name__,
            "loss_function": loss.__class__.__name__,
        }
    )
    mlflow.log_dict(pipeline_config, "pipeline_config.json")
    mlflow.log_figure(fig, "loss_plot.png")
    mlflow.pytorch.log_model(
        proc_transfer.model,
        artifact_path="model",
        input_example=torch.randn(1, 3, 244, 244).numpy()
    )

