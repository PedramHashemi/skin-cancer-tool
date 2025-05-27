"""Generating the classifcation."""

import logging
import torch
import data_loaders
import models
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import yaml
from ray import tune
from ray.tune.search.optuna import OptunaSearch
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient

NUM_CLASSES = 7
NUM_EPOCHS = 2

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# [ ]: train accuracy
# [x]: Logging models
# [x]: Ray for finetuning
# [x]: mlflow for tracking experiments
# [ ]: Final script for training the model
# [ ]: Save the accuracies and print them in a plot
# [ ]: Add the testing after validation


def hyperparameter_tuning(config):
    # Chec if GPU is available
    logger.info("---> Starting Hyperparameter Tuning.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # dataloaders
    logger.info("Calculating data stats.")
    mean, stdev = data_loaders.data_stats(
        data_dir="data/train",
        img_size=(224, 224)
    )

    logger.info("Making the transformers.")
    train_transform, valid_transform = data_loaders.create_transform(
        resize=(244, 244),
        normalize=(mean, stdev),
        random_horizontal_flip=True,
        random_vertical_flip=True,
    )

    logger.info("Creating Train and Valid dataloaders.")
    train_data_loader, valid_data_loader = data_loaders.prepare_data(
        data_dir="data",
        batch_size=config['batch_size'],
        shuffle=True,
        transforms=(train_transform, valid_transform),
    )
    
    # loss function
    logger.info("Defining the loss function.")
    loss_fn = torch.nn.CrossEntropyLoss()

    # model
    logger.info("Creating the model.")
    model = models.get_model(
        num_classes=NUM_CLASSES,
        dropout=config['dropout']
    ).to(device)

    # optimizer
    logger.info("Defining the optimizer.")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],        
    )

    # training
    logger.info("Starting Training.")
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": NUM_EPOCHS,
            "device": device,
            "lr": config['lr'],
            "batch-size": config['batch-size'],
            "dropout": config['dropout'],
        })
        for epoch in range(NUM_EPOCHS):
            logger.info(f"------- starting Epoch {epoch}/{NUM_EPOCHS}")

            mlflow.log_param("epoch", epoch)
            for i, (images, labels) in enumerate(train_data_loader):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                loss_value = loss_fn(outputs, labels)
                loss_value.backward()
                optimizer.step()

            logger.info(f"--> Training_loss: {loss_value.item():.4f}")
            mlflow.log_param("training_loss", loss_value.item())

            logger.info("Starting Testing.")
            y_test = []
            y_pred = []
            model.eval()
            # Validation
            for i, (images, labels) in enumerate(valid_data_loader):
                images = images.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    preds = torch.argmax(model(images), dim=1)
                
                y_test.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

            test_accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"--> Validation accuracy: {test_accuracy:.4f}")
            mlflow.log_param("test_accuracy", test_accuracy)
            tune.report({
                "training_loss":loss_value.item(),
                "test_accuracy":test_accuracy
            })


if __name__ == "__main__":

    with open("./config/ray_config.yaml", "r") as f:
        config_data = yaml.safe_load(f)

    config = {
        "lr": tune.loguniform(1e-5, 1e-2),          # Learning rate between 1e-5 and 1e-2
        "batch_size": tune.choice([8, 16]),   # Choice of three batch sizes
        "dropout_rate": tune.uniform(0.1, 0.5)      # Dropout rate between 0.1 and 0.5
    }
    algo = OptunaSearch()

    tuner = tune.Tuner(
        hyperparameter_tuning,
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            search_alg=algo,
        ),
        run_config=tune.RunConfig(
            stop={"training_iteration": 5},
        ),
        param_space=config,
    )
    results = tuner.fit()
    logger.info("Fine Tuning Results: ", results)
    best_trial = results.get_best_trial("loss", "min", "last")
    logger.info(f"Best trial config: {best_trial.config}")
    logger.info(f"Best trial final validation loss: {best_trial.last_result['test_accuracy']}")
    