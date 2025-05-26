"""Generating the classifcation."""

import torch
import data_loaders
import models
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import yaml
from ray import tune
from ray.tune.search.optuna import OptunaSearch

NUM_CLASSES = 7
NUM_EPOCHS = 2


# [ ]: train accuracy
# [ ]: Logging
# [x]: Ray for finetuning
# [ ]: mlflow for tracking experiments
# [ ]: Final script for training the model
# [ ]: Save the accuracies and print them in a plot
# [ ]: Add the testing after validation


def hyperparameter_tuning(config):
    # Chec if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # dataloaders
    mean, stdev = data_loaders.data_stats(
        data_dir="data/train",
        img_size=(224, 224)
    )
    train_transform, valid_transform = data_loaders.create_transform(
        resize=(244, 244),
        normalize=(mean, stdev),
        random_horizontal_flip=True,
        random_vertical_flip=True,
    )
    train_data_loader, valid_data_loader = data_loaders.prepare_data(
        data_dir="data",
        batch_size=config['batch_size'],
        shuffle=True,
        transforms=(train_transform, valid_transform),
    )
    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # model
    model = models.get_model(
        num_classes=NUM_CLASSES,
        dropout=config['droptout']
    ).to(device)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],        
    )

    # training
    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_data_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss_value = loss_fn(outputs, labels)
            loss_value.backward()
            optimizer.step()

        print(f"------- Epoch {epoch}/{NUM_EPOCHS}, loss: {loss_value.item():.4f}")

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
        print(f"Validation accuracy: {test_accuracy:.4f}")
        tune.report({
            "training_loss":loss_value,
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
    algo = OptunaSearch()  # ②

    tuner = tune.Tuner(  # ③
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
    print(results)
    