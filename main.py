"""Generating the classifcation."""

import torch
import data_loaders
import models
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

BATCH_SIZE = 32
NUM_CLASSES = 7
NUM_EPOCHS = 2

# [ ]: train accuracy
# [ ]: Logging
# [ ]: Hydra for finetuning
# [ ]: mlflow for tracking experiments
# [ ]: Final script for training the model
# [ ]: Save the accuracies and print them in a plot
# [ ]: Add the testing after validation

@hydra.main(version_base=None, config_path="config", config_name="hydra_config")
def hyperparameter_tuning():
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
        batch_size=BATCH_SIZE,
        shuffle=True,
        transforms=(train_transform, valid_transform),
    )
    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # model
    model = models.get_model(num_classes=NUM_CLASSES).to(device)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,        
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

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Validation accuracy: {accuracy:.4f}")
        confusion_matrix(y_test, y_pred)


if __name__ == "__main__":
    hyperparameter_tuning()