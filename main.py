"""Generating the classifcation."""

import torch
import data_loaders
import models

BATCH_SIZE = 32
NUM_CLASSES = 7
NUM_EPOCHS = 2

if __name__ == "__main__":
    # Chec if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        transforms=(train_transform, valid_transform)
    )
    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # model
    model = models.get_model(num_classes=NUM_CLASSES)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,        
    )

    # training
    for epoch in range(NUM_EPOCHS):
        for i, (image, label) in enumerate(train_data_loader):
            images = image.to(device)
            labels = label.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss_value = loss_fn(outputs, labels)
            loss_value.backward()
            optimizer.step()

        print(f"Epoch {epoch}/{NUM_EPOCHS}, loss: {loss_value.item():.4f}")

        # y_test = []
        # y_pred = []
        # with torch.no_grad():
        #     for i, (images, labels) in enumerate(valid_data_loader)