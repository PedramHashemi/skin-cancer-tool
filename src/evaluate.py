"""Test the model"""

import os
import torch
import numpy as np
import argparse
from dotenv import load_dotenv
import mlflow
import torchvision
from processor import Processor
from torch.utils.data import DataLoader
from torchvision import transforms
from models import TailModel

load_dotenv()



def evaluate_mode(model_name: str, mlflow_run_id: str):
    """Evaluate the model on the test dataset."""
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD
        ),
    ])
    test_data = torchvision.datasets.ImageFolder(
        root=os.environ.get("TEST_DATA_PATH"),
        transform=test_transform
    )

    # model_name = "models/skin_cancer_model_resnet18_20251209_105710.pth"
    checkpoint = torch.load(model_name, weights_only=False)
    model=TailModel(num_classes=7, dropout=DROPOUT)
    model.load_state_dict(checkpoint['model_state_dict'])

    processor = Processor(model=model, loss_fn=None, optimizer=None)
    processor.to(DEVICE)

    with mlflow.start_run(run_id=mlflow_run_id):
        tests = []
        actual = []
        for tens, label in test_data:
            tests.append(np.argmax(processor.predict(tens.unsqueeze(0).to(DEVICE))))
            actual.append(label)

        mlflow.log_metric("accuracy", sum(np.array(tests)==np.array(actual))*100/len(actual))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate the trained model.")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Path to the trained model file.",
        required=True
    )
    parser.add_argument(
        "--mlflow_run_id",
        type=str,
        help="The run_id from which the model comes.",
        required=True
    )
    args = parser.parse_args()

    evaluate_mode(
        model_name=args.model_name,
        mlflow_run_id=args.mlflow_run_id
    )