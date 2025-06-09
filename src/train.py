import argparse
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import timm
import datetime

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights, inception_v3, Inception_V3_Weights
from sklearn.metrics import precision_score, recall_score, f1_score
from src.data_utils import transform_fn, clean_bg, get_normalization_terms


# change this to the path to your project
BASE_DIR = '/content/drive/MyDrive/Final_Project'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_ARGS = {
    "batch_size": 32,
    "orientation": "axial",
    "transform": False
}

TRAIN_ARGS = {
    "model": "resnet",
    "num_epochs": 20,
}

optim_funcs = {
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "RMSprop": optim.RMSprop
}


def calculate_metrics(y_true, y_pred, average='binary'):
  precision = precision_score(y_true, y_pred, average=average)
  recall = recall_score(y_true, y_pred, average=average)
  f1 = f1_score(y_true, y_pred, average=average)
  return {
      "precision": precision,
      "recall": recall,
      "f1_score": f1
  }


def get_model_from_name(model_name: str, config: dict):
    if model_name == "resnet":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(in_features, 2)
    )   
    elif model_name == "inception-v3":
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, aux_logits=True)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(in_features, 2)
        )
        # Auxiliary classifier
        if model.aux_logits:
            in_features_aux = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Sequential(
                nn.Dropout(config['dropout']),
                nn.Linear(in_features_aux, 2)
            )
    elif model_name == "vit":
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            drop_rate=config['dropout'],
            num_classes=2
        )
    return model


def train(model, train_loader, val_loader, optimizer, num_epochs, verbose=False, use_output_logits=False, save_every=None, save_dir=None):
    scores = {
        "train_losses": [],
        "val_losses": [],
        "val_accs": [],
        "val_f1": []
    }
    loss_fn = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            if use_output_logits:
              loss = loss_fn(outputs.logits, labels)
              if model.aux_logits:
                loss_aux = loss_fn(outputs.aux_logits, labels)
                loss = loss + 0.4 * loss_aux
            else:
              loss = loss_fn(outputs, labels)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_train_loss /= len(train_loader)
        scores["train_losses"].append(epoch_train_loss)
        if save_every is not None and (epoch + 1) % save_every == 0:
            checkpoint_name = "-".join(["checkpoint", str(epoch + 1) + ".pt"])
            os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_train_loss,
                },
                os.path.join(save_dir, 'checkpoints', checkpoint_name),
            )

        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            epoch_val_loss = 0.0
            y_preds = []
            y_trues = []
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                epoch_val_loss += loss.item()
                total += labels.size(0)
                predicted = torch.argmax(outputs.data, dim=1)
                correct += (predicted == labels).sum().item()

            epoch_val_loss /= len(val_loader)
            accuracy = correct / total

            if len(scores["val_accs"]) and accuracy < scores["val_accs"][-1]:
                patience -= 1
            else:
                patience = 3  # reset on improvement

            scores["val_losses"].append(epoch_val_loss)
            scores["val_accs"].append(accuracy)

            if verbose:
              print(f"\nEpoch {epoch+1}/{num_epochs}")
              print(f"- Train Loss: {epoch_train_loss:.4f}")
              print(f'- Val Loss: {epoch_val_loss:.4f}')
              print(f'- Val Accuracy: {accuracy}')

            if patience == 0:
                print("Early stopping triggered.")
                break

        lr_scheduler.step()

    return scores

def hparam_tune(model_name, train_loader, val_loader, num_trials=20):

    # hparam tuning

    search_space = {
        "learning_rates": [1e-5, 3e-5, 5e-5, 1e-4],
        "optimizers": ["Adam", "AdamW", "RMSprop"],
        "dropout": [0, 0.1, 0.15, 0.2]
    }
    optim_funcs = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "RMSprop": optim.RMSprop
    }

    best_val_acc = 0
    best_model = None
    best_config = None

    for trial in range(num_trials):
        print(f"\n=== Trial {trial+1}/{num_trials} ===")

        # Randomly sample hyperparameters
        config = {k: random.choice(search_space[k]) for k in search_space}
        print(config)

        # get model
        model = get_model_from_name(model_name, config)
        model.to(DEVICE)

        optim_fn = optim_funcs[config['optimizers']]
        optimizer = optim_fn(model.parameters(), lr=config["learning_rates"])
        use_output_logits = (model_name == "inception-v3")

        scores = train(model, train_loader, val_loader, optimizer, num_epochs=3, use_output_logits=use_output_logits)
        train_loss = scores["train_losses"][-1]
        val_loss = scores["val_losses"][-1]
        val_acc = scores["val_accs"][-1]
        print(f"- Train loss: {train_loss}")
        print(f"- Val loss: {val_loss}")
        print(f"- Val acc: {val_acc}")

        if scores["val_accs"][-1] > best_val_acc:
            best_val_acc = scores["val_accs"][-1]
            best_model = model
            best_config = config

    print(f"\nBest model accuracy: {best_val_acc:.4f}")
    print(f"Best config: {best_config}")
    return best_config, best_model

