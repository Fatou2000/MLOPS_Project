"""Modules for make VGG16 model"""
import os
import sys
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from eval_metrics import eval_metrics  


def create_data_loaders_for_vgg16(x_train, x_val, x_test, y_train, y_val, y_test, config):
    """
    Convertit les données et les étiquettes en tensors PyTorch et crée des DataLoaders pour l'entraînement,
    la validation et le test.
    """
    # Conversion des labels one-hot en indices de classe
    y_train_indices = np.argmax(y_train, axis=1)
    y_val_indices = np.argmax(y_val, axis=1)
    y_test_indices = np.argmax(y_test, axis=1)

    # Conversion en tensors PyTorch
    y_train_tensor = torch.tensor(y_train_indices, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val_indices, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_indices, dtype=torch.long)

    # Conversion des tableaux numpy en tensors PyTorch et permutation des dimensions
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).permute(0, 3, 1, 2)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2)

    # Création des jeux de données PyTorch
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Création des DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Vérification des tailles des DataLoaders
    print(f"Nombre de batches dans le DataLoader d'entraînement : {len(train_loader)}")
    print(f"Nombre de batches dans le DataLoader de validation : {len(val_loader)}")
    print(f"Nombre de batches dans le DataLoader de test : {len(test_loader)}")

    return train_loader, val_loader, test_loader


def create_vgg16_model(config):
    """
    Crée un modèle VGG16 en fonction des paramètres de configuration donnés.
    """
    model = models.vgg16(pretrained=config['pretrained'])
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, config['num_classes'])
    return model


def train_vgg16_model(model, train_loader, val_loader, config):
    """
    Entraîne un modèle VGG16 en utilisant les données d'entraînement et de validation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 100 * correct / total

        # Calculate metrics on validation set
        model.eval()
        val_labels = []
        val_preds = []
        val_preds_proba = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
                val_preds_proba.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        # Convert lists to numpy arrays
        val_labels = np.array(val_labels)
        val_preds = np.array(val_preds)
        val_preds_proba = np.array(val_preds_proba)

        # Evaluate metrics
        metrics = eval_metrics(val_labels, val_preds, val_preds_proba)

        # Print metrics
        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        print(f"Validation Metrics - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, Log Loss: {metrics['log_loss']:.4f}, "
              f"Mean ROC AUC: {metrics['mean_roc_auc']:.4f}")

    return model


def test_model_vgg16(model, test_loader):
    """
    Évalue un modèle VGG16 sur un ensemble de test.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Passer le modèle en mode évaluation

    all_labels = []
    all_preds = []
    all_preds_proba = []

    with torch.no_grad():  # Pas besoin de calculer les gradients
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Prédictions et probabilités
            _, preds = torch.max(outputs, 1)
            preds_proba = torch.softmax(outputs, dim=1)  # Probabilités prédites

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_preds_proba.extend(preds_proba.cpu().numpy())

    # Convertir les listes en tableaux numpy
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_pred_proba = np.array(all_preds_proba)

    # Calculer les métriques
    metrics = eval_metrics(y_true, y_pred, y_pred_proba)

    return metrics