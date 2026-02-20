import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import pandas as pd

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=28*28, hidden_sizes=[512, 256], output_size=10):
        super().__init__()
        self.flatten = nn.Flatten()
        
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, X):
        X = self.flatten(X)
        logits = self.linear_relu_stack(X)
        return logits

    def train_model(self, train_loader, val_loader=None, epochs=10, lr=0.01, loss_function=None, optimizer=None):
        criterion = nn.CrossEntropyLoss() if loss_function is None else loss_function
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4) if optimizer is None else optimizer

        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader) if val_loader else 0
        total_steps = n_train_batches + n_val_batches

        metrics = []
        for epoch in range(epochs):
            # --Training--
            self.train()
            running_loss = 0.0

            with tqdm(total=total_steps, desc=f"Epoch {epoch+1}/{epochs}") as epoch_bar:

                for X_batch, y_batch in train_loader:
                    logits = self(X_batch)
                    loss = criterion(logits, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * X_batch.size(0)
                    epoch_bar.update(1)

                epoch_train_loss = running_loss / len(train_loader.dataset)

                # --Validation--
                if val_loader:
                    self.eval()
                    correct = 0
                    total = 0
                    epoch_val_loss = 0.0

                    with torch.no_grad():
                        for X_val, y_val in val_loader:
                            logits = self(X_val)
                            loss = criterion(logits, y_val)
                            epoch_val_loss += loss.item() * X_val.size(0)
                            preds = logits.argmax(dim=1)
                            correct += (preds == y_val).sum().item()
                            total += y_val.size(0)
                            epoch_bar.update(1)

                    epoch_val_loss /= len(val_loader.dataset)
                    val_acc = correct / total

            metrics.append({
                'epoch': epoch+1,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss if val_loader else None,
                'val_acc': val_acc if val_loader else None
            })

        metrics = pd.DataFrame(metrics)
        metrics.set_index('epoch', inplace=True)
        return metrics

    def predict(self, test_loader):
        self.eval()
        all_preds = []

        with torch.no_grad():
            for X_batch, y_batch in tqdm(test_loader):
                logits = self(X_batch)
                preds = logits.argmax(dim=1)
                all_preds.append(preds)
        return torch.cat(all_preds)
    
    def accuracy(self, data_loader):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in tqdm(data_loader):
                logits = self(X_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        return correct/total

    def get_activations(self, X):
        """
        Returns activations for each layer.
        X: either a single input tensor (1 batch/sample) or a DataLoader
        Returns: dict of layer_name -> tensor (single input) or concatenated tensor (DataLoader)
        """
        self.eval()
        activations = {}

        # Prepare layer names
        layer_names = ['flatten'] + [f'layer_{i}_{layer.__class__.__name__}' 
                                    for i, layer in enumerate(self.linear_relu_stack)]

        # Case 1: DataLoader
        if isinstance(X, torch.utils.data.DataLoader):
            # Initialize lists for each layer
            for name in layer_names:
                activations[name] = []

            with torch.no_grad():
                for batch, _ in tqdm(X, desc="Collecting activations", leave=True):
                    out = self.flatten(batch)
                    activations['flatten'].append(out)

                    for idx, layer in enumerate(self.linear_relu_stack):
                        out = layer(out)
                        activations[f'layer_{idx}_{layer.__class__.__name__}'].append(out)

            # Concatenate all batches
            for name in activations:
                activations[name] = torch.cat(activations[name], dim=0)

        # Case 2: Single tensor
        else:
            with torch.no_grad():
                out = self.flatten(X)
                activations['flatten'] = out
                for idx, layer in enumerate(self.linear_relu_stack):
                    out = layer(out)
                    activations[f'layer_{idx}_{layer.__class__.__name__}'] = out

        return activations
