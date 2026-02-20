import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import pandas as pd

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=28*28, hidden_sizes=[512, 256], output_size=10, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.flatten = nn.Flatten()
        
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        
        self.layer_stack = nn.Sequential(*layers)

        self.to(self.device)

    def forward(self, X):
        X = self.flatten(X)
        logits = self.layer_stack(X)
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
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

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
                            X_val = X_val.to(self.device)
                            y_val = y_val.to(self.device)

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
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

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
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self(X_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        return correct/total
    
    def get_layer_data(self, X):
        """
        Returns:
            {
                layer_0: {
                    'weights': tensor,
                    'bias': tensor,
                    'pre_activation': tensor,
                    'post_activation': tensor
                },
                ...
            }

        X can be a single tensor or a DataLoader.
        """
        self.eval()
        layer_data = {}

        is_dataloader = isinstance(X, torch.utils.data.DataLoader)

        # Identify logical layers (Linear followed by optional ReLU)
        linear_indices = []
        for idx, layer in enumerate(self.layer_stack):
            if isinstance(layer, torch.nn.Linear):
                linear_indices.append(idx)

        # Initialize dictionary
        for layer_num, idx in enumerate(linear_indices):
            linear_layer = self.layer_stack[idx]
            layer_data[f"layer_{layer_num}"] = {
                "weights": linear_layer.weight.detach().cpu().clone(),
                "bias": linear_layer.bias.detach().cpu().clone(),
                "pre_activation": [],
                "post_activation": []
            }

        with torch.no_grad():

            if is_dataloader:
                iterator = X
            else:
                iterator = [(X, None)]

            for batch, _ in iterator:
                batch = batch.to(self.device)

                out = self.flatten(batch)
                logical_layer = 0

                i = 0
                while i < len(self.layer_stack):

                    layer = self.layer_stack[i]

                    if isinstance(layer, torch.nn.Linear):

                        # Linear forward
                        out = layer(out)
                        pre_act = out.detach().cpu().clone()

                        # Check if next layer is ReLU
                        if i + 1 < len(self.layer_stack) and \
                        isinstance(self.layer_stack[i + 1], torch.nn.ReLU):

                            out = self.layer_stack[i + 1](out)
                            post_act = out.detach().cpu().clone()
                            i += 1  # skip ReLU
                        else:
                            post_act = out.detach().cpu().clone()

                        layer_name = f"layer_{logical_layer}"
                        layer_data[layer_name]["pre_activation"].append(pre_act)
                        layer_data[layer_name]["post_activation"].append(post_act)

                        logical_layer += 1

                    else:
                        out = layer(out)

                    i += 1

        # Concatenate if DataLoader
        for layer_name in layer_data:
            if is_dataloader:
                layer_data[layer_name]["pre_activation"] = torch.cat(
                    layer_data[layer_name]["pre_activation"], dim=0
                )
                layer_data[layer_name]["post_activation"] = torch.cat(
                    layer_data[layer_name]["post_activation"], dim=0
                )
            else:
                layer_data[layer_name]["pre_activation"] = \
                    layer_data[layer_name]["pre_activation"][0]
                layer_data[layer_name]["post_activation"] = \
                    layer_data[layer_name]["post_activation"][0]

        return layer_data