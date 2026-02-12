import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

        for epoch in range(epochs):
            # --Training--
            self.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                logits = self(X_batch)
                loss = criterion(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * X_batch.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            # --Validation--
            if val_loader:
                self.eval()
                correct = 0
                total = 0
                val_loss = 0.0
                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        logits = self(X_val)
                        loss = criterion(logits, y_val)
                        val_loss += loss.item() * X_val.size(0)
                        preds = logits.argmax(dim=1)
                        correct += (preds == y_val).sum().item()
                        total += y_val.size(0)
                val_loss /= len(val_loader.dataset)
                val_acc = correct / total
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")

    def predict(self, test_loader):
        self.eval()
        all_preds = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                logits = self(X_batch)
                preds = logits.argmax(dim=1)
                all_preds.append(preds)
        return torch.cat(all_preds)
    
    def accuracy(self, data_loader):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                logits = self(X_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        return correct/total


