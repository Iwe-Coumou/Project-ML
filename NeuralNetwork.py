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

    def train_model(self, train_loader, val_loader=None, epochs=10, lr=0.01, loss_function=None, optimizer=None, l1_lambda=1e-5):
        criterion = nn.CrossEntropyLoss() if loss_function is None else loss_function
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4) if optimizer is None else optimizer

        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader) if val_loader else 0
        total_steps = n_train_batches + n_val_batches

        metrics = []
        for epoch in range(epochs):
            running_loss = 0.0
            train_correct = 0
            train_total = 0

            with tqdm(total=total_steps, desc=f"Epoch {epoch+1}/{epochs}", leave=False) as epoch_bar:

                # --Training--
                self.train()
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    logits = self(X_batch)
                    l1_norm = sum(p.abs().sum() for p in self.parameters())
                    loss = criterion(logits, y_batch) + l1_lambda * l1_norm

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * X_batch.size(0)
                    
                    preds = logits.argmax(dim=1)
                    train_correct += (preds == y_batch).sum().item()
                    train_total += y_batch.size(0)

                    epoch_bar.update(1)

                epoch_train_loss = running_loss / len(train_loader.dataset)
                train_acc = train_correct/train_total

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
                'train_acc': train_acc,
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

    def get_layer_params(self):
        """
        Returns a dict with weights and biases for each linear layer.
        Example:
        { 'layer_0': {'weights': tensor, 'bias': tensor}, ... }
        """
        params = {}
        linear_indices = [i for i, l in enumerate(self.layer_stack) if isinstance(l, nn.Linear)]
        
        for layer_num, idx in enumerate(linear_indices):
            layer = self.layer_stack[idx]
            params[f'layer_{layer_num}'] = {
                'weights': layer.weight.detach().cpu().clone(),
                'bias': layer.bias.detach().cpu().clone()
            }
        return params

    def get_activations(self, X, layers_to_record=None):
        """
        Returns pre/post activations for each hidden layer.
        layers_to_record: list of layer numbers to record (optional)
        X: Tensor or DataLoader
        """
        self.eval()
        activations = {}
        is_dataloader = isinstance(X, DataLoader)
        iterator = X if is_dataloader else [(X, None)]

        linear_indices = [i for i, l in enumerate(self.layer_stack) if isinstance(l, nn.Linear)]
        for layer_num in range(len(linear_indices)):
            if layers_to_record is None or layer_num in layers_to_record:
                activations[f'layer_{layer_num}'] = {'pre_activation': [], 'post_activation': []}

        with torch.no_grad():
            for batch, _ in tqdm(iterator):
                batch = batch.to(self.device)
                out = self.flatten(batch)
                logical_layer = 0
                i = 0
                while i < len(self.layer_stack):
                    layer = self.layer_stack[i]
                    if isinstance(layer, nn.Linear):
                        out = layer(out)
                        pre_act = out.detach().cpu().clone()
                        if layers_to_record is None or logical_layer in layers_to_record:
                            post_act = out.detach().cpu().clone()
                            # check for ReLU
                            if i + 1 < len(self.layer_stack) and isinstance(self.layer_stack[i + 1], nn.ReLU):
                                out = self.layer_stack[i + 1](out)
                                post_act = out.detach().cpu().clone()
                                i += 1
                            activations[f'layer_{logical_layer}']['pre_activation'].append(pre_act)
                            activations[f'layer_{logical_layer}']['post_activation'].append(post_act)
                        logical_layer += 1
                    else:
                        out = layer(out)
                    i += 1

        # Concatenate results
        for layer_name in activations:
            activations[layer_name]['pre_activation'] = torch.cat(activations[layer_name]['pre_activation'], dim=0)
            activations[layer_name]['post_activation'] = torch.cat(activations[layer_name]['post_activation'], dim=0)

        return activations
    
    def get_layer_data(self, X):
        """
        Returns combined layer data: weights, biases, pre/post activations.
        
        Args:
            X: tensor or DataLoader for computing activations.
            
        Returns:
            dict: { 'layer_0': {weights, bias, pre_activation, post_activation}, ... }
        """
        # 1. Get weights and biases
        layer_params = self.get_layer_params()

        # 2. Get activations
        layer_activations = self.get_activations(X)

        # 3. Merge the two dicts
        layer_data = {}
        for layer_name in layer_params:
            layer_data[layer_name] = {**layer_params[layer_name], **layer_activations[layer_name]}

        return layer_data
    
    def compute_neuron_importance(self, layer_data=None, X=None, alpha=0.7, type='combined'):
        """
        Computes importance scores for each neuron in hidden layers:
        I(v) = alpha * activation_variance + (1 - alpha) * sum_abs_weights

        Args:
            layer_data: Precomputed dict from get_layer_data (optional).
            X: Input tensor or DataLoader, used only if layer_data is None.
            alpha: Weight for activation variance in combined score.
            type: 'combined', 'var', or 'weight'.

        Returns:
            dict: { 'layer_0': tensor of scores, ... }
        """
        if layer_data is None:
            if X is None:
                raise ValueError("Provide either layer_data or X for activations")
            layer_data = self.get_layer_data(X)

        importance_scores = {}

        for layer_name, data in layer_data.items():
            weights = data['weights']       # [out_features, in_features]
            bias = data['bias']             # [out_features]
            post_act = data['post_activation']  # [N_samples, out_features]

            # Activation-based importance
            I_act = post_act.var(dim=0)
            if type == 'var':
                importance_scores[layer_name] = I_act
                continue

            # Weight-based importance
            I_weight = weights.abs().sum(dim=1) + bias.abs()
            if type == 'weight':
                importance_scores[layer_name] = I_weight
                continue

            # Combined importance
            I = alpha * I_act + (1 - alpha) * I_weight
            importance_scores[layer_name] = I

        return importance_scores


    def prune_hidden_neurons(self, layer_data=None, X=None, prune_rate=0.2, alpha=0.7, importance_scores=None, regrow_frac=0.0, regrow_std=0.01):
        """
        Prunes hidden layers based on neuron importance, with optional random regrowth.

        Args:
            layer_data: Precomputed layer data dict (optional)
            X: Input tensor or DataLoader (used if layer_data is None)
            prune_rate: Fraction of neurons to remove
            alpha: Weight for activation variance in combined score
            importance_scores: Precomputed importance scores (optional)
            regrow_frac: Fraction of pruned neurons to randomly add back
            regrow_std: Std deviation for initializing regrown neurons
        """

        if importance_scores is None:
            if X is None:
                raise ValueError("Must provide X or precomputed importance_scores")
            importance_scores = self.compute_neuron_importance(layer_data=layer_data, X=X, alpha=alpha)

        # Indices of all linear layers
        linear_indices = [i for i, l in enumerate(self.layer_stack) if isinstance(l, nn.Linear)]
        num_layers = len(linear_indices)

        # Only hidden layers (exclude last output layer)
        hidden_linear_indices = linear_indices[0:-1] if num_layers > 1 else []

        for idx, layer_idx in enumerate(hidden_linear_indices):
            layer = self.layer_stack[layer_idx]
            scores = importance_scores[f'layer_{idx}']

            n_remove = int(prune_rate * scores.numel())
            if n_remove == 0:
                continue

            # --- Determine which neurons to keep ---
            keep_idx = scores.argsort(descending=True)[:-n_remove]

            # --- Prune current layer ---
            layer.weight.data = layer.weight.data[keep_idx, :]
            layer.bias.data = layer.bias.data[keep_idx]

            # --- Update next layer weights for kept neurons only ---
            next_layer_idx = linear_indices[idx + 1]
            next_layer = self.layer_stack[next_layer_idx]
            next_layer.weight.data = next_layer.weight.data[:, keep_idx]

            # --- Optional random regrowth ---
            if regrow_frac > 0:
                n_regrow = max(1, int(n_remove * regrow_frac))
                in_features = layer.weight.size(1)  # after pruning
                out_features_next = next_layer.weight.size(0)

                # New neurons for current layer
                new_weights = torch.randn(n_regrow, in_features, device=layer.weight.device) * regrow_std
                new_bias = torch.randn(n_regrow, device=layer.bias.device) * regrow_std
                layer.weight.data = torch.cat([layer.weight.data, new_weights], dim=0)
                layer.bias.data = torch.cat([layer.bias.data, new_bias], dim=0)

                # New corresponding columns for next layer
                new_input_weights = torch.randn(out_features_next, n_regrow, device=next_layer.weight.device) * regrow_std
                next_layer.weight.data = torch.cat([next_layer.weight.data, new_input_weights], dim=1)