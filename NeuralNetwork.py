import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=28*28, hidden_sizes=[512, 256], output_size=10, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = None

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

    def train_model(self, train_loader, epochs=10, lr=0.01, loss_function=None, optimizer=None, l1_lambda=1e-5, early_stop_delta=0.001, patience=3, val_interval=5, val_split=0.1):
        """
        Train the model, validating every `val_interval` epochs for early stopping.
        A fraction (val_split) of the training data is split off internally for
        early stopping only. Set val_split=0 to disable early stopping.
        """
        from torch.utils.data import random_split, DataLoader as _DataLoader

        val_loader = None
        if val_split > 0:
            dataset = train_loader.dataset
            n_val = int(val_split * len(dataset))
            n_train = len(dataset) - n_val
            train_sub, val_sub = random_split(dataset, [n_train, n_val])
            batch_size = train_loader.batch_size
            train_loader = _DataLoader(train_sub, batch_size=batch_size, shuffle=True)
            val_loader = _DataLoader(val_sub, batch_size=batch_size)

        criterion = nn.CrossEntropyLoss() if loss_function is None else loss_function
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4) if optimizer is None else optimizer

        best_val_acc = 0.0
        epochs_no_improve = 0

        epoch_bar = tqdm(range(epochs), desc='Training', leave=False, unit='epoch')
        for epoch in epoch_bar:
            # Training loop with tqdm
            self.train()
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, unit="batches", colour='green')
            for X_batch, y_batch in train_bar:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self(X_batch)
                l1_norm = torch.stack([p.abs().sum() for p in self.parameters()]).sum()
                loss = criterion(logits, y_batch) + l1_lambda * l1_norm

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # --- Enforce permanent connection pruning ---
                if hasattr(self, "connection_masks") and self.connection_masks is not None:
                    for layer_idx, mask in self.connection_masks.items():
                        layer = self.layer_stack[layer_idx]
                        layer.weight.data *= mask

            # Validation every val_interval epochs
            if val_loader and (epoch + 1) % val_interval == 0:
                self.eval()
                correct = 0
                total = 0

                val_bar = tqdm(val_loader, desc="Validation", leave=False, unit='batches', colour='green')
                with torch.no_grad():
                    for X_val, y_val in val_bar:
                        X_val = X_val.to(self.device)
                        y_val = y_val.to(self.device)

                        logits = self(X_val)
                        preds = logits.argmax(dim=1)
                        correct += (preds == y_val).sum().item()
                        total += y_val.size(0)
                val_acc = correct / total
                epoch_bar.set_postfix({'val_acc': f'{val_acc:.4f}'})

                if val_acc - best_val_acc > early_stop_delta:
                    best_val_acc = val_acc
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += val_interval

            # Early stopping
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        return val_acc if val_split > 0 else None

    def predict(self, test_loader):
        self.eval()
        all_preds = []

        with torch.no_grad():
            for X_batch, y_batch in tqdm(test_loader, leave=False, colour='green'):
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
            for X_batch, y_batch in tqdm(data_loader, leave=True, colour='green'):
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
            for batch, _ in tqdm(iterator, leave=False, colour='green'):
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
    
    def get_layer_data(self, X, include_output_layer=False):
        """
        Returns combined layer data: weights, biases, pre/post activations.
        
        Args:
            X: tensor or DataLoader for computing activations.
            include_output_layer: if False, exclude the last layer (logits)
            
        Returns:
            dict: { 'layer_0': {weights, bias, pre_activation, post_activation}, ... }
        """
        # 1. Get weights and biases
        layer_params = self.get_layer_params()

        # 2. Get activations
        layer_activations = self.get_activations(X)

        # 3. Merge the two dicts
        layer_data = {}
        layer_names = sorted(layer_params.keys())  # ensures proper order
        if not include_output_layer:
            layer_names = layer_names[:-1]  # exclude last layer

        for layer_name in layer_names:
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

    def prune_hidden_neurons(self, importance_scores, prune_rate=0.05):
        """
        Prune hidden neurons based on importance scores.

        Returns:
            dict: {layer_idx: n_pruned}
        """
        linear_indices = [i for i, l in enumerate(self.layer_stack) if isinstance(l, nn.Linear)]
        hidden_linear_indices = linear_indices[:-1]  # exclude output layer

        prune_counts = {}

        for idx, layer_idx in enumerate(hidden_linear_indices):
            layer = self.layer_stack[layer_idx]
            scores = importance_scores[f'layer_{idx}']

            n_neurons = scores.numel()
            n_remove = int(prune_rate * n_neurons)
            if n_remove == 0:
                continue

            keep_idx = scores.argsort(descending=True)[:-n_remove]

            # Prune current layer
            layer.weight.data = layer.weight.data[keep_idx, :]
            layer.bias.data = layer.bias.data[keep_idx]

            # Adjust next layer input weights
            next_layer = self.layer_stack[linear_indices[idx + 1]]
            next_layer.weight.data = next_layer.weight.data[:, keep_idx]

            # Record how many neurons were pruned
            prune_counts[layer_idx] = n_remove

            if hasattr(self, 'connection_masks') and layer_idx in self.connection_masks:
                self.connection_masks[layer_idx] = self.connection_masks[layer_idx][keep_idx, :]

            # After pruning next layer's input weights
            next_layer_idx = linear_indices[idx + 1]
            if hasattr(self, 'connection_masks') and next_layer_idx in self.connection_masks:
                self.connection_masks[next_layer_idx] = self.connection_masks[next_layer_idx][:, keep_idx]
        
        if hasattr(self, "connection_masks") and self.connection_masks is not None:
            for layer_idx, mask in self.connection_masks.items() :
                layer = self.layer_stack[layer_idx]
                assert mask.shape == layer.weight.shape, \
                    f"Mask mismatch at layer {layer_idx}: mask {mask.shape} vs weight {layer.weight.shape}"

        return prune_counts

    
    def regrow_hidden_neurons(self, regrow_counts, regrow_std=0.01):
        """
        Regrow neurons in hidden layers.

        Args:
            regrow_counts: dict {layer_idx: n_regrow}
            regrow_std: std for initializing new weights/bias
        """
        linear_indices = [i for i, l in enumerate(self.layer_stack)
                        if isinstance(l, nn.Linear)]
        hidden_linear_indices = linear_indices[:-1] if len(linear_indices) > 1 else []

        for idx, layer_idx in enumerate(hidden_linear_indices):

            if layer_idx not in regrow_counts:
                continue

            n_regrow = regrow_counts[layer_idx]
            if n_regrow <= 0:
                continue

            layer = self.layer_stack[layer_idx]
            next_layer = self.layer_stack[linear_indices[idx + 1]]

            in_features = layer.weight.size(1)
            out_features_next = next_layer.weight.size(0)

            device = layer.weight.device

            # --- Add new neurons to current layer ---
            new_weights = torch.randn(n_regrow, in_features, device=device) * regrow_std
            new_bias = torch.randn(n_regrow, device=device) * regrow_std

            layer.weight.data = torch.cat([layer.weight.data, new_weights], dim=0)
            layer.bias.data = torch.cat([layer.bias.data, new_bias], dim=0)

            # --- Add corresponding input columns in next layer ---
            new_input_weights = torch.randn(out_features_next, n_regrow, device=device) * regrow_std
            next_layer.weight.data = torch.cat([next_layer.weight.data, new_input_weights], dim=1)

        if not hasattr(self, "connection_masks"):
            return
        for layer_idx, mask in self.connection_masks.items():
            layer = self.layer_stack[layer_idx]

            n_new_rows = layer.weight.shape[0] - mask.shape[0]
            n_new_cols = layer.weight.shape[1] - mask.shape[1]

            if n_new_rows > 0:
                mask = torch.cat([mask, torch.ones(n_new_rows, mask.shape[1], device=mask.device)], dim=0)
            
            if n_new_cols > 0:
                mask = torch.cat([mask, torch.ones(mask.shape[0], n_new_cols, device=mask.device)], dim=1)

            self.connection_masks[layer_idx] = mask

        if hasattr(self, "connection_masks") and self.connection_masks is not None:
            for layer_idx, mask in self.connection_masks.items():
                layer = self.layer_stack[layer_idx]
                assert mask.shape == layer.weight.shape, \
                    f"Mask mismatch at layer {layer_idx}: mask {mask.shape} vs weight {layer.weight.shape}"

    def prune_connections(self, prune_frac=0.05):
        linear_indices = [i for i, l in enumerate(self.layer_stack) if isinstance(l, nn.Linear)]
        hidden_linear_indices = linear_indices[:-1]

        if not hasattr(self, "connection_masks"):
            self.connection_masks = {}

        for layer_idx in hidden_linear_indices:
            layer = self.layer_stack[layer_idx]
            W = layer.weight.data

            # 1. Compute new magnitude mask on active weights only
            W_flat = W.abs().view(-1)
            W_active = W_flat[W_flat != 0]
            if W_active.numel() == 0:
                continue
            n_prune = max(1, int(prune_frac * W_active.numel()))
            threshold, _ = torch.kthvalue(W_active, n_prune)
            new_mask = (W.abs() >= threshold).float()

            # 2. Get old mask or initialize
            old_mask = self.connection_masks.get(layer_idx, torch.ones_like(W, device=W.device))

            # 3. Extend old mask to current W shape
            if old_mask.shape != W.shape:
                if old_mask.shape[0] < W.shape[0]:
                    n_new_rows = W.shape[0] - old_mask.shape[0]
                    old_mask = torch.cat([old_mask, torch.ones(n_new_rows, old_mask.shape[1], device=W.device)], dim=0)
                if old_mask.shape[1] < W.shape[1]:
                    n_new_cols = W.shape[1] - old_mask.shape[1]
                    old_mask = torch.cat([old_mask, torch.ones(old_mask.shape[0], n_new_cols, device=W.device)], dim=1)

            # 4. Combine masks
            combined_mask = old_mask * new_mask
            self.connection_masks[layer_idx] = combined_mask

            # 5. Apply mask
            layer.weight.data *= combined_mask

        if hasattr(self, "connection_masks") and self.connection_masks is not None:
            for layer_idx, mask in self.connection_masks.items():
                layer = self.layer_stack[layer_idx]
                assert mask.shape == layer.weight.shape, \
                    f"Mask mismatch at layer {layer_idx}: mask {mask.shape} vs weight {layer.weight.shape}"