# %%
import importlib
import NeuralNetwork
import funcs
import plots

importlib.reload(NeuralNetwork)
importlib.reload(funcs)
importlib.reload(plots)
from NeuralNetwork import NeuralNetwork

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm
import copy
import os

# %%
import importlib
import setup
importlib.reload(setup)
from setup import HIDDEN_LAYERS, BATCH_SIZE

device = setup.get_device()
N_TRAIN_EPOCHS = 15 if device.type == "cuda" else 8
train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = setup.get_dataloaders()

# Pruning parameters
MAX_PRUNE_ROUNDS = 30
MAX_ALLOWED_ACC_DROP = 0.02
N_RETRAIN_EPOCHS = 3

# %%
# Create model and train
model = NeuralNetwork(hidden_sizes=HIDDEN_LAYERS, device=device)
baseline_acc = model.train_model(train_loader=train_loader, epochs=N_TRAIN_EPOCHS)

# %% [markdown]
# ## Prune Neurons and Retrain

# %% [markdown]
# ## Hyperparameter Search

# %%
import pandas as pd

original_params = sum(p.numel() for p in model.parameters())
search_results = []

for prune_frac in [0.05, 0.10, 0.15, 0.20]:
    params = (MAX_PRUNE_ROUNDS, prune_frac, prune_frac * 2, prune_frac * 0.5, N_RETRAIN_EPOCHS, MAX_ALLOWED_ACC_DROP)
    candidate = funcs.pruning(copy.deepcopy(model), train_loader, params, baseline_acc, use_max_rounds=True, mode='full')
    val_acc = candidate.accuracy(val_loader)
    n_params = sum(p.numel() for p in candidate.parameters())
    search_results.append({
        'prune_frac': prune_frac,
        'val_acc': round(val_acc, 4),
        'n_params': n_params,
        'compression': round(original_params / n_params, 2)
    })
    print(search_results[-1])

best_prune_frac = max(search_results, key=lambda r: r['val_acc'])['prune_frac']
print(f"\nBest prune_frac: {best_prune_frac}")
print(pd.DataFrame(search_results).to_string())

# %%
prune_parameters = (MAX_PRUNE_ROUNDS, best_prune_frac, best_prune_frac * 2, best_prune_frac * 0.5, N_RETRAIN_EPOCHS, MAX_ALLOWED_ACC_DROP)
use_max_rounds = False if device.type == "cuda" else True

final_model = funcs.pruning(model, train_loader, prune_parameters, baseline_acc, use_max_rounds=use_max_rounds, mode='full')

# %%
print(f"Test accuracy after pruning: {final_model.accuracy(val_loader):.2f}")

# %%
torch.save(final_model, "pruned_model.pth")


