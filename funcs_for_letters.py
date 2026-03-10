"""
funcs_for_letters.py
--------------------
Factory functions for building and comparing three EMNIST-letters models:

  1. Transfer model  — pruned digit network with layer-0 reset, hidden layers
                       frozen (optionally unfreeze specific clusters).
  2. Reset model     — same sparse architecture but all weights re-initialised.
  3. FC benchmark    — fully-connected MLP with the same neuron counts per layer.

Also provides data loading (EMNIST letters) and a comparison helper.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from NeuralNetwork import NeuralNetwork


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_letters_dataloaders(batch_size=8000, val_frac=0.1):
    """
    Load EMNIST 'letters' split (26 classes, labels 1-26 → remapped to 0-25).

    Applies the same rotate-then-flip transform used for the digit dataset so
    the pixel orientation is consistent.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    def rotate_image(x):
        return torch.rot90(x, k=-1, dims=[1, 2])

    def flip_image(x):
        return torch.flip(x, dims=[2])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(rotate_image),
        transforms.Lambda(flip_image),
    ])

    # EMNIST letters labels are 1-26; shift to 0-25 for CrossEntropyLoss
    target_transform = transforms.Lambda(lambda y: y - 1)

    train_full = datasets.EMNIST(
        root="./data", split="letters", train=True,
        download=True, transform=transform, target_transform=target_transform)

    test_dataset = datasets.EMNIST(
        root="./data", split="letters", train=False,
        download=True, transform=transform, target_transform=target_transform)

    n_val   = int(val_frac * len(train_full))
    n_train = len(train_full) - n_val
    train_dataset, val_dataset = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

    print(f"Letters — train: {n_train}, val: {n_val}, test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _linear_indices(model):
    """Return indices of all nn.Linear layers inside model.layer_stack."""
    return [i for i, l in enumerate(model.layer_stack) if isinstance(l, nn.Linear)]


def _hidden_sizes(model):
    """
    Return the output sizes of every hidden linear layer (all but the last).
    This reflects the *current* (potentially pruned) architecture.
    """
    lin_idx = _linear_indices(model)
    return [model.layer_stack[i].out_features for i in lin_idx[:-1]]


def _copy_connection_masks(src, dst, lin_src, lin_dst):
    """
    Copy connection_masks from src to dst for corresponding hidden layers,
    moving tensors to dst's device.
    """
    if not hasattr(src, 'connection_masks') or src.connection_masks is None:
        return
    dst.connection_masks = {}
    hidden_src = lin_src[:-1]
    hidden_dst = lin_dst[:-1]
    for s_idx, d_idx in zip(hidden_src, hidden_dst):
        if s_idx in src.connection_masks:
            dst.connection_masks[d_idx] = src.connection_masks[s_idx].to(dst.device)


def _zero_cluster_downstream(model, ignore_clusters, layer_mapping, lin_idx):
    """
    Silence neurons in `ignore_clusters` by zeroing the columns that correspond
    to those neurons in the *next* linear layer.  Also zeros the neuron rows
    themselves and their biases.  Returns a set of (layer_stack_idx, local_row)
    pairs that were zeroed so callers can freeze them.
    """
    if not ignore_clusters:
        return

    ignore_global = set()
    for cid in ignore_clusters:
        if cid in (model.final_cluster_map or {}):
            ignore_global.update(model.final_cluster_map[cid])

    if not ignore_global:
        return

    for lname, start, end in layer_mapping:
        k = int(lname.split('_')[1])          # logical layer index (0-based hidden)
        ls_idx  = lin_idx[k]                  # position in layer_stack
        ls_next = lin_idx[k + 1]
        layer      = model.layer_stack[ls_idx]
        next_layer = model.layer_stack[ls_next]

        local_rows = [gi - start for gi in ignore_global if start <= gi < end]
        if not local_rows:
            continue

        rows = torch.tensor(local_rows, dtype=torch.long, device=model.device)
        # Zero this neuron's outgoing row and bias
        layer.weight.data[rows] = 0.0
        layer.bias.data[rows]   = 0.0
        # Zero the downstream columns so nothing listens to these neurons
        next_layer.weight.data[:, rows] = 0.0

        # Extend (or create) connection masks to keep these permanently zero
        if not hasattr(model, 'connection_masks') or model.connection_masks is None:
            model.connection_masks = {}
        if ls_idx not in model.connection_masks:
            model.connection_masks[ls_idx] = (layer.weight.data != 0).float()
        else:
            model.connection_masks[ls_idx][rows] = 0.0

        if ls_next not in model.connection_masks:
            model.connection_masks[ls_next] = (next_layer.weight.data != 0).float()
        else:
            model.connection_masks[ls_next][:, rows] = 0.0


def _make_row_freeze_hook(freeze_rows_tensor):
    """Gradient hook: zero out gradients for frozen output-neuron rows."""
    def hook(grad):
        g = grad.clone()
        g[freeze_rows_tensor] = 0.0
        return g
    return hook


def _make_scalar_freeze_hook(freeze_rows_tensor):
    """Gradient hook: zero out gradient entries for frozen bias elements."""
    def hook(grad):
        g = grad.clone()
        g[freeze_rows_tensor] = 0.0
        return g
    return hook


def _apply_cluster_freeze_hooks(model, cluster_map, layer_mapping, unfreeze_clusters, lin_idx):
    """
    For each hidden layer, register backward hooks that zero gradients of all
    neurons NOT in `unfreeze_clusters`.  This gives per-neuron training control
    without changing requires_grad globally (which would break the computation
    graph through frozen layers).
    """
    if not unfreeze_clusters or not cluster_map:
        return

    unfreeze_global = set()
    for cid in unfreeze_clusters:
        if cid in cluster_map:
            unfreeze_global.update(cluster_map[cid])

    hidden_lin = lin_idx[:-1]  # exclude output layer
    for k, ls_idx in enumerate(hidden_lin):
        lname = f'layer_{k}'
        layer_range = next(((s, e) for n, s, e in layer_mapping if n == lname), None)
        if layer_range is None:
            continue
        start, end = layer_range
        n_neurons   = end - start
        freeze_rows = [i for i in range(n_neurons) if (start + i) not in unfreeze_global]
        if not freeze_rows:
            continue  # all neurons in this layer are unfrozen

        layer = model.layer_stack[ls_idx]
        rows  = torch.tensor(freeze_rows, dtype=torch.long, device=model.device)
        layer.weight.register_hook(_make_row_freeze_hook(rows))
        layer.bias.register_hook(_make_scalar_freeze_hook(rows))


# ---------------------------------------------------------------------------
# Model 1 — Transfer model
# ---------------------------------------------------------------------------

def build_transfer_model(pruned_model, n_letters=26,
                          unfreeze_clusters=None, ignore_clusters=None):
    """
    Build a transfer-learning model from the pruned digit classifier.

    Architecture:
      - Layer 0 (784 → h0): weights RESET (Xavier uniform), always trainable.
      - Layers 1..n-2 (hidden): weights COPIED from pruned_model, FROZEN.
        If `unfreeze_clusters` is given, only those cluster neurons are allowed
        to train (gradient hooks zero out the rest).
      - Output layer (h_last → n_letters): freshly initialised, always trainable.

    Sparse connection structure (connection_masks) from the pruned model is
    preserved for the hidden layers.

    Args:
        pruned_model:      trained NeuralNetwork after pruning (has .final_cluster_map)
        n_letters:         number of letter classes (26 for EMNIST letters)
        unfreeze_clusters: list of cluster IDs whose neurons may train; None = all frozen
        ignore_clusters:   list of cluster IDs to silence (zero + freeze)

    Returns:
        NeuralNetwork configured for letters, ready to call .train_model()
    """
    device = pruned_model.device
    h_sizes = _hidden_sizes(pruned_model)
    model   = NeuralNetwork(input_size=784, hidden_sizes=h_sizes,
                             output_size=n_letters, device=device)

    lin_src = _linear_indices(pruned_model)
    lin_dst = _linear_indices(model)

    # --- Copy hidden layer weights (indices 1..n-2 in logical order) ---
    for k in range(1, len(lin_src) - 1):        # skip layer-0 and output
        src_layer = pruned_model.layer_stack[lin_src[k]]
        dst_layer = model.layer_stack[lin_dst[k]]
        dst_layer.weight.data.copy_(src_layer.weight.data)
        dst_layer.bias.data.copy_(src_layer.bias.data)

    # --- Reset layer 0 (Xavier + zero bias) ---
    layer0 = model.layer_stack[lin_dst[0]]
    init.xavier_uniform_(layer0.weight)
    init.zeros_(layer0.bias)

    # --- New output layer is already randomly initialised by NeuralNetwork.__init__ ---

    # --- Copy connection masks for hidden layers ---
    _copy_connection_masks(pruned_model, model, lin_src, lin_dst)

    # --- Freeze hidden layers ---
    hidden_lin_dst = lin_dst[1:-1]
    if unfreeze_clusters:
        # Per-neuron control via gradient hooks; keep requires_grad=True
        cluster_map   = getattr(pruned_model, 'final_cluster_map', None) or {}
        layer_mapping = getattr(pruned_model, 'final_layer_mapping', None) or []
        _apply_cluster_freeze_hooks(model, cluster_map, layer_mapping,
                                    unfreeze_clusters, lin_dst)
    else:
        # Freeze everything in hidden layers
        for ls_idx in hidden_lin_dst:
            model.layer_stack[ls_idx].weight.requires_grad_(False)
            model.layer_stack[ls_idx].bias.requires_grad_(False)

    # --- Silence ignore_clusters ---
    if ignore_clusters:
        cluster_map   = getattr(pruned_model, 'final_cluster_map', None) or {}
        layer_mapping = getattr(pruned_model, 'final_layer_mapping', None) or []
        # Temporarily attach cluster map so helper can look it up
        model.final_cluster_map   = cluster_map
        model.final_layer_mapping = layer_mapping
        _zero_cluster_downstream(model, ignore_clusters, layer_mapping, lin_dst)

    print(f"Transfer model built: {_arch_str(model)}")
    print(f"  Trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model


# ---------------------------------------------------------------------------
# Model 2 — Reset model (same sparse arch, fresh weights)
# ---------------------------------------------------------------------------

def build_reset_model(pruned_model, n_letters=26, ignore_clusters=None):
    """
    Same pruned architecture as the transfer model but with ALL weights
    re-initialised randomly.  No layers are frozen.

    This is the "same topology, fresh start" baseline: does sparsity alone
    (without pre-trained weights) help for letters?

    Args:
        pruned_model:    NeuralNetwork after pruning (architecture source only)
        n_letters:       number of letter classes
        ignore_clusters: list of cluster IDs to silence (zero + freeze)

    Returns:
        NeuralNetwork configured for letters
    """
    device  = pruned_model.device
    h_sizes = _hidden_sizes(pruned_model)
    model   = NeuralNetwork(input_size=784, hidden_sizes=h_sizes,
                             output_size=n_letters, device=device)

    lin_src = _linear_indices(pruned_model)
    lin_dst = _linear_indices(model)

    # Copy connection masks so the same sparsity pattern is enforced during training
    _copy_connection_masks(pruned_model, model, lin_src, lin_dst)

    # Apply the masks immediately: zero weights that the mask disallows
    if hasattr(model, 'connection_masks') and model.connection_masks:
        for ls_idx, mask in model.connection_masks.items():
            model.layer_stack[ls_idx].weight.data *= mask

    if ignore_clusters:
        cluster_map   = getattr(pruned_model, 'final_cluster_map', None) or {}
        layer_mapping = getattr(pruned_model, 'final_layer_mapping', None) or []
        model.final_cluster_map   = cluster_map
        model.final_layer_mapping = layer_mapping
        _zero_cluster_downstream(model, ignore_clusters, layer_mapping, lin_dst)

    print(f"Reset model built: {_arch_str(model)}")
    return model


# ---------------------------------------------------------------------------
# Model 3 — Fully-connected benchmark
# ---------------------------------------------------------------------------

def build_fc_benchmark(pruned_model, n_letters=26):
    """
    Fully-connected MLP with the same hidden layer widths as the pruned model
    but NO connection masks.  All weights randomly initialised.

    This answers: "How well does a plain MLP of the same depth/width do?"

    Args:
        pruned_model:  NeuralNetwork after pruning (architecture reference only)
        n_letters:     number of letter classes

    Returns:
        NeuralNetwork (fully connected, no masks)
    """
    device  = pruned_model.device
    h_sizes = _hidden_sizes(pruned_model)
    model   = NeuralNetwork(input_size=784, hidden_sizes=h_sizes,
                             output_size=n_letters, device=device)
    # No connection masks → fully connected
    print(f"FC benchmark built: {_arch_str(model)}")
    return model


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_models(models_dict, test_loader):
    """
    Evaluate all models on test_loader and print a comparison table.

    Args:
        models_dict: {name: NeuralNetwork}, e.g. {'transfer': m1, 'reset': m2, 'fc': m3}
        test_loader: DataLoader for EMNIST letters test set

    Returns:
        dict: {name: {'test_acc': float, 'n_params': int, 'n_connections': int}}
    """
    results = {}
    header  = f"{'Model':<20} {'Test Acc':>9} {'Params':>10} {'Connections':>13}"
    print("\n" + header)
    print("-" * len(header))

    for name, model in models_dict.items():
        acc   = model.accuracy(test_loader)
        n_par = sum(p.numel() for p in model.parameters())
        # Count non-zero weights (connections)
        lin_idx  = _linear_indices(model)
        n_conn = sum(
            int((model.layer_stack[i].weight.data != 0).sum().item())
            for i in lin_idx
        )
        results[name] = {'test_acc': acc, 'n_params': n_par, 'n_connections': n_conn}
        print(f"{name:<20} {acc:>9.4f} {n_par:>10,} {n_conn:>13,}")

    return results


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _arch_str(model):
    """Return a readable architecture string like '784 → 24 → 19 → 10'."""
    lin_idx = _linear_indices(model)
    sizes   = [model.layer_stack[lin_idx[0]].in_features]
    sizes  += [model.layer_stack[i].out_features for i in lin_idx]
    return " → ".join(str(s) for s in sizes)
