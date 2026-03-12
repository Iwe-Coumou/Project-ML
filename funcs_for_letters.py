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
import threading
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset
from concurrent.futures import ThreadPoolExecutor, as_completed

from NeuralNetwork import NeuralNetwork

# Thread-safe print to avoid garbled output from parallel variant training
_PRINT_LOCK = threading.Lock()

def _tprint(*args, **kwargs):
    with _PRINT_LOCK:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_letters_dataloaders(batch_size=8000, val_frac=0.1, train_frac=1.0):
    """
    Load EMNIST 'byclass' split filtered to lowercase a-z (classes 36-61 → remapped to 0-25).

    Applies the same rotate-then-flip transform used for the digit dataset so
    the pixel orientation is consistent.

    Args:
        train_frac: fraction of the training split to keep (default 1.0 = full set).
                    Subsampling uses a fixed seed so results are reproducible.

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

    # byclass: digits 0-9, uppercase A-Z (10-35), lowercase a-z (36-61)
    # shift lowercase labels to 0-25
    target_transform = transforms.Lambda(lambda y: y - 36)

    train_full = datasets.EMNIST(
        root="./data", split="byclass", train=True,
        download=True, transform=transform, target_transform=target_transform)

    test_dataset = datasets.EMNIST(
        root="./data", split="byclass", train=False,
        download=True, transform=transform, target_transform=target_transform)

    # Filter to lowercase only at dataloader level (raw targets before transform)
    train_full   = Subset(train_full,   (train_full.targets >= 36).logical_and(train_full.targets <= 61).nonzero(as_tuple=True)[0])
    test_dataset = Subset(test_dataset, (test_dataset.targets >= 36).logical_and(test_dataset.targets <= 61).nonzero(as_tuple=True)[0])

    n_val   = int(val_frac * len(train_full))
    n_train = len(train_full) - n_val
    train_dataset, val_dataset = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    if train_frac < 1.0:
        n_sub = max(26 * 10, int(train_frac * n_train))  # at least 260 samples
        sub_idx = torch.randperm(n_train, generator=torch.Generator().manual_seed(0))[:n_sub]
        train_dataset = Subset(train_dataset, sub_idx.tolist())
        n_train = n_sub

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

    print(f"Letters (lowercase only) — train: {n_train}, val: {n_val}, test: {len(test_dataset)}")
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


# ---------------------------------------------------------------------------
# Experiment — six-variant multi-seed comparison
# ---------------------------------------------------------------------------

_VARIANTS = [
    'frozen_transfer',
    'frozen_regrowth',
    'unfrozen_transfer',
    'fc_baseline',
    'random_frozen',
    'random_frozen_regrowth',
]


def build_random_frozen_model(pruned_model, n_letters=26):
    """
    Same sparse architecture as the transfer model but all hidden weights
    randomly initialised from the empirical weight distribution of the pruned
    model (std of non-zero hidden weights), then frozen.  Layer-0 is reset
    (Xavier), output is fresh.

    Control condition: same topology + same weight scale, no digit knowledge.
    """
    device  = pruned_model.device
    h_sizes = _hidden_sizes(pruned_model)
    model   = NeuralNetwork(input_size=784, hidden_sizes=h_sizes,
                             output_size=n_letters, device=device)

    lin_src = _linear_indices(pruned_model)
    lin_dst = _linear_indices(model)

    # Empirical std from pruned hidden-layer non-zero weights
    all_w = []
    for k in range(1, len(lin_src) - 1):
        w  = pruned_model.layer_stack[lin_src[k]].weight.data
        nz = w[w != 0]
        if nz.numel() > 0:
            all_w.append(nz.abs().cpu())
    emp_std = float(torch.cat(all_w).std()) if all_w else 0.02

    # Copy sparsity masks then re-init hidden weights from N(0, emp_std)
    _copy_connection_masks(pruned_model, model, lin_src, lin_dst)
    for k in range(1, len(lin_dst) - 1):
        ls_idx = lin_dst[k]
        layer  = model.layer_stack[ls_idx]
        layer.weight.data.normal_(mean=0.0, std=emp_std)
        layer.bias.data.zero_()
        if hasattr(model, 'connection_masks') and ls_idx in model.connection_masks:
            layer.weight.data *= model.connection_masks[ls_idx]

    # Reset layer-0 (Xavier)
    init.xavier_uniform_(model.layer_stack[lin_dst[0]].weight)
    init.zeros_(model.layer_stack[lin_dst[0]].bias)

    # Freeze hidden layers
    for ls_idx in lin_dst[1:-1]:
        model.layer_stack[ls_idx].weight.requires_grad_(False)
        model.layer_stack[ls_idx].bias.requires_grad_(False)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Random-frozen model: {_arch_str(model)}  (emp_std={emp_std:.4f}  trainable={n_trainable})")
    return model


def _preload_to_tensors(dataset, batch_size=4096):
    """
    Apply all dataset transforms once and cache the result as CPU float tensors.

    EMNIST transforms (ToTensor + rotate + flip) run in Python per image on every
    DataLoader iteration.  Pre-loading executes them once and stores the result as
    a plain tensor, so subsequent DataLoader iterations over the returned
    TensorDataset are pure C++ tensor slices — zero Python/PIL overhead.

    Returns: (X [N,1,28,28] float32, y [N] long) on CPU.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    Xs, ys = [], []
    for X, y in loader:
        Xs.append(X)
        ys.append(y)
    return torch.cat(Xs), torch.cat(ys).long()


def _frozen_layer_row_counts(model):
    """Return {layer_stack_idx: current_n_rows} for all hidden linear layers."""
    lin = _linear_indices(model)
    return {lin[k]: model.layer_stack[lin[k]].weight.shape[0]
            for k in range(1, len(lin) - 1)}


def _train_one_epoch(model, loader, criterion, device, l1_lambda=1e-5):
    """
    One forward+backward pass through loader.
    Mirrors NeuralNetwork.train_model: respects connection_masks and L1 reg.
    Returns mean batch loss.
    """
    model.train()
    if model.optimizer is None:
        model.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        l1 = torch.stack([p.abs().sum() for p in model.parameters()]).sum()
        loss = criterion(logits, y) + l1_lambda * l1
        model.optimizer.zero_grad()
        loss.backward()
        if hasattr(model, 'connection_masks') and model.connection_masks:
            for li, mask in model.connection_masks.items():
                p = model.layer_stack[li].weight
                if p.grad is not None:
                    p.grad.data *= mask.to(p.grad.device)
        model.optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def _eval_acc(model, loader):
    """Accuracy without tqdm noise."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(model.device), y.to(model.device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total if total > 0 else 0.0


def _make_partial_freeze_hook(n_frozen):
    """Gradient hook: zero gradient entries for indices < n_frozen."""
    def hook(grad):
        if n_frozen <= 0:
            return grad
        g = grad.clone()
        g[:n_frozen] = 0.0
        return g
    return hook


def _refreeze_after_regrowth(model, frozen_row_counts):
    """
    After error_driven_regrowth has extended hidden layer tensors, re-enable
    requires_grad and register gradient hooks that keep the ORIGINAL rows frozen
    while letting newly added rows train freely.

    Args:
        frozen_row_counts: {layer_stack_idx: n_rows_to_keep_frozen}
                           Captured before any regrowth in this run.
    """
    for handle in getattr(model, '_freeze_hook_handles', []):
        handle.remove()
    model._freeze_hook_handles = []

    for ls_idx, n_frozen in frozen_row_counts.items():
        layer = model.layer_stack[ls_idx]
        layer.weight.requires_grad_(True)
        layer.bias.requires_grad_(True)
        hw = layer.weight.register_hook(_make_partial_freeze_hook(n_frozen))
        hb = layer.bias.register_hook(_make_partial_freeze_hook(n_frozen))
        model._freeze_hook_handles.extend([hw, hb])

    model.optimizer = None  # caller must rebuild Adam


def _run_simple_half(model, loader, val_loader, n_epochs, criterion, device, lr, label=''):
    """Train n_epochs, return per-epoch val_acc list."""
    if model.optimizer is None:
        model.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr)
    curves = []
    for ep in range(n_epochs):
        _train_one_epoch(model, loader, criterion, device)
        acc = _eval_acc(model, val_loader)
        curves.append(acc)
        _tprint(f"    {label}ep {ep+1}/{n_epochs}  val={acc:.4f}")
    return curves


def _run_regrowth_half(model, loader, val_loader, n_epochs, criterion, device, lr,
                        cluster_map, layer_mapping, threshold_frac, n_spawn,
                        frozen_row_counts, regrowth_interval=5, label=''):
    """
    Train n_epochs with error_driven_regrowth every `regrowth_interval` epochs.
    cluster_map is modified in-place (new neurons appended by regrowth).
    Calling regrowth every epoch is wasteful: one epoch rarely changes which
    clusters are underperforming, but each call runs a full forward pass through
    the val set once per cluster.  Checking every 5 epochs gives the same
    signal at 5x lower cost.
    """
    import funcs as _funcs
    model.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    curves = []
    for ep in range(n_epochs):
        _train_one_epoch(model, loader, criterion, device)
        n_added = 0
        if (ep + 1) % regrowth_interval == 0:
            n_added = _funcs.error_driven_regrowth(
                model, cluster_map, layer_mapping,
                val_loader, threshold_frac, device, n_spawn)
            if n_added > 0:
                _refreeze_after_regrowth(model, frozen_row_counts)
                model.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        acc = _eval_acc(model, val_loader)
        curves.append(acc)
        marker = f'  +{n_added} neurons' if n_added else ''
        _tprint(f"    {label}ep {ep+1}/{n_epochs}  val={acc:.4f}{marker}")
    return curves


def _run_variant_full(name, model, h1_ds, h2_ds, val_ds, batch_size,
                      n_epochs_half, criterion, device, lr,
                      cluster_map, layer_mapping, threshold_frac, n_spawn,
                      frozen_row_counts, regrowth_interval):
    """
    Run both halves for a single variant.  Designed to be called from a thread.

    Each thread receives **dataset** objects (not DataLoader objects) and builds
    its own independent DataLoaders locally.  This avoids GIL contention that
    occurs when multiple threads share and iterate the same DataLoader: with
    num_workers=0 each `for X, y in loader` call holds the GIL to load data, so
    shared loaders serialize across threads and add switching overhead on top.
    Per-thread loaders give each variant its own iteration state with no sharing.

    Args:
        h1_ds, h2_ds, val_ds: Dataset objects (not loaders)
        batch_size:            used when building the per-thread loaders
        cluster_map:           None for non-regrowth variants; deep-copied dict for
                               regrowth variants (modified in-place by regrowth).
        frozen_row_counts:     None for non-regrowth variants.

    Returns:
        (name, full_curve)   where full_curve has length 2*n_epochs_half.
    """
    # Build independent loaders for this thread
    h1_loader  = DataLoader(h1_ds,  batch_size=batch_size, shuffle=True,  num_workers=0)
    h2_loader  = DataLoader(h2_ds,  batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.optimizer = None
    _tprint(f"\n[{name}] starting half 1")
    c1 = _run_simple_half(model, h1_loader, val_loader, n_epochs_half,
                           criterion, device, lr, label=f'{name} h1 ')

    _tprint(f"\n[{name}] starting half 2")
    if cluster_map is not None:
        c2 = _run_regrowth_half(model, h2_loader, val_loader, n_epochs_half,
                                 criterion, device, lr,
                                 cluster_map, layer_mapping,
                                 threshold_frac, n_spawn, frozen_row_counts,
                                 regrowth_interval=regrowth_interval,
                                 label=f'{name} h2 ')
    else:
        c2 = _run_simple_half(model, h2_loader, val_loader, n_epochs_half,
                               criterion, device, lr, label=f'{name} h2 ')

    _tprint(f"[{name}] done  final_val={c2[-1]:.4f}")
    return name, c1 + c2


def run_experiment(pruned_model, n_seeds=5, n_epochs_half=50, lr=1e-3,
                   batch_size=4096, threshold_frac=1.5, n_spawn=5,
                   regrowth_interval=5, device=None, train_frac=1.0):
    """
    Run six transfer-learning variants on EMNIST letters with multiple seeds,
    training all six variants in parallel within each seed.

    Training is split into two equal halves of n_epochs_half epochs each.
    Regrowth variants call funcs.error_driven_regrowth every `regrowth_interval`
    epochs in half 2 (default every 5 epochs, not every epoch — one epoch is
    rarely enough to change which clusters are underperforming, so checking
    every epoch wastes n_clusters forward passes per epoch for no benefit).

    All six variants run concurrently using a ThreadPoolExecutor.  Each variant
    owns its own model and cluster_map copy so there is no shared mutable state.
    PyTorch gives each thread its own CUDA stream, so GPU work from different
    variants can overlap automatically.

    Variants
    --------
    frozen_transfer        — frozen hidden layers, no regrowth
    frozen_regrowth        — frozen hidden layers, regrowth in half 2
    unfrozen_transfer      — all layers trainable
    fc_baseline            — fully-connected MLP, same width
    random_frozen          — empirical-distribution weights, frozen, no regrowth
    random_frozen_regrowth — same, with regrowth in half 2

    Args:
        regrowth_interval: how many epochs between regrowth checks (default 5)

    Returns
    -------
    dict with keys = variant names plus '_meta', each variant having:
        'curves':   list of n_seeds lists, each of length 2*n_epochs_half
        'ablation': list of n_seeds dicts {cluster_id: mean_letter_drop}
                    (only populated for regrowth variants)
    """
    import analysis as _analysis

    device = device or pruned_model.device
    cluster_map_orig = getattr(pruned_model, 'final_cluster_map',  None) or {}
    layer_mapping    = getattr(pruned_model, 'final_layer_mapping', None) or []
    criterion        = nn.CrossEntropyLoss()

    results = {v: {'curves': [], 'ablation': [], 'test_acc_per_seed': []} for v in _VARIANTS}
    results['_meta'] = {
        'n_epochs_half': n_epochs_half,
        'n_seeds': n_seeds,
        'regrowth_interval': regrowth_interval,
        'train_frac': train_frac,
    }

    # Pre-load all data into CPU tensors once.
    # EMNIST transforms (PIL rotate/flip/ToTensor) run in Python per image on every
    # DataLoader iteration.  Loading here applies them once and stores raw float tensors,
    # so per-thread DataLoaders use TensorDataset whose __getitem__ is a pure C++ tensor
    # slice — zero Python/PIL overhead, no GIL contention on data loading.
    train_loader_base, val_loader_base, test_loader_base = get_letters_dataloaders(batch_size=batch_size, train_frac=train_frac)
    train_dataset = train_loader_base.dataset   # Subset from random_split(seed=42)
    val_ds        = val_loader_base.dataset

    _tprint("Pre-loading letter data into CPU tensors (one-time, applies transforms)...")
    train_X, train_y = _preload_to_tensors(train_dataset, batch_size=batch_size)
    val_X,   val_y   = _preload_to_tensors(val_ds,        batch_size=batch_size)
    val_ds_tensor    = TensorDataset(val_X, val_y)
    _tprint(f"  train: {train_X.shape}  val: {val_X.shape}")
    results['_meta']['n_train'] = train_X.shape[0]

    for seed in range(n_seeds):
        _tprint(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Seed-dependent half split — index into preloaded tensors
        n_total = len(train_X)
        n_h1    = n_total // 2
        n_h2    = n_total - n_h1
        gen     = torch.Generator().manual_seed(seed)
        perm    = torch.randperm(n_total, generator=gen)
        h1_idx, h2_idx = perm[:n_h1], perm[n_h1:]
        h1_ds_tensor = TensorDataset(train_X[h1_idx], train_y[h1_idx])
        h2_ds_tensor = TensorDataset(train_X[h2_idx], train_y[h2_idx])

        # Per-seed cluster map copies for regrowth variants
        cmap_fr  = copy.deepcopy(cluster_map_orig)
        cmap_rfr = copy.deepcopy(cluster_map_orig)

        # ── Build all six models ─────────────────────────────────────────────
        m_ft  = build_transfer_model(pruned_model)
        m_fr  = build_transfer_model(pruned_model)
        m_uf  = build_transfer_model(pruned_model)
        m_fc  = build_fc_benchmark(pruned_model)
        m_rf  = build_random_frozen_model(pruned_model)
        m_rfr = build_random_frozen_model(pruned_model)

        # Unfreeze all hidden layers for unfrozen_transfer
        for ls_idx in _linear_indices(m_uf)[1:-1]:
            m_uf.layer_stack[ls_idx].weight.requires_grad_(True)
            m_uf.layer_stack[ls_idx].bias.requires_grad_(True)

        # Frozen row counts captured before any regrowth
        frc_fr  = _frozen_layer_row_counts(m_fr)
        frc_rfr = _frozen_layer_row_counts(m_rfr)

        # ── Variant specs: (name, model, cmap_or_None, frc_or_None) ─────────
        variant_specs = [
            ('frozen_transfer',        m_ft,  None,      None),
            ('frozen_regrowth',        m_fr,  cmap_fr,   frc_fr),
            ('unfrozen_transfer',      m_uf,  None,      None),
            ('fc_baseline',            m_fc,  None,      None),
            ('random_frozen',          m_rf,  None,      None),
            ('random_frozen_regrowth', m_rfr, cmap_rfr,  frc_rfr),
        ]

        # ── Train all six variants in parallel ───────────────────────────────
        _tprint(f"\n-- Training all 6 variants in parallel (regrowth every "
                f"{regrowth_interval} epochs) --")
        seed_curves = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {
                pool.submit(
                    _run_variant_full,
                    name, m, h1_ds_tensor, h2_ds_tensor, val_ds_tensor, batch_size,
                    n_epochs_half, criterion, device, lr,
                    cmap, layer_mapping, threshold_frac, n_spawn, frc,
                    regrowth_interval,
                ): name
                for name, m, cmap, frc in variant_specs
            }
            for future in as_completed(futures):
                vname, curve = future.result()
                seed_curves[vname] = curve

        for name in _VARIANTS:
            results[name]['curves'].append(seed_curves[name])

        if test_loader_base is not None:
            _tprint("\n-- Test set evaluation --")
            for name, m in [
                ('frozen_transfer',         m_ft),
                ('frozen_regrowth',         m_fr),
                ('unfrozen_transfer',       m_uf),
                ('fc_baseline',             m_fc),
                ('random_frozen',           m_rf),
                ('random_frozen_regrowth',  m_rfr),
            ]:
                m.eval()
                correct = total = 0
                with torch.no_grad():
                    for xb, yb in test_loader_base:
                        xb, yb = xb.to(device), yb.to(device)
                        correct += (m(xb).argmax(1) == yb).sum().item()
                        total += yb.size(0)
                acc = correct / total
                results[name]['test_acc_per_seed'].append(acc)
                _tprint(f"  {name:<30} test acc: {acc:.4f}")

        # ── Ablation (regrowth variants only) — run after training ───────────
        _tprint("\n-- Ablation --")
        ablation_val_loader = DataLoader(val_ds_tensor, batch_size=batch_size, shuffle=False, num_workers=0)
        for name, m, cmap in [
            ('frozen_regrowth',        m_fr,  cmap_fr),
            ('random_frozen_regrowth', m_rfr, cmap_rfr),
        ]:
            _tprint(f"  [{name}]")
            drop_dict = {}
            for cid, neurons in cmap.items():
                res = _analysis.cluster_criticality_per_class(
                    m, neurons, layer_mapping, ablation_val_loader, cid, device=device)
                mean_drop = float(np.mean(
                    [res['pre'][c] - res['post'][c] for c in res['pre']]))
                drop_dict[int(cid)] = round(mean_drop, 4)
            results[name]['ablation'].append(drop_dict)

    return results


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_results(results, pruned_model, output_dir='results'):
    """
    Write all experiment outputs to output_dir/.
    Overwrites existing files.  Creates the folder if absent.
    """
    import json
    import os
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from math import ceil

    os.makedirs(output_dir, exist_ok=True)

    meta          = results.get('_meta', {})
    n_epochs_half = meta.get('n_epochs_half', 50)
    n_seeds       = meta.get('n_seeds', 1)
    total_epochs  = n_epochs_half * 2
    epochs        = list(range(total_epochs))

    COLOURS = {
        'frozen_transfer':        '#1f77b4',
        'frozen_regrowth':        '#ff7f0e',
        'unfrozen_transfer':      '#2ca02c',
        'fc_baseline':            '#d62728',
        'random_frozen':          '#9467bd',
        'random_frozen_regrowth': '#8c564b',
    }

    # ── Compute summary stats ────────────────────────────────────────────────
    means, stds, aucs, milestones, finals = {}, {}, {}, {}, {}
    for v in _VARIANTS:
        arr = np.array(results[v]['curves'])       # [n_seeds, total_epochs]
        m   = arr.mean(axis=0)
        s   = arr.std(axis=0)
        means[v] = m
        stds[v]  = s
        aucs[v]  = round(float(np.trapezoid(m) / total_epochs), 4)
        reach70  = int(np.argmax(m >= 0.70)) if (m >= 0.70).any() else None
        reach80  = int(np.argmax(m >= 0.80)) if (m >= 0.80).any() else None
        milestones[v] = {'70%': reach70, '80%': reach80}
        finals[v] = {
            'mean': round(float(arr[:, -1].mean()), 4),
            'std':  round(float(arr[:, -1].std()),  4),
        }
    
    # test accuracy (if available from run_experiment)
    test_finals = {}
    for v in _VARIANTS:
        per_seed = results[v].get("test_acc_per_seed", [])
        if per_seed:
            test_finals[v] = {
                'mean': round(float(np.mean(per_seed)), 4),
                'std': round(float(np.std(per_seed)), 4),
            }

    # ── learning_curves.html ────────────────────────────────────────────────
    fig_lc = go.Figure()
    for v in _VARIANTS:
        m, s, col = means[v], stds[v], COLOURS[v]
        # Confidence band
        x_fill = epochs + epochs[::-1]
        y_fill = (m + s).tolist() + (m - s).tolist()[::-1]
        fig_lc.add_trace(go.Scatter(
            x=x_fill, y=y_fill, fill='toself',
            fillcolor=col, opacity=0.15,
            line=dict(width=0), showlegend=False, hoverinfo='skip'))
        # Mean line
        fig_lc.add_trace(go.Scatter(
            x=epochs, y=m.tolist(), name=v,
            line=dict(color=col, width=2)))
    # Split marker
    fig_lc.add_vline(x=n_epochs_half - 0.5,
                     line=dict(color='black', dash='dash', width=1),
                     annotation_text='half split', annotation_position='top right')
    fig_lc.update_layout(
        title='Letter transfer — validation accuracy (mean ± std)',
        xaxis_title='Epoch', yaxis_title='Val accuracy',
        yaxis=dict(range=[0, 1]), template='plotly_white')
    fig_lc.write_html(os.path.join(output_dir, 'learning_curves.html'))
    print(f"  Saved learning_curves.html")

    # ── auc ─────────────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, 'auc.json'), 'w') as f:
        json.dump(aucs, f, indent=2)
    with open(os.path.join(output_dir, 'auc.txt'), 'w') as f:
        f.write("Area under learning curve (normalised by n_epochs)\n")
        for v, a in aucs.items():
            f.write(f"  {v:<30} {a:.4f}\n")
    print(f"  Saved auc.json / auc.txt")

    # ── milestone_epochs ────────────────────────────────────────────────────
    with open(os.path.join(output_dir, 'milestone_epochs.json'), 'w') as f:
        json.dump(milestones, f, indent=2)
    with open(os.path.join(output_dir, 'milestone_epochs.txt'), 'w') as f:
        f.write("First epoch reaching accuracy threshold (None = never reached)\n")
        for v, ms in milestones.items():
            f.write(f"  {v:<30}  70%: {ms['70%']}  80%: {ms['80%']}\n")
    print(f"  Saved milestone_epochs.json / milestone_epochs.txt")

    # ── final_accuracy ───────────────────────────────────────────────────────
    with open(os.path.join(output_dir, 'final_accuracy.json'), 'w') as f:
        json.dump(finals, f, indent=2)
    with open(os.path.join(output_dir, 'final_accuracy.txt'), 'w') as f:
        f.write("Final epoch accuracy (mean ± std across seeds)\n")
        for v, fa in finals.items():
            f.write(f"  {v:<30}  {fa['mean']:.4f} ± {fa['std']:.4f}\n")
    print(f"  Saved final_accuracy.json / final_accuracy.txt")

    # ── test_accuracy ───────────────────────────────────────────────────────
    if test_finals:
        with open(os.path.join(output_dir, 'test_accuracy.json'), 'w') as f:
            json.dump(test_finals, f, indent=2)
        with open(os.path.join(output_dir, 'test_accuracy.txt'), 'w') as f:
            f.write("Final test-set accuracy (mean ± std across seeds)\n")
            for v, fa in test_finals.items():
                f.write(f"  {v:<30}  {fa['mean']:.4f} ± {fa['std']:.4f}\n")
        print(f"    Saved test_accuracy.txt")

    # ── per_run_curves.json ─────────────────────────────────────────────────
    raw = {v: results[v]['curves'] for v in _VARIANTS}
    with open(os.path.join(output_dir, 'per_run_curves.json'), 'w') as f:
        json.dump(raw, f)
    print(f"  Saved per_run_curves.json")

    # ── cluster_ablation.html ───────────────────────────────────────────────
    regrowth_variants = ['frozen_regrowth', 'random_frozen_regrowth']
    fig_ab = go.Figure()
    for v in regrowth_variants:
        ablation_runs = results[v]['ablation']
        if not ablation_runs:
            continue
        all_cids = sorted({cid for run in ablation_runs for cid in run})
        mean_drops = []
        for cid in all_cids:
            drops = [run.get(cid, 0.0) for run in ablation_runs]
            mean_drops.append(float(np.mean(drops)))
        fig_ab.add_trace(go.Bar(
            name=v,
            x=[f'C{c}' for c in all_cids],
            y=mean_drops,
            marker_color=COLOURS[v]))
    fig_ab.update_layout(
        barmode='group',
        title='Mean letter-accuracy drop per cluster (ablation)',
        xaxis_title='Cluster', yaxis_title='Mean accuracy drop',
        template='plotly_white')
    fig_ab.write_html(os.path.join(output_dir, 'cluster_ablation.html'))
    print(f"  Saved cluster_ablation.html")

    # ── summary.txt ─────────────────────────────────────────────────────────
    lines = [
        f"Experiment summary — {n_seeds} seed(s), {n_epochs_half} epochs/half",
        f"{'='*60}",
        "",
        "Final accuracy (mean ± std):",
    ]
    for v, fa in finals.items():
        lines.append(f"  {v:<30}  {fa['mean']:.4f} ± {fa['std']:.4f}")
    lines += ["", "AUC (normalised):"]
    for v, a in aucs.items():
        lines.append(f"  {v:<30}  {a:.4f}")
    lines += ["", "Milestone epochs (70% / 80%):"]
    for v, ms in milestones.items():
        lines.append(f"  {v:<30}  70%: {ms['70%']}  80%: {ms['80%']}")
    lines += ["", "Cluster ablation (mean letter drop, regrowth variants):"]
    for v in regrowth_variants:
        ablation_runs = results[v]['ablation']
        if not ablation_runs:
            continue
        all_cids = sorted({cid for run in ablation_runs for cid in run})
        for cid in all_cids:
            drops = [run.get(cid, 0.0) for run in ablation_runs]
            lines.append(f"  {v} C{cid}: {float(np.mean(drops)):.4f}")      
    if test_finals:
        lines += ["", "Test-set accuracy (mean ± std):"]
        for v, fa in test_finals.items():
            lines.append(f"  {v:<30}  {fa['mean']:.4f} ± {fa['std']:.4f}")
    # Collapse detection
    collapse_variants = [v for v in _VARIANTS
                         if finals[v]['mean'] < 0.15]
    if collapse_variants:
        lines += ["", "NOTE — collapsed variants (final acc < 0.15, near chance):"]
        for v in collapse_variants:
            lines.append(f"  {v}: final val acc = {finals[v]['mean']:.4f}. "
                         f"Hypothesis: stale layer_mapping after regrowth corrupts "
                         f"ablation measurement on the second regrowth call, causing "
                         f"mode collapse. Framed as a negative result / future work.")
    
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("\n".join(lines))
    print(f"  Saved summary.txt")

    # ── layer-0 weight heatmaps ──────────────────────────────────────────────
    _save_layer0_heatmaps(pruned_model, output_dir)

    print(f"\nAll results written to '{output_dir}/'")

def run_significance_tests(results):
    """
    Welch's t-test between key variant pairs using per-seed test accuracies.
    Returns dict of {pair_label: {'t': float, 'p': float, 'n': int}}.
    Requires test_loader to have been passed to run_experiment().
    """
    from scipy import stats

    pairs = [
        ('unfrozen_transfer', 'fc_baseline',   'unfrozen_transfer vs fc_baseline'),
        ('frozen_transfer',   'fc_baseline',   'frozen_transfer vs fc_baseline'),
        ('frozen_transfer',   'random_frozen', 'frozen_transfer vs random_frozen'),
    ]
    out = {}
    for a, b, label in pairs:
        accs_a = results[a].get('test_acc_per_seed', [])
        accs_b = results[b].get('test_acc_per_seed', [])
        if len(accs_a) < 2 or len(accs_b) < 2:
            out[label] = {'error': 'not enough seeds for t-test'}
            continue
        t, p = stats.ttest_ind(accs_a, accs_b, equal_var=False)
        out[label] = {'t': round(float(t), 4), 'p': round(float(p), 4), 'n': len(accs_a)}
    return out


def plot_final_comparison(results, sig_tests=None, output_dir='results'):
    """
    Static matplotlib bar chart: final test accuracy per variant with ± std error bars.
    """
    import matplotlib.pyplot as plt
    import os

    variants = _VARIANTS
    means, stds = [], []
    for v in variants:
        per_seed = results[v].get('test_acc_per_seed', [])
        if per_seed:
            means.append(float(np.mean(per_seed)))
            stds.append(float(np.std(per_seed)))
        else:
            fa = results[v].get('curves', [[]])
            last = [c[-1] for c in fa if c]
            means.append(float(np.mean(last)) if last else 0.0)
            stds.append(float(np.std(last))  if last else 0.0)

    colours = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
    x = np.arange(len(variants))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colours, alpha=0.85,
                  error_kw=dict(elinewidth=1.5, ecolor='black'))
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('_', '\n') for v in variants], fontsize=9)
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0, 1)
    ax.set_title(f'Transfer Learning Variants — Final Test Accuracy'
                 f' (mean ± std, n={len(results[variants[0]].get("test_acc_per_seed", [1]))} seeds)')
    ax.axhline(1/26, color='grey', linestyle='--', linewidth=0.8, label='chance (1/26)')
    ax.legend(fontsize=8)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'final_comparison.png')
    fig.savefig(path, dpi=150)
    plt.show()
    print(f"  Saved {path}")

def plot_learning_curves_static(results, output_dir='results'):
    """
    Static matplotlib learning curves with ± std shaded bands.
    """
    import matplotlib.pyplot as plt
    import os

    meta          = results.get('_meta', {})
    n_epochs_half = meta.get('n_epochs_half', 10)
    total_epochs  = n_epochs_half * 2
    epochs        = np.arange(total_epochs)

    colours = {
        'frozen_transfer':        '#1f77b4',
        'frozen_regrowth':        '#ff7f0e',
        'unfrozen_transfer':      '#2ca02c',
        'fc_baseline':            '#d62728',
        'random_frozen':          '#9467bd',
        'random_frozen_regrowth': '#8c564b',
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for v in _VARIANTS:
        arr = np.array(results[v]['curves'])   # [n_seeds, total_epochs]
        m, s = arr.mean(0), arr.std(0)
        col = colours[v]
        ax.plot(epochs, m, label=v, color=col, linewidth=2)
        ax.fill_between(epochs, m - s, m + s, alpha=0.15, color=col)

    ax.axvline(n_epochs_half - 0.5, color='black', linestyle='--',
               linewidth=0.8, label='half split')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_ylim(0, 1)
    ax.set_title('Letter Transfer — Validation Accuracy (mean ± std)')
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'learning_curves.png')
    fig.savefig(path, dpi=150)
    plt.show()
    print(f"  Saved {path}")


def _save_layer0_heatmaps(pruned_model, output_dir):
    """
    Save Plotly heatmaps of layer-0 neuron weight vectors (reshaped to 28×28).
    Produces:
        layer0_all_neurons.html  — full grid of all neurons
        layer0_cluster_{id}.html — one grid per cluster (only neurons feeding that cluster)
    """
    import os
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from math import ceil

    lin_idx = _linear_indices(pruned_model)
    cluster_map  = getattr(pruned_model, 'final_cluster_map',  None) or {}
    layer_mapping = getattr(pruned_model, 'final_layer_mapping', None) or []

    W0 = pruned_model.layer_stack[lin_idx[0]].weight.data.cpu().numpy()  # [h0, 784]
    h0 = W0.shape[0]

    def _make_grid_fig(neuron_indices, title):
        n_cols = min(8, len(neuron_indices))
        n_rows = ceil(len(neuron_indices) / n_cols) if n_cols > 0 else 1
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            horizontal_spacing=0.01, vertical_spacing=0.01)
        for pos, ni in enumerate(neuron_indices):
            r, c = pos // n_cols + 1, pos % n_cols + 1
            hm   = W0[ni].reshape(28, 28)
            fig.add_trace(
                go.Heatmap(z=hm.tolist(), colorscale='RdBu', zmid=0,
                           showscale=False, name=f'n{ni}'),
                row=r, col=c)
        fig.update_layout(
            title=title,
            height=max(200, n_rows * 120),
            template='plotly_white')
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig

    # Full grid
    fig_all = _make_grid_fig(list(range(h0)), 'Layer-0 neuron weight maps (all)')
    fig_all.write_html(os.path.join(output_dir, 'layer0_all_neurons.html'))
    print(f"  Saved layer0_all_neurons.html  ({h0} neurons)")

    # Per-cluster grids
    if not cluster_map or not layer_mapping:
        return

    # Find h1 (size of layer_1 — first primitive layer)
    h1 = pruned_model.layer_stack[lin_idx[1]].weight.shape[0]
    W1 = pruned_model.layer_stack[lin_idx[1]].weight.data.cpu()  # [h1, h0]

    for cid, global_neurons in sorted(cluster_map.items()):
        # Neurons from this cluster that sit in layer_1 (global idx 0..h1-1)
        l1_neurons = [gi for gi in global_neurons if gi < h1]
        if not l1_neurons:
            continue
        # Layer-0 neurons with non-zero weight to at least one cluster-l1 neuron
        feeds = (W1[l1_neurons, :].abs().sum(dim=0) > 0).nonzero(
            as_tuple=True)[0].tolist()
        if not feeds:
            continue
        fig_c = _make_grid_fig(feeds, f'Layer-0 neurons feeding cluster {cid}')
        fig_c.write_html(os.path.join(output_dir, f'layer0_cluster_{cid}.html'))
        print(f"  Saved layer0_cluster_{cid}.html  ({len(feeds)} neurons)")


# ---------------------------------------------------------------------------
# Cluster ablation intersection grid
# ---------------------------------------------------------------------------

def _get_images_by_class(loader, class_ids, n_samples=20):
    """
    Collect up to n_samples images per class from a DataLoader.

    Args:
        loader:     DataLoader yielding (images, labels)
        class_ids:  iterable of class ids to collect (int or str)
        n_samples:  max images per class

    Returns:
        dict {class_id: tensor [k, 1, 28, 28]}  (k ≤ n_samples)
    """
    class_ids = set(class_ids)
    buckets = {c: [] for c in class_ids}
    full = set()

    for X, Y in loader:
        if len(full) == len(class_ids):
            break
        for img, lbl in zip(X, Y):
            key = lbl.item() if isinstance(lbl, torch.Tensor) else lbl
            if key not in buckets:
                continue
            if len(buckets[key]) < n_samples:
                buckets[key].append(img.cpu())
            if len(buckets[key]) >= n_samples:
                full.add(key)

    return {c: torch.stack(imgs) for c, imgs in buckets.items() if imgs}


def _intersection_image(images_by_class, class_ids, pixel_threshold=0.3):
    """
    Compute the intersection of stroke pixels across all sampled images for the
    given class_ids.

    Each image is binarized (pixel > pixel_threshold → 1.0), then the
    element-wise minimum across all binarized images is taken, giving 1.0 only
    where every image has a stroke pixel.

    Args:
        images_by_class:  dict {class_id: tensor [k, 1, 28, 28]}
        class_ids:        which classes to include
        pixel_threshold:  binarization threshold (default 0.3)

    Returns:
        np.ndarray [28, 28], values in {0, 1}
    """
    all_imgs = []
    for c in class_ids:
        if c in images_by_class:
            all_imgs.append(images_by_class[c])  # [k, 1, 28, 28]

    if not all_imgs:
        return np.zeros((28, 28), dtype=np.float32)

    pooled = torch.cat(all_imgs, dim=0)          # [N, 1, 28, 28]
    binary = (pooled > pixel_threshold).float()  # binarize
    binary = binary.squeeze(1)                   # [N, 28, 28]
    intersection = binary.min(dim=0).values      # [28, 28] — 1 iff ALL images lit
    return intersection.numpy()


def _collect_activations_for_prototypes(model, loader, layer_mapping):
    """
    Collect all images and concatenated hidden-layer activations from a loader.

    Mirrors the construction used in funcs.cluster_neurons_fabio: layer_0 is
    excluded (it is the input projection, not a clustered hidden layer) and the
    remaining hidden layers are concatenated in ascending index order to build
    all_activations [N, total_neurons] whose columns match global neuron indices
    defined by layer_mapping.

    Returns:
        images      tensor [N, 1, 28, 28]  (CPU)
        all_acts    tensor [N, total_neurons]  (CPU)
    """
    all_images = []
    for X, _ in loader:
        all_images.append(X.cpu())
    all_images = torch.cat(all_images, dim=0)

    layer_data = model.get_layer_data(loader)
    hidden = sorted(k for k in layer_data if k != 'layer_0')
    all_acts = torch.cat(
        [layer_data[ln]['post_activation'].cpu() for ln in hidden], dim=1
    )
    return all_images, all_acts


def plot_cluster_ablation_grid(
    letter_ablation, digit_ablation,
    m_transfer, pruned_model,
    cluster_map, layer_mapping,
    letter_loader, digit_loader,
    threshold=0.05, n_samples=20,
    pixel_threshold=0.3, top_frac=0.1,
    device=None,
):
    """
    For each cluster plot a 4-row grid:

      Row 0  Letter intersection  — common stroke pixels across significant-ablation letter classes
      Row 1  Digit  intersection  — common stroke pixels across significant-ablation digit classes
      Row 2  Letter prototype     — mean of top-activating letter images (m_transfer)
      Row 3  Digit  prototype     — mean of top-activating digit  images (pruned_model)

    Column title shows which letters/digits drove a significant ablation drop
    (pre_acc - post_acc > threshold).

    Args:
        letter_ablation:  {cluster_id: {'pre': {letter_str: acc}, 'post': {letter_str: acc}}}
        digit_ablation:   {cluster_id: {'pre': {int: acc},        'post': {int: acc}}}
        m_transfer:       letter NeuralNetwork
        pruned_model:     digit NeuralNetwork
        cluster_map:      {cluster_id: [global_neuron_indices]}
        layer_mapping:    [(layer_name, start, end), ...]
        letter_loader:    DataLoader for letter val set
        digit_loader:     DataLoader for digit  val set
        threshold:        minimum accuracy drop considered significant (default 0.05)
        n_samples:        images per class for intersection (default 20)
        pixel_threshold:  binarization cutoff (default 0.3)
        top_frac:         fraction of top-activating samples for prototype (default 0.1)
    """
    import matplotlib.pyplot as plt
    import analysis
    from itertools import chain

    cluster_ids = sorted(cluster_map.keys())
    n_clusters  = len(cluster_ids)
    if n_clusters == 0:
        print("plot_cluster_ablation_grid: cluster_map is empty.")
        return

    # ── significant classes per cluster ──────────────────────────────────────
    def _sig(ablation_dict, cid):
        if cid not in ablation_dict:
            return []
        pre  = ablation_dict[cid]['pre']
        post = ablation_dict[cid]['post']
        return [k for k in pre if pre[k] - post[k] > threshold]

    sig_letters = {cid: _sig(letter_ablation, cid) for cid in cluster_ids}
    sig_digits  = {cid: _sig(digit_ablation,  cid) for cid in cluster_ids}

    # letter_ablation keys are strings ('a','b',...); loader yields integer labels (0-25).
    # Convert to int indices for image collection; keep strings for display.
    _letter_names = [chr(ord('a') + i) for i in range(26)]
    sig_letters_int = {
        cid: [_letter_names.index(c) for c in sl if c in _letter_names]
        for cid, sl in sig_letters.items()
    }

    # ── collect images once (only the classes we actually need) ───────────────
    all_letter_classes = set(chain.from_iterable(sig_letters_int.values()))
    all_digit_classes  = set(chain.from_iterable(sig_digits.values()))

    letter_imgs_by_cls = _get_images_by_class(letter_loader, all_letter_classes, n_samples)
    digit_imgs_by_cls  = _get_images_by_class(digit_loader,  all_digit_classes,  n_samples)

    # ── activation prototypes (one pass per model) ────────────────────────────
    l_images, l_acts = _collect_activations_for_prototypes(m_transfer,  letter_loader, layer_mapping)
    d_images, d_acts = _collect_activations_for_prototypes(pruned_model, digit_loader,  layer_mapping)

    protos_letters = analysis.compute_prototypes_all_clusters(
        cluster_map, l_acts, l_images, top_frac)
    protos_digits  = analysis.compute_prototypes_all_clusters(
        cluster_map, d_acts, d_images, top_frac)

    # ── plot ──────────────────────────────────────────────────────────────────
    row_labels = ['Letter\nintersection', 'Digit\nintersection',
                  'Letter\nprototype',    'Digit\nprototype']

    fig, axes = plt.subplots(4, n_clusters, figsize=(max(n_clusters * 2, 6), 9))
    if n_clusters == 1:
        axes = axes.reshape(4, 1)

    for col, cid in enumerate(cluster_ids):
        # Column title
        sl = sig_letters[cid]
        sd = [str(d) for d in sig_digits[cid]]
        title = (f'Cluster {cid}\n'
                 f'L: {",".join(sl) if sl else "—"}\n'
                 f'D: {",".join(sd) if sd else "—"}')
        axes[0, col].set_title(title, fontsize=8)

        # Row 0 — letter intersection
        img_l = _intersection_image(letter_imgs_by_cls, sig_letters_int[cid], pixel_threshold)
        axes[0, col].imshow(img_l, cmap='gray', vmin=0, vmax=1)
        axes[0, col].axis('off')

        # Row 1 — digit intersection
        img_d = _intersection_image(digit_imgs_by_cls, sig_digits[cid], pixel_threshold)
        axes[1, col].imshow(img_d, cmap='gray', vmin=0, vmax=1)
        axes[1, col].axis('off')

        # Row 2 — letter prototype
        proto_l = protos_letters.get(cid, {}).get('prototype', np.zeros((28, 28)))
        axes[2, col].imshow(proto_l, cmap='viridis')
        axes[2, col].axis('off')

        # Row 3 — digit prototype
        proto_d = protos_digits.get(cid, {}).get('prototype', np.zeros((28, 28)))
        axes[3, col].imshow(proto_d, cmap='viridis')
        axes[3, col].axis('off')

        # Row labels on leftmost column only
        if col == 0:
            for row, lbl in enumerate(row_labels):
                axes[row, 0].set_ylabel(lbl, fontsize=9)

    plt.suptitle('Cluster ablation — intersection & activation prototypes', fontsize=11)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Epoch-level significance — crossover analysis
# ---------------------------------------------------------------------------

def plot_epoch_significance(results, pairs=None, output_dir='results'):
    """
    Show when transfer learning stops being better than the FC baseline.

    For each pair of variants, computes at every epoch:
      - mean accuracy per variant (± std over seeds)
      - Welch's t-test p-value between the two variants

    Two-panel figure per pair:
      Top   — accuracy curves with ± std shading
      Bottom — per-epoch p-value (log scale); p=0.05 line; crossover epochs marked

    Crossover epochs reported:
      - Performance crossover: first epoch where variant A mean < variant B mean
      - Significance crossover: last epoch where p < 0.05 with A still ahead

    Args:
        results:    output of run_experiment()
        pairs:      list of (variant_a, variant_b, label) tuples.
                    Default: the two key comparisons against fc_baseline.
        output_dir: where to save the PNG
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind

    if pairs is None:
        pairs = [
            ('unfrozen_transfer', 'fc_baseline', 'unfrozen_transfer vs fc_baseline'),
            ('frozen_transfer',   'fc_baseline', 'frozen_transfer vs fc_baseline'),
        ]

    colours = {
        'frozen_transfer':        '#1f77b4',
        'unfrozen_transfer':      '#2ca02c',
        'fc_baseline':            '#d62728',
        'random_frozen':          '#9467bd',
        'frozen_regrowth':        '#ff7f0e',
        'random_frozen_regrowth': '#8c564b',
    }

    n_pairs = len(pairs)
    fig, axes = plt.subplots(2, n_pairs, figsize=(7 * n_pairs, 8),
                             gridspec_kw={'height_ratios': [2, 1]})
    if n_pairs == 1:
        axes = axes.reshape(2, 1)

    for col, (va, vb, label) in enumerate(pairs):
        curves_a = np.array(results[va]['curves'])   # [n_seeds, n_epochs]
        curves_b = np.array(results[vb]['curves'])
        n_epochs = curves_a.shape[1]
        epochs   = np.arange(1, n_epochs + 1)

        mean_a = curves_a.mean(axis=0)
        std_a  = curves_a.std(axis=0)
        mean_b = curves_b.mean(axis=0)
        std_b  = curves_b.std(axis=0)

        # Per-epoch t-test
        pvals = np.array([
            ttest_ind(curves_a[:, t], curves_b[:, t], equal_var=False).pvalue
            for t in range(n_epochs)
        ])

        # Crossover epochs
        perf_cross = next((t + 1 for t in range(n_epochs) if mean_a[t] < mean_b[t]), None)
        # Last epoch where p<0.05 and A is still ahead
        sig_cross = None
        for t in range(n_epochs - 1, -1, -1):
            if pvals[t] < 0.05 and mean_a[t] > mean_b[t]:
                sig_cross = t + 1
                break

        # ── Top panel: accuracy curves ──────────────────────────────────────
        ax_acc = axes[0, col]
        ca, cb = colours.get(va, 'C0'), colours.get(vb, 'C1')

        ax_acc.plot(epochs, mean_a, color=ca, label=va)
        ax_acc.fill_between(epochs, mean_a - std_a, mean_a + std_a, alpha=0.15, color=ca)
        ax_acc.plot(epochs, mean_b, color=cb, label=vb)
        ax_acc.fill_between(epochs, mean_b - std_b, mean_b + std_b, alpha=0.15, color=cb)

        if perf_cross is not None:
            ax_acc.axvline(perf_cross, color='gray', linestyle='--', linewidth=1,
                           label=f'perf. crossover (ep {perf_cross})')
        if sig_cross is not None:
            ax_acc.axvline(sig_cross, color='gray', linestyle=':', linewidth=1,
                           label=f'last sig. epoch (ep {sig_cross})')

        ax_acc.set_title(label, fontsize=10)
        ax_acc.set_ylabel('Val accuracy')
        ax_acc.set_xlabel('Epoch')
        ax_acc.legend(fontsize=8)
        ax_acc.grid(True, linestyle='--', alpha=0.4)

        # ── Bottom panel: p-value over epochs ───────────────────────────────
        ax_p = axes[1, col]
        ax_p.semilogy(epochs, pvals, color='black', linewidth=1)
        ax_p.axhline(0.05, color='red', linestyle='--', linewidth=1, label='p = 0.05')
        ax_p.fill_between(epochs, pvals, 0.05,
                          where=(pvals < 0.05) & (mean_a > mean_b),
                          alpha=0.15, color=ca, label=f'{va} significantly better')
        ax_p.fill_between(epochs, pvals, 0.05,
                          where=(pvals < 0.05) & (mean_a < mean_b),
                          alpha=0.15, color=cb, label=f'{vb} significantly better')

        if perf_cross is not None:
            ax_p.axvline(perf_cross, color='gray', linestyle='--', linewidth=1)
        if sig_cross is not None:
            ax_p.axvline(sig_cross, color='gray', linestyle=':', linewidth=1)

        ax_p.set_ylabel('p-value (log scale)')
        ax_p.set_xlabel('Epoch')
        ax_p.legend(fontsize=7)
        ax_p.grid(True, which='both', linestyle='--', alpha=0.4)

        print(f"{label}:")
        print(f"  Performance crossover : epoch {perf_cross or 'never'}")
        print(f"  Last significant epoch: epoch {sig_cross or 'never'}")

    plt.suptitle('Transfer learning advantage over training — crossover analysis',
                 fontsize=11, y=1.01)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'epoch_significance.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Sample-efficiency experiment
# ---------------------------------------------------------------------------

def run_sample_efficiency_experiment(pruned_model, fracs=None, n_seeds=3,
                                     n_epochs_half=25, **kwargs):
    """
    Run run_experiment() once per training-set fraction and collect results.

    Args:
        pruned_model:  trained pruned digit model
        fracs:         list of fractions to test (default [0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
        n_seeds:       seeds per fraction (default 3 — lower than main experiment to save time)
        n_epochs_half: epochs per half (default 25)
        **kwargs:      forwarded to run_experiment() (lr, batch_size, etc.)

    Returns:
        dict {frac: results_dict}  where each results_dict is the standard
        run_experiment() output (with test_acc_per_seed populated).
    """
    if fracs is None:
        fracs = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

    results_by_frac = {}
    for frac in fracs:
        print(f"\n{'#'*60}")
        print(f"# SAMPLE EFFICIENCY — train_frac={frac:.0%}")
        print(f"{'#'*60}")
        results_by_frac[frac] = run_experiment(
            pruned_model,
            n_seeds=n_seeds,
            n_epochs_half=n_epochs_half,
            train_frac=frac,
            **kwargs,
        )
    return results_by_frac


def plot_sample_efficiency(results_by_frac, output_dir='results',
                           variants=None):
    """
    Plot test accuracy vs training-set size for each variant.

    Args:
        results_by_frac: dict {frac: results_dict} from run_sample_efficiency_experiment()
        output_dir:      where to save the PNG
        variants:        list of variant names to include (default: all 4 non-regrowth variants)
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    if variants is None:
        variants = ['frozen_transfer', 'unfrozen_transfer', 'fc_baseline', 'random_frozen']

    colours = {
        'frozen_transfer':        '#1f77b4',
        'unfrozen_transfer':      '#2ca02c',
        'fc_baseline':            '#d62728',
        'random_frozen':          '#9467bd',
        'frozen_regrowth':        '#ff7f0e',
        'random_frozen_regrowth': '#8c564b',
    }

    # Build arrays: x = n_train, y[variant] = (means, stds)
    fracs_sorted = sorted(results_by_frac.keys())
    n_trains = [results_by_frac[f]['_meta']['n_train'] for f in fracs_sorted]

    _, ax = plt.subplots(figsize=(9, 5))

    for v in variants:
        means, stds = [], []
        for frac in fracs_sorted:
            accs = results_by_frac[frac][v].get('test_acc_per_seed', [])
            if accs:
                means.append(float(np.mean(accs)))
                stds.append(float(np.std(accs)))
            else:
                means.append(float('nan'))
                stds.append(0.0)
        means = np.array(means)
        stds  = np.array(stds)
        color = colours.get(v, None)
        ax.plot(n_trains, means, marker='o', label=v, color=color)
        ax.fill_between(n_trains, means - stds, means + stds,
                        alpha=0.15, color=color)

    ax.set_xscale('log')
    ax.set_xlabel('Training set size')
    ax.set_ylabel('Test accuracy')
    ax.set_title('Sample efficiency — test accuracy vs training set size (mean ± std)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'sample_efficiency.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved {out_path}")
