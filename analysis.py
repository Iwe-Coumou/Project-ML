"""
Post-training analysis functions.

These are called from analysis.ipynb after a model has been trained and pruned.
They interpret the cluster structure — what each cluster responds to, which images
activate it, and what visual features it shares.

The pruning pipeline itself lives in funcs.py.
"""

import torch
import numpy as np
from collections import defaultdict


def cluster_criticality_per_class(model, cluster_indices, layer_mapping, data_loader, cluster_id, device=None):
    device = device or model.device
    model.eval()

    # Get all labels
    all_labels = []
    for _, y in data_loader:
        all_labels.append(y)
    all_labels = torch.cat(all_labels).to(device)

    print(f"\n--- Calculating pre and post-ablation accuracy for cluster {cluster_id} ---")

    # Original predictions and per-class accuracy
    orig_preds = model.predict(data_loader).to(device)

    class_total = defaultdict(int)
    class_correct = defaultdict(int)
    for t, p in zip(all_labels, orig_preds):
        class_total[int(t)] += 1
        if t == p:
            class_correct[int(t)] += 1
    orig_acc_per_class = {cls: class_correct[cls]/class_total[cls] for cls in class_total}

    # Backup ALL layers (current and next)
    linear_indices = [i for i, l in enumerate(model.layer_stack) if isinstance(l, torch.nn.Linear)]
    layer_backups = {}
    for linear_layer_idx in linear_indices:
        layer = model.layer_stack[linear_layer_idx]
        layer_backups[linear_layer_idx] = {
            'weight': layer.weight.data.clone(),
            'bias': layer.bias.data.clone()
        }

    # Ablate neurons
    for layer_name, start_idx, end_idx in layer_mapping:
        layer_idx = int(layer_name.split('_')[1])
        linear_layer_idx = linear_indices[layer_idx]
        layer = model.layer_stack[linear_layer_idx]

        local_indices = [i - start_idx for i in cluster_indices if start_idx <= i < end_idx]
        if not local_indices:
            continue

        layer.weight.data[local_indices, :] = 0
        layer.bias.data[local_indices] = 0

        if layer_idx + 1 < len(linear_indices):
            next_layer = model.layer_stack[linear_indices[layer_idx + 1]]
            next_layer.weight.data[:, local_indices] = 0

    # Predictions after ablation
    ablated_preds = model.predict(data_loader).to(device)
    class_correct_after = defaultdict(int)
    for t, p in zip(all_labels, ablated_preds):
        if t == p:
            class_correct_after[int(t)] += 1
    acc_per_class_after = {cls: class_correct_after[cls]/class_total[cls] for cls in class_total}

    # Restore ALL layers
    for linear_layer_idx, backup in layer_backups.items():
        layer = model.layer_stack[linear_layer_idx]
        layer.weight.data = backup['weight']
        layer.bias.data = backup['bias']

    return {
        'pre': orig_acc_per_class,
        'post': acc_per_class_after
    }


def compute_cluster_selectivity(cluster_map, all_activation, labels, n_classes=10):
    results = {}

    for cluster_id, neuron_indices in cluster_map.items():

        cluster_acts = all_activation[:,neuron_indices]
        cluster_strength = cluster_acts.mean(dim=1)

        class_means = []
        class_totals = []

        for c in range(n_classes):
            mask = (labels == c)
            class_strength = cluster_strength[mask]

            mean_activation = class_strength.mean().item()
            total_activation = class_strength.sum().item()

            class_means.append(mean_activation)
            class_totals.append(total_activation)

        class_totals = np.array(class_totals)

        if class_totals.sum() > 0:
            prob_dist = class_totals / class_totals.sum()
        else:
            prob_dist = np.zeros(n_classes)

        entropy = -np.sum(prob_dist*np.log(prob_dist+1e-12))

        max_entropy = np.log(n_classes)
        normalized_entropy = entropy / max_entropy

        results[cluster_id] = {
            "mean_activation_per_class": class_means,
            "prob_distribution": prob_dist,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy
        }

    return results


def compute_prototypes_all_clusters(cluster_map, all_activations, images, top_frac=0.1, use_global_mean=True):
    """
    Compute prototypes and difference maps for all clusters across all images (no class stratification).

    Args:
        cluster_map: dict {cluster_id: list of neuron indices}
        all_activations: tensor (N_samples, total_neurons)
        images: tensor (N_samples, 1, 28, 28)
        top_frac: fraction of top-activating samples to average
        use_global_mean: if True, difference map relative to global mean, else cluster mean

    Returns:
        dict: cluster_id -> {'prototype': 28x28 array, 'diff_map': 28x28 array}
    """
    N_samples = images.shape[0]
    flat_images = images.view(N_samples, -1)

    # Precompute global mean if needed
    if use_global_mean:
        global_mean = flat_images.mean(dim=0).view(28,28)
        global_mean = (global_mean - global_mean.min()) / (global_mean.max() - global_mean.min() + 1e-8)
        global_mean_np = global_mean.cpu().numpy()

    all_prototypes = {}

    for cluster_id, cluster_indices in cluster_map.items():
        cluster_acts = all_activations[:, cluster_indices]
        cluster_strength = cluster_acts.mean(dim=1)

        # Top-k averaging across all samples
        k = max(1, int(top_frac * len(cluster_strength)))
        _, top_idx = torch.topk(cluster_strength, k)

        proto = images[top_idx].mean(dim=0).squeeze()
        proto = (proto - proto.min()) / (proto.max() - proto.min() + 1e-8)
        proto_np = proto.cpu().numpy()

        # Difference map
        if use_global_mean:
            mean_np = global_mean_np
        else:
            cluster_mean = images.mean(dim=0).squeeze()
            cluster_mean = (cluster_mean - cluster_mean.min()) / (cluster_mean.max() - cluster_mean.min() + 1e-8)
            mean_np = cluster_mean.cpu().numpy()

        diff_map = proto_np - mean_np
        diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)

        all_prototypes[cluster_id] = {
            'prototype': proto_np,
            'diff_map': diff_map
        }

    return all_prototypes


def compute_cluster_consensus_pixels(cluster_map, layer_mapping, model, loader, device,
                                      binarize_threshold=0.15, consensus_threshold=0.5):
    """
    For each cluster, collect all images whose dominant cluster is that cluster,
    binarize each image (pixel > binarize_threshold → 1), then compute the
    pixel-wise fraction across those images.

    Returns a dict where each pixel value = fraction of cluster images where
    that pixel is "on" — so dark pixels in the plot are features shared by
    most images that activate the cluster.

    Args:
        cluster_map:         {cluster_id: [global_neuron_indices]}
        layer_mapping:       [(layer_name, start, end), ...]
        model:               NeuralNetwork instance
        loader:              DataLoader (images must be 1×28×28, values in [0,1])
        device:              torch.device
        binarize_threshold:  pixel intensity threshold (0-1) for "on"
        consensus_threshold: stored for caller convenience

    Returns:
        dict: {cluster_id: {'consensus': np.array [28,28],
                             'n_images': int,
                             'threshold': float}}
    """
    # Collect all images in loader order
    all_images = []
    for X, _ in loader:
        all_images.append(X)
    all_images = torch.cat(all_images, dim=0)  # [N, 1, 28, 28]

    # Get activations (same pattern as cluster_digit_alignment)
    layer_data = model.get_layer_data(loader)
    prim_data = {k: v for k, v in layer_data.items() if k != 'layer_0'}
    if not prim_data or not cluster_map:
        return {}

    n_samples   = all_images.shape[0]
    cluster_ids = sorted(cluster_map.keys())

    # Build per-cluster mean activation [N, n_clusters]
    cluster_acts = np.zeros((n_samples, len(cluster_ids)), dtype=np.float32)
    for col, cid in enumerate(cluster_ids):
        neuron_indices = cluster_map[cid]
        layer_vecs = []
        for lname, start, end in layer_mapping:
            if lname not in prim_data:
                continue
            local_idxs = [gi - start for gi in neuron_indices if start <= gi < end]
            if not local_idxs:
                continue
            acts = prim_data[lname]['post_activation'][:, local_idxs]
            layer_vecs.append(acts.numpy())
        if layer_vecs:
            combined = np.concatenate(layer_vecs, axis=1)
            cluster_acts[:, col] = combined.mean(axis=1)

    # Assign each sample to its dominant cluster
    assigned_col = cluster_acts.argmax(axis=1)  # [N]

    # Compute per-cluster consensus pixel map
    flat_images = all_images.squeeze(1).cpu().numpy()  # [N, 28, 28]
    result = {}
    for col, cid in enumerate(cluster_ids):
        mask     = (assigned_col == col)
        n_images = int(mask.sum())
        if n_images == 0:
            result[cid] = {'consensus': np.zeros((28, 28), dtype=np.float32),
                           'n_images': 0, 'threshold': consensus_threshold}
            continue
        cluster_imgs = flat_images[mask]                          # [k, 28, 28]
        binary       = (cluster_imgs > binarize_threshold).astype(np.float32)
        consensus    = binary.mean(axis=0)                        # [28, 28] in [0,1]
        result[cid]  = {'consensus': consensus, 'n_images': n_images,
                        'threshold': consensus_threshold}

    return result


def compute_cluster_digit_consensus(cluster_digit_counts, images, labels,
                                     digit_threshold=0.05):
    """
    For each cluster, find the pixels that are common (shared) across all digit
    types that activate the cluster.

    For cluster C with significant digits {d : weight_d > digit_threshold}:
        consensus = element-wise minimum of mean_image_d across those digits

    The minimum operator keeps only pixels that are bright in ALL relevant digits,
    so the result shows what every activating digit has in common visually.

    Args:
        cluster_digit_counts: {cluster_id: np.array [10]} from cluster_digit_alignment
        images:               tensor [N, 1, 28, 28]
        labels:               tensor [N] integer labels 0-9
        digit_threshold:      minimum fractional weight for a digit to be included
                              (digits below this are treated as non-contributing)

    Returns:
        dict: {cluster_id: {'consensus': np.array [28,28],
                             'n_images': int,
                             'top_digit': int,
                             'weights': np.array [10]}}
    """
    n_digits  = 10
    images_np = images.squeeze(1).cpu().numpy()  # [N, 28, 28]
    labels_np = labels.cpu().numpy()

    # Per-digit mean image
    digit_means = np.zeros((n_digits, 28, 28), dtype=np.float32)
    for d in range(n_digits):
        mask = (labels_np == d)
        if mask.sum() > 0:
            digit_means[d] = images_np[mask].mean(axis=0)

    result = {}
    for cid, counts in cluster_digit_counts.items():
        total = float(counts.sum())
        if total == 0:
            result[cid] = {'consensus': np.zeros((28, 28), dtype=np.float32),
                           'n_images': 0, 'top_digit': -1,
                           'weights': np.zeros(n_digits, dtype=np.float32)}
            continue
        weights = counts.astype(np.float32) / total        # [10] probabilities

        # Digits that meaningfully activate this cluster
        active_digits = [d for d in range(n_digits) if weights[d] > digit_threshold]

        if len(active_digits) == 0:
            # Fall back to top digit only
            active_digits = [int(np.argmax(counts))]

        # Element-wise minimum across active digit mean images:
        # a pixel survives only if it is bright in EVERY relevant digit
        active_means = np.stack([digit_means[d] for d in active_digits], axis=0)  # [k, 28, 28]
        consensus = active_means.min(axis=0)  # [28, 28]

        result[cid] = {
            'consensus': consensus,
            'n_images':  int(total),
            'top_digit': int(np.argmax(counts)),
            'weights':   weights,
        }

    return result


def compute_cluster_class_prototypes(cluster_map, all_neuron_activations, images, labels,
                                      ablation_results, top_frac=0.1):
    """
    For each cluster, find the digit it is most critical for (largest accuracy drop in
    ablation_results), then compute:

      - class_proto : average of the top-k images of that digit ranked by cluster activation
                      → the prototypical instance of the critical digit this cluster fires on
      - class_diff  : class_proto minus the mean of ALL images of that digit
                      → which pixels of that digit are unique to cluster-preferred instances

    Args:
        cluster_map:           {cluster_id: [global_neuron_indices]}
        all_neuron_activations: tensor [N, total_neurons]
        images:                tensor [N, 1, 28, 28]
        labels:                tensor [N] integer class labels
        ablation_results:      {cluster_id: {'pre': {cls: acc}, 'post': {cls: acc}}}
        top_frac:              fraction of per-digit images to average (default 0.1)

    Returns:
        dict: {cluster_id: {'class_proto': np.ndarray [28,28],
                             'class_diff':  np.ndarray [28,28],
                             'top_digit':   int}}
    """
    def _norm(t):
        mn, mx = t.min(), t.max()
        return (t - mn) / (mx - mn + 1e-8)

    result = {}

    for cid, cluster_indices in cluster_map.items():
        # ── find top digit from ablation ──────────────────────────────────────
        if cid not in ablation_results:
            continue
        pre  = ablation_results[cid]['pre']
        post = ablation_results[cid]['post']
        drops = {d: pre[d] - post[d] for d in pre}
        top_digit = max(drops, key=drops.get)

        # ── filter to that digit ──────────────────────────────────────────────
        digit_mask = (labels == top_digit)
        if digit_mask.sum() == 0:
            continue
        digit_imgs = images[digit_mask]          # [M, 1, 28, 28]
        digit_acts = all_neuron_activations[digit_mask][:, cluster_indices]  # [M, k]

        # ── top-k of that digit by cluster activation ─────────────────────────
        strength = digit_acts.mean(dim=1)        # [M]
        k = max(1, int(top_frac * len(strength)))
        _, top_idx = torch.topk(strength, k)

        class_proto = digit_imgs[top_idx].mean(dim=0).squeeze()  # [28, 28]
        class_proto = _norm(class_proto).cpu().numpy()

        # ── diff: class_proto minus mean of all images of that digit ──────────
        digit_mean = digit_imgs.float().mean(dim=0).squeeze()
        digit_mean = _norm(digit_mean).cpu().numpy()

        raw_diff = class_proto - digit_mean
        class_diff = (raw_diff - raw_diff.min()) / (raw_diff.max() - raw_diff.min() + 1e-8)

        result[cid] = {
            'class_proto': class_proto,
            'class_diff':  class_diff,
            'top_digit':   int(top_digit),
        }

    return result
