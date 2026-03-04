import torch
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import copy
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

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

def pruning(model, train_loader, parameters, baseline_acc, use_max_rounds=True, lr=0.01, mode="full", min_width=5, fresh_loader=None):
    max_rounds, prune_frac, prune_con_frac, regrow_frac, retrain_epochs, max_acc_drop = parameters

    current_model = copy.deepcopy(model)
    val_acc = baseline_acc

    round_idx = 0

    while True:
        round_idx += 1
        print(f"\n--- Pruning round {round_idx} ---")

        # 1. Hard round limit
        if use_max_rounds and round_idx > max_rounds:
            print("Reached maximum pruning rounds.")
            break

        prev_model = copy.deepcopy(current_model)
        layer_data = current_model.get_layer_data(train_loader)
        importance_scores = current_model.compute_neuron_importance(layer_data=layer_data)

        new_model = copy.deepcopy(current_model)
        prune_counts = None
        regrow_counts = None
        if fresh_loader is not None:
            mixed = ConcatDataset([train_loader.dataset, fresh_loader.dataset])
            loader_to_use = DataLoader(mixed, batch_size=train_loader.batch_size, shuffle=True)
        else:
            loader_to_use = train_loader    
        # 2. Pruning neurons
        if mode in ["full", "neuron_only", "prune_only"]:
            prune_counts = new_model.prune_hidden_neurons(
                importance_scores=importance_scores,
                prune_rate=prune_frac,
            )

        # 3. Pruning connections
        if mode in ["full", "connections_only", "prune_only"]:
            new_model.prune_connections(prune_frac=prune_con_frac)

        # 4. Retrain after pruning if applicable
        if mode in ["full", "neuron_only", "prune_only"]:
            new_model.optimizer = torch.optim.Adam(new_model.parameters(), lr=lr)
            print("Retraining after pruning...")
            val_acc = new_model.train_model(
                loader_to_use,
                epochs=retrain_epochs,
                lr=lr,
                val_interval=1,
            )

        # 5. Regrowth
        if mode in ["full", "neuron_only", "regrow_only"]:
            # Compute regrow counts
            if prune_counts is not None:
                regrow_counts = compute_regrow_from_pruned(prune_counts, regrow_frac)
            else:
                # growth-only mode: compute based on current width
                regrow_counts = compute_regrow_from_width(new_model, regrow_frac)

            if sum(regrow_counts.values()) > 0:
                new_model.regrow_hidden_neurons(regrow_counts, regrow_std=0.01)
                new_model.optimizer = torch.optim.Adam(new_model.parameters(), lr=lr)
                print("Retraining after regrowth...")
                val_acc = new_model.train_model(
                    loader_to_use,
                    epochs=2,
                    lr=lr,
                    val_interval=1,
                )

        # 6. Structural stopping checks
        for layer in new_model.layer_stack:
            if isinstance(layer, torch.nn.Linear) and layer.out_features < min_width:
                print("Minimum width reached. Stopping.")
                return prev_model

        prev_size = sum(p.numel() for p in prev_model.parameters())
        new_size = sum(p.numel() for p in new_model.parameters())
        if prev_size == new_size:
            print("Model size unchanged. Converged.")
            return prev_model

        # 7. Accuracy stopping condition
        print(f"Validation accuracy: {val_acc:.4f}")
        if baseline_acc - val_acc > max_acc_drop:
            print("Accuracy drop exceeded threshold. Restoring previous model.")
            return prev_model

        # 8. Accept round
        current_model = new_model

    print(f"Returning model after {round_idx} rounds.")
    return current_model

def compute_regrow_from_pruned(prune_counts, regrow_frac):
    """
    Args:
        prune_counts: dict {layer_name: n_pruned}
        regrow_frac: float in [0,1]

    Returns:
        dict {layer_name: n_regrow}
    """
    regrow_counts = {}

    for layer, n_pruned in prune_counts.items():
        n_regrow = int(regrow_frac * n_pruned)
        regrow_counts[layer] = n_regrow

    return regrow_counts

def compute_regrow_from_width(model, regrow_frac):
    """
    Regrow a percentage of current neurons per hidden layer.
    """
    regrow_counts = {}

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear) and name != "output_layer":
            current_width = layer.out_features
            n_regrow = int(regrow_frac * current_width)
            regrow_counts[name] = n_regrow

    return regrow_counts

def _build_neuron_adjacency(layer_data, layer_mapping, weight_threshold=1e-10):
    """
    Build a sparse undirected adjacency matrix for the neuron weight graph.

    Nodes are global neuron indices. Edges are non-zero weight connections between
    neurons in adjacent layers within the subset (layers whose numbers differ by 1).

    Args:
        layer_data:       dict with 'layer_k' keys, each having {'weights': tensor [out, in], ...}
        layer_mapping:    [(layer_name, start_idx, end_idx), ...]
        weight_threshold: minimum abs(weight) to count as an active connection

    Returns:
        scipy.sparse.csr_matrix of shape [total_neurons, total_neurons]
    """
    from scipy.sparse import csr_matrix

    total_neurons = layer_mapping[-1][2]
    rows, cols = [], []

    for i in range(len(layer_mapping) - 1):
        name_k,  start_k,  _ = layer_mapping[i]
        name_k1, start_k1, _ = layer_mapping[i + 1]

        # Only add edges for layers directly adjacent in the full network
        if int(name_k1.split('_')[1]) - int(name_k.split('_')[1]) != 1:
            continue

        # W[j, i] = weight from neuron i in layer_k to neuron j in layer_k1
        W = layer_data[name_k1]['weights']
        W_np = W.numpy() if hasattr(W, 'numpy') else np.array(W)

        j_idx, i_idx = np.where(np.abs(W_np) > weight_threshold)
        g_i = start_k  + i_idx
        g_j = start_k1 + j_idx

        # Undirected: add both directions
        rows += g_i.tolist() + g_j.tolist()
        cols += g_j.tolist() + g_i.tolist()

    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(total_neurons, total_neurons))
    adj.data[:] = 1.0  # binarise (duplicate edges may sum > 1)
    return adj


def cluster_neurons(layer_data, n_clusters, mode='full', nmf_subsample=15000):
    """
    Cluster neurons in a model using NMF on their activation patterns.

    Args:
        layer_data:    dict from model.get_layer_data(), keys like 'layer_0', 'layer_1', ...
                       each value has {'post_activation': tensor [N, n_neurons]}
        n_clusters:    number of clusters
        mode:          'full'      — cluster all neurons together across the whole model
                       'per_layer' — cluster each layer independently
        nmf_subsample: max samples used to fit NMF (subsampled randomly if exceeded)

    Returns:
        cluster_map:           {cluster_id: [global_neuron_indices]}
        layer_mapping:         [(layer_name, start_idx, end_idx), ...]
        all_neuron_activations: tensor [N_samples, total_neurons]
    """
    from sklearn.decomposition import NMF

    hidden_layers = [k for k in layer_data.keys() if 'layer_' in k]

    layer_mapping = []
    start_idx = 0
    for layer_name in hidden_layers:
        n_neurons = layer_data[layer_name]['post_activation'].shape[1]
        layer_mapping.append((layer_name, start_idx, start_idx + n_neurons))
        start_idx += n_neurons

    all_neuron_activations = torch.cat(
        [layer_data[ln]['post_activation'] for ln in hidden_layers], dim=1
    )

    cluster_map = defaultdict(list)

    def _fit_nmf(activations_np, k):
        if activations_np.shape[0] > nmf_subsample:
            rng = np.random.default_rng(42)
            idx = rng.choice(activations_np.shape[0], size=nmf_subsample, replace=False)
            fit_data = activations_np[idx]
        else:
            fit_data = activations_np
        nmf = NMF(n_components=k, random_state=42, max_iter=500)
        nmf.fit(fit_data)
        return nmf.components_  # [k, n_neurons]

    if mode == 'full':
        H = _fit_nmf(all_neuron_activations.numpy(), n_clusters)
        for neuron_idx, cluster_id in enumerate(H.argmax(axis=0)):
            cluster_map[int(cluster_id) + 1].append(neuron_idx)

    elif mode == 'per_layer':
        cluster_offset = 0
        for layer_name, start, end in layer_mapping:
            layer_acts = layer_data[layer_name]['post_activation'].numpy()
            k = min(n_clusters, layer_acts.shape[1])
            H = _fit_nmf(layer_acts, k)
            for local_idx, local_cluster in enumerate(H.argmax(axis=0)):
                global_id = cluster_offset + int(local_cluster) + 1
                cluster_map[global_id].append(start + local_idx)
            cluster_offset += k

    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'full' or 'per_layer'.")

    total_neurons = sum(end - start for _, start, end in layer_mapping)
    print(f"Clustered {total_neurons} neurons into {len(cluster_map)} clusters (mode='{mode}')")

    return cluster_map, layer_mapping, all_neuron_activations

def cluster_neurons_fabio(layer_data, min_clusters=2, max_clusters=10, nmf_subsample=15000, weight_threshold=1e-10):
    from sklearn.decomposition import NMF
    from scipy.sparse.csgraph import connected_components

    hidden_layers = [k for k in layer_data.keys() if 'layer_' in k]

    layer_mapping = []
    start_idx = 0
    for layer_name in hidden_layers:
        n_neurons = layer_data[layer_name]['post_activation'].shape[1]
        layer_mapping.append((layer_name, start_idx, start_idx + n_neurons))
        start_idx += n_neurons

    all_neuron_activations = torch.cat(
        [layer_data[ln]['post_activation'] for ln in hidden_layers], dim=1
    )

    adj = _build_neuron_adjacency(layer_data, layer_mapping, weight_threshold)
    n_comp, labels = connected_components(adj, directed=False, return_labels=True)

    component_to_neurons = defaultdict(list)
    for neuron_idx, comp_id in enumerate(labels):
        component_to_neurons[int(comp_id)].append(neuron_idx)

    acts_np = all_neuron_activations.numpy()
    comp_acts = np.stack(
        [acts_np[:, component_to_neurons[c]].mean(axis=1) for c in range(n_comp)],
        axis=1
    )  # [N_samples, n_comp]

    if comp_acts.shape[0] > nmf_subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(comp_acts.shape[0], size=nmf_subsample, replace=False)
        fit_data = comp_acts[idx]
    else:
        fit_data = comp_acts

    n_nmf = min(max_clusters, n_comp, fit_data.shape[0])
    nmf = NMF(n_components=n_nmf, random_state=42, max_iter=500)
    W = nmf.fit_transform(fit_data)  # [N_samples_sub, n_nmf]
    H = nmf.components_              # [n_nmf, n_comp]

    contribs = np.array([
        np.linalg.norm(W[:, k]) * np.linalg.norm(H[k, :])
        for k in range(n_nmf)
    ])
    fracs = contribs / (contribs.sum() + 1e-12)
    threshold = 1.0 / (2 * n_nmf)
    surviving = np.where(fracs > threshold)[0]
    k_found = len(surviving)

    if k_found < min_clusters:
        raise ValueError(
            f"pruning did not isolate enough components, found {k_found}, "
            f"expected at least {min_clusters}. Prune more aggressively."
        )
    if k_found > max_clusters:
        raise ValueError(
            "found more clusters than max_clusters, raise max_clusters or prune more aggressively"
        )

    H_surviving = H[surviving, :]          # [k_found, n_comp]
    comp_assignments = H_surviving.argmax(axis=0)  # [n_comp] -> index into surviving

    cluster_map = defaultdict(list)
    for comp_id in range(n_comp):
        cluster_id = int(comp_assignments[comp_id]) + 1  # 1-indexed
        cluster_map[cluster_id].extend(component_to_neurons[comp_id])

    print(f"Found {k_found} clusters from {n_comp} structural components.")
    return cluster_map, layer_mapping, all_neuron_activations


def split_clusters_by_layer(cluster_map, layer_mapping):
    """
    Groups a flat cluster_map by layer. Only meaningful after per_layer clustering,
    where each cluster's neurons all belong to a single layer.

    Args:
        cluster_map:   {cluster_id: [global_neuron_indices]}
        layer_mapping: [(layer_name, start_idx, end_idx), ...]

    Returns:
        {layer_name: {cluster_id: [global_neuron_indices]}}
    """
    result = {layer_name: {} for layer_name, _, _ in layer_mapping}

    for cluster_id, neuron_indices in cluster_map.items():
        # In per_layer mode all neurons in a cluster share the same layer
        global_idx = neuron_indices[0]
        for layer_name, start, end in layer_mapping:
            if start <= global_idx < end:
                result[layer_name][cluster_id] = neuron_indices
                break

    return result


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