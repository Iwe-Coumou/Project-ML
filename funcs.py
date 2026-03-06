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

def remove_dead_neurons(model):
    """
    Remove neurons whose incoming AND outgoing connections are all zero.
    These nodes receive no signal and contribute nothing — gradient cannot reach them.
    Returns the total number of neurons removed.
    """
    linear_indices = [i for i, l in enumerate(model.layer_stack) if isinstance(l, torch.nn.Linear)]
    # Skip layer_0 (connects raw pixels → first hidden) and the output layer.
    # Dead-neuron detection requires both incoming (from a hidden layer) and outgoing weights.
    total_removed = 0

    for pos in range(1, len(linear_indices) - 1):
        layer_idx      = linear_indices[pos]
        next_layer_idx = linear_indices[pos + 1]
        layer      = model.layer_stack[layer_idx]
        next_layer = model.layer_stack[next_layer_idx]

        W_in  = layer.weight.data       # [out, in]  — rows are this layer's neurons
        W_out = next_layer.weight.data  # [next, out] — cols are this layer's neurons

        in_dead  = W_in.abs().sum(dim=1) == 0
        out_dead = W_out.abs().sum(dim=0) == 0
        dead     = in_dead & out_dead

        if not dead.any():
            continue

        keep_idx  = (~dead).nonzero(as_tuple=True)[0]
        n_removed = dead.sum().item()

        layer.weight.data  = layer.weight.data[keep_idx, :]
        layer.bias.data    = layer.bias.data[keep_idx]
        layer.out_features = layer.weight.shape[0]

        next_layer.weight.data = next_layer.weight.data[:, keep_idx]
        next_layer.in_features = next_layer.weight.shape[1]

        if hasattr(model, 'connection_masks'):
            if layer_idx in model.connection_masks:
                model.connection_masks[layer_idx] = model.connection_masks[layer_idx][keep_idx, :]
            if next_layer_idx in model.connection_masks:
                model.connection_masks[next_layer_idx] = model.connection_masks[next_layer_idx][:, keep_idx]

        total_removed += n_removed
        print(f"  Removed {n_removed} dead neurons from layer_{pos}")

    return total_removed


def remove_unreachable_neurons(model):
    """
    Remove primitive-layer neurons (layer_1 … layer_{n-1}) that are globally unreachable:
    not reachable from layer_0 outputs (forward path) OR not connected to the output layer
    (backward path). A neuron must appear in BOTH sets to be kept.

    Operates only on positions 1 … n_lin-2 in linear_indices (primitive layers).
    Never touches layer_0 or the output layer.
    Returns the total number of neurons removed.
    """
    linear_indices = [i for i, l in enumerate(model.layer_stack) if isinstance(l, torch.nn.Linear)]
    n_lin = len(linear_indices)
    if n_lin < 3:
        return 0  # need at least layer_0, one primitive layer, and output

    prim_positions = list(range(1, n_lin - 1))  # positions inside linear_indices

    # --- Forward reachability ---
    # fwd[pp] = bool tensor: which neurons in primitive pos pp receive signal from layer_0
    fwd = {}
    W_entry = model.layer_stack[linear_indices[prim_positions[0]]].weight.data  # [prim1, L0]
    fwd[prim_positions[0]] = W_entry.abs().sum(dim=1) > 0  # which prim1 neurons have non-zero input

    for i in range(1, len(prim_positions)):
        pp      = prim_positions[i]
        pp_prev = prim_positions[i - 1]
        W = model.layer_stack[linear_indices[pp]].weight.data  # [curr, prev]
        prev_reach = fwd[pp_prev]
        fwd[pp] = (W[:, prev_reach].abs().sum(dim=1) > 0)

    # --- Backward reachability ---
    # bwd[pp] = bool tensor: which neurons in primitive pos pp connect to the output layer
    bwd = {}
    W_exit = model.layer_stack[linear_indices[-1]].weight.data  # [classes, last_prim]
    bwd[prim_positions[-1]] = W_exit.abs().sum(dim=0) > 0

    for i in range(len(prim_positions) - 2, -1, -1):
        pp      = prim_positions[i]
        pp_next = prim_positions[i + 1]
        W = model.layer_stack[linear_indices[pp_next]].weight.data  # [next, curr]
        next_reach = bwd[pp_next]
        bwd[pp] = (W[next_reach, :].abs().sum(dim=0) > 0)

    # --- Remove unreachable neurons from each primitive layer ---
    total_removed = 0
    counts = {}

    for pp in prim_positions:
        valid = fwd[pp] & bwd[pp]
        n_invalid = int((~valid).sum().item())
        if n_invalid == 0:
            continue

        keep = valid.nonzero(as_tuple=True)[0]
        layer_idx      = linear_indices[pp]
        next_layer_idx = linear_indices[pp + 1]  # always valid: pp <= n_lin-2, so pp+1 <= n_lin-1
        layer      = model.layer_stack[layer_idx]
        next_layer = model.layer_stack[next_layer_idx]

        # Remove rows (output neurons) from this layer
        layer.weight.data  = layer.weight.data[keep, :]
        layer.bias.data    = layer.bias.data[keep]
        layer.out_features = layer.weight.shape[0]

        # Remove corresponding columns (inputs) from the next layer
        next_layer.weight.data = next_layer.weight.data[:, keep]
        next_layer.in_features = next_layer.weight.shape[1]

        # Update connection masks
        if hasattr(model, 'connection_masks'):
            if layer_idx in model.connection_masks:
                model.connection_masks[layer_idx] = model.connection_masks[layer_idx][keep, :]
            if next_layer_idx in model.connection_masks:
                model.connection_masks[next_layer_idx] = model.connection_masks[next_layer_idx][:, keep]

        counts[f'layer_{pp}'] = n_invalid
        total_removed += n_invalid

    if total_removed:
        counts_str = ', '.join(f'{ln}: {n}' for ln, n in counts.items())
        print(f"  Removed {total_removed} unreachable neurons ({counts_str})")

    return total_removed


def cluster_neurons_correlation(layer_data, max_clusters, fixed_k=None):
    """
    Cluster neurons using absolute Pearson correlation as similarity.

    k is chosen automatically via silhouette scoring unless fixed_k is provided
    (used by the pruning loop to avoid re-running the expensive search every round).

    Args:
        layer_data:   dict from get_layer_data(), keys like 'layer_1', 'layer_2', ...
                      Must already exclude layer_0 (primitive layers only).
        max_clusters: upper bound on the number of clusters to return.
        fixed_k:      if not None, skip silhouette search and use this k directly.

    Returns:
        cluster_map:            {cluster_id: [global_neuron_indices]}  (1-indexed)
        layer_mapping:          [(layer_name, start_idx, end_idx), ...]
        all_neuron_activations: tensor [N_samples, total_neurons]
        best_sil:               silhouette score of the chosen k, or None if fixed_k was used
    """
    if 'layer_0' in layer_data:
        raise ValueError("cluster_neurons_correlation: layer_0 must be excluded from layer_data before calling this function")

    from sklearn.cluster import AgglomerativeClustering

    hidden_layers = sorted(k for k in layer_data if 'layer_' in k)

    layer_mapping = []
    start_idx = 0
    for layer_name in hidden_layers:
        n = layer_data[layer_name]['post_activation'].shape[1]
        layer_mapping.append((layer_name, start_idx, start_idx + n))
        start_idx += n

    all_acts = torch.cat(
        [layer_data[ln]['post_activation'] for ln in hidden_layers], dim=1
    )
    acts_np = all_acts.numpy()  # [N_samples, total_neurons]
    n_neurons = acts_np.shape[1]

    # Identify constant-activation neurons (zero variance → NaN in corrcoef).
    # Exclude them from the distance matrix; assign them post-hoc to the nearest cluster.
    stds = acts_np.std(axis=0)
    active_mask = stds > 1e-8                         # True = has signal
    active_idx  = np.where(active_mask)[0]            # global indices of informative neurons
    const_idx   = np.where(~active_mask)[0]
    n_active    = len(active_idx)
    if len(const_idx):
        pct = 100.0 * len(const_idx) / n_neurons
        print(f"  {len(const_idx)}/{n_neurons} constant-activation neurons excluded ({pct:.1f}%)"
              + (" ← HIGH: pruning too aggressive or retraining too short" if pct > 20 else ""))

    acts_active = acts_np[:, active_idx]              # [N_samples, n_active]

    if n_active < 2:
        # Nothing meaningful to cluster — put everything in one group
        cluster_map = {1: list(range(n_neurons))}
        print(f"Correlation clustering: 1 cluster (too few active neurons).")
        return cluster_map, layer_mapping, all_acts, None

    corr = np.corrcoef(acts_active.T)                 # [n_active, n_active]
    np.clip(corr, -1.0, 1.0, out=corr)
    np.nan_to_num(corr, nan=0.0, copy=False)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    np.clip(dist, 0.0, 1.0, out=dist)

    best_sil = None
    if fixed_k is not None and 2 <= fixed_k <= n_active:
        # Use cached k — skip expensive silhouette search
        n_clusters = fixed_k
        print(f"  Using cached k={n_clusters} clusters (silhouette search skipped)")
    elif n_active <= 2:
        n_clusters = n_active
    else:
        from sklearn.metrics import silhouette_score
        k_max = min(max_clusters, n_active - 1)  # silhouette needs at least 2 clusters < n_samples
        best_k, best_sil = 2, -1.0
        for k_try in range(2, k_max + 1):
            labels_try = AgglomerativeClustering(
                n_clusters=k_try, metric='precomputed', linkage='average'
            ).fit_predict(dist)
            if len(set(labels_try)) < 2:
                continue
            sil = silhouette_score(dist, labels_try, metric='precomputed')
            if sil > best_sil:
                best_sil, best_k = sil, k_try
        n_clusters = best_k
        print(f"  Auto-selected k={n_clusters} clusters (silhouette={best_sil:.4f}, max={max_clusters})")

    labels_active = AgglomerativeClustering(
        n_clusters=n_clusters, metric='precomputed', linkage='average'
    ).fit_predict(dist)   # label per active neuron

    # Build initial cluster_map using only active neurons (1-indexed cluster ids)
    cluster_map = defaultdict(list)
    for pos, global_idx in enumerate(active_idx):
        cluster_map[int(labels_active[pos]) + 1].append(int(global_idx))

    # Assign constant neurons to the cluster with nearest mean activation
    if len(const_idx) and len(cluster_map):
        cluster_means = {
            cid: acts_np[:, ns].mean(axis=1)  # [N_samples]
            for cid, ns in cluster_map.items()
        }
        for gi in const_idx:
            neuron_act = acts_np[:, gi]  # constant → same value; L2 dist still works
            best_cid = min(cluster_means,
                           key=lambda cid: float(np.linalg.norm(neuron_act - cluster_means[cid])))
            cluster_map[best_cid].append(int(gi))

    cluster_map = dict(cluster_map)
    print(f"Correlation clustering: {n_clusters} clusters across {n_neurons} neurons "
          f"({', '.join(f'C{k}:{len(v)}' for k, v in sorted(cluster_map.items()))})")
    return cluster_map, layer_mapping, all_acts, best_sil


def _cluster_quality(cluster_map, acts_np):
    """Ratio of mean within-cluster |correlation| to mean cross-cluster |correlation|."""
    corr = np.corrcoef(acts_np.T)
    np.clip(corr, -1.0, 1.0, out=corr)
    np.nan_to_num(corr, nan=0.0, copy=False)
    corr_abs = np.abs(corr)
    n = corr_abs.shape[0]
    # Use -1 sentinel so neurons not covered by cluster_map are excluded from both w and a.
    # np.empty would leave garbage that silently corrupts the ratio.
    labels = np.full(n, -1, dtype=np.int32)
    for cid, ns in cluster_map.items():
        for idx in ns:
            if idx < n:
                labels[idx] = cid

    assigned = labels >= 0
    valid    = assigned[:, None] & assigned[None, :]   # both neurons must be assigned
    triu     = np.triu(np.ones((n, n), dtype=bool), k=1)
    same     = labels[:, None] == labels[None, :]
    w_mask   = same  & valid & triu
    a_mask   = ~same & valid & triu
    w = corr_abs[w_mask].mean() if w_mask.any() else 0.0
    a = corr_abs[a_mask].mean() if a_mask.any() else 1.0
    return float(w) / (float(a) + 1e-8)


def _match_cluster_ids(old_map, new_map):
    """
    Remap new_map cluster IDs to match old_map's numbering by maximum neuron overlap
    (Hungarian algorithm). Keeps cluster identity stable across pruning rounds.
    """
    from scipy.optimize import linear_sum_assignment

    old_ids = sorted(old_map.keys())
    new_ids = sorted(new_map.keys())

    cost = np.zeros((len(old_ids), len(new_ids)))
    for i, oid in enumerate(old_ids):
        old_set = set(old_map[oid])
        for j, nid in enumerate(new_ids):
            cost[i, j] = -len(old_set & set(new_map[nid]))

    row_ind, col_ind = linear_sum_assignment(cost)
    id_remap = {new_ids[c]: old_ids[r] for r, c in zip(row_ind, col_ind)}

    next_fresh = max(old_ids) + 1
    remapped = {}
    for nid, neurons in new_map.items():
        mapped = id_remap.get(nid, next_fresh)
        if mapped == next_fresh:
            next_fresh += 1
        remapped[mapped] = neurons
    return remapped


def _accept_new_clustering(old_map, new_map, acts_np, min_q=1.0):
    """
    Log topology change fraction and cluster quality, then gate on Q.
    Q_old is NOT computed — old_map indices are stale after neuron pruning.
    Returns True (accept, cut cross-cluster connections) if Q_new >= min_q.
    """
    old_assign = {n: cid for cid, ns in old_map.items() for n in ns}
    new_assign = {n: cid for cid, ns in new_map.items() for n in ns}
    common = set(old_assign) & set(new_assign)
    if common:
        changed = sum(1 for n in common if old_assign[n] != new_assign[n])
        print(f"  Topology change: {changed / len(common):.1%}", end="  ")
    q_new = _cluster_quality(new_map, acts_np)
    print(f"Q={q_new:.4f}", end="")
    if q_new >= min_q:
        print(f" ≥ {min_q} — accepting.")
        return True
    print(f" < {min_q} — rejecting.")
    return False


def _prune_connections_native(model, prune_frac, layer_data=None):
    """Prune connections among primitive layers (skip pixel→layer_0 and output).

    Importance per connection (i→j): |W[j,i]| * mean(|a_i|) if activations are available,
    else |W[j,i]|.  Only non-zero connections are candidates — previously-zeroed connections
    are never re-selected, so the pruning budget always removes new connections each round.
    """
    linear_indices = [i for i, l in enumerate(model.layer_stack)
                      if isinstance(l, torch.nn.Linear)]
    if not hasattr(model, 'connection_masks'):
        model.connection_masks = {}

    for j, idx in enumerate(linear_indices[1:-1], start=1):
        # j = position in linear_indices (1-based); source layer = layer_{j-1} in layer_data
        layer = model.layer_stack[idx]
        W = layer.weight.data  # [out, in]

        # Activation-scaled importance: weight × mean absolute activation of source neuron
        if layer_data is not None:
            src_key = f'layer_{j - 1}'
            if src_key in layer_data:
                mean_act = layer_data[src_key]['post_activation'].abs().mean(dim=0)  # [in]
                if mean_act.shape[0] == W.shape[1]:  # activations match current weight shape
                    importance = W.abs() * mean_act.unsqueeze(0)
                else:  # activations are stale (neurons were pruned after layer_data was collected)
                    importance = W.abs()
            else:
                importance = W.abs()
        else:
            importance = W.abs()

        # Prune only among currently non-zero connections so the budget is never wasted
        # on weights that are already zero from previous rounds
        nonzero_mask = W != 0
        n_nonzero = nonzero_mask.sum().item()
        if n_nonzero == 0:
            continue
        n_prune = max(1, int(prune_frac * n_nonzero))

        scores = importance[nonzero_mask]
        threshold = torch.kthvalue(scores, min(n_prune, scores.numel()))[0]
        W.data[(importance <= threshold) & nonzero_mask] = 0.0
        model.connection_masks[idx] = (W != 0).float()


def _neuron_to_cluster_lookup(cluster_map, layer_mapping):
    """Build {(layer_name, local_idx): cluster_id} from cluster_map + layer_mapping."""
    lookup = {}
    for cid, neurons in cluster_map.items():
        for global_idx in neurons:
            for (lname, start, end) in layer_mapping:
                if start <= global_idx < end:
                    lookup[(lname, global_idx - start)] = cid
                    break
    return lookup


def _cluster_label_tensors(cluster_map, layer_mapping):
    """
    Build per-layer 1-D tensors of cluster IDs (dtype=int32, -1 = unassigned).
    Size of each tensor matches the layer's span in layer_mapping.
    """
    result = {}
    for lname, start, end in layer_mapping:
        result[lname] = torch.full((end - start,), -1, dtype=torch.int32)
    for cid, neurons in cluster_map.items():
        for gi in neurons:
            for lname, start, end in layer_mapping:
                if start <= gi < end:
                    local = gi - start
                    if local < result[lname].shape[0]:
                        result[lname][local] = cid
                    break
    return result


def _guard_layer_mapping(fn_name, layer_mapping, linear_indices):
    """Raise if layer_0 or the output layer appear in layer_mapping."""
    _max_prim = len(linear_indices) - 2
    for _lname, _, _ in layer_mapping:
        _n = int(_lname.split('_')[1])
        if _n == 0:
            raise ValueError(f"{fn_name}: layer_0 must not appear in layer_mapping")
        if _n > _max_prim:
            raise ValueError(f"{fn_name}: output layer must not appear in layer_mapping (got layer_{_n})")


def prune_cross_cluster_connections(model, cluster_map, layer_mapping, frac):
    """
    Gradually cut cross-cluster connections between primitive layers.
    Cuts the weakest `frac` fraction of active cross-cluster connections per layer pair.
    Records cuts in connection_masks (permanently zeroed during retraining).
    """
    linear_indices = [i for i, l in enumerate(model.layer_stack) if isinstance(l, torch.nn.Linear)]
    _guard_layer_mapping('prune_cross_cluster_connections', layer_mapping, linear_indices)
    if not hasattr(model, 'connection_masks'):
        model.connection_masks = {}

    lbl = _cluster_label_tensors(cluster_map, layer_mapping)

    for k in range(len(layer_mapping) - 1):
        src_name, _, _ = layer_mapping[k]
        dst_name, _, _ = layer_mapping[k + 1]

        dst_layer_num   = int(dst_name.split('_')[1])
        layer_stack_idx = linear_indices[dst_layer_num]
        layer = model.layer_stack[layer_stack_idx]
        W     = layer.weight.data  # [dst_n, src_n]

        mask = model.connection_masks.get(layer_stack_idx, torch.ones_like(W))

        src_lbl = lbl.get(src_name)
        dst_lbl = lbl.get(dst_name)
        if src_lbl is None or dst_lbl is None:
            continue

        # Trim to actual weight matrix size (guard against stale layer_mapping)
        # Move to same device as W (label tensors are built on CPU)
        src_lbl = src_lbl[:W.shape[1]].to(W.device)
        dst_lbl = dst_lbl[:W.shape[0]].to(W.device)

        # Boolean cross-cluster matrix: True where both neurons are assigned AND in different clusters
        cross = (
            (dst_lbl[:, None] != src_lbl[None, :]) &
            (dst_lbl[:, None] >= 0) &
            (src_lbl[None, :] >= 0) &
            (mask > 0.5)
        )  # [dst_n, src_n]

        cross_j, cross_i = cross.nonzero(as_tuple=True)
        n_cross = cross_j.shape[0]
        if n_cross == 0:
            continue

        # Sort by ascending absolute weight, cut weakest first
        abs_w = W[cross_j, cross_i].abs()
        order = abs_w.argsort()
        n_cut = max(1, int(frac * n_cross))
        cut_j = cross_j[order[:n_cut]]
        cut_i = cross_i[order[:n_cut]]

        W[cut_j, cut_i]    = 0.0
        mask[cut_j, cut_i] = 0.0

        model.connection_masks[layer_stack_idx] = mask
        print(f"  Cut {n_cut}/{n_cross} cross-cluster connections ({src_name}→{dst_name})")


def cut_all_cross_cluster_connections(model, cluster_map, layer_mapping):
    """Cut ALL cross-cluster connections between primitive layers (final isolation)."""
    linear_indices = [i for i, l in enumerate(model.layer_stack) if isinstance(l, torch.nn.Linear)]
    _guard_layer_mapping('cut_all_cross_cluster_connections', layer_mapping, linear_indices)
    if not hasattr(model, 'connection_masks'):
        model.connection_masks = {}

    lbl = _cluster_label_tensors(cluster_map, layer_mapping)
    total_cut = 0

    for k in range(len(layer_mapping) - 1):
        src_name, _, _ = layer_mapping[k]
        dst_name, _, _ = layer_mapping[k + 1]

        dst_layer_num   = int(dst_name.split('_')[1])
        layer_stack_idx = linear_indices[dst_layer_num]
        layer = model.layer_stack[layer_stack_idx]
        W     = layer.weight.data

        mask = model.connection_masks.get(layer_stack_idx, torch.ones_like(W))

        src_lbl = lbl.get(src_name)
        dst_lbl = lbl.get(dst_name)
        if src_lbl is None or dst_lbl is None:
            continue

        src_lbl = src_lbl[:W.shape[1]].to(W.device)
        dst_lbl = dst_lbl[:W.shape[0]].to(W.device)

        cross = (
            (dst_lbl[:, None] != src_lbl[None, :]) &
            (dst_lbl[:, None] >= 0) &
            (src_lbl[None, :] >= 0)
        )  # [dst_n, src_n]

        total_cut += int(cross.sum().item())
        W[cross]    = 0.0
        mask[cross] = 0.0

        model.connection_masks[layer_stack_idx] = mask

    print(f"Final isolation: cut {total_cut} cross-cluster connections.")


def _measure_cluster_ablation(model, cluster_map, layer_mapping, val_loader, device):
    """
    Measure each cluster's contribution to accuracy via ablation.
    Returns {cluster_id: accuracy_drop}  (positive = cluster was useful).
    """
    model.eval()
    linear_indices = [i for i, l in enumerate(model.layer_stack) if isinstance(l, torch.nn.Linear)]
    _guard_layer_mapping('_measure_cluster_ablation', layer_mapping, linear_indices)

    correct = total = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(dim=1) == y).sum().item()
            total   += y.size(0)
    baseline = correct / total

    drops = {}
    for cluster_id, neuron_indices in cluster_map.items():
        backups = {}
        for lname, start, end in layer_mapping:
            layer_num  = int(lname.split('_')[1])
            layer_idx  = linear_indices[layer_num]
            layer      = model.layer_stack[layer_idx]
            local_idxs = [gi - start for gi in neuron_indices if start <= gi < end]
            if not local_idxs:
                continue
            backups[layer_idx] = {
                'w': layer.weight.data[local_idxs, :].clone(),
                'b': layer.bias.data[local_idxs].clone(),
                'idxs': local_idxs,
            }
            layer.weight.data[local_idxs, :] = 0
            layer.bias.data[local_idxs]      = 0
            if layer_num + 1 < len(linear_indices):
                next_layer = model.layer_stack[linear_indices[layer_num + 1]]
                backups[layer_idx]['next_w'] = next_layer.weight.data[:, local_idxs].clone()
                next_layer.weight.data[:, local_idxs] = 0

        correct_abl = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                correct_abl += (model(X).argmax(dim=1) == y).sum().item()
        ablated = correct_abl / total

        for layer_idx, bk in backups.items():
            layer = model.layer_stack[layer_idx]
            layer.weight.data[bk['idxs'], :] = bk['w']
            layer.bias.data[bk['idxs']]      = bk['b']
            layer_num = next((i for i, li in enumerate(linear_indices) if li == layer_idx), None)
            if layer_num is None:
                continue
            if layer_num + 1 < len(linear_indices) and 'next_w' in bk:
                model.layer_stack[linear_indices[layer_num + 1]].weight.data[:, bk['idxs']] = bk['next_w']

        drops[cluster_id] = baseline - ablated

    return drops


def error_driven_regrowth(model, cluster_map, layer_mapping, val_loader,
                          threshold_frac, device):
    """
    Add one neuron to each underperforming cluster (drop < mean_drop / threshold_frac).

    Weight initialisation uses an Exponential distribution scaled by mean network weight,
    then thresholded (weights below epsilon*max zeroed). Sparsity is emergent, not fixed.
    Only within-cluster connections are allowed; cross-cluster stay zero.

    Returns number of neurons added.
    """
    linear_indices = [i for i, l in enumerate(model.layer_stack) if isinstance(l, torch.nn.Linear)]
    _guard_layer_mapping('error_driven_regrowth', layer_mapping, linear_indices)
    drops     = _measure_cluster_ablation(model, cluster_map, layer_mapping, val_loader, device)
    mean_drop = float(np.mean(list(drops.values()))) if drops else 0.0
    threshold = mean_drop / threshold_frac

    all_w = torch.cat([
        model.layer_stack[li].weight.data.abs().view(-1) for li in linear_indices
    ])
    mean_w  = all_w[all_w > 0].mean().item() if (all_w > 0).any() else 1e-3
    epsilon = 0.1

    ntc          = _neuron_to_cluster_lookup(cluster_map, layer_mapping)
    layer_added  = defaultdict(int)  # lname → neurons added this call (index offset)
    n_added = 0

    for cluster_id, drop in drops.items():
        if drop >= threshold:
            continue

        neurons = cluster_map[cluster_id]
        print(f"  Cluster {cluster_id}: drop={drop:.4f} < {threshold:.4f} → adding neuron")

        # Find primitive layer with most of this cluster's neurons
        layer_counts = defaultdict(list)
        for gi in neurons:
            for lname, start, end in layer_mapping:
                if start <= gi < end:
                    layer_counts[lname].append(gi - start)
                    break
        if not layer_counts:
            continue
        target_lname    = max(layer_counts, key=lambda ln: len(layer_counts[ln]))
        target_layer_num = int(target_lname.split('_')[1])
        layer_idx        = linear_indices[target_layer_num]
        next_layer_idx   = linear_indices[target_layer_num + 1]
        layer            = model.layer_stack[layer_idx]
        next_layer       = model.layer_stack[next_layer_idx]
        dev              = layer.weight.device
        in_features      = layer.weight.size(1)
        out_features_next = next_layer.weight.size(0)

        def _expo_sparse(allowed_indices, size, rng_key=None):
            """Sample Exponential weights for allowed indices, threshold the rest to 0."""
            vec = torch.zeros(size, device=dev)
            if not allowed_indices:
                return vec
            raw = torch.from_numpy(
                np.random.exponential(scale=mean_w, size=len(allowed_indices)).astype(np.float32)
            ).to(dev)
            signs = (torch.rand(len(allowed_indices), device=dev) * 2 - 1).sign()
            raw  *= signs
            cutoff = epsilon * raw.abs().max().item()
            for k_i, src_i in enumerate(allowed_indices):
                if raw[k_i].abs().item() >= cutoff:
                    vec[src_i] = raw[k_i]
            return vec

        # Incoming: from same-cluster neurons in the previous primitive layer
        prev_lname = f'layer_{target_layer_num - 1}'
        same_in    = [
            local_i for (ln, local_i), cid in ntc.items()
            if ln == prev_lname and cid == cluster_id
        ] if target_layer_num > 0 else []
        allowed_in = same_in if same_in else list(range(in_features))
        new_row = _expo_sparse(allowed_in, in_features)

        layer.weight.data  = torch.cat([layer.weight.data, new_row.unsqueeze(0)], dim=0)
        layer.bias.data    = torch.cat([layer.bias.data, torch.zeros(1, device=dev)], dim=0)
        layer.out_features = layer.weight.shape[0]

        # Outgoing: to same-cluster neurons in the next primitive layer
        next_lname = f'layer_{target_layer_num + 1}'
        same_out   = [
            local_i for (ln, local_i), cid in ntc.items()
            if ln == next_lname and cid == cluster_id
        ]
        allowed_out = same_out if same_out else list(range(out_features_next))
        new_col = _expo_sparse(allowed_out, out_features_next)

        next_layer.weight.data = torch.cat([next_layer.weight.data, new_col.unsqueeze(1)], dim=1)
        next_layer.in_features = next_layer.weight.shape[1]

        # Update masks
        if not hasattr(model, 'connection_masks'):
            model.connection_masks = {}
        if layer_idx in model.connection_masks:
            model.connection_masks[layer_idx] = torch.cat(
                [model.connection_masks[layer_idx], (new_row != 0).float().unsqueeze(0)], dim=0
            )
        if next_layer_idx in model.connection_masks:
            model.connection_masks[next_layer_idx] = torch.cat(
                [model.connection_masks[next_layer_idx], (new_col != 0).float().unsqueeze(1)], dim=1
            )

        # Register new neuron in cluster_map.
        # Use end2 + layer_added[target_lname] so that if multiple clusters grow in the same
        # layer this round each gets the correct unique global index.
        for lname2, start2, end2 in layer_mapping:
            if lname2 == target_lname:
                new_global_idx = end2 + layer_added[target_lname]
                cluster_map[cluster_id].append(new_global_idx)
                layer_added[target_lname] += 1
                break

        n_added += 1

    return n_added


def _should_cut(model, cluster_map, layer_mapping, train_loader, min_q=1.0):
    """Return True only if cluster quality is high enough to justify a final isolation cut."""
    layer_data = model.get_layer_data(train_loader)
    prim = {k: v for k, v in layer_data.items() if k != 'layer_0'}
    if not prim:
        return False
    acts = torch.cat([prim[ln]['post_activation'] for ln in sorted(prim)], dim=1).numpy()
    q = _cluster_quality(cluster_map, acts)
    print(f"  Final cluster Q={q:.4f}", end="")
    if q > min_q:
        print(" — proceeding with isolation cut.")
        return True
    print(" — Q too low, skipping isolation cut.")
    return False


def _finalise(model, loader_to_use, train_loader, lr, n_final_retrain_epochs, max_clusters, diag_loader=None):
    """
    Post-pruning finalisation:
      1. Run cluster_neurons_fabio once on the compressed model to discover structure.
      2. Cut ALL cross-cluster connections based on those structural components.
      3. Final retrain.
    """
    print(f"\nFinalising: discovering structure in compressed model...")
    layer_data = model.get_layer_data(diag_loader if diag_loader is not None else train_loader)
    prim_layer_data = {k: v for k, v in layer_data.items() if k != 'layer_0'}

    cluster_map = None
    layer_mapping = None
    if prim_layer_data:
        try:
            cluster_map, layer_mapping, _ = cluster_neurons_fabio(
                prim_layer_data, min_clusters=1, max_clusters=max_clusters
            )
            print(f"  Found {len(cluster_map)} structural cluster(s).")
            for cid in sorted(cluster_map):
                print(f"    Cluster {cid}: {len(cluster_map[cid])} neurons")
        except Exception as e:
            print(f"  cluster_neurons_fabio failed ({e}), skipping isolation cut.")

    if cluster_map is not None and len(cluster_map) > 1:
        print("  Cutting all cross-cluster connections...")
        cut_all_cross_cluster_connections(model, cluster_map, layer_mapping)
    else:
        print("  Single component or no clusters — skipping isolation cut.")

    # Final reachability pass: remove any neurons with no path from input to output
    # (ghost neurons left by connection pruning that cut all their in- or out-connections)
    n_ghost = remove_unreachable_neurons(model)
    if n_ghost:
        print(f"  Removed {n_ghost} ghost neuron(s) with no input→output path.")

    model.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Final retraining ({n_final_retrain_epochs} epochs)...")
    model.train_model(loader_to_use, epochs=n_final_retrain_epochs, lr=lr, val_interval=1, patience=20)
    return model


def pruning(model, train_loader, val_loader, parameters, baseline_acc,
            use_max_rounds=True, lr=0.01, mode="full", min_width=5, fresh_loader=None,
            max_clusters=10, cross_cluster_prune_frac=0.3, topology_threshold=0.15,
            error_threshold_frac=1.5, n_final_retrain_epochs=15,
            phase2_min_neurons=150, phase2_min_connections=5000):
    max_rounds, prune_frac, prune_con_frac, regrow_frac, retrain_epochs, max_acc_drop = parameters

    current_model = copy.deepcopy(model)
    val_acc = baseline_acc
    round_idx = 0
    current_cluster_map = None
    current_layer_mapping = None
    in_phase2 = False

    if fresh_loader is not None:
        mixed = ConcatDataset([train_loader.dataset, fresh_loader.dataset])
        loader_to_use = DataLoader(mixed, batch_size=train_loader.batch_size, shuffle=True)
    else:
        loader_to_use = train_loader

    # Fixed random subset for fast activation collection (importance + clustering).
    # 2000 samples is sufficient for both neuron importance ranking and structural clustering.
    _diag_n = min(2000, len(train_loader.dataset))
    _diag_idx = torch.randperm(len(train_loader.dataset))[:_diag_n].tolist()
    _diag_subset = torch.utils.data.Subset(train_loader.dataset, _diag_idx)
    diag_loader = DataLoader(_diag_subset, batch_size=_diag_n)

    while True:
        round_idx += 1
        print(f"\n--- Pruning round {round_idx} ---")

        if use_max_rounds and round_idx > max_rounds:
            print("Reached maximum pruning rounds.")
            break

        prev_model = copy.deepcopy(current_model)

        # 1. Compute neuron importance
        layer_data_all = current_model.get_layer_data(diag_loader)
        importance_scores = current_model.compute_neuron_importance(layer_data=layer_data_all,
                                                                    type='downstream_blend')

        # 2. Prune hidden neurons
        prune_counts = None
        if mode in ["full", "neuron_only", "prune_only"]:
            prune_counts = current_model.prune_hidden_neurons(
                importance_scores=importance_scores,
                prune_rate=prune_frac,
            )
        if prune_counts:
            linear_layers_tmp = [l for l in current_model.layer_stack if isinstance(l, torch.nn.Linear)]
            sizes_after = " → ".join(str(l.weight.shape[0]) for l in linear_layers_tmp[:-1])
            pruned_str = ", ".join(f"layer_{k}: -{v}" for k, v in prune_counts.items())
            print(f"Pruned neurons: {pruned_str}  →  [{sizes_after}]")

        # 3. Prune connections (skip pixel→layer_0 and output layer by index)
        if mode in ["full", "connections_only", "prune_only"]:
            _prune_connections_native(current_model, prune_con_frac, layer_data=layer_data_all)

        # 4. Remove globally unreachable neurons
        n_dead = remove_unreachable_neurons(current_model)
        if n_dead:
            print(f"  Total unreachable neurons removed: {n_dead}")

        # Phase switch check — runs once, then in_phase2 stays True permanently
        linear_layers = [l for l in current_model.layer_stack if isinstance(l, torch.nn.Linear)]
        total_neurons = sum(l.out_features for l in linear_layers[:-1])
        total_connections = sum(
            (l.weight.data != 0).sum().item()
            for l in current_model.layer_stack if isinstance(l, torch.nn.Linear)
        )
        if not in_phase2 and (
            total_neurons < phase2_min_neurons or total_connections < phase2_min_connections
        ):
            in_phase2 = True
            print(f"--- Switching to Phase 2: structure discovery ---")
            print(f"    (neurons={total_neurons}, connections={total_connections})")
            print("  Running reachability cleanup before phase 2 clustering...")
            remove_unreachable_neurons(current_model)

        # 5. Cluster: structural connected components + NMF  (Phase 2 only)
        acts_np = None
        cluster_found = False   # True only when >1 structural cluster is found

        if in_phase2:
            layer_data_prim = current_model.get_layer_data(diag_loader)
            prim_layer_data = {k: v for k, v in layer_data_prim.items() if k != 'layer_0'}
            if prim_layer_data:
                try:
                    new_cluster_map, new_layer_mapping, new_acts = cluster_neurons_fabio(
                        prim_layer_data, min_clusters=1, max_clusters=max_clusters
                    )
                    acts_np = new_acts.numpy()
                    prev_cluster_map = current_cluster_map   # keep pre-match for topology diagnostic

                    if current_cluster_map is not None:
                        matched = _match_cluster_ids(current_cluster_map, new_cluster_map)
                        current_cluster_map = matched
                    else:
                        current_cluster_map = new_cluster_map
                    current_layer_mapping = new_layer_mapping

                    n_clusters = len(current_cluster_map)
                    sizes_str = ", ".join(
                        f"C{cid}:{len(ns)}" for cid, ns in sorted(current_cluster_map.items())
                    )
                    print(f"  Structural clusters: {n_clusters} — {sizes_str}")
                    cluster_found = n_clusters > 1
                except Exception as e:
                    print(f"  cluster_neurons_fabio failed ({e}) — skipping clustering this round.")
                    prev_cluster_map = current_cluster_map

        # 6. Cut cross-cluster connections — only when >1 cluster AND Q gate passes
        if cluster_found and current_cluster_map is not None and acts_np is not None:
            old_map = prev_cluster_map if prev_cluster_map is not None else current_cluster_map
            if _accept_new_clustering(old_map, current_cluster_map, acts_np):
                prune_cross_cluster_connections(
                    current_model, current_cluster_map, current_layer_mapping,
                    cross_cluster_prune_frac
                )
        elif current_cluster_map is not None and not cluster_found:
            print("  Single cluster — cross-cluster cut skipped.")

        # 7. Retrain
        current_model.optimizer = torch.optim.Adam(current_model.parameters(), lr=lr)
        print("Retraining after pruning...")
        val_acc = current_model.train_model(
            loader_to_use, epochs=retrain_epochs, lr=lr, val_interval=1,
        )

        # 8. Error-driven regrowth — gated on regrow_frac > 0 and >1 cluster found
        if regrow_frac > 0 and cluster_found and current_cluster_map is not None:
            n_added = error_driven_regrowth(
                current_model, current_cluster_map, current_layer_mapping,
                val_loader, error_threshold_frac, current_model.device
            )
            if n_added > 0:
                print(f"  Added {n_added} neurons via error-driven regrowth. Brief retraining...")
                current_model.optimizer = torch.optim.Adam(current_model.parameters(), lr=lr)
                val_acc = current_model.train_model(
                    loader_to_use, epochs=2, lr=lr, val_interval=1,
                )

        # 9. Stopping conditions
        for layer in linear_layers[:-1]:
            if layer.out_features < min_width:
                print("Minimum width reached. Stopping early.")
                return _finalise(current_model, loader_to_use, train_loader, lr,
                                 n_final_retrain_epochs, max_clusters, diag_loader)

        prev_size = sum(p.numel() for p in prev_model.parameters())
        curr_size = sum(p.numel() for p in current_model.parameters())
        if prev_size == curr_size:
            print("Model size unchanged. Converged.")
            break

        total_conn = sum(
            (l.weight.data != 0).sum().item()
            for l in current_model.layer_stack if isinstance(l, torch.nn.Linear)
        )
        print(f"Active connections: {total_conn}")
        if val_acc is not None:
            print(f"Validation accuracy: {val_acc:.4f}")
        if val_acc is not None and baseline_acc - val_acc > max_acc_drop:
            print("Accuracy drop exceeded threshold. Restoring previous model.")
            return prev_model

    return _finalise(current_model, loader_to_use, train_loader, lr,
                     n_final_retrain_epochs, max_clusters)

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
    if 'layer_0' in layer_data:
        raise ValueError("_build_neuron_adjacency: layer_0 must be excluded from layer_data before calling this function")

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
    if 'layer_0' in layer_data:
        raise ValueError("cluster_neurons_fabio: layer_0 must be excluded from layer_data before calling this function")

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
    # Shift per-neuron so minimum is 0 — required for NMF (non-negative input)
    # Preserves anti-correlations: neuron A high ↔ neuron B near-zero becomes visible
    acts_np = acts_np - acts_np.min(axis=0, keepdims=True)
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