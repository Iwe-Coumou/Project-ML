import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict


def draw_network(
    model,
    mode='components',
    cluster_map=None,
    layer_mapping=None,
    ignore_layers=None,
    save_path=None,
    figsize=(14, 8),
):
    """
    Visualise the pruned network with neurons coloured by either connected
    components or NMF clusters.

    Args:
        model:         loaded NeuralNetwork (e.g. torch.load('pruned_model.pth'))
        mode:          'components' — colour by structural connected components (default)
                       'clusters'  — colour by cluster_map from cluster_neurons_fabio
        cluster_map:   {cluster_id: [global_neuron_indices]} from cluster_neurons_fabio
                       required when mode='clusters'
        layer_mapping: [(layer_name, start_idx, end_idx), ...] from cluster_neurons_fabio
                       required when mode='clusters'
        ignore_layers: list of original layer indices to hide (default [0, last])
                       e.g. [0, 4] hides input-side layer 0 and output layer 4
        save_path:     if given, saves the figure to this path
        figsize:       figure size tuple
    """
    if mode == 'clusters' and (cluster_map is None or layer_mapping is None):
        raise ValueError("cluster_map and layer_mapping are required when mode='clusters'")

    linear_layers = [l for l in model.layer_stack if isinstance(l, torch.nn.Linear)]
    all_sizes = [l.weight.shape[0] for l in linear_layers]
    n_total_layers = len(all_sizes)

    if ignore_layers is None:
        ignore_layers = [0, n_total_layers - 1]

    layer_indices = [k for k in range(n_total_layers) if k not in ignore_layers]
    sizes = [all_sizes[k] for k in layer_indices]
    display_to_orig = {d: k for d, k in enumerate(layer_indices)}

    # ── BUILD NEURON COLORS ───────────────────────────────────────────────────
    offsets = [sum(sizes[:d]) for d in range(len(sizes))]
    total = sum(sizes)

    if mode == 'components':
        # Union-Find over displayed neurons
        parent = list(range(total))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for d in range(len(sizes) - 1):
            orig_src = display_to_orig[d]
            orig_dst = display_to_orig[d + 1]
            if orig_dst != orig_src + 1:
                continue
            W = linear_layers[orig_dst].weight.detach().cpu().numpy()
            for j in range(sizes[d + 1]):
                for i in range(sizes[d]):
                    if W[j, i] != 0:
                        union(offsets[d] + i, offsets[d + 1] + j)

        unique_comps = list(set(find(n) for n in range(total)))
        rng = np.random.default_rng(42)
        comp_color = {c: rng.random(3) * 0.75 + 0.15 for c in unique_comps}

        def get_color(d, i):
            return comp_color[find(offsets[d] + i)]

        def legend_items():
            comp_counts = Counter(
                find(offsets[d] + i)
                for d in range(len(sizes)) for i in range(sizes[d])
            )
            sorted_comps = sorted(comp_counts.items(), key=lambda x: -x[1])
            return [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=comp_color[c], markersize=9,
                           markeredgecolor='black', markeredgewidth=0.8,
                           label=f'Component {idx+1}  ({count} neurons)')
                for idx, (c, count) in enumerate(sorted_comps)
            ], f'{len(sorted_comps)} components'

    elif mode == 'clusters':
        # Build a lookup: (orig_layer_idx, local_neuron_idx) -> cluster_id
        # layer_mapping uses layer names like 'layer_1', 'layer_2' etc.
        # We need to map global neuron indices from cluster_map back to
        # (orig_layer_idx, local_neuron_idx) in the full model.

        # First build: global_neuron_idx_in_clustering -> cluster_id
        neuron_to_cluster = {}
        for cluster_id, neuron_indices in cluster_map.items():
            for ni in neuron_indices:
                neuron_to_cluster[ni] = cluster_id

        # layer_mapping tells us which layer each global index belongs to
        # e.g. [('layer_1', 0, 21), ('layer_2', 21, 40), ...]
        # We need: orig_layer_idx -> (start_in_clustering, end_in_clustering)
        lm_lookup = {}
        for (layer_name, start, end) in layer_mapping:
            orig_idx = int(layer_name.split('_')[1])
            lm_lookup[orig_idx] = (start, end)

        unique_clusters = sorted(set(neuron_to_cluster.values()))
        rng = np.random.default_rng(7)
        cluster_color = {cid: rng.random(3) * 0.75 + 0.15 for cid in unique_clusters}
        unassigned_color = np.array([0.7, 0.7, 0.7])  # grey for neurons not in any cluster

        def get_color(d, i):
            orig_layer = display_to_orig[d]
            if orig_layer not in lm_lookup:
                return unassigned_color
            start, _ = lm_lookup[orig_layer]
            global_ni = start + i
            cid = neuron_to_cluster.get(global_ni, None)
            if cid is None:
                return unassigned_color
            return cluster_color[cid]

        def legend_items():
            cluster_counts = Counter(neuron_to_cluster.values())
            sorted_clusters = sorted(cluster_counts.items(), key=lambda x: -x[1])
            handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=cluster_color[cid], markersize=9,
                           markeredgecolor='black', markeredgewidth=0.8,
                           label=f'Cluster {cid}  ({count} neurons)')
                for cid, count in sorted_clusters
            ]
            handles.append(
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=unassigned_color, markersize=9,
                           markeredgecolor='black', markeredgewidth=0.8,
                           label='Unassigned')
            )
            return handles, f'{len(sorted_clusters)} clusters'

    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'components' or 'clusters'.")

    # ── DRAW ──────────────────────────────────────────────────────────────────
    def pos(d, i, n):
        return d, (i / (n - 1) if n > 1 else 0.5)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    for d in range(len(sizes) - 1):
        orig_src = display_to_orig[d]
        orig_dst = display_to_orig[d + 1]
        if orig_dst != orig_src + 1:
            continue
        W = linear_layers[orig_dst].weight.detach().cpu().numpy()
        W_max = np.abs(W).max() + 1e-8
        for j in range(sizes[d + 1]):
            for i in range(sizes[d]):
                w = W[j, i]
                if w == 0:
                    continue
                x0, y0 = pos(d,     i, sizes[d])
                x1, y1 = pos(d + 1, j, sizes[d + 1])
                ax.plot([x0, x1], [y0, y1],
                        color=get_color(d, i),
                        alpha=abs(w) / W_max * 0.6,
                        linewidth=0.5, zorder=1)

    for d, n in enumerate(sizes):
        for i in range(n):
            x, y = pos(d, i, n)
            ax.scatter(x, y, s=120, color=get_color(d, i),
                       edgecolors='black', linewidths=1.0, zorder=5)

    all_names = [f'Layer {k}\n({all_sizes[k]}n)' for k in range(n_total_layers)]
    for d, orig_k in display_to_orig.items():
        ax.text(d, -0.05, all_names[orig_k], ha='center', va='top', fontsize=9)

    handles, title = legend_items()
    ax.legend(handles=handles, loc='upper right', fontsize=8,
              title=title, title_fontsize=9, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()