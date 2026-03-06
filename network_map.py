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
        ignore_layers: list of original layer indices to hide (default None = show all)
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
        ignore_layers = []
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
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1),
              borderaxespad=0, fontsize=8, title=title, title_fontsize=9,
              framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def draw_network_interactive(
    model,
    mode='components',
    cluster_map=None,
    layer_mapping=None,
    ignore_layers=None,
    width=1400,
    height=700,
):
    """
    Interactive network visualisation using Plotly — no special backend needed.

    Legend behaviour (built into Plotly):
        single click  → hide / show that cluster
        double click  → isolate that cluster (hide all others); double click again to reset

    Args: same as draw_network except figsize is replaced by width/height (pixels).
    """
    import plotly.graph_objects as go

    if mode == 'clusters' and (cluster_map is None or layer_mapping is None):
        raise ValueError("cluster_map and layer_mapping are required when mode='clusters'")
    if ignore_layers is None:
        ignore_layers = []

    linear_layers   = [l for l in model.layer_stack if isinstance(l, torch.nn.Linear)]
    all_sizes       = [l.weight.shape[0] for l in linear_layers]
    n_total_layers  = len(all_sizes)
    layer_indices   = [k for k in range(n_total_layers) if k not in ignore_layers]
    sizes           = [all_sizes[k] for k in layer_indices]
    display_to_orig = {d: k for d, k in enumerate(layer_indices)}

    offsets = [sum(sizes[:d]) for d in range(len(sizes))]
    total   = sum(sizes)

    def node_pos(d, i, n):
        return d, (i / (n - 1) if n > 1 else 0.5)

    def to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

    # ── CLUSTER / COMPONENT SETUP ─────────────────────────────────────────────
    UNASSIGNED = '__unassigned__'

    if mode == 'components':
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

        rng = np.random.default_rng(42)
        all_roots  = list(set(find(n) for n in range(total)))
        comp_color = {c: rng.random(3) * 0.75 + 0.15 for c in all_roots}

        def get_cid(d, i):
            return find(offsets[d] + i)

        def get_color(cid):
            return comp_color[cid]

        comp_counts = Counter(
            find(offsets[d] + i) for d in range(len(sizes)) for i in range(sizes[d])
        )
        sorted_cids   = [c for c, _ in sorted(comp_counts.items(), key=lambda x: -x[1])]
        legend_labels = {
            c: f'Component {idx + 1}  ({comp_counts[c]} neurons)'
            for idx, c in enumerate(sorted_cids)
        }
        legend_title = f'{len(sorted_cids)} components'

    elif mode == 'clusters':
        neuron_to_cluster = {}
        for cid, nidxs in cluster_map.items():
            for ni in nidxs:
                neuron_to_cluster[ni] = cid

        lm_lookup = {}
        for (layer_name, start, end) in layer_mapping:
            orig_idx = int(layer_name.split('_')[1])
            lm_lookup[orig_idx] = (start, end)

        rng = np.random.default_rng(7)
        unique_clusters = sorted(set(neuron_to_cluster.values()))
        cluster_color   = {cid: rng.random(3) * 0.75 + 0.15 for cid in unique_clusters}
        unassigned_color = np.array([0.7, 0.7, 0.7])

        def get_cid(d, i):
            orig_layer = display_to_orig[d]
            if orig_layer not in lm_lookup:
                return UNASSIGNED
            start, _ = lm_lookup[orig_layer]
            return neuron_to_cluster.get(start + i, UNASSIGNED)

        def get_color(cid):
            return unassigned_color if cid == UNASSIGNED else cluster_color[cid]

        cluster_counts = Counter(neuron_to_cluster.values())
        sorted_cids    = [c for c, _ in sorted(cluster_counts.items(), key=lambda x: -x[1])]
        sorted_cids.append(UNASSIGNED)
        legend_labels  = {c: f'Cluster {c}  ({cluster_counts[c]} neurons)' for c in unique_clusters}
        legend_labels[UNASSIGNED] = 'Unassigned'
        legend_title   = f'{len(unique_clusters)} clusters'

    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'components' or 'clusters'.")

    # ── COLLECT PER-CLUSTER GEOMETRY ──────────────────────────────────────────
    cluster_pts  = defaultdict(list)   # cid -> [(x, y, hover_text)]
    cluster_segs = defaultdict(list)   # cid -> flat [x0,x1,None, ...] for edges
    cluster_segY = defaultdict(list)

    for d, n in enumerate(sizes):
        orig_k = display_to_orig[d]
        for i in range(n):
            x, y = node_pos(d, i, n)
            cluster_pts[get_cid(d, i)].append(
                (x, y, f'Layer {orig_k}, neuron {i}'))

    for d in range(len(sizes) - 1):
        orig_src = display_to_orig[d]
        orig_dst = display_to_orig[d + 1]
        if orig_dst != orig_src + 1:
            continue
        W     = linear_layers[orig_dst].weight.detach().cpu().numpy()
        for j in range(sizes[d + 1]):
            for i in range(sizes[d]):
                if W[j, i] == 0:
                    continue
                cid     = get_cid(d, i)
                x0, y0  = node_pos(d,     i, sizes[d])
                x1, y1  = node_pos(d + 1, j, sizes[d + 1])
                cluster_segs[cid] += [x0, x1, None]
                cluster_segY[cid] += [y0, y1, None]

    # ── BUILD PLOTLY FIGURE ───────────────────────────────────────────────────
    traces = []
    all_cids = sorted(set(list(cluster_pts) + list(cluster_segs)),
                      key=lambda c: sorted_cids.index(c) if c in sorted_cids else 999)

    for cid in all_cids:
        color   = get_color(cid)
        hex_col = to_hex(color)
        label   = legend_labels[cid]
        group   = str(cid)

        # Edge lines (no legend entry)
        if cid in cluster_segs:
            traces.append(go.Scatter(
                x=cluster_segs[cid], y=cluster_segY[cid],
                mode='lines',
                line=dict(color=hex_col, width=0.8),
                opacity=0.35,
                legendgroup=group,
                showlegend=False,
                hoverinfo='skip',
            ))

        # Neuron dots (legend entry)
        if cid in cluster_pts:
            pts = cluster_pts[cid]
            xs, ys, labels = zip(*pts)
            traces.append(go.Scatter(
                x=xs, y=ys,
                mode='markers',
                marker=dict(size=10, color=hex_col,
                            line=dict(color='black', width=1)),
                name=label,
                legendgroup=group,
                legendgrouptitle=None,
                showlegend=True,
                text=labels,
                hovertemplate='%{text}<extra></extra>',
            ))

    # Layer label annotations
    annotations = []
    for d, orig_k in display_to_orig.items():
        annotations.append(dict(
            x=d, y=-0.1,
            text=f'Layer {orig_k}<br>({all_sizes[orig_k]} n)',
            showarrow=False,
            font=dict(size=11),
            xanchor='center',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        width=width, height=height,
        legend=dict(
            title=dict(text=legend_title, font=dict(size=11)),
            x=1.01, y=1, xanchor='left', yanchor='top',
            itemclick='toggle',
            itemdoubleclick='toggleothers',
        ),
        xaxis=dict(visible=False, range=[-0.4, len(sizes) - 0.6]),
        yaxis=dict(visible=False, range=[-0.18, 1.08]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=220, t=20, b=20),
        annotations=annotations,
    )
    from IPython.display import display, HTML
    display(HTML(fig.to_html(full_html=False, include_plotlyjs='cdn')))