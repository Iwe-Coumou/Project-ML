import numpy as np
import matplotlib.pyplot as plt


def plot_cluster_accuracy_bars(cluster_results, target_labels=None, n_cols=4, figsize_per_plot=(4,3)):
    """
    Plots per-cluster pre- vs post-ablation accuracy as bars per class.

    Args:
        cluster_results: dict of dicts {cluster_id: {'pre': {cls: acc}, 'post': {cls: acc}}}
        target_labels: list of class labels in order (optional)
        n_cols: number of columns in the grid
        figsize_per_plot: tuple, figure size per subplot
    """
    n_clusters = len(cluster_results)
    n_rows = int(np.ceil(n_clusters / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*figsize_per_plot[0], n_rows*figsize_per_plot[1]), squeeze=False)
    
    for i, (cluster_id, acc_dict) in enumerate(cluster_results.items()):
        ax = axes[i // n_cols, i % n_cols]
        
        classes = sorted(acc_dict['pre'].keys()) if target_labels is None else target_labels
        pre_acc = [acc_dict['pre'][cls] for cls in classes]
        post_acc = [acc_dict['post'][cls] for cls in classes]
        
        # Draw bars from pre to post
        for x, y_pre, y_post in zip(range(len(classes)), pre_acc, post_acc):
            ax.plot([x, x], [y_pre, y_post], color='skyblue', lw=6, solid_capstyle='round')  # thick line as bar
            ax.scatter(x, y_pre, color='green', label='pre' if x==0 else "", zorder=3)
            ax.scatter(x, y_post, color='red', label='post' if x==0 else "", zorder=3)
        
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45)
        ax.set_ylim(0,1)
        ax.set_title(f'Cluster {cluster_id}')
        if i == 0:
            ax.legend()
        
        row = i // n_cols
        col = i % n_cols
        # Only leftmost column gets y-label
        if col == 0:
            ax.set_ylabel("Accuracy")
        # Only bottom row gets x-label
        if row == n_rows - 1:
            ax.set_xlabel("Class label")
    
    # Hide empty subplots
    for j in range(i+1, n_rows*n_cols):
        fig.delaxes(axes[j//n_cols, j%n_cols])
    
    fig.tight_layout()
    plt.show()

def plot_accuracy(metrics):
    """
    Plots training and validation accuracy over epochs.

    Args:
        metrics (pd.DataFrame): Must contain 'train_acc' and 'val_acc' columns.
    """
    epochs = range(1, len(metrics)+1)

    plt.figure()
    plt.plot(epochs, metrics['train_acc'], label='Train Accuracy')
    plt.plot(epochs, metrics['val_acc'], label='Validation Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss(metrics):
    """
    Plots training and validation accuracy over epochs.

    Args:
        metrics (pd.DataFrame): Must contain 'train_loss' and 'val_loss' columns.
    """
    epochs = range(1, len(metrics)+1)

    plt.figure()
    plt.plot(epochs, metrics['train_loss'], label='Train loss')
    plt.plot(epochs, metrics['val_loss'], label='Validation loss')

    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cluster_prototypes_and_diff_all(all_prototypes):
    """
    Plot prototypes and difference maps for all clusters.

    Args:
        all_prototypes: dict {cluster_id: {'prototype': 28x28 array, 'diff_map': 28x28 array}}
    """
    n_clusters = len(all_prototypes)
    fig, axes = plt.subplots(2, n_clusters, figsize=(n_clusters*2, 4))

    for i, cluster_id in enumerate(sorted(all_prototypes.keys())):
        proto = all_prototypes[cluster_id]['prototype']
        diff = all_prototypes[cluster_id]['diff_map']

        # Top row = prototype
        ax_top = axes[0, i] if n_clusters > 1 else axes[0]
        ax_top.imshow(proto, cmap='viridis')
        ax_top.axis('off')
        if i == 0:
            ax_top.set_ylabel('Prototype', fontsize=10)

        # Bottom row = diff map
        ax_bottom = axes[1, i] if n_clusters > 1 else axes[1]
        ax_bottom.imshow(diff, cmap='viridis')
        ax_bottom.axis('off')
        if i == 0:
            ax_bottom.set_ylabel('Diff Map', fontsize=10)

        # Column title = cluster ID
        ax_top.set_title(f'Cluster {cluster_id}', fontsize=10)

    plt.suptitle('All Clusters - Prototypes & Difference Maps', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_cluster_activation_heatmap(selectivity_results, n_classes=10):
    cluster_ids = sorted(selectivity_results.keys())
    activation_matrix = np.array([selectivity_results[c]['mean_activation_per_class'] for c in cluster_ids])

    plt.figure(figsize=(10, len(cluster_ids)*0.5 + 2))
    im = plt.imshow(activation_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Mean Activation')
    plt.xlabel('Digit')
    plt.ylabel('Cluster ID')
    plt.title('Cluster Mean Activation per Digit')
    plt.xticks(range(n_classes))
    plt.yticks(range(len(cluster_ids)), cluster_ids)
    plt.show()

def plot_cluster_entropy(selectivity_results):
    cluster_ids = sorted(selectivity_results.keys())
    entropy_values = [selectivity_results[c]['normalized_entropy'] for c in cluster_ids]

    plt.figure(figsize=(10,4))
    plt.bar(cluster_ids, entropy_values, color='steelblue')
    plt.xlabel('Cluster ID')
    plt.ylabel('Normalized Entropy')
    plt.title('Cluster Selectivity (Lower = More Selective)')

    plt.xticks(cluster_ids, rotation=45)

    plt.show()

def plot_cluster_prob_distribution(selectivity_results, cluster_id):
    probs = selectivity_results[cluster_id]['prob_distribution']
    plt.figure(figsize=(6,4))
    plt.bar(range(len(probs)), probs, color='orange')
    plt.xticks(range(len(probs)))
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title(f'Cluster {cluster_id} Class Probability Distribution')
    plt.show()