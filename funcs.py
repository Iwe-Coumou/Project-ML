import torch
from collections import defaultdict
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import copy

def cluster_criticality_per_class(model, cluster_indices, layer_mapping, data_loader, cluster_id, device=None, ):
    """
    Computes per-class impact of ablating a neuron cluster using model.predict.

    Args:
        model: trained NeuralNetwork instance
        cluster_indices: list of global neuron indices in the cluster
        layer_mapping: list of tuples [(layer_name, start_idx, end_idx), ...]
        data_loader: DataLoader for evaluation
        device: torch device

    Returns:
        class_acc_drop: dict {class_label: accuracy_drop}
    """
    device = device or model.device
    model.eval()

    # Get all labels
    all_labels = []
    for _, y in data_loader:
        all_labels.append(y)
    all_labels = torch.cat(all_labels).to(device)

    clear_output(wait=True)
    print(f"\n--- Calculating pre and post-ablation accuracy for cluster {cluster_id} ---")
    # Original predictions and per-class accuracy
    orig_preds = model.predict(data_loader)
    orig_preds = orig_preds.to(device)

    class_total = defaultdict(int)
    class_correct = defaultdict(int)
    for t, p in zip(all_labels, orig_preds):
        class_total[int(t)] += 1
        if t == p:
            class_correct[int(t)] += 1
    orig_acc_per_class = {cls: class_correct[cls]/class_total[cls] for cls in class_total}

    # Backup weights/biases
    linear_indices = [i for i, l in enumerate(model.layer_stack) if isinstance(l, torch.nn.Linear)]
    layer_backups = {}
    for layer_name, start_idx, end_idx in layer_mapping:
        layer_idx = int(layer_name.split('_')[1])
        linear_layer_idx = linear_indices[layer_idx]
        layer = model.layer_stack[linear_layer_idx]
        layer_backups[layer_name] = {
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

        # Zero out next layer input weights
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

    # Restore weights/biases
    for layer_name in layer_backups:
        layer_idx = int(layer_name.split('_')[1])
        linear_layer_idx = linear_indices[layer_idx]
        layer = model.layer_stack[linear_layer_idx]
        layer.weight.data = layer_backups[layer_name]['weight']
        layer.bias.data = layer_backups[layer_name]['bias']

    return {
        'pre': orig_acc_per_class,
        'post': acc_per_class_after
    }

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

def pruning(model, train_loader, val_loader, parameters, use_max_rounds=True):
    max_rounds, prune_frac, regrow_frac, retrain_epochs, max_acc_drop = parameters
    metrics_history = []
    prune_history = []

    current_model = copy.deepcopy(model)

    baseline_val_acc = current_model.accuracy(val_loader)
    best_val_acc = baseline_val_acc

    round_idx = 0
    while True:
        round_idx+=1
        print(f"\n--- Pruning round {round_idx} ---")

        if use_max_rounds and round_idx >= max_rounds:
            print("Reached maximum pruning rounds.")
            break

        prev_model = copy.deepcopy(current_model)

        print("Getting layer data:")
        layer_data = current_model.get_layer_data(train_loader)
        importance_scores = current_model.compute_neuron_importance(layer_data=layer_data)
        prune_history.append(importance_scores)

        new_model = copy.deepcopy(current_model)
        new_model.prune_hidden_neurons(importance_scores=importance_scores, prune_rate=prune_frac, alpha=0.7, regrow_frac=regrow_frac)

        print("Retraining:")
        metrics = new_model.train_model(train_loader, val_loader, epochs=retrain_epochs, lr=0.01)
        metrics_history.append(metrics)

        val_acc = metrics['val_acc'].iloc[-1]
        print(f"Validation accuracy after pruning round {round_idx+1}: {val_acc:.4f}")

        acc_drop = baseline_val_acc - val_acc
        
        if acc_drop > max_acc_drop:
            print("Accuracy drop exceeded threshold.")
            print("Restoring previous model")
            current_model = prev_model
            break

        current_model = new_model
        clear_output(wait=True)
    
    best_val_acc = current_model.accuracy(val_loader)
    return (current_model, best_val_acc, metrics_history)

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

def compute_class_conditional_prototypes_with_diff(cluster_id, cluster_map, all_activations, images, labels, top_frac=0.1, use_class_mean_diff=True):
    """
    Compute prototypes and difference maps for all digits for a single cluster.

    Args:
        cluster_id: int, the cluster to compute
        cluster_map: dict {cluster_id: list of neuron indices}
        all_activations: tensor (N_samples, total_neurons)
        images: tensor (N_samples, 1, 28, 28)
        labels: tensor (N_samples,)
        top_frac: fraction of top-activating samples to average
        use_class_mean_diff: if True, difference map relative to class mean, else global mean

    Returns:
        dict: digit -> {'prototype': 28x28 array, 'diff_map': 28x28 array}
    """
    cluster_indices = cluster_map[cluster_id]
    cluster_acts = all_activations[:, cluster_indices]
    cluster_strength = cluster_acts.mean(dim=1)

    N_samples = images.shape[0]
    flat_images = images.view(N_samples, -1)

    # Precompute global mean if needed
    if not use_class_mean_diff:
        global_mean = flat_images.mean(dim=0).view(28,28)
        global_mean = (global_mean - global_mean.min()) / (global_mean.max() - global_mean.min() + 1e-8)
        global_mean_np = global_mean.cpu().numpy()

    prototypes = {}

    for digit in range(10):
        mask = (labels == digit)
        digit_strength = cluster_strength[mask]
        digit_images = images[mask]

        if len(digit_strength) == 0:
            continue

        # Top-k averaging
        k = max(1, int(top_frac * len(digit_strength)))
        _, top_idx = torch.topk(digit_strength, k)

        proto = digit_images[top_idx].mean(dim=0).squeeze()
        proto = (proto - proto.min()) / (proto.max() - proto.min() + 1e-8)
        proto_np = proto.cpu().numpy()

        # Difference map
        if use_class_mean_diff:
            class_mean = digit_images.mean(dim=0).squeeze()
            class_mean = (class_mean - class_mean.min()) / (class_mean.max() - class_mean.min() + 1e-8)
            mean_np = class_mean.cpu().numpy()
        else:
            mean_np = global_mean_np

        diff_map = proto_np - mean_np
        diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)

        prototypes[digit] = {
            'prototype': proto_np,
            'diff_map': diff_map
        }

    return prototypes

def plot_cluster_prototypes_and_diff(cluster_id, prototypes_dict):
    """
    Plot prototypes and difference maps for a single cluster.

    Args:
        cluster_id: int, the cluster ID
        prototypes_dict: dict {digit: {'prototype': 28x28 array, 'diff_map': 28x28 array}}
    """
    n_digits = len(prototypes_dict)
    fig, axes = plt.subplots(2, n_digits, figsize=(n_digits*1.8, 4))

    for i, digit in enumerate(sorted(prototypes_dict.keys())):
        proto = prototypes_dict[digit]['prototype']
        diff = prototypes_dict[digit]['diff_map']

        # Top row = prototype
        ax_top = axes[0, i]
        ax_top.imshow(proto, cmap='viridis')
        ax_top.axis('off')
        if i == 0:
            ax_top.set_ylabel('Prototype', fontsize=10)

        # Bottom row = diff map
        ax_bottom = axes[1, i]
        ax_bottom.imshow(diff, cmap='viridis')
        ax_bottom.axis('off')
        if i == 0:
            ax_bottom.set_ylabel('Diff Map', fontsize=10)

        # Column title = digit
        ax_top.set_title(str(digit), fontsize=10)

    plt.suptitle(f'Cluster {cluster_id} - Prototypes & Difference Maps', fontsize=12)
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