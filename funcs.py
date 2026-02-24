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
    best_model = copy.deepcopy(current_model)

    baseline_val_acc = current_model.accuracy(val_loader)
    best_val_acc = baseline_val_acc
    round_idx = 0
    use_max_rounds = False
    while True:
        round_idx+=1
        clear_output(wait=True)
        print(f"\n--- Pruning round {round_idx+1} ---")

        if use_max_rounds and round_idx > max_rounds:
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
            print("Restoring previous best model")
            current_model = best_model
            break

        best_model = copy.deepcopy(new_model)
        best_val_acc = val_acc
        current_model = new_model

        return (best_model, best_val_acc, metrics_history)

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