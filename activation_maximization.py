import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def visualize_cluster(model, cluster_map, layer_mapping, cluster_id, steps=500, lr=0.1, tv_weight=0.05, show=True):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    neuron_indices = cluster_map[cluster_id]
    device = next(model.parameters()).device

    linear_layers = [l for l in model.layer_stack if isinstance(l, torch.nn.Linear)]
    last_linear = linear_layers[-1]

    def get_cluster_activation(x):
        out = x.view(1, -1)
        hidden_acts = []
        for layer in model.layer_stack:
            out = layer(out)
            if isinstance(layer, torch.nn.Linear) and layer is not last_linear:
                hidden_acts.append(out)
        flat = torch.cat(hidden_acts, dim=1)  # [1, total_hidden_neurons]
        return flat[0, neuron_indices].mean()

    img = torch.randn(1, 28, 28, device=device) * 0.01
    img.requires_grad_(True)

    optimizer = torch.optim.Adam([img], lr=lr)

    gauss_kernel = torch.tensor(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32
    ).view(1, 1, 3, 3).to(device) / 16.0

    for step in range(steps):
        optimizer.zero_grad()

        activation = get_cluster_activation(img)

        tv_h = (img[:, :, 1:] - img[:, :, :-1]).abs().mean()
        tv_v = (img[:, 1:, :] - img[:, :-1, :]).abs().mean()
        tv = tv_h + tv_v

        loss = -activation + tv_weight * tv
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            with torch.no_grad():
                smoothed = F.conv2d(img.data.unsqueeze(0), gauss_kernel, padding=1).squeeze(0)
                img.data.copy_(smoothed)

    with torch.no_grad():
        result = img.squeeze().cpu()
        result = (result - result.min()) / (result.max() - result.min() + 1e-8)
        result_np = result.numpy()

    if show:
        plt.figure()
        plt.imshow(result_np, cmap='gray')
        plt.title(f'Cluster {cluster_id} — activation maximization')
        plt.axis('off')
        plt.show()

    return result_np
