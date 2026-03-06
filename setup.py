import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# model parameters
HIDDEN_LAYERS = [64, 64, 64, 32]
TRAIN_VAL_SPLIT = 0.8
TRAIN_FRESH_SPLIT = 0.1
NUM_WORKERS = 4
BATCH_SIZE = 8000

# clustering — upper bound on clusters; actual k is auto-selected by dendrogram gap
MAX_CLUSTERS = 10

# pruning loop
N_FINAL_RETRAIN_EPOCHS   = 50
CROSS_CLUSTER_PRUNE_FRAC = 0.3
TOPOLOGY_THRESHOLD       = 0.15
PHASE2_MIN_NEURONS       = 50
PHASE2_MIN_CONNECTIONS   = 200
ERROR_THRESHOLD_FRAC     = 1.5


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device being used: {device.type}")
    return device


def get_dataloaders(batch_size=BATCH_SIZE):
    def rotate_image(x):
        return torch.rot90(x, k=-1, dims=[1, 2])

    def flip_image(x):
        return torch.flip(x, dims=[2])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(rotate_image),
        transforms.Lambda(flip_image),
    ])

    train_dataset = datasets.EMNIST(
        root="./data",
        split="digits",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.EMNIST(
        root="./data",
        split="digits",
        train=False,
        download=True,
        transform=transform
    )

    total_train = int(TRAIN_VAL_SPLIT * len(train_dataset))
    fresh_size = int(TRAIN_FRESH_SPLIT * total_train)
    train_size = total_train - fresh_size
    val_size = len(train_dataset) - total_train 
    train_dataset, val_dataset, fresh_dataset = random_split(
        train_dataset, [train_size, val_size, fresh_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"train size: {train_size}, val size: {val_size}, fresh size: {fresh_size}, test size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    fresh_loader = DataLoader(fresh_dataset, batch_size=batch_size)
    return train_dataset, val_dataset, test_dataset, fresh_dataset, train_loader, val_loader, test_loader, fresh_loader