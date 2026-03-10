import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# model parameters
HIDDEN_LAYERS = [128, 128, 128, 64]
TRAIN_VAL_SPLIT = 0.8
TRAIN_FRESH_SPLIT = 0.1
NUM_WORKERS = 4
BATCH_SIZE = 8000

# training
N_TRAIN_EPOCHS = 10   # for CPU runs this is overridden to 8 in the notebook

# clustering — upper bound on clusters; actual k is auto-selected by dendrogram gap
MAX_CLUSTERS = 10

# pruning loop — core
MAX_PRUNE_ROUNDS     = 150
SEARCH_MAX_ROUNDS    = 25    # reduced rounds for hyperparameter search
SEARCH_SUBSET_SIZE   = 20000 # samples used for retraining during search (vs ~172k full)
MAX_ALLOWED_ACC_DROP = 0.2
N_RETRAIN_EPOCHS     = 8
PRUNE_FRAC           = 0.025
PRUNE_CON_FRAC       = 0.35
MIN_WIDTH            = 25

# pruning loop — advanced
N_FINAL_RETRAIN_EPOCHS   = 50
CROSS_CLUSTER_PRUNE_FRAC = 0.7
TOPOLOGY_THRESHOLD       = 0.15
PHASE2_MIN_NEURONS       = 100
PHASE2_MIN_CONNECTIONS   = 2000
ERROR_THRESHOLD_FRAC     = 1.5
REGROW_FRAC              = 0.2     # fraction of pruned neurons to regrow; 0 = disabled
N_SPAWN                  = 5       # neurons spawned per underperforming cluster per round

# Stage 1: search pruning rates (structural params fixed at defaults above)
HP_SEARCH_GRID = [
    {'prune_frac': 0.025, 'prune_con_frac': 0.35},
    {'prune_frac': 0.025, 'prune_con_frac': 0.50},
    {'prune_frac': 0.05,  'prune_con_frac': 0.35},
    {'prune_frac': 0.05,  'prune_con_frac': 0.50},
]

# Stage 2: search structural params (run after stage 1, plug in best prune rates)
# PHASE2_MIN_NEURONS controls how long the network stays in cluster-guided mode —
# higher = more phase 2 rounds = more cluster reinforcement but less total pruning time.
# MIN_WIDTH is the neuron floor per hidden layer.
HP_SEARCH_GRID_STAGE2 = [
    {'phase2_min_neurons': 80,  'min_width': 15},
    {'phase2_min_neurons': 150, 'min_width': 15},
    {'phase2_min_neurons': 150, 'min_width': 25},
    {'phase2_min_neurons': 250, 'min_width': 25},
]


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