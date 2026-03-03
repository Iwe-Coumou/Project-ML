import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# model parameters
HIDDEN_LAYERS = [512, 256, 128, 64]
TRAIN_VAL_SPLIT = 0.8
NUM_WORKERS = 4
BATCH_SIZE = 4000

# pruning loop
MAX_ALLOWED_ACC_DROP = 0.02
MAX_PRUNE_ROUNDS = 30
PRUNE_FRAC = 0.05
PRUNE_CON_FRAC = 0.1
REGROW_FRAC = 0.1
MIN_VAL_ACC = 0.9
N_RETRIAN_EPOCHS = 3

# clustering
N_CLUSTERS = 15


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

    train_size = int(TRAIN_VAL_SPLIT * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    print(f"train size: {train_size}, val size: {val_size}, test size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
