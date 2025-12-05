import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import config

# load MNIST dataset
def load_data(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

# prepare data for each client
def prepare_data(train_dataset, test_dataset, num_clients, batch_size, data_split_mode, n_classes_per_client=2):
    print(f"Preparing '{data_split_mode}' data split for {num_clients} clients...")
    
    client_loaders = []
    labels = np.array(train_dataset.targets)
    
    if data_split_mode == 'iid':
        # IID split logic
        num_samples = len(train_dataset)
        indices = list(range(num_samples))
        np.random.shuffle(indices)
        samples_per_client = num_samples // num_clients
        label_splits = {}
        for i in range(num_clients):
            client_indices = indices[i * samples_per_client: (i + 1) * samples_per_client]
            label_splits[i] = np.unique(labels[client_indices]).tolist()
            client_subset = Subset(train_dataset, client_indices)
            client_loaders.append(DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS))
            
    elif data_split_mode == 'non-iid':
        # Non-IID split logic
        num_classes = len(train_dataset.classes)
        idx_by_class = {i: np.where(labels == i)[0] for i in range(num_classes)}

        shards_per_class = (num_clients * n_classes_per_client) // num_classes
        if shards_per_class == 0:
            raise ValueError(f"Cannot create non-iid split. Not enough data for {n_classes_per_client} classes per client.")

        shards = []
        for i in range(num_classes):
            np.random.shuffle(idx_by_class[i])
            split_points = np.array_split(idx_by_class[i], shards_per_class)
            shards.extend([sp.tolist() for sp in split_points])
        
        np.random.shuffle(shards)
        
        client_indices_map = {i: [] for i in range(num_clients)}
        label_splits = {} # {i: [] for i in range(num_clients)}
        shards_per_client = len(shards) // num_clients
        for i in range(num_clients):
            assigned_shards = shards[i * shards_per_client : (i + 1) * shards_per_client]
            client_indices = [idx for shard in assigned_shards for idx in shard]
            client_indices_map[i] = client_indices
            label_splits[i] = np.unique(labels[client_indices]).tolist()

            client_subset = Subset(train_dataset, client_indices_map[i])
            client_loaders.append(DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS))
    else:
        raise ValueError("Invalid data_split_mode in config.py. Choose 'iid' or 'non-iid'.")

    # Create a single test loader (common for all clients)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
    print("Data preparation complete.")
    return client_loaders, test_loader, label_splits
