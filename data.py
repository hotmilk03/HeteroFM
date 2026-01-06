import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import config

class Dataset:
    def __init__(self, dataset_name, data_dir):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.train_dataset, self.test_dataset = self.load_data()

    # load MNIST dataset
    def load_data(self):
        """Loads the specified dataset and applies appropriate transformations."""
        if self.dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(self.data_dir, train=False, download=True, transform=transform)
            
        elif self.dataset_name == 'cifar10':
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            train_dataset = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR10(self.data_dir, train=False, download=True, transform=test_transform)
            
        elif self.dataset_name == 'imagenet':
            # Assumes ImageNet is structured in 'train' and 'val' subdirectories
            train_dir = os.path.join(self.data_dir, 'train')
            val_dir = os.path.join(self.data_dir, 'val')
            
            if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
                raise FileNotFoundError(
                    "ImageNet data directory not found. "
                    f"Expected 'train' and 'val' folders in '{self.data_dir}'"
                )

            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
            test_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
            
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}. Supported datasets are 'mnist', 'cifar10', 'imagenet'.")
            
        return train_dataset, test_dataset

    # prepare data for each client
    def prepare_data(self, num_clients, batch_size, data_split_mode, n_classes_ratio):
        """Prepares data loaders for federated learning clients."""
        print(f"Preparing '{data_split_mode}' data split for {num_clients} clients...")
        
        client_loaders = []
        
        # Get labels for splitting
        if hasattr(self.train_dataset, 'targets'):
            labels = np.array(self.train_dataset.targets)
        elif hasattr(self.train_dataset, 'samples'):
            labels = np.array([s[1] for s in self.train_dataset.samples])
        else:
            raise TypeError("Could not find labels in the dataset.")
            
        if hasattr(self.train_dataset, 'classes'):
            num_classes = len(self.train_dataset.classes)
        else:
            num_classes = len(np.unique(labels))

        if data_split_mode == 'iid':
            # IID split logic
            num_samples = len(self.train_dataset)
            indices = list(range(num_samples))
            np.random.shuffle(indices)
            samples_per_client = num_samples // num_clients
            label_splits = {}
            for i in range(num_clients):
                client_indices = indices[i * samples_per_client: (i + 1) * samples_per_client]
                label_splits[i] = np.unique(labels[client_indices]).tolist()
                client_subset = Subset(self.train_dataset, client_indices)
                client_loaders.append(DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True))
                
        elif data_split_mode == 'non-iid':
            # Non-IID split logic
            # Calculate the number of classes per client based on the ratio
            n_classes_per_client = max(1, int(num_classes * n_classes_ratio))
            print(f"Non-IID split: Each client will have {n_classes_per_client} classes out of {num_classes}.")

            idx_by_class = {i: np.where(labels == i)[0] for i in range(num_classes)}

            shards_per_class = (num_clients * n_classes_per_client) // num_classes
            if shards_per_class < 1:
                raise ValueError(f"Cannot create non-iid split. Not enough data for {n_classes_per_client} classes per client.")

            class_shards = {i: [] for i in range(num_classes)}
            for i in range(num_classes):
                np.random.shuffle(idx_by_class[i])
                split_points = np.array_split(idx_by_class[i], shards_per_class)
                class_shards[i] = [sp.tolist() for sp in split_points]
            
            client_indices_map = {i: [] for i in range(num_clients)}
            label_splits = {}
            
            # Distribute shards to clients
            client_class_map = {i: [] for i in range(num_clients)}
            classes = list(range(num_classes))
            for i in range(n_classes_per_client):
                for j in range(num_clients):
                    cls_idx = (j + i * (num_clients // num_classes)) % len(classes)
                    client_class_map[j].append(classes[cls_idx])
            
            for i in range(num_clients):
                client_indices = []
                target_classes = client_class_map[i]
                
                for cls in target_classes:
                    if len(class_shards[cls]) > 0:
                        shard = class_shards[cls].pop(0)
                        client_indices.extend(shard)
                    else:
                        print(f"Warning: Class {cls} is out of data shards! This may lead to data imbalance.")

                client_indices_map[i] = client_indices
                if client_indices:
                    label_splits[i] = np.unique(labels[client_indices]).tolist()
                else:
                    label_splits[i] = []


                client_subset = Subset(self.train_dataset, client_indices_map[i])
                client_loaders.append(DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True))
        else:
            raise ValueError("Invalid data_split_mode in config.py. Choose 'iid' or 'non-iid'.")

        # Create a single test loader (common for all clients)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
        print("Data preparation complete.")
        return client_loaders, test_loader, label_splits