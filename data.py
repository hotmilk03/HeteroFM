import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset as TorchDataset
import numpy as np
import os
import config

class LabelRemappingDataset(TorchDataset):
    """Wrapper dataset that remaps labels to contiguous range [0, num_classes-1]."""
    def __init__(self, base_dataset, label_mapping):
        """
        Args:
            base_dataset: The original dataset (can be Subset or regular dataset)
            label_mapping: Dict mapping original labels to new labels {old: new}
        """
        self.base_dataset = base_dataset
        self.label_mapping = label_mapping
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        # Remap label
        new_label = self.label_mapping.get(label, label)
        return img, new_label

class Dataset:
    def __init__(self, dataset_name, data_dir, data_dir_imagenet):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.data_dir_imagenet = data_dir_imagenet
        self.train_dataset, self.test_dataset = self.load_data()
        self.num_classes = self._get_num_classes()
    
    def _get_num_classes(self):
        """Get the actual number of classes in the loaded dataset."""
        if isinstance(self.train_dataset, LabelRemappingDataset):
            # For remapped datasets, count unique remapped labels
            return len(self.train_dataset.label_mapping)
        elif isinstance(self.train_dataset, Subset):
            base = self.train_dataset.dataset
            idxs = self.train_dataset.indices
            if hasattr(base, 'targets'):
                labels = np.array(base.targets)[idxs]
            elif hasattr(base, 'samples'):
                labels = np.array([base.samples[i][1] for i in idxs])
            else:
                labels = np.array([base[i][1] for i in idxs])
            return len(np.unique(labels))
        else:
            if hasattr(self.train_dataset, 'classes'):
                return len(self.train_dataset.classes)
            elif hasattr(self.train_dataset, 'targets'):
                return len(np.unique(self.train_dataset.targets))
            elif hasattr(self.train_dataset, 'samples'):
                return len(np.unique([s[1] for s in self.train_dataset.samples]))
            else:
                return len(np.unique([y for _, y in self.train_dataset]))

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
            train_dir = os.path.join(self.data_dir_imagenet, 'train')
            val_dir = os.path.join(self.data_dir_imagenet, 'val')
            
            if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
                raise FileNotFoundError(
                    "ImageNet data directory not found. "
                    f"Expected 'train' and 'val' folders in '{self.data_dir_imagenet}'"
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
            
            # Optionally restrict to a subset of classes for faster training
            subset_ratio = getattr(config, 'IMAGENET_SUBSET_RATIO', 1.0)
            subset_num_classes = getattr(config, 'IMAGENET_SUBSET_NUM_CLASSES', None)
            if subset_num_classes is not None or (hasattr(config, 'IMAGENET_SUBSET_RATIO') and subset_ratio < 1.0):
                total_classes = len(train_dataset.classes)
                num_keep = subset_num_classes if subset_num_classes is not None else max(1, int(total_classes * subset_ratio))
                # Randomly select classes to keep
                rng = np.random.default_rng(seed=getattr(config, 'IMAGENET_SUBSET_SEED', 42))
                selected_class_indices = sorted(rng.choice(total_classes, size=num_keep, replace=False).tolist())

                # Build indices for samples whose label is in selected classes
                train_keep_idxs = [i for i, (_, y) in enumerate(train_dataset.samples) if y in selected_class_indices]
                test_keep_idxs = [i for i, (_, y) in enumerate(test_dataset.samples) if y in selected_class_indices]

                # Create label mapping: old label -> new label (0-indexed)
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(selected_class_indices)}

                # Create subsets
                train_subset = Subset(train_dataset, train_keep_idxs)
                test_subset = Subset(test_dataset, test_keep_idxs)
                
                # Wrap with label remapping
                train_dataset = LabelRemappingDataset(train_subset, label_mapping)
                test_dataset = LabelRemappingDataset(test_subset, label_mapping)
                
                print(f"Using {num_keep}/{total_classes} ImageNet classes: {len(train_dataset)} train, {len(test_dataset)} test samples")
                print(f"Labels remapped from {selected_class_indices[:5]}... to [0, 1, 2, 3, 4...]")
            
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}. Supported datasets are 'mnist', 'cifar10', 'imagenet'.")
            
        return train_dataset, test_dataset

    # prepare data for each client
    def prepare_data(self, num_clients, batch_size, data_split_mode, n_classes_ratio):
        """Prepares data loaders for federated learning clients."""
        print(f"Preparing '{data_split_mode}' data split for {num_clients} clients...")
        
        client_loaders = []
        
        # Helper: extract labels and num_classes, robust to Subset
        def _extract_labels_and_classes(ds):
            # Returns (labels np.array, num_classes int)
            if isinstance(ds, LabelRemappingDataset):
                # For remapped datasets, labels are already 0-indexed
                base = ds.base_dataset
                labels = []
                for i in range(len(base)):
                    _, label = ds[i]  # Use remapped label
                    labels.append(label)
                labels_arr = np.array(labels, dtype=np.int64)
                n_classes = len(ds.label_mapping)
                return labels_arr, n_classes
            elif isinstance(ds, Subset):
                base = ds.dataset
                idxs = ds.indices
                if hasattr(base, 'targets'):
                    base_labels = np.array(base.targets)
                elif hasattr(base, 'samples'):
                    base_labels = np.array([s[1] for s in base.samples])
                else:
                    # Fallback iterate
                    base_labels = np.array([y for _, y in base])
                labels_arr = base_labels[idxs]
                # Use unique labels present in the subset to determine number of classes
                n_classes = int(np.unique(labels_arr).size)
                return labels_arr, n_classes
            else:
                if hasattr(ds, 'targets'):
                    labels_arr = np.array(ds.targets)
                elif hasattr(ds, 'samples'):
                    labels_arr = np.array([s[1] for s in ds.samples])
                else:
                    labels_arr = np.array([y for _, y in ds])
                n_classes = int(np.unique(labels_arr).size)
                return labels_arr, n_classes

        # Get labels and class count for splitting (handles Subset from ImageNet subset)
        labels, num_classes = _extract_labels_and_classes(self.train_dataset)
        
        # Only remap if labels are not already contiguous [0, num_classes-1]
        unique_labels = np.unique(labels)
        if not (len(unique_labels) == num_classes and unique_labels[0] == 0 and unique_labels[-1] == num_classes - 1):
            # Labels need remapping
            label_to_new = {old: new for new, old in enumerate(unique_labels)}
            new_to_old = {new: old for old, new in label_to_new.items()}
            labels = np.array([label_to_new[l] for l in labels], dtype=np.int64)
        else:
            # Labels already contiguous, no remapping needed
            new_to_old = {i: i for i in range(num_classes)}

        if data_split_mode == 'iid':
            # IID split logic
            client_indices_map = {}
            num_samples = len(self.train_dataset)
            indices = list(range(num_samples))
            np.random.shuffle(indices)
            samples_per_client = num_samples // num_clients
            label_splits = {}
            for i in range(num_clients):
                client_indices = indices[i * samples_per_client: (i + 1) * samples_per_client]
                client_indices_map[i] = client_indices
                label_splits[i] = np.unique(labels[client_indices]).tolist()
                client_subset = Subset(self.train_dataset, client_indices)
                client_loaders.append(DataLoader(
                    client_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=config.NUM_WORKERS,
                    drop_last=True,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=(config.NUM_WORKERS > 0)
                ))
                
        elif data_split_mode == 'non-iid': # for mlp and vgg now (just for class# == client#)
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

                if not config.DYNAMIC_ON_NON_IID_SPLIT:
                    # Static split: equal shards
                    split_points = np.array_split(idx_by_class[i], shards_per_class)
                    class_shards[i] = [sp.tolist() for sp in split_points]
                else:
                    # Dynamic split: 1:4 ratio
                    total_samples = len(idx_by_class[i])
                    weights = np.random.uniform(1, 4, shards_per_class)
                    weights /= weights.sum()
                    sizes = (total_samples * weights).astype(int)
                    
                    # Distribute remainder
                    remainder = total_samples - sizes.sum()
                    for j in range(remainder):
                        sizes[j] += 1
                    
                    # Create shards
                    start = 0
                    shards = []
                    for size in sizes:
                        shards.append(idx_by_class[i][start : start + size].tolist())
                        start += size
                    class_shards[i] = shards
            
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
                client_loaders.append(DataLoader(
                    client_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=config.NUM_WORKERS,
                    drop_last=True,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=(config.NUM_WORKERS > 0)
                ))
        else:
            raise ValueError("Invalid data_split_mode in config.py. Choose 'iid' or 'non-iid'.")

        # Create client-specific test loaders (Subset of test_dataset based on label_splits)
        client_test_loaders = []
        
        # Get test labels (handles Subset)
        def _extract_labels(ds):
            if isinstance(ds, LabelRemappingDataset):
                # For remapped datasets, use remapped labels
                labels = []
                for i in range(len(ds)):
                    _, label = ds[i]
                    labels.append(label)
                return np.array(labels, dtype=np.int64)
            elif isinstance(ds, Subset):
                base = ds.dataset
                idxs = ds.indices
                if hasattr(base, 'targets'):
                    base_labels = np.array(base.targets)
                elif hasattr(base, 'samples'):
                    base_labels = np.array([s[1] for s in base.samples])
                else:
                    base_labels = np.array([y for _, y in base])
                return base_labels[idxs]
            else:
                if hasattr(ds, 'targets'):
                    return np.array(ds.targets)
                elif hasattr(ds, 'samples'):
                    return np.array([s[1] for s in ds.samples])
                else:
                    return np.array([y for _, y in ds])

        test_labels = _extract_labels(self.test_dataset)
        
        # Apply same label remapping to test labels if needed
        if not (len(unique_labels) == num_classes and unique_labels[0] == 0 and unique_labels[-1] == num_classes - 1):
            # Test labels also need remapping
            test_labels = np.array([label_to_new.get(l, -1) for l in test_labels], dtype=np.int64)

        for i in range(num_clients):
            target_labels = label_splits[i]
            if not target_labels:
                # Handle case with no assigned labels (empty list)
                client_test_subset = Subset(self.test_dataset, [])
            else:
                # Find indices in test dataset that match the client's target labels
                test_indices = np.where(np.isin(test_labels, target_labels))[0]
                client_test_subset = Subset(self.test_dataset, test_indices)
            
            client_test_loaders.append(DataLoader(
                client_test_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(config.NUM_WORKERS > 0)
            ))

        # Create a single test loader (common for all clients)
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(config.NUM_WORKERS > 0)
        )
        
        # Print class 0 (first class) distribution across clients
        original_label_0 = new_to_old.get(0, 0)
        if original_label_0 != 0:
            print(f"\n=== Class 0 Distribution (Originally Class {original_label_0}) ===")
        else:
            print(f"\n=== Class 0 Distribution Across Clients ===")
        for i in range(num_clients):
            if i in client_indices_map:
                client_indices = client_indices_map[i]
                class_0_count = np.sum(labels[client_indices] == 0)
                print(f"Client {i}: {class_0_count} samples of class 0")
            else:
                print(f"Client {i}: 0 samples of class 0")
        print("=" * 40 + "\n")
        
        print("Data preparation complete.")
        return client_loaders, client_test_loaders, test_loader, label_splits