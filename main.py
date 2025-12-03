import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import time

# Local imports
from federated import client_update, aggregate_heterofl
import config
from model import init_model

# =============================================================================
# 1. DATA LOADING & PREPARATION
# =============================================================================
def load_data(data_dir):
    """Loads the MNIST dataset."""
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    print("Dataset loaded.")
    return train_dataset, test_dataset

def prepare_data(train_dataset, test_dataset, num_clients, batch_size, data_split_mode, n_classes_per_client):
    """
    Prepares data loaders for clients and a single test loader,
    supporting both IID and Non-IID splits.
    """
    print(f"Preparing '{data_split_mode}' data for {num_clients} clients...")
    
    client_loaders = []
    
    if data_split_mode == 'iid':
        # IID split logic (same as before)
        num_samples = len(train_dataset)
        indices = list(range(num_samples))
        np.random.shuffle(indices)
        samples_per_client = num_samples // num_clients
        for i in range(num_clients):
            client_indices = indices[i * samples_per_client: (i + 1) * samples_per_client]
            client_subset = Subset(train_dataset, client_indices)
            client_loaders.append(DataLoader(client_subset, batch_size=batch_size, shuffle=True))
            
    elif data_split_mode == 'non-iid':
        # Non-IID split logic (adapted from HeteroFL)
        # 1. Group samples by class
        num_classes = len(train_dataset.classes)
        labels = np.array(train_dataset.targets)
        idx_by_class = {i: np.where(labels == i)[0] for i in range(num_classes)}

        # 2. Create shards
        shards_per_class = (num_clients * n_classes_per_client) // num_classes
        shards = []
        for i in range(num_classes):
            np.random.shuffle(idx_by_class[i])
            # Split indices of a class into shards_per_class
            split_points = np.array_split(idx_by_class[i], shards_per_class)
            shards.extend([sp.tolist() for sp in split_points])
        
        # 3. Assign shards to clients
        np.random.shuffle(shards)
        client_indices_map = {i: [] for i in range(num_clients)}
        for i in range(num_clients):
            # Assign n_classes_per_client shards to each client
            assigned_shards = shards[i * n_classes_per_client : (i + 1) * n_classes_per_client]
            client_indices_map[i] = [idx for shard in assigned_shards for idx in shard]

        # 4. Create DataLoaders
        for i in range(num_clients):
            client_subset = Subset(train_dataset, client_indices_map[i])
            client_loaders.append(DataLoader(client_subset, batch_size=batch_size, shuffle=True))

    else:
        raise ValueError("Invalid data_split_mode. Choose 'iid' or 'non-iid'.")

    # Create a single test loader (common for all clients)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Data preparation complete.")
    return client_loaders, test_loader

def evaluate(model, test_loader, device):
    """Evaluates the model on the test dataset."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# =============================================================================
# 3. MAIN EXPERIMENT
# =============================================================================
if __name__ == '__main__':
    
    print("Starting Heterogeneous Federated Learning Experiment...")
    print(f"  - Clients: {config.NUM_CLIENTS}")
    print(f"  - Client Sizes: {config.CLIENT_SIZES}")
    print(f"  - Communication Rounds: {config.COMMUNICATION_ROUNDS}")
    print(f"  - Local Epochs: {config.LOCAL_EPOCHS}\n")
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Initialize global model with the maximum client size
    global_model = init_model(config.MAX_HIDDEN_SIZE)
    
    # Load data and create client loaders
    train_dataset, test_dataset = load_data(config.DATA_DIR)
    client_loaders, test_loader = prepare_data(
        train_dataset, 
        test_dataset, 
        config.NUM_CLIENTS, 
        config.BATCH_SIZE,
        config.DATA_SPLIT_MODE,
        config.NON_IID_N_CLASSES_PER_CLIENT
    )
    
    # --- Federated Training Loop ---
    start_time = time.time()
    
    for comm_round in range(1, config.COMMUNICATION_ROUNDS + 1):
        round_start_time = time.time()
        
        client_contributions = []
        
        # Client selection and local training
        for i in range(config.NUM_CLIENTS):
            client_size = config.CLIENT_SIZES[i]
            client_state, size = client_update(
                client_loader=client_loaders[i],
                global_model_state=global_model.state_dict(),
                client_size=client_size,
                local_epochs=config.LOCAL_EPOCHS,
                learning_rate=config.LEARNING_RATE,
                device=device
            )
            client_contributions.append((client_state, size))
            
        # Server aggregation
        global_model_state = aggregate_heterofl(global_model.state_dict(), client_contributions)
        global_model.load_state_dict(global_model_state)
        
        # Evaluate the global model
        test_loss, accuracy = evaluate(global_model, test_loader, device)
        
        round_duration = time.time() - round_start_time
        
        print(f"Round {comm_round:3d}/{config.COMMUNICATION_ROUNDS} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Accuracy: {accuracy:6.2f}% | "
              f"Round Time: {round_duration:.2f}s")

    total_time = time.time() - start_time
    print(f"\nExperiment finished in {total_time/60:.2f} minutes.")
    
    # --- Final Evaluation ---
    final_loss, final_accuracy = evaluate(global_model, test_loader, device)
    print(f"\nFinal Global Model Performance:")
    print(f"  - Test Loss: {final_loss:.4f}")
    print(f"  - Accuracy: {final_accuracy:.2f}%")