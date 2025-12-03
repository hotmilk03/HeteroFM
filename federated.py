import torch
import torch.nn as nn
import torch.optim as optim
import copy

from model import init_model

def client_update(client_loader, global_model_state, client_size, local_epochs, learning_rate, device):
    """
    Creates a local model by slicing the global model, trains it on local data,
    and returns the updated local model state.
    """
    # 1. Create a local model with the client's specific size
    local_model = init_model(client_size).to(device)
    
    # 2. Slice the global model's state to fit the local model
    local_model_state = local_model.state_dict()
    
    # Layer 1 (Linear 784 -> hidden_size)
    local_model_state['layers.1.weight'] = global_model_state['layers.1.weight'][:client_size, :].clone()
    local_model_state['layers.1.bias'] = global_model_state['layers.1.bias'][:client_size].clone()
    
    # Layer 3 (Linear hidden_size -> 10)
    local_model_state['layers.3.weight'] = global_model_state['layers.3.weight'][:, :client_size].clone()
    local_model_state['layers.3.bias'] = global_model_state['layers.3.bias'].clone()
    
    local_model.load_state_dict(local_model_state)
    
    # 3. Standard training loop
    local_model.train()
    optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(local_epochs):
        for data, target in client_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = local_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
    # 4. Return the updated local state and its size
    return local_model.state_dict(), client_size

def aggregate_heterofl(global_model_state, client_contributions):
    """
    Aggregates heterogeneous client models into the global model using the HeteroFL logic.
    """
    # Create a zero-initialized state for aggregation and a counter
    agg_state = copy.deepcopy(global_model_state)
    count_state = copy.deepcopy(global_model_state)
    for key in agg_state:
        agg_state[key] = torch.zeros_like(agg_state[key], dtype=torch.float32)
        count_state[key] = torch.zeros_like(count_state[key], dtype=torch.float32)
        
    # Sum up all client model contributions
    for client_state, client_size in client_contributions:
        # Layer 1 (Linear 784 -> hidden_size)
        agg_state['layers.1.weight'][:client_size, :] += client_state['layers.1.weight']
        agg_state['layers.1.bias'][:client_size] += client_state['layers.1.bias']
        count_state['layers.1.weight'][:client_size, :] += 1
        count_state['layers.1.bias'][:client_size] += 1
        
        # Layer 3 (Linear hidden_size -> 10)
        agg_state['layers.3.weight'][:, :client_size] += client_state['layers.3.weight']
        agg_state['layers.3.bias'] += client_state['layers.3.bias'] # Bias is same size for all
        count_state['layers.3.weight'][:, :client_size] += 1
        count_state['layers.3.bias'] += 1

    # Create the new global state by averaging
    new_global_state = copy.deepcopy(global_model_state)
    
    for key in new_global_state:
        # Use torch.where to avoid division by zero and perform a safe update
        new_global_state[key] = torch.where(
            count_state[key] > 0,
            agg_state[key] / count_state[key],
            new_global_state[key]
        )
        
    return new_global_state
