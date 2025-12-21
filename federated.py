import torch
import torch.nn as nn
import torch.optim as optim
import copy
from model import init_model

def client_update(client_loader, global_model_state, client_size, scaler_rate, label_split, use_masked_loss, grad_clip_norm, local_epochs, learning_rate, momentum, weight_decay, device):
    # Create a local model with the client's specific size
    local_model = init_model(client_size, scaler_rate).to(device)
    local_model_state = local_model.state_dict()
    
    # Slice the global model's state to fit the local model
    for key in local_model_state:
        if key in global_model_state:
            global_param = global_model_state[key]
            local_shape = local_model_state[key].shape
            
            slices = [slice(0, dim) for dim in local_shape]
            
            local_model_state[key] = global_param[tuple(slices)].clone()
    
    local_model.load_state_dict(local_model_state)
    
    # Standard training loop
    local_model.train()
    optimizer = optim.SGD(local_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(local_epochs):
        for data, target in client_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = local_model(data)

            if use_masked_loss and label_split is not None:
                mask = torch.full_like(output, -float('inf'))
                mask[:, label_split] = 0.0
                output = output + mask

            loss = criterion(output, target)
            loss.backward()

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), grad_clip_norm)
            
            optimizer.step()
            
    return local_model.cpu().state_dict(), client_size

def aggregate_heterofl(global_model_state, client_contributions):
    """
    Aggregates heterogeneous client models into the global model using the HeteroFL logic.
    It averages the weights of the sub-models on the corresponding parts of the global model.
    """
    # Create a zero-initialized state for aggregation and a counter
    agg_state = copy.deepcopy(global_model_state)
    count_state = copy.deepcopy(global_model_state)
    for key in agg_state:
        agg_state[key] = torch.zeros_like(agg_state[key], dtype=torch.float32)
        count_state[key] = torch.zeros_like(count_state[key], dtype=torch.float32)
        
    # Sum up all client model contributions to the appropriate slices
    for client_state, client_size in client_contributions:
        for key, client_param in client_state.items():
            if key in agg_state:
                slices = [slice(0, dim) for dim in client_param.shape]

                agg_state[key][tuple(slices)] += client_param
                count_state[key][tuple(slices)] += 1

    # Create the new global state by averaging
    new_global_state = copy.deepcopy(global_model_state)
    
    for key in new_global_state:
        # Average the weights where count > 0, otherwise keep the old global model weights
        new_global_state[key] = torch.where(
            count_state[key] > 0,
            agg_state[key] / count_state[key],
            new_global_state[key]
        )
        
    return new_global_state
