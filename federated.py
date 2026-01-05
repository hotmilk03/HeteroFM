import torch
import torch.nn as nn
import torch.optim as optim
import copy
from model import init_model
from weight_matching import mlp2_permutation_spec, mlp3_permutation_spec, vgg_permutation_spec, resnet50_permutation_spec, weight_matching, apply_permutation
import config

def client_update(client_loader, global_model_state, client_size_ratio, scaler_rate, label_split, use_masked_loss, grad_clip_norm, local_epochs, learning_rate, momentum, weight_decay, device):
    # Create a local model with the client's specific size
    local_model = init_model(client_size_ratio, scaler_rate).to(device)
    local_model_state = local_model.state_dict()
    
    # Slice the global model's state to fit the local model
    for key in local_model_state:
        if key in global_model_state:
            global_param = global_model_state[key]
            global_shape = global_model_state[key].shape
            local_shape = local_model_state[key].shape
            
            slices = [slice(0, min(global_dim, local_dim)) for global_dim, local_dim in zip(global_shape, local_shape)]
            slices_tuple = tuple(slices)
            
            local_model_state[key][slices_tuple] = global_param[slices_tuple].clone()
    
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
            
    return local_model.cpu().state_dict(), client_size_ratio

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
    for client_state, _ in client_contributions:
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

def aggregate_rearrange(global_model_state, client_contributions):
    """
    Aggregates heterogeneous client models by first finding a permutation to align neurons
    and then averaging. The reference model for permutation is the one with the smallest client size.
    """
    if not client_contributions:
        return global_model_state

    # 1. Find the client with the smallest size to use as the reference model
    min_size_ratio = float('inf')
    ref_client_state = None
    for state, size_ratio in client_contributions:
        if size_ratio < min_size_ratio:
            min_size_ratio = size_ratio
            ref_client_state = state
    
    if ref_client_state is None:
        # This should not happen if client_contributions is not empty
        return global_model_state

    # print(f"\n--- Rearranging models based on the smallest model (size: {min_size}) ---")

    # Define permutation spec for the model
    if config.MODEL == 'mlp2':
        ps = mlp2_permutation_spec()
    elif config.MODEL == 'mlp3':
        ps = mlp3_permutation_spec()
    elif config.MODEL == 'vgg':
        ps = vgg_permutation_spec()
    elif config.MODEL == 'resnet':
        ps = resnet50_permutation_spec()
    else:
        raise ValueError(f"No permutation spec defined for model: {config.MODEL}")
        
    params_a = ref_client_state  # Reference for permutation

    permuted_client_contributions = []
    for client_state, client_size_ratio in client_contributions:
        if client_state is ref_client_state:
            # The reference model does not need to be permuted
            permuted_client_contributions.append((client_state, client_size_ratio))
            # print(f"Skipping permutation for the reference model (size: {client_size}).")
            continue

        # print(f"Rearranging client model with size {client_size}...")
        
        params_b = client_state
        
        # Find the permutation that aligns params_b with params_a (the smallest model)
        perm = weight_matching(
            ps, 
            params_a, 
            params_b, 
            permute_mode=config.PERMUTE,
            match_mode=config.MATCH,
            silent=True
        )
        
        # Apply the permutation to params_b
        permuted_params_b = apply_permutation(ps, perm, params_b)
        
        permuted_client_contributions.append((permuted_params_b, client_size_ratio))
        # print(f"Finished rearranging client model with size {client_size}.")

    # 2. Aggregate the permuted models
    # Create a zero-initialized state for aggregation and a counter
    agg_state = copy.deepcopy(global_model_state)
    count_state = copy.deepcopy(global_model_state)
    for key in agg_state:
        agg_state[key] = torch.zeros_like(agg_state[key], dtype=torch.float32)
        count_state[key] = torch.zeros_like(count_state[key], dtype=torch.float32)

    # Sum up all permuted client model contributions
    for client_state, _ in permuted_client_contributions:
        for key, client_param in client_state.items():
            if key in agg_state:
                slices = [slice(0, min(client_dim, agg_dim)) for client_dim, agg_dim in zip(client_param.shape, agg_state[key].shape)]
                agg_state[key][tuple(slices)] += client_param[tuple(slices)]
                count_state[key][tuple(slices)] += 1

    # Create the new global state by averaging
    new_global_state = copy.deepcopy(global_model_state)
    
    for key in new_global_state:
        new_global_state[key] = torch.where(
            count_state[key] > 0,
            agg_state[key] / count_state[key],
            new_global_state[key]
        )
        
    return new_global_state
