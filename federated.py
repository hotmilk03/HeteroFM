import torch
import torch.nn as nn
import torch.optim as optim
import copy
from model import init_model
from weight_matching import mlp2_permutation_spec, mlp3_permutation_spec, vgg_permutation_spec, resnet50_permutation_spec, weight_matching, hierarchical_weight_matching, apply_permutation
import config
import time

def evaluate(model, test_loader, device):
    """
    Evaluates the model's performance on the test dataset.
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            test_loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

def client_update(client_loader, test_loader, local_test_loader, global_model_state, client_size_ratio, scaler_rate, label_split, use_masked_loss, grad_clip_norm, local_epochs, learning_rate, momentum, weight_decay, device):
    # Create a local model with the client's specific size
    local_model = init_model(client_size_ratio, scaler_rate).to(device)
    local_model_state = local_model.state_dict()
    use_amp = bool(config.USE_AMP) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
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
    
    optimizer = optim.SGD(local_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Standard training loop
    local_model.train()

    try:
        for _ in range(local_epochs):
            for data, target in client_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    output = local_model(data)

                    if use_masked_loss and label_split is not None:
                        mask = torch.full_like(output, -float('inf'))
                        mask[:, label_split] = 0.0
                        output = output + mask

                    loss = criterion(output, target)

                scaler.scale(loss).backward()

                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), grad_clip_norm)
                
                scaler.step(optimizer)
                scaler.update()

        # Evaluation on Global test_loader
        global_test_loss, global_test_acc = evaluate(local_model, test_loader, device)

        # Evaluation on Local test_loader (specific classes)
        local_test_loss, local_test_acc = evaluate(local_model, local_test_loader, device)

        metrics = {
            'loss': global_test_loss,
            'accuracy': global_test_acc,
            'local_loss': local_test_loss,
            'local_accuracy': local_test_acc
        }

        # Move weights back to CPU before freeing CUDA memory
        final_state_dict = local_model.cpu().state_dict()

    finally:
        # Ensure optimizer/memory cleanup even on error
        del local_model
        del optimizer
        del criterion
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return final_state_dict, client_size_ratio, metrics

def aggregate_heterofl(global_model_state, client_contributions, device='cpu'):
    """
    Aggregates heterogeneous client models into the global model using the HeteroFL logic.
    It averages the weights of the sub-models on the corresponding parts of the global model.
    """
    # Create a zero-initialized state for aggregation and a counter
    agg_state = copy.deepcopy(global_model_state)
    count_state = copy.deepcopy(global_model_state)
    for key in agg_state:
        agg_state[key] = torch.zeros_like(agg_state[key], dtype=torch.float32).to(device)
        count_state[key] = torch.zeros_like(count_state[key], dtype=torch.float32).to(device)
        
    # Sum up all client model contributions to the appropriate slices
    for client_state, _ in client_contributions:
        for key, client_param in client_state.items():
            if key in agg_state:
                client_param = client_param.to(device)
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
            new_global_state[key].to(device)
        ).cpu()
        
    return new_global_state

def aggregate_rearrange(global_model_state, client_contributions, device='cpu'):
    """
    Aggregates heterogeneous client models using hierarchical weight matching.
    Uses hierarchical matching to align all models progressively from smallest to largest.
    """
    if not client_contributions:
        return global_model_state

    # Define permutation spec for the model
    if config.MODEL == 'mlp2':
        ps = mlp2_permutation_spec()
    elif config.MODEL == 'mlp3':
        ps = mlp3_permutation_spec()
    elif config.MODEL == 'vgg11':
        ps = vgg_permutation_spec()
    elif config.MODEL == 'resnet50':
        ps = resnet50_permutation_spec()
    else:
        raise ValueError(f"No permutation spec defined for model: {config.MODEL}")

    # Extract all client states and size ratios
    client_states = [state for state, _ in client_contributions]
    client_size_ratios = [size_ratio for _, size_ratio in client_contributions]
    
    if not config.SILENT:
        print(f"\n--- Hierarchical Weight Matching for {len(client_states)} clients ---")
        print(f"Client sizes: {client_size_ratios}")
    
    # Perform hierarchical weight matching
    start_time = time.time()
    
    permutations = hierarchical_weight_matching(
        ps=ps,
        model_params_list=client_states,
        permute_mode=config.PERMUTE,
        max_iter=config.SINKHORN_MAX_ITER,
        silent=config.SILENT
    )
    
    elapsed = time.time() - start_time
    if not config.SILENT or elapsed > 30:
        print(f"[Aggregation] Hierarchical matching completed in {elapsed:.2f}s")
    
    # Apply permutations to all clients
    permuted_client_contributions = []
    for client_idx, ((client_state, client_size_ratio), perm) in enumerate(zip(client_contributions, permutations)):
        if not config.SILENT:
            print(f"[Aggregation] Applying permutation to client {client_idx+1}/{len(client_contributions)} (size: {client_size_ratio})")
        
        # Apply the permutation
        permuted_params = apply_permutation(ps, perm, client_state)
        
        if config.PERM_WARNING and all(torch.allclose(client_state[k], permuted_params[k]) for k in client_state):
            print(f"  Warning: Permutation did not change client {client_idx+1} parameters.")
        
        permuted_client_contributions.append((permuted_params, client_size_ratio))

    # 2. Aggregate the permuted models on GPU
    agg_state = copy.deepcopy(global_model_state)
    count_state = copy.deepcopy(global_model_state)
    for key in agg_state:
        agg_state[key] = torch.zeros_like(agg_state[key], dtype=torch.float32).to(device)
        count_state[key] = torch.zeros_like(count_state[key], dtype=torch.float32).to(device)

    # Sum up all permuted client model contributions
    # After permutation, all models should have compatible shapes with global model
    for client_state, _ in permuted_client_contributions:
        for key, client_param in client_state.items():
            if key in agg_state:
                client_param = client_param.to(device)
                global_shape = agg_state[key].shape
                client_shape = client_param.shape
                
                # Create slices for valid region (intersection of client and global dimensions)
                slices = [slice(0, min(client_dim, global_dim)) 
                         for client_dim, global_dim in zip(client_shape, global_shape)]
                
                # Add client contribution to global state
                agg_state[key][tuple(slices)] += client_param[tuple(slices)]
                count_state[key][tuple(slices)] += 1

    # Create the new global state by averaging
    new_global_state = copy.deepcopy(global_model_state)
    
    for key in new_global_state:
        new_global_state[key] = torch.where(
            count_state[key] > 0,
            agg_state[key] / count_state[key],
            new_global_state[key].to(device)
        ).cpu()
        
    return new_global_state
