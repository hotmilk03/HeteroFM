import torch
import torch.nn as nn
import torch.optim as optim
import copy
from model import init_model
from weight_matching import mlp2_permutation_spec, mlp3_permutation_spec, vgg_permutation_spec, resnet50_permutation_spec, weight_matching, apply_permutation
import config

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
    Aggregates heterogeneous client models by first finding a permutation to align neurons
    and then averaging. The reference model selection depends on MATCH mode:
    - Extension (E): Use largest model as reference (since global model is MAX_W)
    - Contraction (C): Use smallest model as reference (since global model is MIN_W)
    """
    if not client_contributions:
        return global_model_state

    # 1. Find the reference client (smallest model)
    target_size_ratio = float('inf')
    ref_client_state = None
    for state, size_ratio in client_contributions:
        if size_ratio < target_size_ratio:
            target_size_ratio = size_ratio
            ref_client_state = state
    
    if ref_client_state is None:
        # This should not happen if client_contributions is not empty
        return global_model_state

    # print(f"\n--- Rearranging models based on reference model (size: {target_size_ratio}, mode: {config.MATCH}) ---")

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
        
    params_a = ref_client_state  # Reference for permutation

    permuted_client_contributions = []
    for client_idx, (client_state, client_size_ratio) in enumerate(client_contributions):
        if client_state is ref_client_state:
            # The reference model does not need to be permuted
            permuted_client_contributions.append((client_state, client_size_ratio))
            # print(f"Skipping permutation for the reference model (size: {client_size}).")
            continue

        if not config.SILENT:
            print(f"[Aggregation] Processing client {client_idx+1}/{len(client_contributions)} (size: {client_size_ratio})...")
        
        params_b = client_state

        if not config.SILENT:
            # print(params_a)
            # print(params_b)
            print(params_a.keys())
            print(params_b.keys())
            for key in params_a:
                if key in params_b:
                    print(f"Layer {key}: params_a shape = {params_a[key].shape}, params_b shape = {params_b[key].shape}")
        
        # Find the permutation that aligns params_b with params_a (the reference model)
        import time
        start_time = time.time()
        perm = weight_matching(
            ps, 
            params_a, 
            params_b, 
            permute_mode=config.PERMUTE,
            match_mode=config.MATCH,
            max_iter=config.SINKHORN_MAX_ITER,
            silent=config.SILENT
        )
        elapsed = time.time() - start_time
        if not config.SILENT or elapsed > 30:
            print(f"[Aggregation] Client {client_idx+1} permutation completed in {elapsed:.2f}s")
        
        # Apply the permutation to params_b
        permuted_params_b = apply_permutation(ps, perm, params_b)

        # print shape of permuted_params_b & params_b
        # print(f"params_b shapes: {[params_b[k].shape for k in params_b]}")
        # print(f"permuted_params_b shapes: {[permuted_params_b[k].shape for k in permuted_params_b]}")
        
        if config.PERM_WARNING and all(torch.allclose(params_b[k], permuted_params_b[k]) for k in params_b):
            print("Warning: Permutation did not change the parameters.")
        else:
            pass
            # print("Permutation applied successfully.")
            # print(f"perm: {perm}")

        permuted_client_contributions.append((permuted_params_b, client_size_ratio))
        # print(f"Finished rearranging client model with size {client_size}.")

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
