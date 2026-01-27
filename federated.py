import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from model import init_model
from weight_matching import mlp2_permutation_spec, mlp3_permutation_spec, vgg_permutation_spec, resnet50_permutation_spec, weight_matching, apply_permutation
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

def client_update(client_loader, test_loader, local_test_loader, global_model_state, client_size_ratio, scaler_rate, label_split, use_masked_loss, grad_clip_norm, local_epochs, learning_rate, momentum, weight_decay, device, previous_client_state=None, client_id=None):
    # Initialize local model state from previous round or create fresh weights
    # Assumption: client model shape remains constant across rounds
    if previous_client_state is not None:
        # Use previous round's state (preserves non-overlapping weights when local > global)
        local_model_state = copy.deepcopy(previous_client_state)
    else:
        # First round: initialize with fresh random weights from init_model
        # Use client-specific seed for reproducibility while maintaining independence
        client_seed = None
        if client_id is not None and hasattr(config, 'SEED') and config.SEED is not None:
            client_seed = config.SEED + client_id + 1  # +1 to avoid using base seed
        local_model = init_model(client_size_ratio, scaler_rate, seed=client_seed).to(device)
        local_model_state = local_model.state_dict()
    
    use_amp = bool(config.USE_AMP) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # Overwrite overlapping regions with global model weights
    for key in local_model_state:
        if key in global_model_state:
            global_param = global_model_state[key]
            global_shape = global_model_state[key].shape
            local_shape = local_model_state[key].shape
            
            # Copy only the overlapping region from global model
            slices = [slice(0, min(global_dim, local_dim)) for global_dim, local_dim in zip(global_shape, local_shape)]
            slices_tuple = tuple(slices)
            
            local_model_state[key][slices_tuple] = global_param[slices_tuple].clone()
    
    # Load the state into a model instance for training
    local_model = init_model(client_size_ratio, scaler_rate).to(device)
    local_model.load_state_dict(local_model_state)

    prev_test_loss, prev_test_acc = evaluate(local_model, test_loader, device)
    
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

        metrics = {
            'loss': global_test_loss,
            'accuracy': global_test_acc,
            'prev_loss': prev_test_loss,
            'prev_accuracy': prev_test_acc
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

def interpolation_experiment(params_a, params_b, size_a, size_b, test_loader, device='cpu', match_mode=None):
    """
    Performs interpolation experiment between two aligned models.
    Lambda ranges from -0.2 to 1.2 in steps of 0.02.
    
    Args:
        params_a: First model parameters (already permuted)
        params_b: Second model parameters (already permuted)
        size_a: Size ratio of first model
        size_b: Size ratio of second model
        test_loader: Test dataset loader for evaluation
        device: Device for evaluation
        match_mode: 'E' (Extension), 'C' (Contraction), or None (HeteroFL - zero padding)
    """
    print("\n" + "="*80)
    print("Starting Interpolation Experiment between first two clients")
    print(f"Match mode: {match_mode if match_mode else 'HeteroFL (zero padding)'}")
    print("="*80)
    
    lambdas = np.arange(config.INTERPOLATION_LAMBDA_START, config.INTERPOLATION_LAMBDA_END + config.INTERPOLATION_LAMBDA_STEP, config.INTERPOLATION_LAMBDA_STEP)  # -0.2 to 1.2, step 0.02
    results = []
    
    # Prepare aligned models based on match_mode
    aligned_params_a = {}
    aligned_params_b = {}
    
    if match_mode is None or match_mode == 'E':
        # HeteroFL or Extension mode: Align to the larger model size with zero padding
        max_size = max(size_a, size_b)
        
        for key in params_a:
            if key in params_b:
                shape_a = params_a[key].shape
                shape_b = params_b[key].shape
                
                # Determine target shape (maximum of both dimensions)
                target_shape = tuple(max(dim_a, dim_b) for dim_a, dim_b in zip(shape_a, shape_b))
                
                # Zero-pad params_a to target shape
                param_a_padded = torch.zeros(target_shape, dtype=params_a[key].dtype, device=params_a[key].device)
                slices_a = tuple(slice(0, dim) for dim in shape_a)
                param_a_padded[slices_a] = params_a[key]
                
                # Zero-pad params_b to target shape
                param_b_padded = torch.zeros(target_shape, dtype=params_b[key].dtype, device=params_b[key].device)
                slices_b = tuple(slice(0, dim) for dim in shape_b)
                param_b_padded[slices_b] = params_b[key]
                
                aligned_params_a[key] = param_a_padded
                aligned_params_b[key] = param_b_padded
        
        eval_size = max_size
        
    elif match_mode == 'C':
        # Contraction mode: Align to the smaller model size
        min_size = min(size_a, size_b)
        
        for key in params_a:
            if key in params_b:
                shape_a = params_a[key].shape
                shape_b = params_b[key].shape
                
                # Determine target shape (minimum of both dimensions)
                target_shape = tuple(min(dim_a, dim_b) for dim_a, dim_b in zip(shape_a, shape_b))
                
                # Contract params_a to target shape
                slices = tuple(slice(0, dim) for dim in target_shape)
                param_a_contracted = params_a[key][slices]
                
                # Contract params_b to target shape
                param_b_contracted = params_b[key][slices]
                
                aligned_params_a[key] = param_a_contracted
                aligned_params_b[key] = param_b_contracted
        
        eval_size = min_size
    
    else:
        raise ValueError(f"Invalid match_mode: {match_mode}")
    
    # Now interpolate between aligned models
    for lam in lambdas:
        interpolated_state = {}
        
        for key in aligned_params_a:
            # Interpolate: (1 - lambda) * param_a + lambda * param_b
            interpolated_param = (1 - lam) * aligned_params_a[key] + lam * aligned_params_b[key]
            interpolated_state[key] = interpolated_param
        
        # Create a model with the interpolated weights
        eval_model = init_model(eval_size).to(device)
        
        # Load interpolated weights into the model
        model_state = eval_model.state_dict()
        for key in interpolated_state:
            if key in model_state:
                model_state[key] = interpolated_state[key].to(device)
        
        eval_model.load_state_dict(model_state)
        
        # Evaluate on test dataset
        test_loss, test_acc = evaluate(eval_model, test_loader, device)
        results.append((lam, test_loss, test_acc))
        
        print(f"lambda = {lam:5.2f} | Test Loss: {test_loss:.6f} | Test Accuracy: {test_acc:6.2f}%")
        
        # Clean up
        del eval_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print("="*80)
    print("Interpolation Experiment Complete")
    print("="*80 + "\n")
    
    return results

def aggregate_heterofl(global_model_state, client_contributions, device='cpu', comm_round=None, test_loader=None):
    """
    Aggregates heterogeneous client models into the global model using the HeteroFL logic.
    It averages the weights of the sub-models on the corresponding parts of the global model.
    
    Args:
        comm_round: Current communication round (for interpolation experiment)
        test_loader: Test dataset loader (for interpolation experiment)
    """
    # Interpolation experiment for first round with first two clients (without permutation)
    if (comm_round <= config.INTERPOLATION_ROUND and config.INTERPOLATION and 
        len(client_contributions) >= 2 and test_loader is not None):
        # Find the client with width == 1 to use as params_a
        width1_state = None
        width1_size = None
        width1_idx = None
        other_state = None
        other_size = None
        other_idx = None
        
        for idx, (state, size) in enumerate(client_contributions[:2]):
            # Check if this client has width == 1
            effective_width = config.W_CLIENT[idx]
            if config.MODEL == 'vgg11':
                effective_width *= config.VGG_WIDTH_MULTIPLIER
            
            if effective_width == 1.0:
                width1_state = state
                width1_size = size
                width1_idx = idx
            else:
                other_state = state
                other_size = size
                other_idx = idx
        
        # Use width==1 client as params_a if found, otherwise use first client
        if width1_state is not None and other_state is not None:
            params_a, size_a = width1_state, width1_size
            params_b, size_b = other_state, other_size
            print(f"\n[Interpolation - HeteroFL] Starting experiment with client {width1_idx} (width=1, size={size_a}) vs client {other_idx} (size={size_b})")
        else:
            params_a, size_a = client_contributions[0]
            params_b, size_b = client_contributions[1]
            print(f"\n[Interpolation - HeteroFL] Starting experiment between client 0 (size={size_a}) and client 1 (size={size_b})")
        
        interpolation_experiment(
            params_a=params_a,
            params_b=params_b,
            size_a=size_a,
            size_b=size_b,
            test_loader=test_loader,
            device=device,
            match_mode=None  # HeteroFL uses zero padding
        )
    
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

def aggregate_rearrange(global_model_state, client_contributions, device='cpu', comm_round=None, test_loader=None):
    """
    Aggregates heterogeneous client models by first finding a permutation to align neurons
    and then averaging. The reference model selection depends on MATCH mode:
    - Extension (E): Use largest model as reference (since global model is MAX_W)
    - Contraction (C): Use smallest model as reference (since global model is MIN_W)
    
    Args:
        comm_round: Current communication round (for interpolation experiment)
        test_loader: Test dataset loader (for interpolation experiment)
    """
    if not client_contributions:
        return global_model_state

    # Find the reference client (smallest model)
    target_size_ratio = float('inf')
    ref_client_state = None
    for state, size_ratio in client_contributions:
        if size_ratio < target_size_ratio:
            target_size_ratio = size_ratio
            ref_client_state = state
    
    if ref_client_state is None:
        # This should not happen if client_contributions is not empty
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
    
    params_a = ref_client_state  # Reference for permutation

    # Track width==1 client for interpolation experiment
    width1_client_state = None
    width1_client_size = None
    width1_client_idx = None
    width1_permuted_state = None

    permuted_client_contributions = []
    for client_idx, (client_state, client_size_ratio) in enumerate(client_contributions):
        # Check if this client has width == 1
        effective_width = config.W_CLIENT[client_idx]
        if config.MODEL == 'vgg11':
            effective_width *= config.VGG_WIDTH_MULTIPLIER
        
        is_width1 = (effective_width == 1.0)
        
        if client_state is ref_client_state:
            # The reference model does not need to be permuted
            permuted_client_contributions.append((client_state, client_size_ratio))
            if is_width1:
                width1_client_state = client_state
                width1_client_size = client_size_ratio
                width1_client_idx = client_idx
                width1_permuted_state = client_state  # No permutation for ref
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
        
        # Track width==1 client
        if is_width1:
            width1_client_state = client_state
            width1_client_size = client_size_ratio
            width1_client_idx = client_idx
            width1_permuted_state = permuted_params_b
        
        # print(f"Finished rearranging client model with size {client_size}.")
        
        # Interpolation experiment for first round with first two clients
        # print(f"comm_round: {comm_round}, INTERPOLATION_ROUND: {config.INTERPOLATION_ROUND}, INTERPOLATION: {config.INTERPOLATION}, client_idx: {client_idx}, test_loader: {test_loader is not None}")
        if (comm_round <= config.INTERPOLATION_ROUND and config.INTERPOLATION and 
            client_idx <= 1 and test_loader is not None):
            # Determine which state to use as params_a (prefer width==1)
            if is_width1:
                # Current client is width==1, use it as params_a
                interp_params_a = permuted_params_b
                interp_size_a = client_size_ratio
                # Use the other client (ref) as params_b
                interp_params_b = params_a
                interp_size_b = target_size_ratio
                print(f"\n[Interpolation - Rearrange] Starting experiment with client {client_idx} (width=1, size={interp_size_a}) vs client 0 (ref, size={interp_size_b})")
            elif width1_client_state is not None:
                # We already processed width==1 client, use it as params_a
                interp_params_a = width1_permuted_state
                interp_size_a = width1_client_size
                interp_params_b = permuted_params_b
                interp_size_b = client_size_ratio
                print(f"\n[Interpolation - Rearrange] Starting experiment with client {width1_client_idx} (width=1, size={interp_size_a}) vs client {client_idx} (size={interp_size_b})")
            else:
                # No width==1 client yet, use ref as params_a
                interp_params_a = params_a
                interp_size_a = target_size_ratio
                interp_params_b = permuted_params_b
                interp_size_b = client_size_ratio
                print(f"\n[Interpolation - Rearrange] Starting experiment between client 0 (ref, size={interp_size_a}) and client {client_idx} (size={interp_size_b})")
            
            interpolation_experiment(
                params_a=interp_params_a,
                params_b=interp_params_b,
                size_a=interp_size_a,
                size_b=interp_size_b,
                test_loader=test_loader,
                device=device,
                match_mode=config.MATCH  # Use the same match mode as weight matching
            )

    # Aggregate the permuted models on GPU
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
