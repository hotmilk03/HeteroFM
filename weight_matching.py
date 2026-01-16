from collections import defaultdict
from typing import NamedTuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import config
from sinkhorn import Sinkhorn

class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict

def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis)) # P_0 : (layers.1.weight, 0), (layers.3.weight, 0), ...
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)

def mlp2_permutation_spec() -> PermutationSpec:
    """Permutation spec for the MLP2 model."""
    return permutation_spec_from_axes_to_perm({
        "layers.1.weight": ("P_0", None),

        "layers.3.weight": ("P_0",),
        "layers.3.bias": ("P_0",),
        
        "layers.5.weight": (None, "P_0"),
        "layers.5.bias": (None,),
    })

def mlp3_permutation_spec() -> PermutationSpec:
    """Permutation spec for the MLP3 model."""
    return permutation_spec_from_axes_to_perm({
        "layers.1.weight": ("P_0", None), 

        "layers.3.weight": ("P_0",), 
        "layers.3.bias": ("P_0",),

        "layers.5.weight": ("P_1", "P_0"), 

        "layers.7.weight": ("P_1",), 
        "layers.7.bias": ("P_1",),

        "layers.9.weight": (None, "P_1"),
        "layers.9.bias": (None,),
    })

def vgg_permutation_spec() -> PermutationSpec:
    """Permutation spec for the VGG11 model, matching the structure in model.py."""
    spec = {}
    layer_idx = 0
    perm_idx = 0
    in_perm = None
    # Block is Conv-Scaler-BN-ReLU (4 layers), MaxPool is 1 layer
    for layer_spec in config.VGG_CFG['VGG11']:
        if layer_spec == 'M':
            layer_idx += 1
            continue
        
        conv_layer = f"features.{layer_idx}"
        bn_layer = f"features.{layer_idx + 2}"
        out_perm = f"P_{perm_idx}"
        
        spec[f"{conv_layer}.weight"] = (out_perm, in_perm, None, None)
        spec[f"{bn_layer}.weight"] = (out_perm,)
        spec[f"{bn_layer}.bias"] = (out_perm,)
        
        layer_idx += 4
        in_perm = out_perm
        perm_idx += 1

    # If switching to VGG16, note that two additional dense layers are added at the end

    spec["classifier.weight"] = (None, in_perm)
    spec["classifier.bias"] = (None,)
        
    return permutation_spec_from_axes_to_perm(spec)

def resnet50_permutation_spec() -> PermutationSpec:
    """Permutation spec for the ResNet50 model."""
    spec = {}
    spec["conv1.weight"] = ("P0", None, None, None)
    spec["bn1.weight"] = ("P0",)
    spec["bn1.bias"] = ("P0",)

    def _add_block_spec(prefix, p_in, p_out_stage, block_num, has_shortcut_conv):
        p_out_block = p_out_stage if has_shortcut_conv else p_in
        p_mid_a = f"{p_out_stage}_B{block_num}_A"
        p_mid_b = f"{p_out_stage}_B{block_num}_B"
        
        spec[f"{prefix}.{block_num}.conv1.weight"] = (p_mid_a, p_in, None, None)
        spec[f"{prefix}.{block_num}.bn1.weight"] = (p_mid_a,)
        spec[f"{prefix}.{block_num}.bn1.bias"] = (p_mid_a,)
        
        spec[f"{prefix}.{block_num}.conv2.weight"] = (p_mid_b, p_mid_a, None, None)
        spec[f"{prefix}.{block_num}.bn2.weight"] = (p_mid_b,)
        spec[f"{prefix}.{block_num}.bn2.bias"] = (p_mid_b,)

        spec[f"{prefix}.{block_num}.conv3.weight"] = (p_out_block, p_mid_b, None, None)
        spec[f"{prefix}.{block_num}.bn3.weight"] = (p_out_block,)
        spec[f"{prefix}.{block_num}.bn3.bias"] = (p_out_block,)
        
        if has_shortcut_conv:
            spec[f"{prefix}.{block_num}.shortcut.0.weight"] = (p_out_block, p_in, None, None)
            spec[f"{prefix}.{block_num}.shortcut.1.weight"] = (p_out_block,)
            spec[f"{prefix}.{block_num}.shortcut.1.bias"] = (p_out_block,)
        return p_out_block

    p_in = "P0"
    p_stages = ["P1", "P2", "P3", "P4"]
    num_blocks_list = [3, 4, 6, 3]
    for i_stage, (p_out_stage, num_blocks) in enumerate(zip(p_stages, num_blocks_list)):
        for i_block in range(num_blocks):
            has_shortcut = (i_block == 0)
            p_in = _add_block_spec(f"layer{i_stage+1}", p_in, p_out_stage, i_block, has_shortcut)

    spec["linear.weight"] = (None, p_in)
    spec["linear.bias"] = (None,)

    return permutation_spec_from_axes_to_perm(spec)

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    if k not in ps.axes_to_perm:
        return w
    for axis, p in enumerate(ps.axes_to_perm[k]):
        if axis == except_axis:
            continue
        if p is not None:
            # Handle both hard permutations (LongTensor) and soft permutations (FloatTensor)
            perm_p = perm[p]
            if perm_p.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                # Hard permutation: use index_select
                w = torch.index_select(w, axis, perm_p)
            else:
                # Soft permutation: use matrix multiplication
                # Move the axis to position 0, apply soft perm, move back
                w = w.movedim(axis, 0)
                original_shape = w.shape
                w_flat = w.reshape(original_shape[0], -1)
                
                # Apply soft permutation: perm_p @ w_flat
                if perm_p.shape[1] == w_flat.shape[0]:
                    w_flat = perm_p @ w_flat
                else:
                    # Handle size mismatch
                    min_size = min(perm_p.shape[1], w_flat.shape[0])
                    w_flat = perm_p[:, :min_size] @ w_flat[:min_size, :]
                
                # Reshape back
                new_shape = (w_flat.shape[0],) + original_shape[1:]
                w = w_flat.reshape(new_shape)
                w = w.movedim(0, axis)
    return w

def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply a `perm` to `params`."""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}

# TODO : with fast_wm
def weight_matching(ps: PermutationSpec, params_a, params_b, permute_mode, match_mode, max_iter=100, init_perm=None, silent=False):
    """
    Find a permutation of `params_b` to make them match `params_a`.
    This implementation is adapted for PyTorch and handles heterogeneous layer sizes
    based on the `permute_mode` and `match_mode`.
    """
    # Reference model 'a' is always the smaller one
    assert all(params_a[k].shape[d] <= params_b[k].shape[d] 
               for k, p_spec in ps.axes_to_perm.items() 
               for d, p_name in enumerate(p_spec) if p_name is not None and k in params_a), "params_a must be the smaller model"

    # Initialize permutations for model b
    perm_sizes_b = {p: params_b[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()} # P_0 : size of {layer.1.weight, axis 0}
    perm = {p: torch.arange(n) for p, n in perm_sizes_b.items()} if init_perm is None else init_perm
    perm_names = list(perm.keys()) # [P_0, P_1, ...]

    # Gradient Descent mode with Sinkhorn
    if config.SINKHORN:
        return _weight_matching_sinkhorn(
            ps,
            params_a,
            params_b,
            permute_mode,
            match_mode,
            perm_sizes_b,
            perm_names,
            max_iter,
            init_perm,
            silent
        )

    for iteration in range(max_iter):
        progress = False
        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            
            size_a = params_a[ps.perm_to_axes[p][0][0]].shape[ps.perm_to_axes[p][0][1]]
            size_b = perm_sizes_b[p]

            if not silent:
                print(perm_names)
                print(ps.perm_to_axes)
                print(f"Processing permutation {p} (size_a: {size_a}, size_b: {size_b})")

            cost_matrix = torch.zeros(size_a, size_b)
            
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk].clone()
                w_b_permuted = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                
                # Define valid region slice: keep 'axis' full, crop others to A's size
                slices = [slice(0, w_a.shape[d]) if d != axis else slice(None) for d in range(w_a.ndim)]

                if match_mode == 'C':
                    # Cut B to match A's dimensions
                    w_b_permuted = w_b_permuted[tuple(slices)]

                elif match_mode == 'E':
                    # Prepare target shape: A's size on 'axis', B's size on others
                    target_shape = list(w_b_permuted.shape)
                    target_shape[axis] = w_a.shape[axis]

                    if permute_mode == 'M':
                        # Init with B's values (crop 'axis' to match A)
                        axis_subset = [slice(None)] * w_a.ndim
                        axis_subset[axis] = slice(0, w_a.shape[axis])
                        w_a_new = w_b_permuted[tuple(axis_subset)].clone()

                    elif permute_mode == 'Z':
                        # Init with zeros
                        w_a_new = torch.zeros(target_shape, dtype=w_a.dtype, device=w_a.device)
                    
                    # Inject A into the valid region
                    w_a_new[tuple(slices)] = w_a
                    w_a = w_a_new

                # Align axes for dot product
                w_a_flat = w_a.movedim(axis, 0).reshape(w_a.shape[axis], -1)
                w_b_flat = w_b_permuted.movedim(axis, 0).reshape(w_b_permuted.shape[axis], -1)
                
                dot_product = w_a_flat @ w_b_flat.T
                
                if not silent:
                    print(f"w_a_flat shape: {w_a_flat.shape}, w_b_flat shape: {w_b_flat.shape}")
                    print(f"Dot product shape for {wk}, axis {axis}: {dot_product.shape}")

                # Handle PERMUTE mode for cost objective
                if permute_mode == 'M':
                    cost = dot_product
                elif permute_mode == 'Z':
                    cost = -torch.abs(dot_product)
                else:
                    raise ValueError(f"Unknown PERMUTE mode: {permute_mode}")
                
                # The cost matrix size is determined by the dot_product shape
                cost_matrix[:cost.shape[0], :cost.shape[1]] += cost

                if not silent:
                    # should be same
                    print(f"Cost shape for {wk}, axis {axis}: {cost.shape}")
                    print(f"cost_matrix.shape: {cost_matrix.shape}")
                
            # Solve assignment problem
            ri, ci = linear_sum_assignment(cost_matrix.numpy(), maximize=True)

            # --- Update permutation for model b ---
            new_perm_for_b = perm[p].clone()
            
            # The old permutation maps the first size_a indices of b to some other indices.
            old_L = cost_matrix[torch.arange(size_a), perm[p][:size_a]].sum()

            # Reorder the chosen 'ci' neurons to come first, aligned with 'ri'
            sorted_ci = torch.from_numpy(ci[np.argsort(ri)]).long()
            unmatched_b_indices = torch.tensor([i for i in range(size_b) if i not in sorted_ci], dtype=torch.long)
            new_perm_for_b = torch.cat([sorted_ci, unmatched_b_indices])

            new_L = cost_matrix[ri, ci].sum()

            if not silent:
                print(f"Iteration {iteration}/{p}: Improvement {new_L - old_L:.4f}")

            if new_L > old_L + 1e-12: # np.isclose(new_L, old_L)
                # print("progress == TRUE !!")
                progress = True
            else:
                pass
                # print("**progress not permuted")
            
            perm[p] = new_perm_for_b

        if not progress:
            break
            
    return perm

def _weight_matching_sinkhorn(ps: PermutationSpec, params_a, params_b, permute_mode, match_mode,
                            perm_sizes_b, perm_names, max_iter, init_perm, silent):
    """
    Gradient Descent based weight matching using Sinkhorn for soft permutations.
    Optimizes soft permutation matrices by minimizing alignment loss.
    """
    # Initialize learnable permutation matrices
    P_matrices = {}
    size_a_dict = {}
    
    for p in perm_names:
        size_a = params_a[ps.perm_to_axes[p][0][0]].shape[ps.perm_to_axes[p][0][1]]
        size_b = perm_sizes_b[p]
        size_a_dict[p] = size_a
        
        # Initialize with identity-like soft matrix (larger noise for better gradient flow)
        if size_a == size_b:
            P_init = torch.eye(size_a, dtype=torch.float32) + torch.randn(size_a, size_b) * 0.1
        else:
            # For heterogeneous sizes: create block-diagonal identity-like matrix
            P_init = torch.zeros(size_a, size_b, dtype=torch.float32)
            min_size = min(size_a, size_b)
            # Initialize block diagonal with identity + noise (symmnet approach)
            P_init[:min_size, :min_size] = torch.eye(min_size, dtype=torch.float32) + torch.randn(min_size, min_size) * 0.1
        
        P_matrices[p] = torch.nn.Parameter(P_init, requires_grad=True)
    
    # Optimizer for permutation matrices
    lr = getattr(config, "SINKHORN_LR", 0.1)
    optimizer = torch.optim.Adam(P_matrices.values(), lr=lr)
    
    num_sink = getattr(config, "SINKHORN_NUM_ITER", 50)
    lambd_sink = getattr(config, "SINKHORN_LAMBDA", 0.1)
    
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        # Apply Sinkhorn to get soft permutations
        soft_perms = {}
        l = getattr(config, "SINKHORN_SCALING", 1.0)  # Cost scaling factor (matches symmnet: self.l)
        
        for p in perm_names:
            size_a = size_a_dict[p]
            size_b = perm_sizes_b[p]
            
            # Cost matrix: scale by lambda (matches symmnet.py: -self.p[i] * self.l)
            c = -P_matrices[p] * l
            # Marginals: unnormalized ones to match symmnet.py
            # In Sinkhorn.forward: log_a = log(a), so uniform prior with log(1) = 0
            a = torch.ones(size_a, dtype=torch.float32)
            b = torch.ones(size_b, dtype=torch.float32)
            
            soft_perms[p] = Sinkhorn.apply(c, a, b, num_sink, lambd_sink)
        
        # Compute alignment loss using soft permutations
        total_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)        
        
        for p in perm_names:
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk].detach().clone()
                w_b_permuted = get_permuted_param(ps, soft_perms, wk, params_b, except_axis=axis)
                
                # Define valid region slice: keep 'axis' full, crop others to A's size
                slices = [slice(0, w_a.shape[d]) if d != axis else slice(None) for d in range(w_a.ndim)]
                
                if match_mode == 'C':
                    # Cut B to match A's dimensions
                    w_b_permuted = w_b_permuted[tuple(slices)]
                elif match_mode == 'E':
                    # Prepare target shape: A's size on 'axis', B's size on others
                    target_shape = list(w_b_permuted.shape)
                    target_shape[axis] = w_a.shape[axis]
                    
                    if permute_mode == 'M':
                        # Init with B's values (crop 'axis' to match A)
                        axis_subset = [slice(None)] * w_a.ndim
                        axis_subset[axis] = slice(0, w_a.shape[axis])
                        w_a_new = w_b_permuted[tuple(axis_subset)].clone()
                    elif permute_mode == 'Z':
                        # Init with zeros
                        w_a_new = torch.zeros(target_shape, dtype=w_a.dtype, device=w_a.device)
                    
                    # Inject A into the valid region
                    w_a_new[tuple(slices)] = w_a
                    w_a = w_a_new
                
                # Align axes for dot product - same as original weight_matching
                w_a_flat = w_a.movedim(axis, 0).reshape(w_a.shape[axis], -1)
                w_b_flat = w_b_permuted.movedim(axis, 0).reshape(w_b_permuted.shape[axis], -1)
                
                # Compute similarity matrix (not scalar!) for proper gradient flow
                # This creates a (size_a, size_b) matrix where each entry is similarity between neuron pairs
                similarity_matrix = w_a_flat @ w_b_flat.T
                
                # Compute alignment loss using soft permutation
                P_soft = soft_perms[p]
                
                if permute_mode == 'M':
                    # Maximize alignment: -trace(P^T @ similarity_matrix)
                    # This encourages P to assign high weights to high similarity pairs
                    alignment = torch.sum(P_soft * similarity_matrix)
                    total_loss = total_loss - alignment
                elif permute_mode == 'Z':
                    # For zero-padding mode, minimize absolute difference
                    # Use Frobenius norm weighted by P
                    diff_matrix = torch.abs(similarity_matrix)
                    total_loss = total_loss + torch.sum(P_soft * diff_matrix)
        
        # Backward and optimize
        if total_loss.requires_grad:
            total_loss.backward()
            optimizer.step()
        
        if not silent and iteration % 10 == 0:
            print(f"GD Iteration {iteration}: Loss {total_loss.item():.4f}")
        
        # Early stopping if loss is very small
        if total_loss.item() < 1e-6:
            break
    
    # Convert final soft permutations to hard permutations
    final_perm = {}
    for p in perm_names:
        with torch.no_grad():
            size_a = size_a_dict[p]
            size_b = perm_sizes_b[p]
            c = -P_matrices[p]
            a = torch.ones(size_a, dtype=torch.float32) / size_a
            b = torch.ones(size_b, dtype=torch.float32) / size_b
            P_soft = Sinkhorn.apply(c, a, b, num_sink, lambd_sink)
            
            ri, ci = linear_sum_assignment(P_soft.detach().cpu().numpy(), maximize=True)
            sorted_ci = torch.from_numpy(ci[np.argsort(ri)]).long()
            unmatched_b_indices = torch.tensor([i for i in range(size_b) if i not in sorted_ci], dtype=torch.long)
            final_perm[p] = torch.cat([sorted_ci, unmatched_b_indices])
    
    return final_perm

def get_model_params_as_dict(model):
    """Converts a PyTorch model's state_dict to a flattened dictionary."""
    return {k: v.clone() for k, v in model.state_dict().items()}
