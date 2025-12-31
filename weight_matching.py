from collections import defaultdict
from typing import NamedTuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

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

# make it scalable
def mlp2_permutation_spec() -> PermutationSpec:
    """Permutation spec for the MLP model in HeteroFM."""
    return permutation_spec_from_axes_to_perm({
        "layers.1.weight": ("P_0", None),

        "layers.3.weight": ("P_0",),
        "layers.3.bias": ("P_0",),
        
        "layers.5.weight": (None, "P_0"),
        "layers.5.bias": (None,),
    })

def mlp3_permutation_spec() -> PermutationSpec:
    """Permutation spec for the MLP model in HeteroFM."""
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

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        if axis == except_axis:
            continue
        if p is not None:
            w = torch.index_select(w, axis, perm[p])
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
               for d, p_name in enumerate(p_spec) if p_name is not None), "params_a must be the smaller model"

    # Initialize permutations for model b
    perm_sizes_b = {p: params_b[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()} # P_0 : size of {layer.1.weight, axis 0}
    perm = {p: torch.arange(n) for p, n in perm_sizes_b.items()} if init_perm is None else init_perm
    perm_names = list(perm.keys()) # [P_0, P_1, ...]

    for iteration in range(max_iter):
        progress = False
        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            
            size_a = params_a[ps.perm_to_axes[p][0][0]].shape[ps.perm_to_axes[p][0][1]]
            size_b = perm_sizes_b[p]

            # print(f"Processing permutation {p} (size_a: {size_a}, size_b: {size_b})")

            # Cost matrix dimensions depend on MATCH mode
            if match_mode == 'C':
                cost_matrix = torch.zeros(size_a, size_b)
            elif match_mode == 'E':
                cost_matrix = torch.zeros(size_b, size_b)
            else:
                raise ValueError(f"Unknown MATCH mode: {match_mode}")

            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk].clone()
                w_b_permuted = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)

                # Handle padding for 'E' mode (zero-padding the smaller model 'a')
                if match_mode == 'E' and size_a < size_b:
                    pad_shape = list(w_a.shape)
                    pad_shape[axis] = size_b - size_a
                    if permute_mode == 'Z':
                        padding = torch.zeros(pad_shape, dtype=w_a.dtype, device=w_a.device)
                    elif permute_mode == 'M':
                        padding = w_b_permuted.narrow(dim=axis, start=size_a, length=size_b - size_a)
                    w_a = torch.cat([w_a, padding], dim=axis)

                # Align axes for dot product
                w_a_flat = w_a.movedim(axis, 0).reshape(w_a.shape[axis], -1)
                w_b_flat = w_b_permuted.movedim(axis, 0).reshape(w_b_permuted.shape[axis], -1)

                if w_a_flat.shape[1] != w_b_flat.shape[1]:
                    if w_a_flat.shape[0] == w_b_flat.shape[1]: w_b_flat = w_b_flat.T
                    else: w_a_flat = w_a_flat.T
                
                dot_product = w_a_flat @ w_b_flat.T
                # print(f"Dot product shape for {wk}, axis {axis}: {dot_product.shape}")

                # Handle PERMUTE mode for cost objective
                if permute_mode == 'M':
                    cost = dot_product
                elif permute_mode == 'Z':
                    cost = -torch.abs(dot_product)
                else:
                    raise ValueError(f"Unknown PERMUTE mode: {permute_mode}")
                
                # The cost matrix size is determined by the dot_product shape
                cost_matrix[:cost.shape[0], :cost.shape[1]] += cost
            
            # Solve assignment problem
            ri, ci = linear_sum_assignment(cost_matrix.numpy(), maximize=True)

            # --- Update permutation for model b ---
            new_perm_for_b = torch.arange(size_b)
            
            if match_mode == 'C':
                # The old permutation maps the first size_a indices of b to some other indices.
                old_L = cost_matrix[torch.arange(size_a), perm[p][:size_a]].sum()
                # Reorder the chosen 'ci' neurons to come first, aligned with 'ri'
                sorted_ci = torch.from_numpy(ci[np.argsort(ri)]).long()
                unmatched_b_indices = torch.tensor([i for i in range(size_b) if i not in ci], dtype=torch.long)
                new_perm_for_b = torch.cat([sorted_ci, unmatched_b_indices])

            elif match_mode == 'E':
                old_L = torch.diag(cost_matrix[:, perm[p]]).sum()
                # The assignment is a full permutation of size_b
                new_perm_for_b = torch.from_numpy(ci[np.argsort(ri)]).long()

            new_L = cost_matrix[ri, ci].sum()
            
            if not silent:
                print(f"Iteration {iteration}/{p}: Improvement {new_L - old_L:.4f}")

            if new_L > old_L + 1e-12: # np.isclose(new_L, old_L)
                progress = True
            
            perm[p] = new_perm_for_b

        if not progress:
            break
            
    return perm

def get_model_params_as_dict(model):
    """Converts a PyTorch model's state_dict to a flattened dictionary."""
    return {k: v.clone() for k, v in model.state_dict().items()}
