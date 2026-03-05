from datetime import timedelta
import os
import torch
import torch.optim as optim
import torch.distributed as dist
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import numpy as np
import argparse
import random

import config
import data
import federated
import model

# Reduce CUDA memory fragmentation if the env var is unset.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def set_seed(seed):
    """
    Set random seed for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value. If None, does nothing.
    """
    if seed is None:
        return
    
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _init_distributed():
    """Initialize torch.distributed from SLURM or torchrun envs if available."""
    if dist.is_available() and (os.environ.get("RANK") is not None or os.environ.get("SLURM_PROCID") is not None):
        # Derive ranks/world_size
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        # Master addr/port
        master_addr = os.environ.get("MASTER_ADDR")
        
        # Fallback: on SLURM, use first hostname
        master_addr = master_addr or "127.0.0.1"
            
        # Use job ID to avoid port conflicts when multiple jobs run on same node
        base_port = 29500
        job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
        port_offset = job_id % 1000  # Keep offset within reasonable range
        master_port = str(base_port + port_offset)

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        if not config.SILENT:
            print(f"[Rank {rank}] Master: {master_addr}:{master_port}, World Size: {world_size}")
        
        dist.init_process_group(
            backend=backend, 
            rank=rank, 
            world_size=world_size,
            device_id=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None,
            timeout=timedelta(minutes=360)
        )
        
        if not config.SILENT:
            print(f"[Rank {rank}] Process group initialized successfully")

        # Set device to local_rank if CUDA
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        return True, rank, world_size, local_rank
    return False, 0, 1, 0

def run_client_gpu(args):
    client_id, gpu_index, global_state, client_loader, test_loader, client_test_loader, client_size_ratio, scaler_rate, label_split, use_masked_loss, learning_rate, momentum, weight_decay, grad_clip_norm, previous_client_state = args
    device = torch.device(f'cuda:{gpu_index}')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    try:
        client_state, size_ratio, metrics = federated.client_update(
            client_loader=client_loader,
            test_loader=test_loader,
            local_test_loader=client_test_loader,
            global_model_state=global_state,
            client_size_ratio=client_size_ratio,
            scaler_rate=scaler_rate,
            label_split=label_split,
            use_masked_loss=use_masked_loss,
            local_epochs=config.LOCAL_EPOCHS,
            learning_rate=learning_rate,
            grad_clip_norm=grad_clip_norm,
            momentum=momentum,
            weight_decay=weight_decay,
            device=device,
            previous_client_state=previous_client_state,
            client_id=client_id
        )
        return client_state, size_ratio, metrics

    except Exception as e:
        print(f"Error in client {client_id} on GPU {gpu_index}: {e}")
        traceback.print_exc()
        raise

    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def print_info(label_splits):
    print("=== HeteroFM Configuration ===")
    print("=================================")
    print(f"  - Model: {config.MODEL}")
    print(f"  - Norm: {config.NORM}")
    print(f"  - Dataset: {config.DATA_SET}")
    if config.MODEL == 'vgg11':
        print(f"    - VGG Width Multiplier: {config.VGG_WIDTH_MULTIPLIER}")
    if config.DATA_SET == 'imagenet':
        print(f"    - ImageNet Subset Ratio: {config.IMAGENET_SUBSET_RATIO}")
    print(f"  - Client Interpolation: {config.INTERPOLATION}")
    print(f"  - Data Split: {config.DATA_SPLIT_MODE}")
    if config.DATA_SPLIT_MODE == 'non-iid':
        print(f"    - Dynamic Split: {config.DYNAMIC_ON_NON_IID_SPLIT}")
    print(f"  - Batch Size: {config.BATCH_SIZE}")
    print(f"  - Number of Workers: {config.NUM_WORKERS}")
    print(f"  - Number of Clients: {config.NUM_CLIENTS}")
    print(f"  - Client Hidden Sizes Ratio: {config.W_CLIENT}")
    print(f"  - Classes per Client Ratio: {config.NON_IID_CLASSES_RATIO_PER_CLIENT}")
    if config.REARRANGE:
        print("  - Model Aggregation: Rearrangement")
        print(f"    - Permute Mode: {config.PERMUTE}")
        print(f"    - Match Mode: {config.MATCH}")
        if config.REARRANGE_INTERPOLATE:
            print(f"    - Interpolation Permute Mode: {config.PERMUTE_INTERPOLATE}")
            print(f"    - Interpolation Match Mode: {config.MATCH_INTERPOLATE}")
            print(f"    - Interpolation Weight: {config.INTERPOLATE_WEIGHT}")
            mode_label = {'E': 'E-dominant', 'C': 'C-dominant', 'interp': 'Weighted interpolation'}.get(config.GLOBAL_MODEL_SIZE_MODE, config.GLOBAL_MODEL_SIZE_MODE)
            print(f"    - Global Model Size Mode: {mode_label}")
        print(f"  - Sinkhorn: {config.SINKHORN}")
        if config.SINKHORN:
            print(f"    - Sinkhorn Lambda: {config.SINKHORN_LAMBDA}")
            print(f"    - Sinkhorn Num Iterations: {config.SINKHORN_NUM_ITER}")
            print(f"    - Sinkhorn Learning Rate: {config.SINKHORN_LR}")
    else:
        print("  - Model Aggregation: HeteroFL")
    print(f"  - Communication Rounds: {config.COMMUNICATION_ROUNDS}")
    print(f"  - Local Epochs: {config.LOCAL_EPOCHS}")
    print(f"  - Learning Rate: {config.LEARNING_RATE}")
    print("=================================\n")

    if config.DATA_SPLIT_MODE == 'non-iid' and config.MODEL in ['mlp2', 'mlp3', 'vgg11']:
        print("\n--- Non-IID Label Distribution ---")
        print(f"  - Classes per Client Ratio: {config.NON_IID_CLASSES_RATIO_PER_CLIENT}")
        for client_id, labels in label_splits.items():
            print(f"    - Client {client_id}: Classes {sorted(labels)}")
        print("----------------------------------\n")

def parse_args():
    """Parse command line arguments to override config values"""
    parser = argparse.ArgumentParser(description='HeteroFM Federated Learning')
    
    # Model parameters
    parser.add_argument('--model', type=str, default=None, choices=['mlp2', 'mlp3', 'vgg11', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='Model architecture')
    parser.add_argument('--w-client', type=str, default=None,
                        help='Client width ratios (comma-separated, e.g., "1,1,0.5,0.5")')
    parser.add_argument('--vgg-width-multiplier', type=int, default=None,
                        help='VGG channel width multiplier (e.g., 1, 2, 4, 8)')
    parser.add_argument('--norm', type=str, default=None, choices=['bn', 'in', 'ln', 'gn', 'none'],
                        help='Normalization layer type (bn=BatchNorm, in=InstanceNorm, ln=LayerNorm, gn=GroupNorm, none=no norm)')
    
    # Training parameters
    parser.add_argument('--rounds', type=int, default=None,
                        help='Number of communication rounds')
    parser.add_argument('--local-epochs', type=int, default=None,
                        help='Number of local epochs per round')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--grad-clip', type=float, default=None,
                        help='Gradient clipping norm')
    
    # Data parameters
    parser.add_argument('--data-split', type=str, default=None, choices=['iid', 'non-iid'],
                        help='Data split mode')
    parser.add_argument('--non-iid-ratio', type=float, default=None,
                        help='Classes ratio per client in non-iid setting')
    parser.add_argument('--imagenet-subset', type=float, default=None,
                        help='ImageNet subset ratio')
    
    # Aggregation parameters
    parser.add_argument('--rearrange', action='store_true', default=None,
                        help='Use rearrangement aggregation')
    parser.add_argument('--no-rearrange', action='store_false', dest='rearrange',
                        help='Use HeteroFL aggregation')
    parser.add_argument('--permute', type=str, default=None, choices=['Z', 'M'],
                        help='Permutation mode')
    parser.add_argument('--match', type=str, default=None, choices=['C', 'E'],
                        help='Match mode')
    parser.add_argument('--sinkhorn', action='store_true', default=None,
                        help='Enable Sinkhorn')
    parser.add_argument('--no-sinkhorn', action='store_false', dest='sinkhorn',
                        help='Disable Sinkhorn')
    
    # Rearrangement Interpolation parameters
    parser.add_argument('--rearrange-interpolate', action='store_true', default=None,
                        help='Enable interpolation between two mode pairs')
    parser.add_argument('--no-rearrange-interpolate', action='store_false', dest='rearrange_interpolate',
                        help='Disable interpolation between two mode pairs')
    parser.add_argument('--permute-interpolate', type=str, default=None, choices=['Z', 'M'],
                        help='Second permute mode to interpolate with')
    parser.add_argument('--match-interpolate', type=str, default=None, choices=['C', 'E'],
                        help='Second match mode to interpolate with')
    parser.add_argument('--interpolate-weight', type=float, default=None,
                        help='Weight for interpolation (0.0 to 1.0)')
    parser.add_argument('--global-model-size-mode', type=str, default=None, choices=['E', 'C', 'interp'],
                        help='Global model size rule when REARRANGE_INTERPOLATE=True: '
                             'E=E-dominant (MAX_W if either is E), '
                             'C=C-dominant (MIN_W if either is C), '
                             'interp=weighted interpolation')
    
    # Model Interpolation parameters
    parser.add_argument('--client-interpolation', action='store_true', default=None,
                        help='Enable model interpolation analysis')
    parser.add_argument('--no-client-interpolation', action='store_false', dest='interpolation',
                        help='Disable model interpolation analysis')
    parser.add_argument('--interpolation-lambda-step', type=float, default=None,
                        help='Lambda step size for interpolation')
    
    args = parser.parse_args()
    return args

def apply_args_to_config(args):
    """Override config values with command line arguments"""
    if args.model is not None:
        config.MODEL = args.model
        # Update dataset accordingly
        model_to_dataset = {
            'mlp2': 'mnist',
            'mlp3': 'mnist',
            'vgg11': 'cifar10',
            'resnet18': 'cifar10',
            'resnet34': 'cifar10',
            'resnet50': 'imagenet',
            'resnet101': 'imagenet',
            'resnet152': 'imagenet'
        }
        config.DATA_SET = model_to_dataset[config.MODEL]
    
    if args.w_client is not None:
        w_values = [float(x.strip()) for x in args.w_client.split(',')]
        config.W_CLIENT = w_values
        config.NUM_CLIENTS = len(w_values)
        config.MAX_W = max(w_values)
        config.MIN_W = min(w_values)

    # Update VGG width settings (must recompute dependent configs)
    if getattr(args, 'vgg_width_multiplier', None) is not None:
        config.VGG_WIDTH_MULTIPLIER = int(args.vgg_width_multiplier)
        # Recompute VGG config and base width derived values
        config.VGG_CFG = {
            key: [item if item == 'M' else item * config.VGG_WIDTH_MULTIPLIER for item in config.VGG_CFG_ORIGIN[key]]
            for key in config.VGG_CFG_ORIGIN
        }
        config.VGG_BASE_WIDTH = 512 * config.VGG_WIDTH_MULTIPLIER
    
    if args.norm is not None:
        config.NORM = args.norm
    
    if args.rounds is not None:
        config.COMMUNICATION_ROUNDS = args.rounds
    
    if args.local_epochs is not None:
        config.LOCAL_EPOCHS = args.local_epochs
    
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    
    if args.grad_clip is not None:
        config.GRAD_CLIP_NORM = args.grad_clip
    
    if args.data_split is not None:
        config.DATA_SPLIT_MODE = args.data_split
    
    if args.non_iid_ratio is not None:
        config.NON_IID_CLASSES_RATIO_PER_CLIENT = args.non_iid_ratio
    
    if args.imagenet_subset is not None:
        config.IMAGENET_SUBSET_RATIO = args.imagenet_subset
    
    if args.rearrange is not None:
        config.REARRANGE = args.rearrange
    
    if args.permute is not None:
        config.PERMUTE = args.permute
    
    if args.match is not None:
        config.MATCH = args.match
    
    if args.sinkhorn is not None:
        config.SINKHORN = args.sinkhorn
    
    if args.rearrange_interpolate is not None:
        config.REARRANGE_INTERPOLATE = args.rearrange_interpolate
    
    if args.permute_interpolate is not None:
        config.PERMUTE_INTERPOLATE = args.permute_interpolate
    
    if args.match_interpolate is not None:
        config.MATCH_INTERPOLATE = args.match_interpolate
    
    if args.interpolate_weight is not None:
        config.INTERPOLATE_WEIGHT = args.interpolate_weight
    
    if args.global_model_size_mode is not None:
        config.GLOBAL_MODEL_SIZE_MODE = args.global_model_size_mode
    
    if args.interpolation is not None:
        config.INTERPOLATION = args.interpolation
    
    if args.interpolation_lambda_step is not None:
        config.INTERPOLATION_LAMBDA_STEP = args.interpolation_lambda_step

def main():
    # Parse command line arguments and override config
    args = parse_args()
    apply_args_to_config(args)
    
    # Set random seed for reproducibility
    if hasattr(config, 'SEED') and config.SEED is not None:
        set_seed(config.SEED)
        if not hasattr(config, 'SILENT') or not config.SILENT:
            print(f"Set random seed to {config.SEED} for reproducibility\n")
    
    # setup
    torch.backends.cudnn.benchmark = (not hasattr(config, 'SEED') or config.SEED is None)
    is_dist, rank, world_size, local_rank = _init_distributed()
    num_gpus = torch.cuda.device_count()
    if not config.SILENT:
        if is_dist:
            # In distributed mode, each process can use multiple local GPUs
            print(f"[Rank {rank}/{world_size}] Local GPUs: {num_gpus}")
        print(f"Number of available GPUs (visible to this process): {num_gpus}\n")

    server_device = torch.device("cpu")
    # Initialize global model with base seed (0 for global model)
    global_seed = config.SEED if (hasattr(config, 'SEED') and config.SEED is not None) else None
    if config.REARRANGE:
        if config.REARRANGE_INTERPOLATE:
            def _match_to_w(match):
                return config.MAX_W if match == 'E' else config.MIN_W
            mode = config.GLOBAL_MODEL_SIZE_MODE
            if mode == 'interp':
                # Weighted interpolation of global sizes
                w = config.INTERPOLATE_WEIGHT
                global_w = _match_to_w(config.MATCH) * (1 - w) + _match_to_w(config.MATCH_INTERPOLATE) * w
            elif mode == 'C':
                # C-dominant: if either is 'C', use MIN_W
                if config.MATCH == 'C' or config.MATCH_INTERPOLATE == 'C':
                    global_w = config.MIN_W
                else:
                    global_w = config.MAX_W
            else:  # 'E' (default)
                # E-dominant: if either is 'E', use MAX_W
                if config.MATCH == 'E' or config.MATCH_INTERPOLATE == 'E':
                    global_w = config.MAX_W
                else:
                    global_w = config.MIN_W
            global_model = model.init_model(global_w, seed=global_seed).to(server_device)
        else:
            if config.MATCH == 'C':
                global_model = model.init_model(config.MIN_W, seed=global_seed).to(server_device)
            elif config.MATCH == 'E':
                global_model = model.init_model(config.MAX_W, seed=global_seed).to(server_device)
            else:
                raise ValueError(f"Unknown MATCH mode: {config.MATCH}")
    else:
        global_model = model.init_model(config.MAX_W, seed=global_seed).to(server_device)

    # Optimizer and LR Scheduler
    optimizer = optim.SGD(global_model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=config.LR_DECAY_MILESTONES, 
        gamma=config.LR_DECAY_GAMMA
    )

    # Data preparation
    if is_dist:
        # Distributed mode: rank 0 computes indices, broadcast to all ranks
        if not config.SILENT:
            print(f"[Rank {rank}] Preparing data in distributed mode...")
        
        # All ranks create dataset instance (lightweight)
        dataset = data.Dataset(config.DATA_SET, config.DATA_DIR)
        
        # Rank 0 computes client indices, then broadcast to all
        if rank == 0:
            client_indices_map, label_splits = dataset.compute_client_indices(
                num_clients=config.NUM_CLIENTS,
                data_split_mode=config.DATA_SPLIT_MODE,
                n_classes_ratio=config.NON_IID_CLASSES_RATIO_PER_CLIENT
            )
            data_split_info = [{'indices': client_indices_map, 'labels': label_splits}]
            if not config.SILENT:
                print(f"[Rank 0] Broadcasting client indices to all ranks...")
        else:
            data_split_info = [None]
        
        # Broadcast indices to all ranks (much faster than file I/O)
        dist.broadcast_object_list(data_split_info, src=0)
        client_indices_map = data_split_info[0]['indices']
        label_splits = data_split_info[0]['labels']
        
        # Each rank determines its assigned clients
        assigned_clients = [i for i in range(config.NUM_CLIENTS) if (i % world_size) == rank]
        
        if not config.SILENT:
            print(f"[Rank {rank}] Creating DataLoaders for assigned clients {assigned_clients}...")
        
        # Each rank creates DataLoaders ONLY for its assigned clients
        client_loaders, client_test_loaders, test_loader = dataset.create_dataloaders(
            client_indices_map=client_indices_map,
            label_splits=label_splits,
            batch_size=config.BATCH_SIZE,
            client_ids=assigned_clients
        )
        
        # Sync before proceeding
        dist.barrier()
    else:
        # Single process mode
        if not config.SILENT:
            print("Preparing data on single process...")
        dataset = data.Dataset(config.DATA_SET, config.DATA_DIR)
        client_loaders, client_test_loaders, test_loader, label_splits = dataset.prepare_data(
            config.NUM_CLIENTS,
            config.BATCH_SIZE,
            config.DATA_SPLIT_MODE,
            config.NON_IID_CLASSES_RATIO_PER_CLIENT
        )

    # Federated Training Loop
    if not config.SILENT:
        print("Starting Federated Training...\n")
    start_time = time.time()

    # Initialize dictionary to store each client's state across rounds
    # Key: client_id, Value: client state dict
    previous_client_states = {}

    # init evaluation before training (only on rank 0)
    comm_round = 0
    if (not is_dist) or (is_dist and rank == 0):
        eval_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test_loss, accuracy = federated.evaluate(global_model, test_loader, eval_device)
        global_model.cpu()

        print_info(label_splits)

        print(f"Round {comm_round:3d}/{config.COMMUNICATION_ROUNDS} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Accuracy: {accuracy:6.2f}%")
    
    for comm_round in range(1, config.COMMUNICATION_ROUNDS + 1):
        round_start_time = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        # print(f"\nRound {comm_round:3d}/{config.COMMUNICATION_ROUNDS} | Learning Rate: {current_lr:.5f}")

        client_contributions = []
        client_metrics = []
        global_model_state_cpu = copy.deepcopy(global_model.state_dict())

        if not is_dist:
            # parallel client updates using ThreadPoolExecutor on single process
            with ThreadPoolExecutor(max_workers=num_gpus if num_gpus > 0 else 1) as executor:
                tasks = []
                global_model_state_cpu = copy.deepcopy(global_model.state_dict())
                for i in range(config.NUM_CLIENTS):
                    gpu_index = i % num_gpus if num_gpus > 0 else 0
                    scaler_rate = config.W_CLIENT[i] / config.MAX_W
                    label_split = label_splits.get(i)
                    
                    # Get previous client state if it exists
                    previous_state = previous_client_states.get(i, None)

                    args = (
                        i,
                        gpu_index,
                        global_model_state_cpu,
                        client_loaders[i],
                        test_loader,
                        client_test_loaders[i],
                        config.W_CLIENT[i],
                        scaler_rate,
                        label_split,
                        config.USE_MASKED_LOSS,
                        current_lr,
                        config.MOMENTUM,
                        config.WEIGHT_DECAY,
                        config.GRAD_CLIP_NORM,
                        previous_state
                    )
                    tasks.append(args)

                futures = [executor.submit(run_client_gpu, task) for task in tasks]

                for idx, future in enumerate(as_completed(futures)):
                    client_state, client_size, metrics = future.result()
                    client_contributions.append((client_state, client_size))
                    client_metrics.append(metrics)
        else:
            # Distributed: Each rank processes a partition of clients using local GPUs via ThreadPool
            assigned_clients = [i for i in range(config.NUM_CLIENTS) if (i % world_size) == rank]
            
            # Use ThreadPool to parallelize across local GPUs
            with ThreadPoolExecutor(max_workers=num_gpus if num_gpus > 0 else 1) as executor:
                tasks = []
                for idx, i in enumerate(assigned_clients):
                    gpu_index = idx % num_gpus if num_gpus > 0 else 0  # Cycle through local GPUs
                    scaler_rate = config.W_CLIENT[i] / config.MAX_W
                    label_split = label_splits.get(i)
                    
                    # Get previous client state if it exists
                    previous_state = previous_client_states.get(i, None)
                    
                    args = (
                        i,
                        gpu_index,
                        global_model_state_cpu,
                        client_loaders[i],
                        test_loader,
                        client_test_loaders[i],
                        config.W_CLIENT[i],
                        scaler_rate,
                        label_split,
                        config.USE_MASKED_LOSS,
                        current_lr,
                        config.MOMENTUM,
                        config.WEIGHT_DECAY,
                        config.GRAD_CLIP_NORM,
                        previous_state
                    )
                    tasks.append(args)
                
                futures = [executor.submit(run_client_gpu, task) for task in tasks]
                
                for future in as_completed(futures):
                    client_state, client_size, metrics = future.result()
                    client_contributions.append((client_state, client_size))
                    client_metrics.append(metrics)
            
            # Collect all contributions from all ranks using broadcast_object_list
            all_contributions = [None for _ in range(world_size)]
            dist.all_gather_object(all_contributions, client_contributions)
            
            # Use all_gather_object only for lightweight metrics
            all_metrics = [None for _ in range(world_size)]
            dist.all_gather_object(all_metrics, client_metrics)
            
            if rank == 0:
                # Flatten the list of lists
                client_contributions = [item for sublist in all_contributions for item in sublist]
                client_metrics = [item for sublist in all_metrics for item in sublist]
                        
            # Sync before aggregation
            dist.barrier()

        # Update previous_client_states with the current round's results
        # This needs to be done before aggregation so we have it for next round
        if not is_dist or (is_dist and rank == 0):
            for idx, (client_state, client_size) in enumerate(client_contributions):
                previous_client_states[idx] = copy.deepcopy(client_state)

        # Server aggregation phase
        if not is_dist or (is_dist and rank == 0):
            agg_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if config.REARRANGE:
                print("--- Aggregating with rearrangement ---")
                # print(f"[DEBUG - Before aggregate_rearrange] client_contributions length={len(client_contributions)}")
                # for idx, (state, size) in enumerate(client_contributions):
                #     print(f"[DEBUG - client {idx}] features.0.weight shape={state['features.0.weight'].shape}, size={size}")
                global_model_state = federated.aggregate_rearrange(
                    global_model.state_dict(), 
                    client_contributions, 
                    device=agg_device,
                    comm_round=comm_round,
                    test_loader=test_loader
                )
            else:
                global_model_state = federated.aggregate_heterofl(
                    global_model.state_dict(), 
                    client_contributions, 
                    device=agg_device,
                    comm_round=comm_round,
                    test_loader=test_loader
                )
        else:
            global_model_state = None

        # Broadcast updated global state to all ranks
        if is_dist:
            obj_list = [global_model_state]
            dist.broadcast_object_list(obj_list, src=0)
            global_model_state = obj_list[0]

        global_model.load_state_dict(global_model_state)
        
        # no optimizer.step() on global model
        # scheduler.step() updates learning rate for next round
        scheduler.step()

        # Evaluate the global model
        if (not is_dist) or (is_dist and rank == 0):
            eval_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            test_loss, accuracy = federated.evaluate(global_model, test_loader, eval_device)
            global_model.cpu()

            round_duration = time.time() - round_start_time
            print(f"Round {comm_round:3d}/{config.COMMUNICATION_ROUNDS} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Accuracy: {accuracy:6.2f}% | "
                f"Round Time: {round_duration:.2f}s")
        
        if ((not is_dist) or (is_dist and rank == 0)) and config.CLIENT_EVAL:
            # Log Client Metrics
            losses = [m['loss'] for m in client_metrics]
            accs = [m['accuracy'] for m in client_metrics]
            prev_losses = [m['prev_loss'] for m in client_metrics]
            prev_accs = [m['prev_accuracy'] for m in client_metrics]
            
            print("  - Client Global Accuracy: Min: {:.2f}%, Max: {:.2f}%, Mean: {:.2f}%".format(
                np.min(accs), np.max(accs), np.mean(accs)))
            print("  - Client Global Loss:     Min: {:.4f}, Max: {:.4f}, Mean: {:.4f}".format(
                np.min(losses), np.max(losses), np.mean(losses)))
                
            print("  - Client Prev Accuracy:  Min: {:.2f}%, Max: {:.2f}%, Mean: {:.2f}%".format(
                np.min(prev_accs), np.max(prev_accs), np.mean(prev_accs)))
            print("  - Client Prev Loss:      Min: {:.4f}, Max: {:.4f}, Mean: {:.4f}".format(
                np.min(prev_losses), np.max(prev_losses), np.mean(prev_losses)))

            # Optional: Print detail for each client if not too many
            print("  - Client Details:")
            for i, m in enumerate(client_metrics):
                print(f"    Client {i}: Acc: {m['accuracy']:6.2f}%, Loss: {m['loss']:.4f}")
            print("  - Client Prev Details:")
            for i, m in enumerate(client_metrics):
                print(f"    Client {i}: Prev Acc: {m['prev_accuracy']:6.2f}%, Prev Loss: {m['prev_loss']:.4f}")

        # Proactively release CUDA caches between rounds to avoid fragmentation across clients.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    total_time = time.time() - start_time
    if (not is_dist) or (is_dist and rank == 0):
        print(f"\nFederated Training finished in {total_time/60:.2f} minutes.")
        
        # Final Evaluation
        eval_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        final_loss, final_accuracy = federated.evaluate(global_model, test_loader, eval_device)
        print(f"\nFinal Global Model Performance:")
        print(f"  - Test Loss: {final_loss:.4f}")
        print(f"  - Accuracy: {final_accuracy:.2f}%")

    if is_dist and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("\n--- AN ERROR OCCURRED ---")
        print(traceback.format_exc())
        if dist.is_initialized():
            dist.destroy_process_group()