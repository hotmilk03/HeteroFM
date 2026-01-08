import torch
import torch.nn as nn
import torch.optim as optim
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import numpy as np

import config
import data
import federated
import model

def run_client_gpu(args):
    client_id, gpu_index, global_state, client_loader, test_loader, client_test_loader, client_size_ratio, scaler_rate, label_split, use_masked_loss, learning_rate, momentum, weight_decay, grad_clip_norm = args
    device = torch.device(f'cuda:{gpu_index}')

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
            device=device
        )
        return client_state, size_ratio, metrics

    except Exception as e:
        print(f"Error in client {client_id} on GPU {gpu_index}: {e}")
        traceback.print_exc()
        raise

def main():
    print("Starting HeteroFM Experiment...")
    print("=================================")
    print(f"  - Model: {config.MODEL}")
    print(f"  - Dataset: {config.DATA_SET}")
    print(f"  - Data Split: {config.DATA_SPLIT_MODE}")
    print(f"  - Number of Clients: {config.NUM_CLIENTS}")
    print(f"  - Client Hidden Sizes Ratio: {config.W_CLIENT}")
    print(f"  - Classes per Client Ratio: {config.NON_IID_CLASSES_RATIO_PER_CLIENT}")
    if config.REARRANGE:
        print("  - Model Aggregation: Rearrangement")
        print(f"  - Permute Mode: {config.PERMUTE}")
        print(f"  - Match Mode: {config.MATCH}")
    else:
        print("  - Model Aggregation: HeteroFL")
    print(f"  - Communication Rounds: {config.COMMUNICATION_ROUNDS}")
    print(f"  - Local Epochs: {config.LOCAL_EPOCHS}")
    print(f"  - Learning Rate: {config.LEARNING_RATE}")
    print("=================================\n")

    # setup
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}\n")

    server_device = torch.device("cpu")
    if config.REARRANGE:
        if config.MATCH == 'C':
            global_model = model.init_model(config.MIN_W).to(server_device)
        elif config.MATCH == 'E':
            global_model = model.init_model(config.MAX_W).to(server_device)
        else:
            raise ValueError(f"Unknown MATCH mode: {config.MATCH}")
    else:
        global_model = model.init_model(config.MAX_W).to(server_device)

    # Optimizer and LR Scheduler
    optimizer = optim.SGD(global_model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=config.LR_DECAY_MILESTONES, 
        gamma=config.LR_DECAY_GAMMA
    )

    # data
    dataset = data.Dataset(config.DATA_SET, config.DATA_DIR)
    client_loaders, client_test_loaders,test_loader, label_splits = dataset.prepare_data(
        config.NUM_CLIENTS, 
        config.BATCH_SIZE,
        config.DATA_SPLIT_MODE,
        config.NON_IID_CLASSES_RATIO_PER_CLIENT
    )

    if config.DATA_SPLIT_MODE == 'non-iid' and config.MODEL in ['mlp2', 'mlp3', 'vgg']:
        print("\n--- Non-IID Label Distribution ---")
        print(f"  - Classes per Client Ratio: {config.NON_IID_CLASSES_RATIO_PER_CLIENT}")
        for client_id, labels in label_splits.items():
            print(f"    - Client {client_id}: Classes {sorted(labels)}")
        print("----------------------------------\n")
    
    # Federated Training Loop
    print("Starting Federated Training...\n")
    start_time = time.time()

    # init evaluation before training
    comm_round = 0
    eval_device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
    test_loss, accuracy = federated.evaluate(global_model, test_loader, eval_device)
    global_model.cpu()

    print(f"Round {comm_round:3d}/{config.COMMUNICATION_ROUNDS} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Accuracy: {accuracy:6.2f}%")
    
    for comm_round in range(1, config.COMMUNICATION_ROUNDS + 1):
        round_start_time = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nRound {comm_round:3d}/{config.COMMUNICATION_ROUNDS} | Learning Rate: {current_lr:.5f}")

        client_contributions = []
        client_metrics = []
        global_model_state_cpu = copy.deepcopy(global_model.state_dict())

        # parallel client updates using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_gpus if num_gpus > 0 else 1) as executor:
            tasks = []
            global_model_state_cpu = copy.deepcopy(global_model.state_dict())
            for i in range(config.NUM_CLIENTS):
                gpu_index = i % num_gpus if num_gpus > 0 else 0
                scaler_rate = config.W_CLIENT[i] / config.MAX_W
                label_split = label_splits.get(i)

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
                    config.GRAD_CLIP_NORM
                )
                tasks.append(args)

            futures = [executor.submit(run_client_gpu, task) for task in tasks]

            for future in as_completed(futures):
                client_state, client_size, metrics = future.result()
                client_contributions.append((client_state, client_size))
                client_metrics.append(metrics)

        # Server aggregation phase
        if config.REARRANGE:
            print("--- Aggregating with rearrangement ---")
            global_model_state = federated.aggregate_rearrange(
                global_model.state_dict(), client_contributions
            )
        else:
            global_model_state = federated.aggregate_heterofl(
                global_model.state_dict(), client_contributions
            )
        global_model.load_state_dict(global_model_state)
        
        optimizer.step() # nothing # erase?
        scheduler.step() # erase?

        # Evaluate the global model
        eval_device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
        test_loss, accuracy = federated.evaluate(global_model, test_loader, eval_device)
        global_model.cpu()

        round_duration = time.time() - round_start_time
        print(f"Round {comm_round:3d}/{config.COMMUNICATION_ROUNDS} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Accuracy: {accuracy:6.2f}% | "
              f"Round Time: {round_duration:.2f}s")
        
        if config.CLIENT_EVAL:
            # Log Client Metrics
            losses = [m['loss'] for m in client_metrics]
            accs = [m['accuracy'] for m in client_metrics]
            local_losses = [m['local_loss'] for m in client_metrics]
            local_accs = [m['local_accuracy'] for m in client_metrics]
            
            print("  - Client Global Accuracy: Min: {:.2f}%, Max: {:.2f}%, Mean: {:.2f}%".format(
                np.min(accs), np.max(accs), np.mean(accs)))
            print("  - Client Global Loss:     Min: {:.4f}, Max: {:.4f}, Mean: {:.4f}".format(
                np.min(losses), np.max(losses), np.mean(losses)))
                
            print("  - Client Local Accuracy:  Min: {:.2f}%, Max: {:.2f}%, Mean: {:.2f}%".format(
                np.min(local_accs), np.max(local_accs), np.mean(local_accs)))
            print("  - Client Local Loss:      Min: {:.4f}, Max: {:.4f}, Mean: {:.4f}".format(
                np.min(local_losses), np.max(local_losses), np.mean(local_losses)))

            # Optional: Print detail for each client if not too many
            print("  - Client Details:")
            for i, m in enumerate(client_metrics):
                print(f"    Client {i}: Acc: {m['accuracy']:6.2f}%, Loss: {m['loss']:.4f}")
            print("  - Client Local Details:")
            for i, m in enumerate(client_metrics):
                print(f"    Client {i}: Local Acc: {m['local_accuracy']:6.2f}%, Local Loss: {m['local_loss']:.4f}")

    total_time = time.time() - start_time
    print(f"\nFederated Training finished in {total_time/60:.2f} minutes.")
    
    # Final Evaluation
    eval_device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
    final_loss, final_accuracy = federated.evaluate(global_model, test_loader, eval_device)
    print(f"\nFinal Global Model Performance:")
    print(f"  - Test Loss: {final_loss:.4f}")
    print(f"  - Accuracy: {final_accuracy:.2f}%")

if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("\n--- AN ERROR OCCURRED ---")
        print(traceback.format_exc())