import torch
import torch.nn as nn
import torch.optim as optim
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

import config
import data
import federated
import model

def run_client_gpu(args):
    client_id, gpu_index, global_state, client_loader, client_size, scaler_rate, label_split, use_masked_loss, learning_rate, grad_clip_norm = args
    device = torch.device(f'cuda:{gpu_index}')

    try:
        client_state, _ = federated.client_update(
            client_loader=client_loader,
            global_model_state=global_state,
            client_size=client_size,
            scaler_rate=scaler_rate,
            label_split=label_split,
            use_masked_loss=use_masked_loss,
            local_epochs=config.LOCAL_EPOCHS,
            learning_rate=learning_rate,
            grad_clip_norm=grad_clip_norm,
            device=device
        )
        return client_state, client_size

    except Exception as e:
        print(f"Error in client {client_id} on GPU {gpu_index}: {e}")
        traceback.print_exc()
        raise

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

def main():
    print("Starting HeteroFM Experiment...")
    print("=================================")
    print(f"  - Data Split: {config.DATA_SPLIT_MODE}")
    print(f"  - Number of Clients: {config.NUM_CLIENTS}")
    print(f"  - Client Hidden Sizes: {config.CLIENT_SIZES}")
    print(f"  - Communication Rounds: {config.COMMUNICATION_ROUNDS}")
    print(f"  - Local Epochs: {config.LOCAL_EPOCHS}")
    print("=================================\n")

    # setup
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}\n")

    server_device = torch.device("cpu")
    global_model = model.init_model(config.MAX_HIDDEN_SIZE).to(server_device)

    # Optimizer and LR Scheduler
    optimizer = optim.SGD(global_model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=config.LR_DECAY_MILESTONES, 
        gamma=config.LR_DECAY_GAMMA
    )

    # data
    train_dataset, test_dataset = data.load_data(config.DATA_DIR)
    client_loaders, test_loader, label_splits = data.prepare_data(
        train_dataset, 
        test_dataset, 
        config.NUM_CLIENTS, 
        config.BATCH_SIZE,
        config.DATA_SPLIT_MODE,
        config.NON_IID_N_CLASSES_PER_CLIENT
    )

    if config.DATA_SPLIT_MODE == 'non-iid':
        print("\n--- Non-IID Label Distribution ---")
        print(f"  - Classes per Client: {config.NON_IID_N_CLASSES_PER_CLIENT}")
        for client_id, labels in label_splits.items():
            print(f"    - Client {client_id}: Classes {sorted(labels)}")
        print("----------------------------------\n")

    
    # Federated Training Loop
    print("Starting Federated Training...\n")
    start_time = time.time()

    # init evaluation before training
    comm_round = 0
    eval_device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
    test_loss, accuracy = evaluate(global_model, test_loader, eval_device)
    global_model.cpu()

    print(f"Round {comm_round:3d}/{config.COMMUNICATION_ROUNDS} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Accuracy: {accuracy:6.2f}%")
    
    for comm_round in range(1, config.COMMUNICATION_ROUNDS + 1):
        round_start_time = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nRound {comm_round:3d}/{config.COMMUNICATION_ROUNDS} | Learning Rate: {current_lr:.5f}")

        client_contributions = []
        global_model_state_cpu = copy.deepcopy(global_model.state_dict())

        # parallel client updates using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_gpus if num_gpus > 0 else 1) as executor:
            tasks = []
            global_model_state_cpu = copy.deepcopy(global_model.state_dict())
            for i in range(config.NUM_CLIENTS):
                gpu_index = i % num_gpus if num_gpus > 0 else 0
                scaler_rate = config.CLIENT_SIZES[i] / config.MAX_HIDDEN_SIZE
                label_split = label_splits.get(i)

                args = (
                    i,
                    gpu_index,
                    global_model_state_cpu,
                    client_loaders[i],
                    config.CLIENT_SIZES[i],
                    scaler_rate,
                    label_split,
                    config.USE_MASKED_LOSS,
                    current_lr,
                    config.GRAD_CLIP_NORM
                )
                tasks.append(args)

            futures = [executor.submit(run_client_gpu, task) for task in tasks]

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    client_contributions.append(result)

        # Server aggregation phase
        global_model_state = federated.aggregate_heterofl(
            global_model.state_dict(), client_contributions
        )
        global_model.load_state_dict(global_model_state)
        
        optimizer.step() # nothing
        scheduler.step()

        # Evaluate the global model
        eval_device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
        test_loss, accuracy = evaluate(global_model, test_loader, eval_device)
        global_model.cpu()

        round_duration = time.time() - round_start_time
        print(f"Round {comm_round:3d}/{config.COMMUNICATION_ROUNDS} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Accuracy: {accuracy:6.2f}% | "
              f"Round Time: {round_duration:.2f}s")

    total_time = time.time() - start_time
    print(f"\nFederated Training finished in {total_time/60:.2f} minutes.")
    
    # Final Evaluation
    eval_device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
    final_loss, final_accuracy = evaluate(global_model, test_loader, eval_device)
    print(f"\nFinal Global Model Performance:")
    print(f"  - Test Loss: {final_loss:.4f}")
    print(f"  - Accuracy: {final_accuracy:.2f}%")

if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("\n--- AN ERROR OCCURRED ---")
        print(traceback.format_exc())