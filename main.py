import torch
import torch.nn as nn
import time
import traceback
import concurrent.futures
import copy

import config
import data
import federated
import model

def run_client_gpu(client_id, gpu_index, global_state, client_loader, client_size):
    device = torch.device(f'cuda:{gpu_index}')

    try:
        client_state, _ = federated.client_update(
            client_loader=client_loader,
            global_model_state=global_state,
            client_size=client_size,
            local_epochs=config.LOCAL_EPOCHS,
            learning_rate=config.LEARNING_RATE,
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
    if config.DATA_SPLIT_MODE == 'non-iid':
        print(f"  - Classes per Client: {config.NON_IID_N_CLASSES_PER_CLIENT}")
    print(f"  - Number of Clients: {config.NUM_CLIENTS}")
    print(f"  - Client Hidden Sizes: {config.CLIENT_SIZES}")
    print(f"  - Communication Rounds: {config.COMMUNICATION_ROUNDS}")
    print(f"  - Local Epochs: {config.LOCAL_EPOCHS}")
    print("=================================\n")

    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}\n")

    server_device = torch.device("cpu")
    global_model = model.init_model(config.MAX_HIDDEN_SIZE).to(server_device)

    train_dataset, test_dataset = data.load_data(config.DATA_DIR)
    client_loaders, test_loader = data.prepare_data(
        train_dataset, 
        test_dataset, 
        config.NUM_CLIENTS, 
        config.BATCH_SIZE,
        config.DATA_SPLIT_MODE,
        config.NON_IID_N_CLASSES_PER_CLIENT
    )
    
    # Federated Training Loop
    print("Starting Federated Training...\n")
    start_time = time.time()

    MAX_WORKERS = num_gpus

    # init evaluation before training
    comm_round = 0
    eval_device = torch.device("cuda:0")
    test_loss, accuracy = evaluate(global_model, test_loader, eval_device)
    global_model.cpu()

    print(f"Round {comm_round:3d}/{config.COMMUNICATION_ROUNDS} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Accuracy: {accuracy:6.2f}%")
    
    for comm_round in range(1, config.COMMUNICATION_ROUNDS + 1):
        round_start_time = time.time()
        
        client_contributions = []

        global_model_state = copy.deepcopy(global_model.state_dict())
        print(f"Round {comm_round}: Training on {num_gpus} GPUs in parallel...")

        # parallel client updates using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i in range(config.NUM_CLIENTS):
                gpu_index = i % num_gpus

                futures.append(
                    executor.submit(
                        run_client_gpu,
                        client_id=i,
                        gpu_index=gpu_index,
                        global_state=global_model_state,
                        client_loader=client_loaders[i],
                        client_size=config.CLIENT_SIZES[i]
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    client_contributions.append(result)

        # Server aggregation phase
        global_model_state = federated.aggregate_heterofl(
            global_model.state_dict(), client_contributions
        )
        global_model.load_state_dict(global_model_state)
        
        # Evaluate the global model
        eval_device = torch.device("cuda:0")
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
    eval_device = torch.device("cuda:0")
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
