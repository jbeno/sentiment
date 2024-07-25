import warnings
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.", category=FutureWarning)
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.nn as nn
import sys
import os
import time
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report
import sst
import utils
from torch_ddp_neural_classifier import TorchDDPNeuralClassifier

def setup(backend, rank, world_size, debug=False):
    """
    Initialize the distributed environment.
    This function is called on each process.

    Args:
        backend: Backend for distributed training
        rank: Unique identifier of each process
        world_size: Total number of processes
        debug: Debug mode
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if debug:
        print(f"Rank {rank} - Process group initialized with '{backend}' backend on {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    elif rank == 0:
        print(f"Process groups initialized with '{backend}' backend on {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

def cleanup(rank, debug=False):
    """
    Clean up the distributed environment.
    This function is called on each process.

    Args:
        rank: Unique identifier of the process
        debug: Debug mode
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        if debug:
            print(f"Rank {rank} - Process group destroyed")

def bert_phi(texts, tokenizer, model, device, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer.batch_encode_plus(
            batch_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        embeddings.append(batch_embeddings)
    
    return torch.cat(embeddings, dim=0)

def convert_numeric_to_labels(numeric_preds):
    label_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return [label_dict[pred] for pred in numeric_preds]

def gather_tensors(tensor, world_size):
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list)

def main(rank,
         world_size,
         device_type,
         backend,
         batch_size=32,
         epochs=10,
         lr=0.001,
         sample_percent=None,
         random_seed=42,
         early_stop=True,
         debug=False):
    """
    Main function to train a distributed PyTorch model.

    Args:
        rank: Unique identifier of the process
        world_size: Total number of processes
        device_type: Device to use for training
        backend: Backend for distributed training
        batch_size: Batch size for training on each GPU
        lr: Learning rate
        epochs: Number of epochs to train
        sample_percent: Percentage of data to use for training
        random_seed: Random seed for reproducibility
        early_stop: Use early stopping
        debug: Debug mode
    """
    try:       
        start_time = time.time()
        if device_type == "cuda":
            device = torch.device('cuda', rank)
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')
        print(f"Rank {rank} - Device: {device}, Device Type: {device.type}, Backend: {backend}")

        setup(backend, rank, world_size, debug)
        
        # Set random seed
        if debug:
            print(f"Rank {rank} - Setting random seed to {random_seed}")
        elif rank == 0:
            print(f"Setting random seed to {random_seed}")
        utils.fix_random_seeds(random_seed)

        # Initialize tokenizer and model
        model_init_start = time.time()
        weights_name = 'bert-base-uncased'
        bert_tokenizer = BertTokenizer.from_pretrained(weights_name)
        bert_model = BertModel.from_pretrained(weights_name).to(device)
        if device_type == "cuda":
            bert_model = DDP(bert_model, device_ids=[rank])
        else:
            bert_model = DDP(bert_model, device_ids=None)
        if debug:
            print(f"Rank {rank} - {weights_name} model initialized ({time.time() - model_init_start:.2f}s)")
        elif rank == 0:
            print(f"{weights_name} model initialized ({time.time() - model_init_start:.2f}s)")

        # Load data
        data_load_start = time.time()
        SST_HOME = os.path.join("data", "sentiment")
        train = sst.train_reader(SST_HOME)
        dev = sst.dev_reader(SST_HOME)
        if sample_percent is not None:
            if debug:
                print(f"Rank {rank} - Using {sample_percent * 100:.2f}% of the data")
            elif rank == 0:
                print(f"Using {sample_percent * 100:.2f}% of the data")
            train = train.sample(frac=sample_percent)
            dev = dev.sample(frac=sample_percent)
        if debug:
            print(f"Rank {rank} - Train size: {len(train)}, Dev size: {len(dev)}")
        elif rank == 0:
            print(f"Train size: {len(train)}, Dev size: {len(dev)}")

        X_str_train = train.sentence.values
        y_train = train.label.values

        X_str_dev = dev.sentence.values
        y_dev = dev.label.values
        if debug:
            print(f"Rank {rank} - SST data loaded ({time.time() - data_load_start:.2f}s)")
        elif rank == 0:
            print(f"SST data loaded ({time.time() - data_load_start:.2f}s)")

        # Process training data
        data_process_start = time.time()
        if debug:
            print(f"Rank {rank} - Processing training data...")
        elif rank == 0:
            print("Processing training data...")
        X_train_tensor = bert_phi(X_str_train, bert_tokenizer, bert_model, device, batch_size).to(device)
        X_dev_tensor = bert_phi(X_str_dev, bert_tokenizer, bert_model, device, batch_size).to(device)
        if debug:
            print(f"Rank {rank} - X_train shape: {list(X_train_tensor.shape)}, X_dev shape: {list(X_dev_tensor.shape)}")
            print(f"Rank {rank} - Data processed ({time.time() - data_process_start:.2f}s)")
        elif rank == 0:
            print(f"X_train shape: {list(X_train_tensor.shape)}, X_dev shape: {list(X_dev_tensor.shape)}")
            print(f"Data processed ({time.time() - data_process_start:.2f}s)")
        
        # Initialize classifier
        if debug:
            print(f"Rank {rank} - Initializing classifier...")
            print(f"Rank {rank} - Batch size: {batch_size}, Max epochs: {epochs}, Early stop: {early_stop}")
        elif rank == 0:
            print(f"Initializing classifier...")
            print(f"Batch size: {batch_size}, Max epochs: {epochs}, Early stop: {early_stop}")
        model = TorchDDPNeuralClassifier(
            early_stopping=early_stop,
            hidden_dim=300,
            batch_size=batch_size,
            max_iter=epochs)

        # Fit the model
        training_start = time.time()
        model.fit(X_train_tensor.cpu().numpy(), y_train, rank, world_size, device, debug)

        if debug:
            print(f"Rank {rank} - Training completed in {time.time() - training_start:.2f} seconds")
        elif rank == 0:
            print(f"Training completed in {time.time() - training_start:.2f} seconds")

        # Evaluation
        eval_start = time.time()
        if debug:
            print(f"Rank {rank} - Evaluating model")
        elif rank == 0:
            print("Evaluating model...")

        model.model.eval()
        with torch.no_grad():
            preds = model.model(X_dev_tensor)
            
            # Gather predictions from all processes
            all_preds = [torch.zeros_like(preds) for _ in range(world_size)]
            dist.all_gather(all_preds, preds)
            
            if rank == 0:
                # Concatenate all predictions
                all_preds = torch.cat(all_preds, dim=0)
                
                # Trim predictions to match the original dev set size
                all_preds = all_preds[:len(y_dev)]
                
                preds_labels = convert_numeric_to_labels(all_preds.argmax(dim=1).cpu().numpy())
                
                print("Classification report:")
                print(classification_report(y_dev, preds_labels, digits=3, zero_division=0))

                print(f"Evaluation completed ({time.time() - eval_start:.2f}s)")
                print(f"TOTAL Time: {time.time() - start_time:.2f} seconds")

        # Synchronize all processes before cleanup
        dist.barrier()
    finally:
        cleanup(rank, debug)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="DDP PyTorch Training")
    parser.add_argument('--device', type=str, default=None, help="Device: auto-detected, or specify 'cuda' or 'cpu'")
    parser.add_argument('--gpus', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training on each GPU (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--sample_percent', type=float, default=None, help='Percentage of data to use for training')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--early_stop', type=bool, default=True, help='Use early stopping (default: True)')
    parser.add_argument('--num_threads', type=int, default=None, help='Number of threads for CPU training')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode (default: False)')
    args = parser.parse_args()

    print("Starting DDP PyTorch Training...")
    # Check if device is specified, else auto-detect
    if args.device is not None:
        device_type = args.device
    else:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    # If device is cuda, check GPUs and set backend to 'nccl'
    if device_type == "cuda":
        if args.gpus is not None:
            world_size = args.gpus
        else:
            world_size = torch.cuda.device_count()
        print(f"Device: {device_type}, Number of GPUs: {world_size}")
        backend = "nccl"
    # If device is cpu, ignore --gpus argument and set backend to 'gloo'
    elif device_type == "cpu":
        world_size = 1
        print(f"Device: {device_type}")
        if args.gpus is not None:
            print("Warning: Ignoring --gpus argument because device is set to 'cpu'")
        backend = "gloo"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        world_size = 1
        print(f"Warning: Device {device_type} not recognized. Defaulting to 'cpu'")
        device_type = "cpu"
        backend = "gloo"

    if args.num_threads is not None:
        os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
        os.environ['MKL_NUM_THREADS'] = str(args.num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(args.num_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(args.num_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(args.num_threads)
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(args.num_threads)
        print(f"Using {args.num_threads} threads for PyTorch")

    # Spawn processes
    suffix = '' if world_size == 1 else 'es'
    print(f"Spawning {world_size} process{suffix} using '{backend}' backend...")
    try:
        mp.spawn(main,
                args=(world_size,
                    device_type,
                    backend,
                    args.batch_size,
                    args.epochs,
                    args.lr,
                    args.sample_percent,
                    args.random_seed,
                    args.early_stop,
                    args.debug),
                nprocs=world_size,
                join=True)
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
    finally:
        pass
