import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.", category=FutureWarning)
warnings.filterwarnings("ignore", message="TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class.", category=UserWarning, module=r"torch\._utils")
warnings.filterwarnings("ignore", message="promote has been superseded by promote_options='default'", category=FutureWarning, module=r"datasets\.table")

# Standard library imports
import argparse
import os
import sys
import signal
import time
from collections import Counter
import traceback
import zipfile

# PyTorch imports
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

# Third-party library imports
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report
import sst
from datasets import load_dataset

# Custom utility imports
from utils import (
    setup_environment, 
    prepare_device, 
    fix_random_seeds,
    convert_numeric_to_labels, 
    convert_labels_to_tensor,
    format_time, 
    convert_sst_label, 
    get_activation,
    set_threads,
    signal_handler,
    cleanup_and_exit
)
from torch_ddp_neural_classifier import TorchDDPNeuralClassifier

# Suppress Hugging Face library warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.repocard").setLevel(logging.ERROR)

from torch.utils.data import Dataset

class SentencesDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

def save_data_archive(X_train, X_dev, y_train, y_dev, X_dev_sent, world_size, device_type, data_dir):
    # Create directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Create filename with appropriate suffix and timestamp
    suffix = f'_{world_size}_gpu' if device_type == 'cuda' else '_1_cpu'
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    filename = f'data{suffix}_{timestamp}.npz'
    filepath = os.path.join(data_dir, filename)

    # Save data to archive file
    np.savez_compressed(filepath, X_train=X_train, X_dev=X_dev, y_train=y_train, y_dev=y_dev, X_dev_sent=X_dev_sent)
    print(f"Data saved to: {filepath}")

def load_data_archive(archive_file, device, rank):
    load_archive_start = time.time()
    
    # Check if the archive file path is provided
    if archive_file is None:
        raise ValueError("No archive file provided to load data from")
    
    # Check if the archive file exists
    if not os.path.exists(archive_file):
        raise FileNotFoundError(f"Archive file not found: {archive_file}")
    
    # Attempt to load the data from the archive file
    try:
        print(f"\nLoading archived data from: {archive_file}...") if rank == 0 else None
        with np.load(archive_file, allow_pickle=True) as data:
            X_train = data['X_train']
            X_dev = data['X_dev']
            y_train = data['y_train']
            y_dev = data['y_dev']
            X_dev_sent = data['X_dev_sent']
        print(f"Archived data loaded ({format_time(time.time() - load_archive_start)})") if rank == 0 else None
    except Exception as e:
        raise RuntimeError(f"Failed to load data from archive file {archive_file}: {str(e)}")
    
    return X_train, X_dev, y_train, y_dev, X_dev_sent

def initialize_bert_model(weights_name, device, rank, device_type, debug):
    torch.manual_seed(42)  # Set a fixed seed for model initialization
    model_init_start = time.time()
    print(f"\nInitializing '{weights_name}' tokenizer and model...") if rank == 0 else None
    bert_tokenizer = BertTokenizer.from_pretrained(weights_name)
    bert_model = BertModel.from_pretrained(weights_name).to(device)
    device_ids_list = [rank] if device_type == "cuda" else None
    bert_model = DDP(bert_model, device_ids=device_ids_list)
    dist.barrier()
    if rank == 0:
        if debug:
            print(f"Bert Tokenizer:\n{bert_tokenizer}")
            print(f"Bert Model:\n{bert_model}")
        print(f"Tokenizer and model initialized ({format_time(time.time() - model_init_start)})")
    return bert_tokenizer, bert_model

def load_data(dataset, eval_dataset, sample_percent, rank, debug):
    data_load_start = time.time()
    print(f"\nLoading data...") if rank == 0 else None
    if eval_dataset is not None:
        print("Using different datasets for training and evaluation") if rank == 0 else None
    else:
        eval_dataset = dataset
        print("Using the same dataset for training and evaluation") if rank == 0 else None
    
    # Function to get a subset of data based on split name
    def get_split(data, split):
            if split == 'train':
                data_split = data['train'].to_pandas()
            elif split == 'dev':
                data_split = data['validation'].to_pandas()
            elif split == 'test':
                data_split = data['test'].to_pandas()
            else:
                raise ValueError(f"Unknown split: {split}")
            return data_split
    
    def print_label_dist(dataset, label_name='label'):
        dist = sorted(Counter(dataset[label_name]).items())
        for k, v in dist:
            print(f"\t{k.capitalize():>14s}: {v}")

    # Function to load data from Hugging Face or local based on ID and split name
    def get_data(id, split, rank, debug):
        # Identify the dataset and path from the ID
        dataset_source = 'Hugging Face'
        dataset_subset = None
        dataset_url = None
        if id == 'sst_local':
            dataset_name = 'Stanford Sentiment Treebank (SST)'
            dataset_source = 'Local'
            dataset_path = os.path.join('data', 'sentiment')
        elif id == 'sst':
            dataset_name = 'Stanford Sentiment Treebank (SST)'
            dataset_url = 'https://huggingface.co/datasets/gimmaru/SetFit-sst5'
            dataset_path = 'SetFit/sst5'
        elif id in ['dynasent', 'dynasent_r1']:
            dataset_name = 'DynaSent Round 1'
            dataset_url = 'https://huggingface.co/datasets/dynabench/dynasent'
            dataset_path = 'dynabench/dynasent'
            dataset_subset = 'dynabench.dynasent.r1.all'
        elif id == 'dynasent_r2':
            dataset_name = 'DynaSent Round 2'
            dataset_url = 'https://huggingface.co/datasets/dynabench/dynasent'
            dataset_path = 'dynabench/dynasent'
            dataset_subset = 'dynabench.dynasent.r2.all'
        elif id == 'mteb_tweet':
            dataset_name = 'MTEB Tweet Sentiment Extraction'
            dataset_url = 'https://huggingface.co/datasets/mteb/tweet_sentiment_extraction'
            dataset_path = 'mteb/tweet_sentiment_extraction'
        else:
            raise ValueError(f"Unknown dataset: {id}")
        print(f"{split.capitalize()} Data: {dataset_name} from {dataset_source}: '{dataset_path}'") if rank == 0 else None
        print(f"Dataset URL: {dataset_url}") if dataset_url is not None and rank == 0 else None

        # Load the dataset, do any pre-processing, and select appropriate split
        if id == 'sst_local':
            data_split = sst.train_reader(dataset_path) if split == 'train' else sst.dev_reader(dataset_path)
        elif id == 'sst':
            data = load_dataset(dataset_path)
            data = data.rename_column('label', 'label_orig') 
            for split_name in ('train', 'validation', 'test'):
                dis = [convert_sst_label(s) for s in data[split_name]['label_text']]
                data[split_name] = data[split_name].add_column('label', dis)
                data[split_name] = data[split_name].add_column('sentence', data[split_name]['text'])
            data_split = get_split(data, split)
        elif id in ['dynasent', 'dynasent_r1']:
            data = load_dataset(dataset_path, dataset_subset)
            data = data.rename_column('gold_label', 'label')
            data_split = get_split(data, split)
        elif id == 'dynasent_r2':
            data = load_dataset(dataset_path, dataset_subset)
            data = data.rename_column('gold_label', 'label')
            data_split = get_split(data, split)
        elif id == 'mteb_tweet':
            data = load_dataset(dataset_path)
            data = data.rename_column('label', 'label_orig') 
            data = data.rename_column('label_text', 'label')
            data = data.rename_column('text', 'sentence')
            split = 'test' if split == 'dev' else split
            data_split = get_split(data, split)

        return data_split
    
    train = get_data(dataset, 'train', rank, debug)
    dev = get_data(eval_dataset, 'dev', rank, debug)
    
    if sample_percent is not None:
        print(f"Sampling {sample_percent:.0%} of data") if rank == 0 else None
        train = train.sample(frac=sample_percent)
        dev = dev.sample(frac=sample_percent)

    dist.barrier()
    if rank == 0:
        print(f"Train size: {len(train)}, Dev size: {len(dev)}")
        if debug:
            print("Train label distribution:")
            print_label_dist(train)
            print("Dev label distribution:")
            print_label_dist(dev)
        print(f"Data loaded ({format_time(time.time() - data_load_start)})")

    return train, dev

def process_data(bert_tokenizer, bert_model, pooling, world_size, train, dev, device, batch_size, rank, debug, save_archive, save_dir):
    data_process_start = time.time()
    if rank == 0:
        print(f"\nProcessing data (Batch size: {batch_size}, Pooling: {pooling.upper() if pooling == 'cls' else pooling.capitalize()})...")
        print(f"Extracting sentences and labels...")
        
    # Extract y labels
    y_train = train.label.values
    y_dev = dev.label.values
    
    # Extract X sentences
    X_train_sent = train.sentence.values
    X_dev_sent = dev.sentence.values
    
    # Generate random indices
    train_indices = np.random.choice(len(X_train_sent), 3, replace=False)
    dev_indices = np.random.choice(len(X_dev_sent), 3, replace=False)
    
    # Print samples and collect sample sentences
    train_samples = []
    dev_samples = []
    if debug and rank == 0:
        for i in train_indices:
            train_samples.append((f'Train[{i}]: ', X_train_sent[i], f' - {y_train[i].upper()}'))
        for i in dev_indices:
            dev_samples.append((f'Dev[{i}]: ', X_dev_sent[i], f' - {y_dev[i].upper()}'))

    # Process X sentences (tokenize and encode with BERT)
    X_train_tensor = bert_phi(X_train_sent, bert_tokenizer, bert_model, pooling, world_size, device, batch_size, train_samples, rank, debug, split='train').to(device)
    X_dev_tensor = bert_phi(X_dev_sent, bert_tokenizer, bert_model, pooling, world_size, device, batch_size, dev_samples, rank, debug, split='dev').to(device)
    
    # Convert tensors to numpy arrays on CPU
    X_train = X_train_tensor.cpu().numpy()
    X_dev = X_dev_tensor.cpu().numpy()
    
    if save_archive and rank == 0:
        save_data_archive(X_train, X_dev, y_train, y_dev, X_dev_sent, world_size, device.type, save_dir)
    
    dist.barrier()
    if rank == 0:
        print(f"Train shape: {list(np.shape(X_train))}, Dev shape: {list(np.shape(X_dev))}")
        print(f"Data processed ({format_time(time.time() - data_process_start)})")
    
    return X_train, X_dev, y_train, y_dev, X_dev_sent


def bert_phi(texts, tokenizer, model, pooling, world_size, device, batch_size, sample_texts, rank, debug, split):
    encoding_start = time.time()
    total_texts = len(texts)
    
    # Divide texts among GPUs
    texts_per_gpu = (total_texts + world_size - 1) // world_size  # Round up
    start_idx = rank * texts_per_gpu
    end_idx = min(start_idx + texts_per_gpu, total_texts)
    local_texts = texts[start_idx:end_idx]
    
    total_batches_per_rank = (len(local_texts) + batch_size - 1) // batch_size
    total_batches = total_batches_per_rank * world_size

    if rank == 0:
        print(f"\nEncoding {split.capitalize()} data of {total_texts} texts...")
        print(f"Texts per GPU: {texts_per_gpu}")
        print(f"Total batches per rank: {total_batches_per_rank}")
        print(f"Total batches across all ranks: {total_batches}")

    embeddings = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(local_texts), batch_size):
            batch_start = time.time()
            batch_texts = local_texts[i:i + batch_size]
            encoded = tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)

            if pooling == 'cls':
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
            elif pooling == 'mean':
                batch_embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            elif pooling == 'max':
                batch_embeddings = torch.max(outputs.last_hidden_state * attention_mask.unsqueeze(-1), dim=1)[0]
            
            embeddings.append(batch_embeddings)
            
            batch_num = i // batch_size + 1
            print(f"Rank {rank}: Batch {batch_num:2d} / {total_batches_per_rank}, Shape: {list(batch_embeddings.shape)}, Time: {format_time(time.time() - batch_start)}")

    local_embeddings = torch.cat(embeddings, dim=0)
    
    # Pad local embeddings to ensure all have the same size
    padding_size = texts_per_gpu - local_embeddings.shape[0]
    if padding_size > 0:
        padding = torch.zeros((padding_size, local_embeddings.shape[1]), device=device)
        local_embeddings = torch.cat([local_embeddings, padding], dim=0)
    
    # Gather embeddings from all processes
    gathered_embeddings = [torch.zeros_like(local_embeddings, device=device) for _ in range(world_size)]
    dist.all_gather(gathered_embeddings, local_embeddings)
    
    # Remove padding and concatenate
    final_embeddings = torch.cat([emb[:texts_per_gpu] for emb in gathered_embeddings], dim=0)
    final_embeddings = final_embeddings[:total_texts]  # Trim to the correct size

    dist.barrier()

    if rank == 0 and debug:
        print(f"\nEncoding completed ({format_time(time.time() - encoding_start)})")
        print(f"Final embeddings shape: {final_embeddings.shape}")

    return final_embeddings

def initialize_classifier(num_layers, hidden_dim, batch_size, epochs, lr, early_stop, hidden_activation, n_iter_no_change,
                          tol, rank, device, debug, checkpoint_dir, checkpoint_interval, resume_from_checkpoint,
                          load_model=False, filename=None, use_saved_params=True):
    class_init_start = time.time()
    hidden_activation = get_activation(hidden_activation)
    print(f"\nInitializing DDP Neural Classifier...") if rank == 0 else None
    print(f"Number of Layers: {num_layers}, Hidden dimension: {hidden_dim}, Hidden activation: {hidden_activation}, Batch size: {batch_size}, Max epochs: {epochs}, Early stop: {early_stop}") if rank == 0 else None
    
    classifier = TorchDDPNeuralClassifier(
        num_layers=num_layers,
        early_stopping=early_stop,
        hidden_dim=hidden_dim,
        hidden_activation=hidden_activation,
        batch_size=batch_size,
        max_iter=epochs,
        n_iter_no_change=n_iter_no_change,
        tol=tol,
        eta=lr,
        rank=rank,
        debug=debug,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        resume_from_checkpoint=resume_from_checkpoint,
        device=device
    )

    if load_model and filename is not None:
        print(f"Loading model from: {checkpoint_dir}/{filename}...") if rank == 0 else None
        start_epoch, optimizer_state_dict = classifier.load_model(directory=checkpoint_dir, filename=filename, pattern=None, use_saved_params=use_saved_params, rank=rank, debug=debug)
    elif load_model and filename is None:
        print("Loading the latest final model...") if rank == 0 else None
        start_epoch, model_state_dict, optimizer_state_dict = classifier.load_model(directory=checkpoint_dir, filename=None, pattern='final_model', use_saved_params=use_saved_params, rank=rank, debug=debug)
    elif resume_from_checkpoint:
        print("Resuming training from the latest checkpoint...") if rank == 0 else None
        start_epoch, model_state_dict, optimizer_state_dict = classifier.load_model(directory=checkpoint_dir, filename=None, pattern='checkpoint_epoch', use_saved_params=use_saved_params, rank=rank, debug=debug)
    else:
        start_epoch = 1
        model_state_dict = None
        optimizer_state_dict = None

    dist.barrier()
    if rank == 0:
        print(classifier) if debug else None
        print(f"Classifier initialized ({format_time(time.time() - class_init_start)})")
    return classifier, start_epoch, model_state_dict, optimizer_state_dict

def evaluate_model(model, X_dev, y_dev, label_dict, numeric_dict, world_size, device, rank, debug, save_preds,
                   save_dir, X_dev_sent):
    eval_start = time.time()
    print("\nEvaluating model...") if rank == 0 else None
    model.model.eval()
    with torch.no_grad():
        print("Making predictions...") if rank == 0 else None
        if not torch.is_tensor(X_dev):
            X_dev = torch.tensor(X_dev, device=device)
        preds = model.model(X_dev)
        all_preds = [torch.zeros_like(preds) for _ in range(world_size)]
        dist.all_gather(all_preds, preds)
        if rank == 0:
            all_preds = torch.cat(all_preds, dim=0)[:len(y_dev)]
            preds_labels = convert_numeric_to_labels(all_preds.argmax(dim=1).cpu().numpy(), numeric_dict)
            print(f"Predictions: {len(preds_labels)}, True labels: {len(y_dev)}") if debug else None
            # Save predictions if requested
            if save_preds:
                df = pd.DataFrame({
                    'X_dev_sent': X_dev_sent,
                    'y_dev': y_dev,
                    'preds_labels': preds_labels
                })
                # Create a save directory if it doesn't exist
                if not os.path.exists(save_dir):
                    print(f"Creating save directory: {save_dir}")
                    os.makedirs(save_dir)
                # Create a filename timestamp
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(save_dir, f'predictions_{timestamp}.csv')
                df.to_csv(save_path, index=False)
                print(f"Saved predictions: {save_dir}/predictions_{timestamp}.csv")
            print("\nClassification report:")
            print(classification_report(y_dev, preds_labels, digits=3, zero_division=0))
            macro_f1_score = model.score(X_dev, y_dev, device, debug)
            print(f"Macro F1 Score: {macro_f1_score:.2f}")
            print(f"\nEvaluation completed ({format_time(time.time() - eval_start)})")

def main(rank, world_size, device_type, backend, dataset, eval_dataset, weights_name, hidden_activation, label_dict, numeric_dict,
         num_layers, hidden_dim, batch_size=32, epochs=10, lr=0.001, sample_percent=None, random_seed=42, early_stop=None, 
         n_iter_no_change=10, tol=1e-5, pooling='cls', debug=False, checkpoint_dir='checkpoints', checkpoint_interval=10, resume_from_checkpoint=False,
         save_preds=False, save_dir='saves', load_model=False, filename=None, use_saved_params=True, save_archive=False, archive_file=None):
    try:
        start_time = time.time()
        # Initialize the distributed environment
        device = prepare_device(rank, device_type)
        setup_environment(rank, world_size, backend, device, debug)
        fix_random_seeds(random_seed)

        if archive_file is not None:
            # Load previously processed data from an archive file
            X_train, X_dev, y_train, y_dev, X_dev_sent = load_data_archive(archive_file, device, rank)
        else:
            # Initialize BERT model and tokenizer. Load, tokenize and encode data
            bert_tokenizer, bert_model = initialize_bert_model(weights_name, device, rank, device_type, debug)
            train, dev = load_data(dataset, eval_dataset, sample_percent, rank, debug)
            X_train, X_dev, y_train, y_dev, X_dev_sent = process_data(bert_tokenizer, bert_model, pooling, world_size, train, dev, device, batch_size, rank, debug, save_archive, save_dir)

        # Initialize and train the neural classifier
        classifier, start_epoch, model_state_dict, optimizer_state_dict = initialize_classifier(
            num_layers, hidden_dim, batch_size, epochs, lr, early_stop, hidden_activation,
            n_iter_no_change, tol, rank, device, debug, checkpoint_dir, checkpoint_interval, resume_from_checkpoint,
            load_model, filename, use_saved_params
        )
        classifier.fit(X_train, y_train, rank, world_size, debug, start_epoch, model_state_dict, optimizer_state_dict)

        # Evaluate the model
        evaluate_model(classifier, X_dev, y_dev, label_dict, numeric_dict, world_size, device, rank, debug, save_preds,
                       save_dir, X_dev_sent)
        print(f"TOTAL Time: {format_time(time.time() - start_time)}") if rank == 0 else None
        dist.barrier()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Terminating all processes...")
        cleanup_and_exit(rank, debug)
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        traceback.print_exc()
        cleanup_and_exit(rank, debug)
    finally:
        cleanup_and_exit(rank, debug)
    
    return



if __name__ == '__main__':
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="DDP PyTorch Training")
    parser.add_argument('--dataset', type=str, default='sst_local', help='Training dataset to use: sst or dynasent')
    parser.add_argument('--eval_dataset', type=str, default=None, help='Evaluation dataset to use: sst or dynasent')
    parser.add_argument('--weights_name', type=str, default='bert-base-uncased', help='Pre-trained model name')
    parser.add_argument('--hidden_activation', type=str, default='tanh', help='Hidden activation function')
    parser.add_argument('--label_dict', type=dict, default={'negative': 0, 'neutral': 1, 'positive': 2}, help='Text label dictionary, string to numeric')
    parser.add_argument('--numeric_dict', type=dict, default={0: 'negative', 1: 'neutral', 2: 'positive'}, help='Numeric label dictionary, numeric to string')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of hidden layers for neural classifier')
    parser.add_argument('--hidden_dim', type=int, default=300, help='Hidden dimension for neural classifier')
    parser.add_argument('--device', type=str, default=None, help="Device: auto-detected, or specify 'cuda' or 'cpu'")
    parser.add_argument('--gpus', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training on each GPU (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--sample_percent', type=float, default=None, help='Percentage of data to use for training')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--early_stop', type=str, default=None, help="Early stopping method, 'score' or 'loss' (default: None)")
    parser.add_argument('--n_iter_no_change', type=int, default=10, help='Number of iterations with no improvement to stop training')
    parser.add_argument('--tol', type=float, default=1e-5, help='Tolerance for early stopping')
    parser.add_argument('--pooling', type=str, default='cls', help='Pooling method for BERT embeddings')
    parser.add_argument('--num_threads', type=int, default=None, help='Number of threads for CPU training')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode (default: False)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save and load checkpoints (default: checkpoints)')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Checkpoint interval in epochs (default: 10)')
    parser.add_argument('--resume_from_checkpoint', action='store_true', default=False, help='Resume training from latest checkpoint (default: False)')
    parser.add_argument("--save_preds", action='store_true', default=False, help="Save predictions to CSV (default: False)")
    parser.add_argument("--save_dir", type=str, default='saves', help="Directory to save data, predictions (default: saves)")
    parser.add_argument("--load_model", action='store_true', default=False, help="Load a pre-trained model or checkpoint (default: False)")
    parser.add_argument("--filename", type=str, default=None, help="Filename of the model or checkpoint to load")
    parser.add_argument("--use_saved_params", action='store_true', default=False, help="Use saved parameters for training, if loading a model (default: False)")
    parser.add_argument("--save_archive", action='store_true', default=False, help="Save processed data to disk as a .zip archive (X_train, X_dev, y_train, y_dev, y_dev_sent) (default: False)")
    parser.add_argument("--archive_file", type=str, default=None, help="Filename of the processed data to load as a .zip archive (default: None)")
    args = parser.parse_args()

    print("\nStarting DDP PyTorch Training...")

    if args.debug:
        print("Arguments:")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")

    device_type = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    if device_type == "cuda":
        world_size = args.gpus if args.gpus is not None else torch.cuda.device_count()
        print(f"Device: {device_type}, Number of GPUs: {world_size}")
        backend = "nccl"
    elif device_type == "cpu":
        world_size = 1
        backend = "gloo"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print(f"Device: {device_type}")
    else:
        world_size = 1
        backend = "gloo"
        device_type = "cpu"
        print(f"Warning: Device {device_type} not recognized. Defaulting to 'cpu'")

    if args.num_threads is not None:
        set_threads(args.num_threads)
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(args.num_threads)
        print(f"Using {args.num_threads} threads for PyTorch")

    suffix = '' if world_size == 1 else 'es'
    print(f"Spawning {world_size} process{suffix}...")
    try:
        mp.spawn(main,
                args=(world_size,
                      device_type,
                      backend,
                      args.dataset,
                      args.eval_dataset,
                      args.weights_name,
                      args.hidden_activation,
                      args.label_dict,
                      args.numeric_dict,
                      args.num_layers,
                      args.hidden_dim,
                      args.batch_size,
                      args.epochs,
                      args.lr,
                      args.sample_percent,
                      args.random_seed,
                      args.early_stop,
                      args.n_iter_no_change,
                      args.tol,
                      args.pooling,
                      args.debug,
                      args.checkpoint_dir,
                      args.checkpoint_interval,
                      args.resume_from_checkpoint,
                      args.save_preds,
                      args.save_dir,
                      args.load_model,
                      args.filename,
                      args.use_saved_params,
                      args.save_archive,
                      args.archive_file),
                nprocs=world_size,
                join=True)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Terminating all processes...")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        traceback.print_exc()
    finally:
        cleanup_and_exit(0, args.debug)

    print("All processes finished.")
