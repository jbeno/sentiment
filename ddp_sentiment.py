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
import math

# PyTorch imports
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import torch.optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer

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
    cleanup_and_exit,
    get_optimizer,
    get_shape_color,
    print_rank_memory_summary,
    #get_scheduler
)
from torch_ddp_neural_classifier import TorchDDPNeuralClassifier
from colors import *

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
    print(f"\nData saved to: {filepath}")

def load_data_archive(data_file, device, rank):
    load_archive_start = time.time()
    
    # Check if the archive file path is provided
    if data_file is None:
        raise ValueError("No archive file provided to load data from")
    
    # Check if the archive file exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Archive file not found: {data_file}")
    
    # Attempt to load the data from the archive file
    try:
        print(f"\nLoading archived data from: {data_file}...") if rank == 0 else None
        with np.load(data_file, allow_pickle=True) as data:
            X_train = data['X_train']
            X_dev = data['X_dev']
            y_train = data['y_train']
            y_dev = data['y_dev']
            X_dev_sent = data['X_dev_sent']
        print(f"Archived data loaded ({format_time(time.time() - load_archive_start)})") if rank == 0 else None
    except Exception as e:
        raise RuntimeError(f"Failed to load data from archive file {data_file}: {str(e)}")
    
    return X_train, X_dev, y_train, y_dev, X_dev_sent

def initialize_bert_model(weights_name, device, rank, debug):
    model_init_start = time.time()
    print(f"\nInitializing '{weights_name}' tokenizer and model...") if rank == 0 else None
    bert_tokenizer = BertTokenizer.from_pretrained(weights_name)
    bert_model = BertModel.from_pretrained(weights_name).to(device)
    # if device.type == 'cuda':
    #     bert_model = DDP(bert_model, device_ids=[rank], output_device=rank, static_graph=True)
    # else:
    #     bert_model = DDP(bert_model, device_ids=None, output_device=None, static_graph=True)
    dist.barrier()
    if rank == 0:
        if debug:
            print(f"Bert Tokenizer:\n{bert_tokenizer}")
            print(f"Bert Model:\n{bert_model}")
        print(f"Tokenizer and model initialized ({format_time(time.time() - model_init_start)})")
    return bert_tokenizer, bert_model

def load_data(dataset, eval_dataset, sample_percent, world_size, rank, debug):
    data_load_start = time.time()

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

    if rank == 0:
        print(f"\nLoading data...")
        if eval_dataset is not None:
            print("Using different datasets for training and evaluation")
        else:
            eval_dataset = dataset
            print("Using the same dataset for training and evaluation")
        
        train = get_data(dataset, 'train', rank, debug)
        dev = get_data(eval_dataset, 'dev', rank, debug)
        
        print(f"Train size: {len(train)}, Dev size: {len(dev)}")

        if sample_percent is not None:
            print(f"Sampling {sample_percent:.0%} of data...")
            train = train.sample(frac=sample_percent)
            dev = dev.sample(frac=sample_percent)
            print(f"Sampled Train size: {len(train)}, Sampled Dev size: {len(dev)}")

    else:
        train = None
        dev = None

    # Broadcast the data to all ranks
    if world_size > 1:
        object_list = [train, dev]
        dist.broadcast_object_list(object_list, src=0)
        train, dev = object_list

        dist.barrier()
        print(f"Data broadcasted to all ranks") if rank == 0 and debug else None
        print(f"Rank {rank}: Train size: {len(train)}, Dev size: {len(dev)}") if debug else None

    if rank == 0:
        print("Train label distribution:")
        print_label_dist(train)
        print("Dev label distribution:")
        print_label_dist(dev)
        print(f"Data loaded ({format_time(time.time() - data_load_start)})")
    dist.barrier()
        
    return train, dev

def process_data(bert_tokenizer, bert_model, pooling, world_size, train, dev, device, batch_size, rank, debug, save_archive, save_dir, num_workers, prefetch, empty_cache):
    data_process_start = time.time()

    print(f"\nProcessing data (Batch size: {batch_size}, Pooling: {pooling.upper() if pooling == 'cls' else pooling.capitalize()})...") if rank == 0 else None
    print(f"Extracting sentences and labels...") if rank == 0 else None
    
    # Extract y labels
    y_train = train.label.values
    y_dev = dev.label.values
    
    # Extract X sentences
    X_train_sent = train.sentence.values
    X_dev_sent = dev.sentence.values
    
    if rank == 0:
        # Generate random indices
        train_indices = np.random.choice(len(X_train_sent), 3, replace=False)
        dev_indices = np.random.choice(len(X_dev_sent), 3, replace=False)
        
        # Collect sample sentences
        train_samples = []
        dev_samples = []
        for i in train_indices:
            train_samples.append((f'Train[{i}]: ', X_train_sent[i], f' - {y_train[i].upper()}'))
        for i in dev_indices:
            dev_samples.append((f'Dev[{i}]: ', X_dev_sent[i], f' - {y_dev[i].upper()}'))
    else:
        train_samples = None
        dev_samples = None
    
    # Process X sentences (tokenize and encode with BERT)
    X_train = bert_phi(X_train_sent, bert_tokenizer, bert_model, pooling, world_size, device, batch_size, train_samples, rank, debug, split='train', num_workers=num_workers, prefetch=prefetch, empty_cache=empty_cache).cpu().numpy()
    X_dev = bert_phi(X_dev_sent, bert_tokenizer, bert_model, pooling, world_size, device, batch_size, dev_samples, rank, debug, split='dev', num_workers=num_workers, prefetch=prefetch, empty_cache=empty_cache).cpu().numpy()
    
    # Data integrity check, make sure the sizes are consistent across ranks
    if device.type == 'cuda' and world_size > 1:
        # Gather sizes from all ranks
        train_sizes = [torch.tensor(X_train.shape[0], device=device) for _ in range(world_size)]
        dev_sizes = [torch.tensor(X_dev.shape[0], device=device) for _ in range(world_size)]
        
        dist.all_gather(train_sizes, train_sizes[rank])
        dist.all_gather(dev_sizes, dev_sizes[rank])

        if rank == 0:
            # Convert to CPU for easier handling
            train_sizes = [size.cpu().item() for size in train_sizes]
            dev_sizes = [size.cpu().item() for size in dev_sizes]

            if debug:
                print("\nDataset size summary:")
                print(f"Train sizes across ranks: {train_sizes}")
                print(f"Dev sizes across ranks: {dev_sizes}")
                
                if len(set(train_sizes)) > 1 or len(set(dev_sizes)) > 1:
                    print("WARNING: Mismatch in dataset sizes across ranks!")
                    print(f"Train size mismatch: {max(train_sizes) - min(train_sizes)}")
                    print(f"Dev size mismatch: {max(dev_sizes) - min(dev_sizes)}")
                else:
                    print("All ranks have consistent dataset sizes.")
                
                print(f"Total train samples: {sum(train_sizes)}")
                print(f"Total dev samples: {sum(dev_sizes)}")

            # Check for significant mismatch and raise error if necessary
            max_mismatch = max(max(train_sizes) - min(train_sizes), max(dev_sizes) - min(dev_sizes))
            if max_mismatch > world_size:  # Allow for small mismatches due to uneven division
                raise ValueError(f"Significant mismatch in dataset sizes across ranks. Max difference: {max_mismatch}")

    if save_archive and rank == 0:
        save_data_archive(X_train, X_dev, y_train, y_dev, X_dev_sent, world_size, device.type, save_dir)

    dist.barrier()
    if rank == 0:
        print(f"X Train shape: {list(np.shape(X_train))}, X Dev shape: {list(np.shape(X_dev))}")
        print(f"y Train shape: {list(np.shape(y_train))}, y Dev shape: {list(np.shape(y_dev))}")
        print(f"Data processed ({format_time(time.time() - data_process_start)})")
    
    return X_train, X_dev, y_train, y_dev, X_dev_sent

def bert_phi(texts, tokenizer, model, pooling, world_size, device, batch_size, sample_texts, rank, debug, split, num_workers, prefetch, empty_cache):
    encoding_start = time.time()
    total_texts = len(texts)
    embeddings = []

    # Process and display sample texts first
    def display_sample_texts(sample_texts):
        for text in sample_texts:
            # Tokenize the text and get the tokens
            tokens = tokenizer.tokenize(text[1])
            print(f"{text[0]}{text[1]}{text[2]}")
            print(f"Tokens: {tokens}")
            
            # Encode the text (including special tokens) and get embeddings
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            
            if pooling == 'cls':
                embedding = outputs.last_hidden_state[:, 0, :]
            elif pooling == 'mean':
                embedding = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            elif pooling == 'max':
                embedding = torch.max(outputs.last_hidden_state * attention_mask.unsqueeze(-1), dim=1)[0]
            
            print(f"Embedding: {embedding[0, :6].cpu().numpy()} ...")
            print()

            if device.type == 'cuda':
                del encoded, input_ids, attention_mask, outputs, embedding
                torch.cuda.empty_cache()

    # Use DDP to distribute the encoding process across multiple GPUs
    if device.type == 'cuda' and world_size > 1: 
        if rank == 0:

            # Display sample texts
            print(f"\nDisplaying samples from {split.capitalize()} data:")
            display_sample_texts(sample_texts)

            print(f"\nEncoding {split.capitalize()} data of {total_texts} texts distributed across {world_size} GPUs...")
            print(f"Batch Size: {batch_size}, Pooling: {pooling.upper() if pooling == 'cls' else pooling.capitalize()}, Empty Cache: {empty_cache}")

        dist.barrier()
        # Calculate the number of texts that make the dataset evenly divisible by world_size
        texts_per_rank = math.ceil(total_texts / world_size)
        padded_total = texts_per_rank * world_size
        
        if padded_total > total_texts:
            print(f"Padding {split.capitalize()} data to {padded_total} texts for even distribution across {world_size} ranks...") if rank == 0 else None
        
        # Calculate number of padding texts needed
        padding_texts = padded_total - total_texts

        # Create padding texts using [PAD] token
        pad_text = tokenizer.pad_token * 10  # Arbitrary length, will be truncated if too long
        texts_with_padding = list(texts) + [pad_text] * padding_texts  # Convert texts to list and then concatenate

        # Distribute texts evenly across ranks
        start_idx = rank * texts_per_rank
        end_idx = start_idx + texts_per_rank
        local_texts = texts_with_padding[start_idx:end_idx]
        local_batch_count = len(local_texts) // batch_size + 1

        batch_count = len(texts) // batch_size + 1
        total_batches = batch_count

        if rank == 0:
            print(f"Texts per rank: {texts_per_rank}, Total batches: {total_batches}")
        
        dist.barrier()
        print(f"Rank {rank}: Processing {len(local_texts)} texts (indices {start_idx} to {end_idx-1}) in {local_batch_count} batches...")
        
        dist.barrier()
        model.eval()

        for i in range(0, len(local_texts), batch_size):
            batch_start = time.time()
            batch_texts = local_texts[i:i + batch_size]
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

            if pooling == 'cls':
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
            elif pooling == 'mean':
                batch_embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            elif pooling == 'max':
                batch_embeddings = torch.max(outputs.last_hidden_state * attention_mask.unsqueeze(-1), dim=1)[0]
            
            embeddings.append(batch_embeddings)

            batch_shape = list(batch_embeddings.shape)

            if empty_cache:
                # Delete the unused objects
                del outputs, input_ids, attention_mask, batch_embeddings
                # Empty CUDA cache
                torch.cuda.empty_cache()

            shape_color = get_shape_color(batch_size, batch_shape)
            print(f"Rank {bright_white}{bold}{rank}{reset}: Batch {bright_white}{bold}{(i // batch_size) + 1:2d}{reset} / {local_batch_count}, Shape: {shape_color}{bold}{batch_shape}{reset}, Time: {format_time(time.time() - batch_start)}")

            if rank == 0:
                if (i // batch_size) % 5 == 0:
                    print_rank_memory_summary(world_size, rank, all_local=True, verbose=False)

        local_embeddings = torch.cat(embeddings, dim=0)

        #dist.barrier()

        # Gather embeddings from all processes
        if world_size > 1:
            gathered_embeddings = [torch.zeros_like(local_embeddings) for _ in range(world_size)]
            dist.all_gather(gathered_embeddings, local_embeddings)
            all_embeddings = torch.cat(gathered_embeddings, dim=0)
            
            if rank == 0:  # Only one process needs to do this check
                total_embeddings = all_embeddings.shape[0]
                padding_embeddings = total_embeddings - total_texts
                
                print(f"Total embeddings: {total_embeddings}")
                print(f"Original texts: {total_texts}")
                print(f"Expected padding: {padding_texts}")
                print(f"Actual padding: {padding_embeddings}")
                
                if padding_embeddings != padding_texts:
                    print("WARNING: Mismatch in padding count!")
                
                if padding_embeddings > 0:
                    # Get the embeddings of the padded texts
                    padding_embeds = all_embeddings[-padding_embeddings:]
                    
                    # Calculate the maximum difference between any two padding embeddings
                    max_diff = torch.max(torch.pdist(padding_embeds))
                    
                    print(f"Maximum difference between padding embeddings: {max_diff}")
                    
                    # You can adjust this threshold based on your observations
                    if max_diff > 1e-6:
                        print("WARNING: Padding embeddings are not similar.")
                    else:
                        print("Padding embeddings verified as very similar.")

            # Now slice off the padding
            final_embeddings = all_embeddings[:total_texts]
        else:
            final_embeddings = local_embeddings

        dist.barrier()
        if rank == 0:
            print(f"Final embeddings shape: {list(final_embeddings.shape)}") if debug else None
            print(f"Encoding completed ({format_time(time.time() - encoding_start)})")

    else:
        # Take a more straightforward approach for CPU or single GPU
        if rank == 0:
            device_string = 'GPU' if device.type == 'cuda' else 'CPU'
            print(f"\nEncoding {split.capitalize()} data of {total_texts} texts on a single {device_string}...")
            print(f"Batch Size: {batch_size}, Pooling: {pooling.upper() if pooling == 'cls' else pooling.capitalize()}, Empty Cache: {empty_cache}")

            # Display sample texts
            print(f"\nDisplaying samples from {split.capitalize()} data:")
            display_sample_texts(sample_texts)

        total_batches = len(texts) // batch_size + 1

        for i in range(0, len(texts), batch_size):
            batch_start = time.time()
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

            # Perform only the selected pooling strategy for all batches
            if pooling == 'cls':
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
            elif pooling == 'mean':
                batch_embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            elif pooling == 'max':
                batch_embeddings = torch.max(outputs.last_hidden_state * attention_mask.unsqueeze(-1), dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")
            
            embeddings.append(batch_embeddings)
            final_embeddings = torch.cat(embeddings, dim=0)
            print(f"Batch {(i // batch_size) + 1:2d} / {total_batches}, Shape: {list(batch_embeddings.shape)}, Time: {format_time(time.time() - batch_start)}")
    
    return final_embeddings


def initialize_classifier(num_layers, hidden_dim, batch_size, epochs, lr, early_stop, hidden_activation, n_iter_no_change,
                          tol, rank, world_size, device, debug, checkpoint_dir, checkpoint_interval, resume_from_checkpoint,
                          load_model=False, filename=None, use_saved_params=True, optimizer_name=None, scheduler_name=None):
    class_init_start = time.time()
    print(f"\nInitializing DDP Neural Classifier...") if rank == 0 else None
    hidden_activation = get_activation(hidden_activation)
    optimizer_class = get_optimizer(optimizer_name, device, rank, world_size)
    #scheduler_class = get_scheduler(scheduler_name, device, rank, world_size)
    print(f"Layers: {num_layers}, Hidden Dim: {hidden_dim}, Hidden Act: {hidden_activation.__class__.__name__}, Optimizer: {optimizer_class.__name__},  Batch Size: {batch_size}, Max Epochs: {epochs}, LR: {lr}, Early Stop: {early_stop.capitalize()}") if rank == 0 else None
    
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
        device=device,
        optimizer_class=optimizer_class
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
        print("Making predictions...") if rank == 0 and debug else None
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

# Main function for DDP training
def main(rank, world_size, device_type, backend, dataset, eval_dataset, weights_name, hidden_activation, label_dict, numeric_dict,
         num_layers, hidden_dim, batch_size, epochs, lr, sample_percent, random_seed, early_stop, n_iter_no_change, tol,
         pooling, debug, checkpoint_dir, checkpoint_interval, resume_from_checkpoint, save_preds, save_dir, model_file,
         use_saved_params, save_data, data_file, num_workers, prefetch, optimizer_name, empty_cache, decimal, scheduler_name):
    try:
        start_time = time.time()
        # Initialize the distributed environment
        device = prepare_device(rank, device_type)
        setup_environment(rank, world_size, backend, device, debug)
        fix_random_seeds(random_seed)

        if data_file is not None:
            # Load previously processed data from an archive file
            X_train, X_dev, y_train, y_dev, X_dev_sent = load_data_archive(data_file, device, rank)
        else:
            # Initialize BERT model and tokenizer. Load, tokenize and encode data
            bert_tokenizer, bert_model = initialize_bert_model(weights_name, device, rank, debug)
            train, dev = load_data(dataset, eval_dataset, sample_percent, world_size, rank, debug)
            X_train, X_dev, y_train, y_dev, X_dev_sent = process_data(bert_tokenizer, bert_model, pooling, world_size, train, dev, device, batch_size, rank, debug, save_data, save_dir, num_workers, prefetch, empty_cache)

        # Initialize and train the neural classifier
        classifier, start_epoch, model_state_dict, optimizer_state_dict = initialize_classifier(
            num_layers, hidden_dim, batch_size, epochs, lr, early_stop, hidden_activation,
            n_iter_no_change, tol, rank, world_size, device, debug, checkpoint_dir, checkpoint_interval, resume_from_checkpoint,
            model_file, use_saved_params, optimizer_name, scheduler_name
        )
        classifier.fit(X_train, y_train, rank, world_size, debug, start_epoch, model_state_dict, optimizer_state_dict, num_workers, prefetch, empty_cache, decimal)

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
    parser = argparse.ArgumentParser(description="DDP Distributed PyTorch Training for Sentiment Analysis using BERT")

    # Dataset configuration
    dataset_group = parser.add_argument_group('Dataset configuration')
    dataset_group.add_argument('--dataset', type=str, default='sst_local', help="Training dataset to use: 'sst', 'sst_local', 'dynasent_r1', 'dynasent_r2', 'mteb_tweet' (default: sst_local)")
    dataset_group.add_argument('--eval_dataset', type=str, default=None, help="(Optional) Different evaluation dataset to use: 'sst', 'sst_local', 'dynasent_r1', 'dynasent_r2', 'mteb_tweet' (default: None)")
    dataset_group.add_argument('--sample_percent', type=float, default=None, help='Percentage of data to use for training and evaluation (default: None)')
    dataset_group.add_argument('--label_dict', type=dict, default={'negative': 0, 'neutral': 1, 'positive': 2}, help="Text label dictionary, string to numeric (default: {'negative': 0, 'neutral': 1, 'positive': 2})")
    dataset_group.add_argument('--numeric_dict', type=dict, default={0: 'negative', 1: 'neutral', 2: 'positive'}, help="Numeric label dictionary, numeric to string (default: {0: 'negative', 1: 'neutral', 2: 'positive'})")

    # BERT tokenizer/model configuration
    bert_group = parser.add_argument_group('BERT tokenizer/model configuration')
    bert_group.add_argument('--weights_name', type=str, default='bert-base-uncased', help="Pre-trained model/tokenizer name from a Hugging Face repo. Can be root-level or namespaced (default: 'bert-base-uncased')")
    bert_group.add_argument('--pooling', type=str, default='cls', help="Pooling method for BERT embeddings: 'cls', 'mean', 'max' (default: 'cls')")

    # Classifier configuration
    classifier_group = parser.add_argument_group('Classifier configuration')
    classifier_group.add_argument('--num_layers', type=int, default=1, help='Number of hidden layers for neural classifier (default: 1)')
    classifier_group.add_argument('--hidden_dim', type=int, default=300, help='Hidden dimension for neural classifier layers (default: 300)')
    classifier_group.add_argument('--hidden_activation', type=str, default='tanh', help="Hidden activation function: 'tanh', 'relu', 'sigmoid', 'leaky_relu' (default: 'tanh')")

    # Training configuration
    training_group = parser.add_argument_group('Training configuration')
    training_group.add_argument('--batch_size', type=int, default=32, help='Batch size for both encoding text and training classifier (default: 32)')
    training_group.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    training_group.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    training_group.add_argument('--optimizer', type=str, default=None, help="Optimizer to use, will auto-detect multiple GPUs and use 'zero', otherwise 'adam': 'zero', 'adam', 'sgd', 'adagrad', 'rmsprop' (default: None)")
    training_group.add_argument('--scheduler', type=str, default=None, help="Learning rate scheduler to use: 'step', 'plateau', 'exponent', 'cosine' (default: None)")
    training_group.add_argument('--random_seed', type=int, default=42, help='Random seed (default: 42)')

    # Checkpoint configuration
    checkpoint_group = parser.add_argument_group('Checkpoint configuration')
    checkpoint_group.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save and load checkpoints (default: checkpoints)')
    checkpoint_group.add_argument('--checkpoint_interval', type=int, default=50, help='Checkpoint interval in epochs (default: 50)')
    checkpoint_group.add_argument('--resume_from_checkpoint', action='store_true', default=False, help='Resume training from latest checkpoint (default: False)')

    # Early stopping
    early_stopping_group = parser.add_argument_group('Early stopping')
    early_stopping_group.add_argument('--early_stop', type=str, default=None, help="Early stopping method, 'score' or 'loss' (default: None)")
    early_stopping_group.add_argument('--n_iter_no_change', type=int, default=10, help='Number of iterations with no improvement to stop training (default: 10)')
    early_stopping_group.add_argument('--tol', type=float, default=1e-5, help='Tolerance for early stopping (default: 1e-5)')

    # Saving options
    saving_group = parser.add_argument_group('Saving options')
    saving_group.add_argument('--save_data', action='store_true', default=False, help="Save processed data to disk as an .npz archive (X_train, X_dev, y_train, y_dev, y_dev_sent)")
    saving_group.add_argument('--save_model', action='store_true', default=False, help="Save the final model after training in PyTorch .pth format")
    saving_group.add_argument('--save_preds', action='store_true', default=False, help="Save predictions to CSV")
    saving_group.add_argument('--save_dir', type=str, default='saves', help="Directory to save archived data and predictions (default: saves)")

    # Loading options
    loading_group = parser.add_argument_group('Loading options')
    loading_group.add_argument('--data_file', type=str, default=None, help="Filename of the processed data to load as an .npz archive (default: None)")
    loading_group.add_argument('--model_file', type=str, default=None, help="Filename of the classifier model or checkpoint to load (default: None)")
    loading_group.add_argument('--use_saved_params', action='store_true', default=False, help="Use saved parameters for training, if loading a model")

    # GPU and CPU processing
    gpu_cpu_group = parser.add_argument_group('GPU and CPU processing')
    gpu_cpu_group.add_argument('--device', type=str, default=None, help="Device will be auto-detected, or specify 'cuda' or 'cpu' (default: None)")
    gpu_cpu_group.add_argument('--gpus', type=int, default=None, help="Number of GPUs to use if device is 'cuda', will be auto-detected (default: None)")
    gpu_cpu_group.add_argument('--num_threads', type=int, default=None, help='Number of threads for CPU training (default: None)')
    gpu_cpu_group.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader (default: 0)')
    gpu_cpu_group.add_argument('--prefetch', type=int, default=None, help='Number of batches to prefetch (default: None)')
    gpu_cpu_group.add_argument('--empty_cache', action='store_true', default=False, help='Empty CUDA cache after each batch (default: False)')
 
    # Debugging and logging
    debug_group = parser.add_argument_group('Debugging and logging')
    debug_group.add_argument('--debug', action='store_true', default=False, help='Debug or verbose mode to print more details (default: False)')
    debug_group.add_argument('--decimal', type=int, default=6, help='Decimal places for floating point numbers (default: 6)')

    args = parser.parse_args()

    print("\nStarting DDP PyTorch Training...")

    if args.debug:
        for group in parser._action_groups:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            print(f"\n{group.title}:")
            for arg, value in group_dict.items():
                print(f"  {arg}: {value}")
        print()

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
                      args.model_file,
                      args.use_saved_params,
                      args.save_data,
                      args.data_file,
                      args.num_workers,
                      args.prefetch,
                      args.optimizer,
                      args.empty_cache,
                      args.decimal,
                      args.scheduler),
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





