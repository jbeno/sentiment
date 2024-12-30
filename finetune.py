import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", message="resume_download is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use force_download=True.", category=FutureWarning)
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
from multiprocessing import Value
from queue import Empty

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
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import sst
from datasets import load_dataset
import wandb

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
    tensor_to_numpy,
    print_label_dist,
    get_scheduler,
    parse_dict
)
from datawaza_funcs import eval_model
from classifier import TorchDDPNeuralClassifier, SentimentDataset
from colors import *

# Suppress Hugging Face library warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.repocard").setLevel(logging.ERROR)

from torch.utils.data import Dataset

from multiprocessing import Queue, Pipe, set_start_method

def load_label_dicts(label_template):
    if label_template == 'neg_neu_pos':
        label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
        numeric_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}
    elif label_template == 'bin_neu':
        label_dict = {'non-neutral': 0, 'neutral': 1}
        numeric_dict = {0: 'non-neutral', 1: 'neutral'}
    elif label_template == 'bin_pos':
        label_dict = {'non-positive': 0, 'positive': 1}
        numeric_dict = {0: 'non-positive', 1: 'positive'}
    elif label_template == 'bin_neg':
        label_dict = {'non-negative': 0, 'negative': 1}
        numeric_dict = {0: 'non-negative', 1: 'negative'}
    else:
        raise ValueError(f"Unknown label template: {label_template}. Options are: 'neg_neu_pos', 'bin_neu', 'bin_pos', 'bin_neg'")
    
    return label_dict, numeric_dict

def save_data_archive(X_train, X_val, X_test, y_train, y_val, y_test, X_test_sent, world_size, device_type, data_dir):
    # Create directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Create filename with appropriate suffix and timestamp
    suffix = f'_{world_size}_gpu' if device_type == 'cuda' else '_1_cpu'
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    filename = f'data{suffix}_{timestamp}.npz'
    filepath = os.path.join(data_dir, filename)

    # Convert tensors to NumPy arrays if necessary
    X_train = tensor_to_numpy(X_train)
    X_val = tensor_to_numpy(X_val) if X_val is not None else None
    X_test = tensor_to_numpy(X_test)
    y_train = tensor_to_numpy(y_train)
    y_val = tensor_to_numpy(y_val) if y_val is not None else None
    y_test = tensor_to_numpy(y_test)

    # Save data to archive file
    if X_val is not None:
        np.savez_compressed(filepath, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test, X_test_sent=X_test_sent)
    else:
        np.savez_compressed(filepath, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, X_test_sent=X_test_sent)
    print(f"\nData saved to: {filepath}")

def load_data_archive(data_file, device, rank, sample_percent=None):
    load_archive_start = time.time()
    
    # Check if the archive file path is provided
    if data_file is None:
        raise ValueError(f"{red}No archive file provided to load data from{reset}")
    
    # Check if the archive file exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{red}Archive file not found: {data_file}{reset}")
    
    # Attempt to load the data from the archive file
    try:
        print(f"\n{sky_blue}Loading archived data from: {data_file}...{reset}") if rank == 0 else None
        with np.load(data_file, allow_pickle=True) as data:
            X_train = data['X_train']
            X_val = data['X_val'] if 'X_val' in data else None
            X_test = data['X_test']
            y_train = data['y_train']
            y_val = data['y_val'] if 'y_val' in data else None
            y_test = data['y_test']
            X_test_sent = data['X_test_sent']
        
        # Sample data if sample_percent is provided
        if sample_percent is not None:
            print(f"Sampling {sample_percent:.0%} of data...") if rank == 0 else None
            num_train_samples = int(len(X_train) * sample_percent)
            num_val_samples = int(len(X_val) * sample_percent) if X_val is not None else None
            num_test_samples = int(len(X_test) * sample_percent)
            
            # Create a permutation of indices
            train_indices = np.random.permutation(len(X_train))[:num_train_samples]
            val_indices = np.random.permutation(len(X_val))[:num_val_samples] if X_val is not None else None
            test_indices = np.random.permutation(len(X_test))[:num_test_samples]
            
            # Sample the data
            X_train = X_train[train_indices]
            y_train = y_train[train_indices]
            X_val = X_val[val_indices] if X_val is not None else None
            y_val = y_val[val_indices] if y_val is not None else None
            X_test = X_test[test_indices]
            y_test = y_test[test_indices]
            X_test_sent = X_test_sent[test_indices]
            
            if X_val is not None:
                print(f"Sampled Train size: {len(X_train)}, Sampled Validation size: {len(X_val)}, Sampled Evaluation size: {len(X_test)}") if rank == 0 else None
            else:
                print(f"Sampled Train size: {len(X_train)}, Sampled Evaluation size: {len(X_test)}") if rank == 0 else None
        
        if rank == 0:
            # Print a summary of the loaded data
            print(f"X Train shape: {list(X_train.shape)}, y Train shape: {list(y_train.shape)}")
            print(f"X Validation shape: {list(X_val.shape)}, y Validation shape: {list(y_val.shape)}") if X_val is not None else None
            print(f"X Test shape: {list(X_test.shape)}, y Dev shape: {list(y_test.shape)}")
            print(f"X Test Sentences shape: {list(X_test_sent.shape)}")
            # Print label distributions
            print("Train label distribution:")
            print_label_dist(y_train)
            if X_val is not None:
                print("Validation label distribution:")
                print_label_dist(y_val)
            print("Test label distribution:")
            print_label_dist(y_test)
        print(f"Archived data loaded ({time.time() - load_archive_start:.2f}s)") if rank == 0 else None
    except Exception as e:
        raise RuntimeError(f"Failed to load data from archive file {data_file}: {str(e)}")
        
    return X_train, X_val, X_test, y_train, y_val, y_test, X_test_sent

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

def initialize_transformer_model(weights_name, device, rank, debug):
    model_init_start = time.time()
    print(f"\n{sky_blue}Initializing '{weights_name}' tokenizer and model...{reset}") if rank == 0 else None
    
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            # Only rank 0 checks and downloads files
            if rank == 0:
                # Try loading with local_files_only first to check if files exist
                try:
                    print(f"Checking for local files...")
                    _ = AutoTokenizer.from_pretrained(weights_name, local_files_only=True)
                    _ = AutoModel.from_pretrained(weights_name, local_files_only=True)
                    print(f"Found all files in cache, skipping download")
                except Exception as e:
                    print(f"Some files not found locally, downloading...")
                    # Download tokenizer files
                    tokenizer = AutoTokenizer.from_pretrained(weights_name, local_files_only=False)
                    # Download model files
                    _ = AutoModel.from_pretrained(weights_name, local_files_only=False)
                    print(f"Download complete")
            
            # Wait for rank 0 to finish checking/downloading
            dist.barrier()
            
            # Now all ranks can load from local files
            if rank == 0:
                print(f"All ranks loading tokenizer from local files...")
            tokenizer = AutoTokenizer.from_pretrained(weights_name, local_files_only=True)
            
            if rank == 0:
                print(f"All ranks loading model from local files...")
            model = AutoModel.from_pretrained(weights_name, local_files_only=True).to(device)
            
            # Final sync point
            dist.barrier()
            
            if rank == 0:
                if debug:
                    print(f"Tokenizer:\n{tokenizer}")
                    print(f"Model:\n{model}")
                print(f"Tokenizer and model initialized ({format_time(time.time() - model_init_start)})")
            return tokenizer, model
            
        except Exception as e:
            if attempt < max_retries - 1:
                if rank == 0:
                    print(f"\n{yellow}Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...{reset}")
                    print(f"Error: {str(e)}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                if rank == 0:
                    print(f"\n{red}Failed to initialize tokenizer/model after {max_retries} attempts{reset}")
                raise e

    raise RuntimeError("Failed to initialize transformer model and tokenizer")

def load_data(dataset, eval_dataset, sample_percent, eval_split, use_val_split, val_percent,
              world_size, rank, debug):
    data_load_start = time.time()

    # Function to get a subset of data based on split name
    def get_split(data, split):
            split = 'validation' if split == 'dev' else split
            if split in ['train', 'validation', 'test']:
                data_split = data[split].to_pandas()
            else:
                raise ValueError(f"Unknown split: {split}")
            return data_split
    
    # Function to load data from Hugging Face or local based on ID and split name
    def get_data(id, split, purpose, rank, debug):
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
        elif id == 'merged_local':
            dataset_name = 'Merged DynaSent Round 1, Round 2 and SST'
            dataset_source = 'Local'
            dataset_path = os.path.join('data', 'merged')
        elif id == 'merged_neutral':
            dataset_name = 'Merged DynaSent Round 1, Round 2 and SST: Neutral Only'
            dataset_source = 'Local'
            dataset_path = os.path.join('data', 'merged')
        elif id == 'merged_positive':
            dataset_name = 'Merged DynaSent Round 1, Round 2 and SST: Positive Only'
            dataset_source = 'Local'
            dataset_path = os.path.join('data', 'merged')
        elif id == 'merged_negative':
            dataset_name = 'Merged DynaSent Round 1, Round 2 and SST: Negative Only'
            dataset_source = 'Local'
            dataset_path = os.path.join('data', 'merged')
        elif id == 'merged_balanced':
            dataset_name = 'Merged DynaSent Round 1, Round 2 and SST: Balanced'
            dataset_source = 'Local'
            dataset_path = os.path.join('data', 'merged')
        else:
            raise ValueError(f"Unknown dataset: {id}")
        print(f"{purpose} Data: {dataset_name} from {dataset_source}: '{dataset_path}'") if rank == 0 else None
        #print(f"Dataset URL: {dataset_url}") if dataset_url is not None and rank == 0 else None

        # Load the dataset, do any pre-processing, and select appropriate split
        if id == 'sst_local':
            if split == 'train':
                src = os.path.join(dataset_path, 'sst3-train.csv')
                data_split = sst.sentiment_reader(src, include_subtrees=False, dedup=False)
            elif split in ['dev', 'validation']:
                src = os.path.join(dataset_path, 'sst3-dev.csv')
                data_split = sst.sentiment_reader(src, include_subtrees=False, dedup=False)
            elif split in ['test', 'test-labeled']:
                src = os.path.join(dataset_path, 'sst3-test-labeled.csv')
                data_split = sst.sentiment_reader(src, include_subtrees=False, dedup=False)
            elif split == 'test-unlabeled':
                src = os.path.join(dataset_path, 'sst3-test-unlabeled.csv')
                data_split = sst.sentiment_reader(src, include_subtrees=False, dedup=False)
            else:
                raise ValueError(f"Unknown split: {split}")
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
        elif id == 'merged_local':
            if split == 'train':
                data_split = pd.read_csv(os.path.join(dataset_path, 'train_all.csv'), index_col=None)
            elif split in ['dev', 'validation']:
                data_split = pd.read_csv(os.path.join(dataset_path, 'val_all.csv'), index_col=None)
            elif split == 'test':
                data_split = pd.read_csv(os.path.join(dataset_path, 'test_all.csv'), index_col=None)
        elif id == 'merged_balanced':
            if split == 'train':
                data_split = pd.read_csv(os.path.join(dataset_path, 'train_balanced.csv'), index_col=None)
            elif split in ['dev', 'validation']:
                data_split = pd.read_csv(os.path.join(dataset_path, 'val_all.csv'), index_col=None)
            elif split == 'test':
                data_split = pd.read_csv(os.path.join(dataset_path, 'test_all.csv'), index_col=None)
        elif id == 'merged_neutral':
            if split == 'train':
                data_split = pd.read_csv(os.path.join(dataset_path, 'train_all_binary.csv'), index_col=None)
                data_split = data_split.rename(columns={'label': 'label_orig'})
                data_split = data_split.rename(columns={'neutral_label': 'label'})
            elif split in ['dev', 'validation']:
                data_split = pd.read_csv(os.path.join(dataset_path, 'val_all_binary.csv'), index_col=None)
                data_split = data_split.rename(columns={'label': 'label_orig'})
                data_split = data_split.rename(columns={'neutral_label': 'label'})
            elif split == 'test':
                data_split = pd.read_csv(os.path.join(dataset_path, 'test_all_binary.csv'), index_col=None)
                data_split = data_split.rename(columns={'label': 'label_orig'})
                data_split = data_split.rename(columns={'neutral_label': 'label'})
        elif id == 'merged_positive':
            if split == 'train':
                data_split = pd.read_csv(os.path.join(dataset_path, 'train_all_binary.csv'), index_col=None)
                data_split = data_split.rename(columns={'label': 'label_orig'})
                data_split = data_split.rename(columns={'positive_label': 'label'})
            elif split in ['dev', 'validation']:
                data_split = pd.read_csv(os.path.join(dataset_path, 'val_all_binary.csv'), index_col=None)
                data_split = data_split.rename(columns={'label': 'label_orig'})
                data_split = data_split.rename(columns={'positive_label': 'label'})
            elif split == 'test':
                data_split = pd.read_csv(os.path.join(dataset_path, 'test_all_binary.csv'), index_col=None)
                data_split = data_split.rename(columns={'label': 'label_orig'})
                data_split = data_split.rename(columns={'positive_label': 'label'})
        elif id == 'merged_negative':
            if split == 'train':
                data_split = pd.read_csv(os.path.join(dataset_path, 'train_all_binary.csv'), index_col=None)
                data_split = data_split.rename(columns={'label': 'label_orig'})
                data_split = data_split.rename(columns={'negative_label': 'label'})
            elif split in ['dev', 'validation']:
                data_split = pd.read_csv(os.path.join(dataset_path, 'val_all_binary.csv'), index_col=None)
                data_split = data_split.rename(columns={'label': 'label_orig'})
                data_split = data_split.rename(columns={'negative_label': 'label'})
            elif split == 'test':
                data_split = pd.read_csv(os.path.join(dataset_path, 'test_all_binary.csv'), index_col=None)
                data_split = data_split.rename(columns={'label': 'label_orig'})
                data_split = data_split.rename(columns={'negative_label': 'label'})
        else:
            raise ValueError(f"Unknown dataset: {id}")

        return data_split

    if rank == 0:
        print(f"\n{sky_blue}Loading data...{reset}")
        if eval_dataset is not None:
            print(f"Using different datasets for training and evaluation")
        else:
            eval_dataset = dataset
            print(f"Using the same dataset for training and evaluation")
        print(f"Splits:")
        print(f"- Train: Using {dataset} 'train' split")
        if use_val_split:
            print(f"- Validation: Using {dataset} 'validation' split")
        else:
            print(f"- Validation: Using {val_percent} of {dataset} 'train' split")
        print(f"- Evaluation: Using {eval_dataset} '{eval_split}' split")

        train = get_data(dataset, 'train', 'Train', rank, debug)
        if use_val_split:
            validation = get_data(dataset, 'validation', 'Validation', rank, debug)
        else:
            validation = None
        test = get_data(eval_dataset, eval_split, 'Evaluation', rank, debug)
        
        print(f"Train size: {len(train)}")
        print(f"Validation size: {len(validation)}") if validation is not None else None
        print(f"Evaluation size: {len(test)}")

        if sample_percent is not None:
            print(f"Sampling {sample_percent:.0%} of data...")
            train = train.sample(frac=sample_percent)
            if validation is not None:
                validation = validation.sample(frac=sample_percent)
            test = test.sample(frac=sample_percent)
            print(f"Sampled Train size: {len(train)}")
            print(f"Sampled Validation size: {len(validation)}") if validation is not None else None
            print(f"Sampled Evaluation size: {len(test)}")

    else:
        train = None
        validation = None
        test = None

    # Broadcast the data to all ranks
    if world_size > 1:
        object_list = [train, validation, test]
        dist.broadcast_object_list(object_list, src=0)
        train, validation, test = object_list

        dist.barrier()
        print(f"Data broadcasted to all ranks") if rank == 0 and debug else None
        print(f"Rank {rank}: Train size: {len(train)}, Validation size: {len(validation) if validation is not None else None}, Evaluation size: {len(test)}") if debug else None

    if rank == 0:
        print("Train label distribution:")
        print_label_dist(train)
        if validation is not None:
            print("Validation label distribution:")
            print_label_dist(validation)
        print("Evaluation label distribution:")
        print_label_dist(test)
        print(f"Data loaded ({format_time(time.time() - data_load_start)})")
    dist.barrier()
        
    return train, validation, test

def process_data(bert_tokenizer, bert_model, pooling, world_size, train, validation, test, device, batch_size, rank, debug, save_archive,
                 save_dir, num_workers, prefetch, empty_cache, finetune_bert, freeze_bert, chunk_size=None):
    data_process_start = time.time()

    print(f"\n{sky_blue}Processing data...{reset}") if rank == 0 else None
    print(f"(Batch size: {batch_size}, Pooling: {pooling.upper() if pooling == 'cls' else pooling.capitalize()}, Fine Tune BERT: {finetune_bert}, Chunk size: {chunk_size})...") if rank == 0 else None
    print(f"Extracting sentences and labels...") if rank == 0 else None
    
    # Extract y labels
    y_train = train.label.values
    y_val = validation.label.values if validation is not None else None
    y_test = test.label.values
    
    # Extract X sentences
    X_train_sent = train.sentence.values
    X_val_sent = validation.sentence.values if validation is not None else None
    X_test_sent = test.sentence.values

    if rank == 0:
        # Generate random indices
        train_indices = np.random.choice(len(X_train_sent), 3, replace=False)
        val_indices = np.random.choice(len(X_val_sent), 3, replace=False) if validation is not None else None
        test_indices = np.random.choice(len(X_test_sent), 3, replace=False)
        
        # Collect sample sentences
        train_samples = []
        val_samples = []
        test_samples = []
        for i in train_indices:
            train_samples.append((f'Train[{i}]: ', X_train_sent[i], f' - {y_train[i].upper()}'))
        if validation is not None:
            for i in val_indices:
                val_samples.append((f'Validation[{i}]: ', X_val_sent[i], f' - {y_val[i].upper()}'))
        for i in test_indices:
            test_samples.append((f'Evaluation[{i}]: ', X_test_sent[i], f' - {y_test[i].upper()}'))
    else:
        train_samples = None
        val_samples = None
        test_samples = None
    
    # Process X sentences (tokenize and encode with BERT) if we're not fine-tuning BERT
    if finetune_bert:
        # For fine-tuning, we just return the sentences
        X_train = X_train_sent
        X_val = X_val_sent
        X_test = X_test_sent
    else:
        # Process X sentences (tokenize and encode with BERT) for non-fine-tuning workflow
        X_train = process_data_chunks(X_train_sent, bert_tokenizer, bert_model, pooling, world_size, device, batch_size, 
                                      train_samples, rank, debug, split='Train', num_workers=num_workers, prefetch=prefetch,
                                      empty_cache=empty_cache, chunk_size=chunk_size)
        X_val = process_data_chunks(X_val_sent, bert_tokenizer, bert_model, pooling, world_size, device, batch_size, 
                                    val_samples, rank, debug, split='Validation', num_workers=num_workers, prefetch=prefetch,
                                    empty_cache=empty_cache, chunk_size=chunk_size) if validation is not None else None
        X_test = process_data_chunks(X_test_sent, bert_tokenizer, bert_model, pooling, world_size, device, batch_size, 
                                    test_samples, rank, debug, split='Evaluation', num_workers=num_workers, prefetch=prefetch,
                                    empty_cache=empty_cache, chunk_size=chunk_size)
    
    # Data integrity check, make sure the sizes are consistent across ranks
    if not finetune_bert and device.type == 'cuda' and world_size > 1:
        # Gather sizes from all ranks
        train_sizes = [torch.tensor(X_train.shape[0], device=device) for _ in range(world_size)]
        val_sizes = [torch.tensor(X_val.shape[0], device=device) for _ in range(world_size)] if validation is not None else None
        test_sizes = [torch.tensor(X_test.shape[0], device=device) for _ in range(world_size)]
        
        dist.all_gather(train_sizes, train_sizes[rank])
        dist.all_gather(val_sizes, val_sizes[rank]) if validation is not None else None
        dist.all_gather(test_sizes, test_sizes[rank])

        if rank == 0:
            # Convert to CPU for easier handling
            train_sizes = [size.cpu().item() for size in train_sizes]
            val_sizes = [size.cpu().item() for size in val_sizes] if validation is not None else None
            test_sizes = [size.cpu().item() for size in test_sizes]

            if debug:
                print("\nDataset size summary:")
                print(f"Train sizes across ranks: {train_sizes}")
                print(f"Validation sizes across ranks: {val_sizes}") if validation is not None else None
                print(f"Test sizes across ranks: {test_sizes}")
                
                if len(set(train_sizes)) > 1 or len(set(test_sizes)) > 1 or (validation is not None and len(set(val_sizes)) > 1):
                    print(f"{red}WARNING: Mismatch in dataset sizes across ranks!{red}")
                    print(f"Train size mismatch: {max(train_sizes) - min(train_sizes)}")
                    print(f"Validation size mismatch: {max(val_sizes) - min(val_sizes)}") if validation is not None else None
                    print(f"Test size mismatch: {max(test_sizes) - min(test_sizes)}")
                else:
                    print("All ranks have consistent dataset sizes.")
                
                print(f"Total train samples: {sum(train_sizes)}")
                print(f"Total validation samples: {sum(val_sizes)}") if validation is not None else None
                print(f"Total test samples: {sum(test_sizes)}")

            # Check for significant mismatch and raise error if necessary
            max_mismatch = max(max(train_sizes) - min(train_sizes), max(test_sizes) - min(test_sizes))
            if max_mismatch > world_size:  # Allow for small mismatches due to uneven division
                raise ValueError(f"{red}Significant mismatch in dataset sizes across ranks. Max difference: {max_mismatch}{reset}")

    if save_archive and rank == 0:
        save_data_archive(X_train, X_val, X_test, y_train, y_val, y_test, X_test_sent, world_size, device.type, save_dir)

    dist.barrier()
    if rank == 0:
        if validation is not None:
            print(f"X Train shape: {list(np.shape(X_train))}, X Validation shape: {list(np.shape(X_val))}, X Test shape: {list(np.shape(X_test))}")
            print(f"y Train shape: {list(np.shape(y_train))}, y Validation shape: {list(np.shape(y_val))}, y Test shape: {list(np.shape(y_test))}")
        else:
            print(f"X Train shape: {list(np.shape(X_train))}, X Test shape: {list(np.shape(X_test))}")
            print(f"y Train shape: {list(np.shape(y_train))}, y Test shape: {list(np.shape(y_test))}")
        print(f"Data processed ({format_time(time.time() - data_process_start)})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, X_test_sent

def process_data_chunks(texts, tokenizer, model, pooling, world_size, device, batch_size, sample_texts, rank, debug, split,
                        num_workers, prefetch, empty_cache, chunk_size=None):
    if chunk_size is None or chunk_size >= len(texts):
        return bert_phi(texts, tokenizer, model, pooling, world_size, device, batch_size, sample_texts, rank, debug, split,
                        num_workers, prefetch, empty_cache)
    
    print(f"\n{sky_blue}Processing {split} data in chunks of size {chunk_size}...{reset}") if rank == 0 else None
    
    all_embeddings = []
    num_chunks = math.ceil(len(texts) / chunk_size)
    
    for i in range(num_chunks):
        chunk_start = i * chunk_size
        chunk_end = min((i + 1) * chunk_size, len(texts))
        chunk_texts = texts[chunk_start:chunk_end]
        
        print(f"\n{sky_blue}Processing chunk {i+1}/{num_chunks} (samples {chunk_start} to {chunk_end-1})...{reset}") if rank == 0 else None
        
        # Only pass sample_texts for the first chunk
        current_sample_texts = sample_texts if i == 0 else None
        
        chunk_embeddings = bert_phi(chunk_texts, tokenizer, model, pooling, world_size, device, batch_size, 
                                    current_sample_texts, rank, debug, f"{split}_chunk_{i+1}", 
                                    num_workers, prefetch, empty_cache, i+1, num_chunks)
        
        # Move embeddings to CPU and convert to numpy to save GPU memory
        chunk_embeddings = chunk_embeddings.cpu().numpy()
        all_embeddings.append(chunk_embeddings)
        
        # Clear CUDA cache
        if empty_cache and device.type == 'cuda':
            torch.cuda.empty_cache()
        
        dist.barrier()
    
    # Concatenate all chunk embeddings
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    
    print(f"Finished processing all chunks for {split} data.") if rank == 0 else None
    
    return final_embeddings


def bert_phi(texts, tokenizer, model, pooling, world_size, device, batch_size, sample_texts, rank, debug, split, num_workers, prefetch, empty_cache,
             chunk_id=None, num_chunks=None):
    encoding_start = time.time()
    total_texts = len(texts)
    embeddings = []

    # Ensure texts is a list
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    elif not isinstance(texts, (list, tuple)):
        raise TypeError(f"{red}texts must be a list, tuple, or numpy array. Got {type(texts)}{reset}")
    
    def tokenize(texts, tokenizer, device):
        # Convert NumPy array to list if necessary
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()

        encoded = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        return input_ids, attention_mask

    def pool(last_hidden_state, attention_mask, pooling):
        if pooling == 'cls':
            return last_hidden_state[:, 0, :]
        elif pooling == 'mean':
            return (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif pooling == 'max':
            return torch.max(last_hidden_state * attention_mask.unsqueeze(-1), dim=1)[0]
        else:
            raise ValueError(f"{red}Unknown pooling method: {pooling}{reset}")

    # Process and display sample texts first
    def display_sample_texts(sample_texts):
        if sample_texts is None:
            return
        print(f"\n{sky_blue}Displaying samples from {split.capitalize()} data:{reset}")
        for text in sample_texts:
            # Tokenize the text and get the tokens
            tokens = tokenizer.tokenize(text[1])
            print(f"{text[0]}{text[1]}{text[2]}")
            print(f"Tokens: {tokens}")
            
            # Encode the text (including special tokens) and get embeddings
            input_ids, attention_mask = tokenize([text[1]], tokenizer, device)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            
            embedding = pool(outputs.last_hidden_state, attention_mask, pooling)
            
            print(f"Embedding: {embedding[0, :6].cpu().numpy()} ...")
            print()

            if device.type == 'cuda':
                del input_ids, attention_mask, outputs, embedding
                torch.cuda.empty_cache()

    # Use DDP to distribute the encoding process across multiple GPUs
    if device.type == 'cuda' and world_size > 1: 
        if rank == 0:

            # Display sample texts
            display_sample_texts(sample_texts)

            print(f"\n{sky_blue}Encoding {split.capitalize()} data of {total_texts} texts distributed across {world_size} GPUs...{reset}")
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
            input_ids, attention_mask = tokenize(batch_texts, tokenizer, device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

            batch_embeddings = pool(outputs.last_hidden_state, attention_mask, pooling)
            
            embeddings.append(batch_embeddings)

            batch_shape = list(batch_embeddings.shape)

            if empty_cache:
                # Delete the unused objects
                del outputs, input_ids, attention_mask, batch_embeddings
                # Empty CUDA cache
                torch.cuda.empty_cache()

            shape_color = get_shape_color(batch_size, batch_shape)
            if chunk_id is not None:
                print(f"Rank {bright_white}{bold}{rank}{reset}: Chunk {purple}{bold}{chunk_id}{reset} / {num_chunks}, Batch {sky_blue}{bold}{(i // batch_size) + 1:2d}{reset} / {local_batch_count}, Shape: {shape_color}{bold}{batch_shape}{reset}, Time: {format_time(time.time() - batch_start)}")
            else:
                print(f"Rank {bright_white}{bold}{rank}{reset}: Batch {sky_blue}{bold}{(i // batch_size) + 1:2d}{reset} / {local_batch_count}, Shape: {shape_color}{bold}{batch_shape}{reset}, Time: {format_time(time.time() - batch_start)}")

            if rank == 0:
                if (i // batch_size) % 5 == 0:
                    print_rank_memory_summary(world_size, rank, all_local=True, verbose=False)

        local_embeddings = torch.cat(embeddings, dim=0)

        #dist.barrier()

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
                    print(f"{red}WARNING: Mismatch in padding count!{reset}")

                if padding_embeddings > 0:
                    padding_embeds = all_embeddings[-padding_embeddings:]
                    
                    if padding_embeds.shape[0] > 1:  # Ensure there are at least 2 embeddings to compare
                        max_diff = torch.max(torch.pdist(padding_embeds))
                        print(f"Maximum difference between padding embeddings: {max_diff}")
                        
                        if max_diff > 1e-6:
                            print(f"{red}WARNING: Padding embeddings are not similar.{reset}")
                        else:
                            print("Padding embeddings verified as very similar.")
                    else:
                        print(f"{yellow}Not enough padding embeddings to calculate differences (found {padding_embeds.shape[0]}).{reset}")

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
            print(f"\n{sky_blue}Encoding {split.capitalize()} data of {total_texts} texts on a single {device_string}...{reset}")
            print(f"Batch Size: {batch_size}, Pooling: {pooling.upper() if pooling == 'cls' else pooling.capitalize()}, Empty Cache: {empty_cache}")

            # Display sample texts
            display_sample_texts(sample_texts)

        total_batches = len(texts) // batch_size + 1

        for i in range(0, len(texts), batch_size):
            batch_start = time.time()
            batch_texts = texts[i:i + batch_size]
            input_ids, attention_mask = tokenize(batch_texts, tokenizer, device)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

            # Perform only the selected pooling strategy for all batches
            batch_embeddings = pool(outputs.last_hidden_state, attention_mask, pooling)
            
            embeddings.append(batch_embeddings)
            final_embeddings = torch.cat(embeddings, dim=0)
            print(f"Batch {bright_white}{bold}{(i // batch_size) + 1:2d}{reset} / {total_batches}, Shape: {list(batch_embeddings.shape)}, Time: {format_time(time.time() - batch_start)}")
    
            if empty_cache and device.type == 'cuda':
                # Delete the unused objects
                del outputs, input_ids, attention_mask, batch_embeddings
                # Empty CUDA cache
                torch.cuda.empty_cache()

    return final_embeddings


def initialize_classifier(bert_model, bert_tokenizer, finetune_bert, finetune_layers, num_layers, hidden_dim, batch_size,
                          epochs, lr, lr_decay, early_stop, hidden_activation, n_iter_no_change, tol, rank, world_size, device, debug,
                          checkpoint_dir, checkpoint_interval, resume_from_checkpoint, filename=None, use_saved_params=True,
                          optimizer_name=None, use_zero=True, scheduler_name=None, l2_strength=0.0, pooling='cls',
                          target_score=None, interactive=False, response_pipe=None, accumulation_steps=1, max_grad_norm=None,
                          freeze_bert=False, dropout_rate=0.0, show_progress=False, advance_epochs=1, wandb_run=None, val_percent=0.1,
                          random_seed=42, label_dict=None, optimizer_kwargs={}, scheduler_kwargs={}):
    class_init_start = time.time()
    print(f"\n{sky_blue}Initializing DDP Neural Classifier...{reset}") if rank == 0 else None
    hidden_activation = get_activation(hidden_activation, hidden_dim)
    optimizer_class = get_optimizer(optimizer_name, use_zero, device, rank, world_size)
    scheduler_class = get_scheduler(scheduler_name, device, rank, world_size)
    print(f"Layers: {num_layers}, Hidden Dim: {hidden_dim}, Hidden Act: {hidden_activation.__class__.__name__}, Dropout: {dropout_rate}, Optimizer: {optimizer_class.__name__}, L2 Strength: {l2_strength}, Pooling: {pooling.upper()}, Accumulation Steps: {accumulation_steps}, Max Grad Norm: {max_grad_norm}") if rank == 0 else None
    print(f"Batch Size: {batch_size}, Max Epochs: {epochs}, LR: {lr}, Early Stop: {early_stop}, Fine-tune BERT: {finetune_bert}, Fine-tune Layers: {finetune_layers}, Freeze BERT: {freeze_bert}, Target Score: {target_score}, Interactive: {interactive}") if rank == 0 else None
    
    classifier = TorchDDPNeuralClassifier(
        bert_model=bert_model,
        bert_tokenizer=bert_tokenizer,
        finetune_bert=finetune_bert,
        finetune_layers=finetune_layers,
        pooling=pooling,
        num_layers=num_layers,
        early_stopping=early_stop,
        hidden_dim=hidden_dim,
        hidden_activation=hidden_activation,
        batch_size=batch_size,
        max_iter=epochs,
        n_iter_no_change=n_iter_no_change,
        tol=tol,
        eta=lr,
        lr_decay=lr_decay,
        rank=rank,
        world_size=world_size,
        debug=debug,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        resume_from_checkpoint=resume_from_checkpoint,
        device=device,
        optimizer_class=optimizer_class,
        use_zero=use_zero,
        scheduler_class=scheduler_class,
        target_score=target_score,
        interactive=interactive,
        response_pipe=response_pipe,
        gradient_accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        freeze_bert=freeze_bert,
        dropout_rate=dropout_rate,
        l2_strength=l2_strength,
        show_progress=show_progress,
        advance_epochs=advance_epochs,
        wandb_run=wandb_run,
        validation_fraction=val_percent,
        random_seed=random_seed,
        label_dict=label_dict,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_kwargs=scheduler_kwargs
    )

    if filename is not None:
        print(f"Loading model from: {checkpoint_dir}/{filename}...") if rank == 0 else None
        start_epoch, model_state_dict, optimizer_state_dict = classifier.load_model(directory=checkpoint_dir, filename=filename, pattern=None, use_saved_params=use_saved_params, rank=rank, debug=debug)
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

def evaluate_model(model, bert_tokenizer, X_test, y_test, label_dict, numeric_dict, world_size, device, rank, debug, save_preds,
                   save_dir, X_test_sent, wandb_run=None, decimal=2, pos_label=1, threshold=0.5, save_plots=False,
                   model_name=None, weights_name=None):
    eval_start = time.time()
    print(f"\n{sky_blue}Evaluating model...{reset}") if rank == 0 else None
    model.model.eval()
    with torch.no_grad():
        print("Making predictions...") if rank == 0 and debug else None
        if model.finetune_bert:
            dataset = SentimentDataset(X_test, [0] * len(X_test), bert_tokenizer)  # Dummy labels
            dataloader = DataLoader(dataset, batch_size=model.batch_size, shuffle=False)
            preds = []
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model.model(input_ids, attention_mask=attention_mask)
                preds.append(outputs)
            preds = torch.cat(preds, dim=0)
        else:
            if not torch.is_tensor(X_test):
                X_test = torch.tensor(X_test, device=device)
            preds = model.model(X_test)
        all_preds = [torch.zeros_like(preds) for _ in range(world_size)]
        dist.all_gather(all_preds, preds)
        if rank == 0:
            all_preds = torch.cat(all_preds, dim=0)[:len(y_test)]
            y_pred = convert_numeric_to_labels(all_preds.argmax(dim=1).cpu().numpy(), numeric_dict)
            print(f"Predictions: {len(y_pred)}, True labels: {len(y_test)}") if debug else None

            # Convert text labels to numeric labels
            y_test_numeric = np.array([label_dict[label] for label in y_test])
            y_pred_numeric = np.array([label_dict[label] for label in y_pred])

            # Set model name based on run name
            if model_name is None:
                if wandb_run is not None:
                    model_name = wandb_run.name
                elif weights_name is not None:
                    model_name = weights_name
                else:
                    model_name = 'Neural Classifier'
            
            # Use the DataWaza eval_model function
            metrics = eval_model(
                y_test=y_test_numeric,
                y_pred=y_pred_numeric,
                class_map=numeric_dict,
                estimator=model,
                x_test=X_test,
                class_type='multi' if len(numeric_dict) > 2 else 'binary',
                model_name=model_name,
                plot=False,
                save_plots=save_plots,
                save_dir=save_dir,
                debug=debug,
                pos_label=pos_label,
                decimal=decimal,
                return_metrics=True,
                threshold=threshold,
                wandb_run=wandb_run
            )

            # Save predictions if requested
            if save_preds:
                df = pd.DataFrame({
                    'X_test_sent': X_test_sent,
                    'y_test': y_test,
                    'y_pred': y_pred
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
                if wandb_run is not None:
                    wandb_run.log({
                        "eval/predictions": wandb.Table(
                            data=[[sent, true, pred] for sent, true, pred in zip(X_test_sent, y_test, y_pred)],
                            columns=["X_test_sent", "y_test", "y_pred"]
                        )
                    })
            #print(f"\n{bright_white}{bold}Classification report:{reset}")
            #print(classification_report(y_dev, preds_labels, digits=3, zero_division=0))
            class_report = classification_report(y_test, y_pred, digits=decimal, zero_division=0, output_dict=True)

            # Create a confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=list(numeric_dict.values()))

            macro_f1_score = model.score(X_test, y_test, device, debug)
            print(f"\n{bright_white}{bold}Macro F1 Score:{reset} {bright_cyan}{bold}{macro_f1_score:.2f}{reset}")

            # Log evaluation metrics to Weights & Biases
            if wandb_run is not None:
                wandb.log({
                    'eval/macro_f1_score': macro_f1_score,
                    'eval/classification_report': class_report,
                    #'eval/confusion_matrix': cm,
                    'eval/metrics': metrics,
                })

            print(f"\nEvaluation completed ({format_time(time.time() - eval_start)})")

def make_predictions(classifier, tokenizer, transformer_model, predict_file, numeric_dict, rank, debug, save_dir, device, pooling,
                     world_size, batch_size, num_workers, prefetch, empty_cache, finetune_transformer, freeze_transformer, chunk_size):
    predictions_start = time.time()
    print(f"\n{sky_blue}Predicting on unlabled test dataset...{reset}") if rank == 0 else None
    # Load the test dataset
    test_df = pd.read_csv(predict_file, index_col=None)
    test_texts = test_df.sentence.values
    print(f"Loaded test dataset at: {predict_file}") if rank == 0 else None
    print(f"Test dataset size: {len(test_texts)}") if rank == 0 else None
    print(f"Test dataset columns: {list(test_df.columns)}") if rank == 0 else None
    print(f"Test dataset sample:\n{test_df[['sentence']].sample(3)}") if rank == 0 else None
            
    # Tokenize and encode the test dataset
    if not finetune_transformer:
        X_test = bert_phi(test_texts, tokenizer, transformer_model, pooling, world_size, device, batch_size, 
                                    None, rank, debug, 'Test', 
                                    num_workers, prefetch, empty_cache, None, None)
    else:
        X_test = test_texts

    if rank == 0:
        dataset = SentimentDataset(X_test, None, tokenizer)
        dataloader = DataLoader(dataset, batch_size=classifier.batch_size, shuffle=False)
        classifier.model.eval()
        preds = []
        with torch.no_grad():
            if finetune_transformer:
                dataset = SentimentDataset(X_test, [0] * len(X_test), tokenizer)
                dataloader = DataLoader(dataset, batch_size=classifier.batch_size, shuffle=False)
                preds = []
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = classifier.model(input_ids, attention_mask=attention_mask)
                    preds.append(outputs)
                preds = torch.cat(preds, dim=0)
            else:
                if not torch.is_tensor(X_test):
                    X_test = torch.tensor(X_test, device=device)
                preds = classifier.model(X_test)
            preds_labels = convert_numeric_to_labels(preds.argmax(dim=1).cpu().numpy(), numeric_dict)
            test_df['prediction'] = preds_labels
            print(f"Sample test predictions:\n{test_df[['sentence', 'prediction']].sample(3)}")

            # Create a save directory if it doesn't exist
            if not os.path.exists(save_dir):
                print(f"Creating save directory: {save_dir}")
                os.makedirs(save_dir)
            # Create a filename timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(save_dir, f'test_predictions_{timestamp}.csv')
            test_df.to_csv(save_path, index=False)
            
            print(f"Saved test predictions: {save_dir}/test_predictions_{timestamp}.csv")
            print(f"Test prediction completed ({format_time(time.time() - predictions_start)})")
    dist.barrier()

# Main function for DDP training
def main(rank, world_size, device_type, backend, dataset, eval_dataset, weights_name, hidden_activation, label_dict,
         numeric_dict, num_layers, hidden_dim, batch_size, epochs, lr, lr_decay, sample_percent, random_seed, early_stop,
         n_iter_no_change, tol, pooling, debug, checkpoint_dir, checkpoint_interval, resume_from_checkpoint, save_preds,
         save_dir, model_file, use_saved_params, save_data, data_file, num_workers, prefetch, optimizer_name, optimizer_kwargs,
         use_zero, l2_strength, empty_cache, decimal, scheduler_name, schedular_kwargs, finetune_transformer, finetune_layers,
         target_score, interactive, mem_interval, accumulation_steps, freeze_transformer, dropout_rate, chunk_size,
         show_progress, predict, predict_file, save_final_model, save_pickle, save_hf, max_grad_norm, port, color_theme,
         use_wandb, wandb_project, wandb_run_name, wandb_alerts, val_percent, use_val_split, eval_split, label_template, pos_label,
         threshold, save_plots, model_name, advance_epochs, input_queue, pipes, running):
    try:
        if interactive:
            response_pipe = pipes[rank][1]  # Get the specific pipe for this rank
        else:
            response_pipe = None
        start_time = time.time()
        # Initialize the distributed environment
        device = prepare_device(rank, device_type)
        setup_environment(rank, world_size, backend, device, debug, port)
        fix_random_seeds(random_seed)

        # Load the dictionary for labels and numeric values if template provided
        if label_template is not None:
            label_dict, numeric_dict = load_label_dicts(label_template)

        # Initialize wandb
        if use_wandb and rank == 0:
            wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, config={
                "dataset": dataset,
                "eval_dataset": eval_dataset,
                "weights_name": weights_name,
                "hidden_activation": hidden_activation,
                "num_layers": num_layers,
                "hidden_dim": hidden_dim,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": lr,
                "sample_percent": sample_percent,
                "random_seed": random_seed,
                "early_stop": early_stop,
                "n_iter_no_change": n_iter_no_change,
                "tolerance": tol,
                "pooling": pooling,
                "optimizer": optimizer_name,
                "scheduler": scheduler_name,
                "finetune_transformer": finetune_transformer,
                "finetune_layers": finetune_layers,
                "dropout_rate": dropout_rate,
                "l2_strength": l2_strength,
                "freeze_transformer": freeze_transformer,
                "accumulation_steps": accumulation_steps,
                "max_grad_norm": max_grad_norm,
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_interval": checkpoint_interval,
                "resume_from_checkpoint": resume_from_checkpoint,
                "model_file": model_file,
                "use_saved_params": use_saved_params,
                "save_data": save_data,
                "data_file": data_file,
                "num_workers": num_workers,
                "prefetch": prefetch,
                "empty_cache": empty_cache,
                "decimal": decimal,
                "target_score": target_score,
                "interactive": interactive,
                "mem_interval": mem_interval,
                "save_final_model": save_final_model,
                "save_pickle": save_pickle,
                "predict": predict,
                "predict_file": predict_file,
                "save_preds": save_preds,
                "save_dir": save_dir,
                "port": port,
                "color_theme": color_theme,
                "val_percent": val_percent,
                "advance_epochs": advance_epochs,
                "show_progress": show_progress,
                "threshold": threshold,
                "use_val_split": use_val_split,
                "eval_with": eval_split
            })
            print(f"Wand run initialized.") if rank == 0 else None
        else:
            wandb_run = None

        # Initialize transformer model and tokenizer
        tokenizer, transformer_model = initialize_transformer_model(weights_name, device, rank, debug)

        if data_file is not None and not finetune_transformer:
            # Load previously processed data from an archive file
            X_train, X_val, X_test, y_train, y_val, y_test, X_test_sent = load_data_archive(data_file, device, rank, sample_percent)
        else:
            # Load, tokenize and encode data
            train, validation, test = load_data(dataset, eval_dataset, sample_percent, eval_split, use_val_split,
                                                val_percent, world_size, rank, debug)
            X_train, X_val, X_test, y_train, y_val, y_test, X_test_sent = process_data(tokenizer, transformer_model, pooling, world_size, train,
                validation, test, device, batch_size, rank, debug, save_data, save_dir, num_workers, prefetch, empty_cache, finetune_transformer,
                freeze_transformer, chunk_size)

        # Initialize and train the transformer classifier
        classifier, start_epoch, model_state_dict, optimizer_state_dict = initialize_classifier(
            transformer_model if finetune_transformer else None, tokenizer if finetune_transformer else None,
            finetune_transformer, finetune_layers, num_layers, hidden_dim,
            batch_size, epochs, lr, lr_decay, early_stop, hidden_activation, n_iter_no_change, tol, rank, world_size, device, debug,
            checkpoint_dir, checkpoint_interval, resume_from_checkpoint, model_file, use_saved_params, optimizer_name, use_zero,
            scheduler_name, l2_strength, pooling, target_score, interactive, response_pipe, accumulation_steps, max_grad_norm,
            freeze_transformer, dropout_rate, show_progress, advance_epochs, wandb_run, val_percent, random_seed, label_dict,
            optimizer_kwargs, schedular_kwargs)

        classifier.fit(X_train, X_val, y_train, y_val, rank, world_size, debug, start_epoch, model_state_dict, optimizer_state_dict,
                       num_workers, prefetch, empty_cache, decimal, input_queue, mem_interval, save_final_model, save_pickle, save_hf,
                       save_dir, weights_name)
        
        # Evaluate the model
        evaluate_model(classifier, tokenizer, X_test, y_test, label_dict, numeric_dict, world_size, device, rank, debug, save_preds,
                       save_dir, X_test_sent, wandb_run, decimal, pos_label, threshold, save_plots, model_name, weights_name)
        
        # Make predictions on unlabled test dataset
        if predict:
            make_predictions(classifier, tokenizer, transformer_model, predict_file, numeric_dict, rank, debug, save_dir, device, pooling,
                             world_size, batch_size, num_workers, prefetch, empty_cache, finetune_transformer,
                             freeze_transformer, chunk_size)

        # Finish the wandb run
        if rank == 0 and use_wandb:
            if wandb_alerts:
                wandb.alert(title=f"{wandb_run} Completed", text="Training completed successfully.")
            wandb.finish()

        print(f"TOTAL Time: {format_time(time.time() - start_time)}") if rank == 0 else None
        dist.barrier()
        if rank == 0:
            # Signal that all processes have finished
            for _ in range(world_size):
                if input_queue is not None:
                    input_queue.put(None)
            with running.get_lock():
                running.value = False
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Terminating all processes...")
        cleanup_and_exit(rank, debug, response_pipe, input_queue)
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        if wandb_alerts:
            wandb.alert(title=f"{wandb_run} Error", text=f"An error occurred during training: {str(e)}")
        traceback.print_exc()
        cleanup_and_exit(rank, debug, response_pipe, input_queue)
    finally:
        if rank == 0:
            with running.get_lock():
                running.value = False
        cleanup_and_exit(rank, debug, response_pipe, input_queue)
    
    return


if __name__ == '__main__':
    set_start_method('spawn', force=True)

    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="DDP Distributed PyTorch Training for Sentiment Analysis using BERT")

    # Dataset configuration
    dataset_group = parser.add_argument_group('Dataset configuration')
    dataset_group.add_argument('--dataset', type=str, default='sst_local', help="Training dataset to use: 'sst', 'sst_local', 'dynasent_r1', 'dynasent_r2', 'mteb_tweet', 'merged_local' (default: sst_local)")
    dataset_group.add_argument('--eval_dataset', type=str, default=None, help="(Optional) Different test dataset to use: 'sst', 'sst_local', 'dynasent_r1', 'dynasent_r2', 'mteb_tweet', 'merged_local' (default: None)")
    dataset_group.add_argument('--eval_split', type=str, choices=['validation', 'test'], default='validation', help="Specify whether to evaluate with 'validation' or 'test' split (default: validation)")
    dataset_group.add_argument('--sample_percent', type=float, default=None, help='Percentage of data to use for training, validation and test (default: None)')
    dataset_group.add_argument('--chunk_size', type=int, default=None, help='Number of dataset samples to encode in each chunk (default: None, process all data at once)')
    dataset_group.add_argument('--label_dict', type=parse_dict, default={'negative': 0, 'neutral': 1, 'positive': 2}, help="Text label dictionary, string to numeric (default: {'negative': 0, 'neutral': 1, 'positive': 2})")
    dataset_group.add_argument('--numeric_dict', type=parse_dict, default={0: 'negative', 1: 'neutral', 2: 'positive'}, help="Numeric label dictionary, numeric to string (default: {0: 'negative', 1: 'neutral', 2: 'positive'})")
    dataset_group.add_argument('--label_template', type=str, default=None, help="Predefined class label template with dictionary mappings: 'neg_neu_pos', 'bin_neu', 'bin_pos', 'bin_neg' (default: None)")
    dataset_group.add_argument('--pos_label', type=int, default=1, help="Positive class label for binary classification, must be integer (default: 1)")

    # BERT tokenizer/model configuration
    bert_group = parser.add_argument_group('BERT tokenizer/model configuration')
    bert_group.add_argument('--weights_name', type=str, default='bert-base-uncased', help="Pre-trained model/tokenizer name from a Hugging Face repo. Can be root-level or namespaced (default: 'bert-base-uncased')")
    bert_group.add_argument('--pooling', type=str, default='cls', help="Pooling method for BERT embeddings: 'cls', 'mean', 'max' (default: 'cls')")
    bert_group.add_argument('--finetune_bert', action='store_true', default=False, help='Whether to fine-tune BERT weights. If True, specify number of finetune_layers (default: False)')
    bert_group.add_argument('--finetune_layers', type=int, default=1, help='Number of BERT layers to fine-tune. For example: 0 to freeze all, 12 or 24 to tune all, 1 to tune the last layer, etc. (default: 1)')
    bert_group.add_argument('--freeze_bert', action='store_true', default=False, help='Whether to freeze BERT weights during training (default: False)')

    # Classifier configuration
    classifier_group = parser.add_argument_group('Classifier configuration')
    classifier_group.add_argument('--num_layers', type=int, default=1, help='Number of hidden layers for neural classifier (default: 1)')
    classifier_group.add_argument('--hidden_dim', type=int, default=300, help='Hidden dimension for neural classifier layers (default: 300)')
    classifier_group.add_argument('--hidden_activation', type=str, default='tanh', help="Hidden activation function: 'tanh', 'relu', 'sigmoid', 'leaky_relu', 'gelu', 'swish', 'swishglu' (default: 'tanh')")
    classifier_group.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate for neural classifier (default: 0.0)')
 
    # Training configuration
    training_group = parser.add_argument_group('Training configuration')
    training_group.add_argument('--batch_size', type=int, default=32, help='Batch size for both encoding text and training classifier (default: 32)')
    training_group.add_argument('--accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients before updating weights (default: 1)')
    training_group.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    training_group.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    training_group.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay factor, defaults to none, 0.95 is 5%% per layer (default: 1.0)')
    training_group.add_argument('--optimizer', type=str, default='adam', help="Optimizer to use: 'adam', 'sgd', 'adagrad', 'rmsprop', 'zero', 'adamw' (default: 'adam')")
    training_group.add_argument('--use_zero', action='store_true', default=False, help='Use Zero Redundancy Optimizer for efficient DDP training, with the optimizer specified in --optimizer (default: False)')
    training_group.add_argument('--l2_strength', type=float, default=0.0, help='L2 regularization strength for optimizer (default: 0.0)')
    training_group.add_argument('--optimizer_kwargs', type=parse_dict, default={}, help="Additional optimizer keyword arguments as a dictionary (default: None)")
    training_group.add_argument('--scheduler', type=str, default=None, help="Learning rate scheduler to use: 'none', 'step', 'multi_step', 'exponential', 'cosine', 'reduce_on_plateau', 'cyclic' (default: None)")
    training_group.add_argument('--scheduler_kwargs', type=parse_dict, default={}, help="Additional scheduler keyword arguments as a dictionary (default: None)") 
    training_group.add_argument('--max_grad_norm', type=float, default=None, help='Maximum gradient norm for clipping (default: None)')
    training_group.add_argument('--random_seed', type=int, default=42, help='Random seed (default: 42)')
    training_group.add_argument('--interactive', action='store_true', default=False, help='Interactive mode for training (default: False)')
    training_group.add_argument('--show_progress', action='store_true', default=False, help='Show progress bars for training and evaluation (default: False)')

    # Checkpoint configuration
    checkpoint_group = parser.add_argument_group('Checkpoint configuration')
    checkpoint_group.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save and load checkpoints (default: checkpoints)')
    checkpoint_group.add_argument('--checkpoint_interval', type=int, default=50, help='Checkpoint interval in epochs (default: 50)')
    checkpoint_group.add_argument('--resume_from_checkpoint', action='store_true', default=False, help='Resume training from latest checkpoint (default: False)')

    # Early stopping
    early_stopping_group = parser.add_argument_group('Early stopping')
    early_stopping_group.add_argument('--early_stop', type=str, default=None, help="Early stopping method, 'score' or 'loss' (default: None)")
    early_stopping_group.add_argument('--n_iter_no_change', type=int, default=5, help='Number of iterations with no improvement to stop training (default: 5)')
    early_stopping_group.add_argument('--tol', type=float, default=1e-5, help='Tolerance for early stopping (default: 1e-5)')
    early_stopping_group.add_argument('--target_score', type=float, default=None, help='Target score for early stopping (default: None)')
    early_stopping_group.add_argument('--val_percent', type=float, default=0.1, help='Fraction of training data to use for validation (default: 0.1)')
    early_stopping_group.add_argument('--use_val_split', action='store_true', default=False, help='Use a validation split instead of a proportion of the train data (default: False)')

    # Weights and bias integration
    wandb_group = parser.add_argument_group('Weights and bias integration')
    wandb_group.add_argument('--wandb', action='store_true', default=False, help='Use Weights and Biases for logging (default: False)')
    wandb_group.add_argument('--wandb_project', type=str, default=None, help="Weights and Biases project name (default: None)")
    wandb_group.add_argument('--wandb_run', type=str, default=None, help="Weights and Biases run name (default: None)")
    wandb_group.add_argument('--wandb_alerts', action='store_true', default=False, help='Enable Weights and Biases alerts (default: False)')
    # Evaluation options
    evaluation_group = parser.add_argument_group('Evaluation options')
    evaluation_group.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification evaluation (default: 0.5)')
    evaluation_group.add_argument('--model_name', type=str, default=None, help='Model name for display in evaluation plots (default: None)')
 
    # Saving options
    saving_group = parser.add_argument_group('Saving options')
    saving_group.add_argument('--save_data', action='store_true', default=False, help="Save processed data to disk as an .npz archive (X_train, X_dev, y_train, y_dev, y_dev_sent)")
    saving_group.add_argument('--save_model', action='store_true', default=False, help="Save the final model state after training in PyTorch .pth format (default: False)")
    saving_group.add_argument('--save_pickle', action='store_true', default=False, help="Save the final model after training in pickle .pkl format (default: False)")
    saving_group.add_argument('--save_hf', action='store_true', default=False, help="Save the final model after training in Hugging Face format (default: False)")
    saving_group.add_argument('--save_preds', action='store_true', default=False, help="Save predictions to CSV (default: False)")
    saving_group.add_argument('--save_plots', action='store_true', default=False, help="Save evaluation plots (default: False)")
    saving_group.add_argument('--save_dir', type=str, default='saves', help="Directory to save archived data, predictions, plots (default: saves)")

    # Loading options
    loading_group = parser.add_argument_group('Loading options')
    loading_group.add_argument('--data_file', type=str, default=None, help="Filename of the processed data to load as an .npz archive (default: None)")
    loading_group.add_argument('--model_file', type=str, default=None, help="Filename of the classifier model or checkpoint to load (default: None)")
    loading_group.add_argument('--use_saved_params', action='store_true', default=False, help="Use saved parameters for training, if loading a model")

    # Prediction options
    prediction_group = parser.add_argument_group('Prediction options')
    prediction_group.add_argument('--predict', action='store_true', default=False, help='Make predictions on a provided unlabled dataset (default: False)')
    prediction_group.add_argument('--predict_file', type=str, default=None, help='Filename of the unlabeled dataset to make predictions on (default: None)')

    # GPU and CPU processing
    gpu_cpu_group = parser.add_argument_group('GPU and CPU processing')
    gpu_cpu_group.add_argument('--device', type=str, default=None, help="Device will be auto-detected, or specify 'cuda' or 'cpu' (default: None)")
    gpu_cpu_group.add_argument('--gpus', type=int, default=None, help="Number of GPUs to use if device is 'cuda', will be auto-detected (default: None)")
    gpu_cpu_group.add_argument('--num_threads', type=int, default=None, help='Number of threads for CPU training (default: None)')
    gpu_cpu_group.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader (default: 0)')
    gpu_cpu_group.add_argument('--prefetch', type=int, default=None, help='Number of batches to prefetch (default: None)')
    gpu_cpu_group.add_argument('--empty_cache', action='store_true', default=False, help='Empty CUDA cache after each batch (default: False)')
    gpu_cpu_group.add_argument('--port', type=int, default=12355, help='Port number for DDP distributed training (default: 12355)')
 
    # Debugging and logging
    debug_group = parser.add_argument_group('Debugging and logging')
    debug_group.add_argument('--debug', action='store_true', default=False, help='Debug or verbose mode to print more details (default: False)')
    debug_group.add_argument('--mem_interval', type=int, default=10, help='Memory check interval in epochs (default: 10)')
    debug_group.add_argument('--decimal', type=int, default=4, help='Decimal places for floating point numbers (default: 4)')
    debug_group.add_argument('--color_theme', type=str, default='dark', help="Color theme for console output: 'light', 'dark' (default: 'dark')")

    args = parser.parse_args()

    if args.color_theme == 'dark':
        black = "\033[30m"
        white = "\033[37m"
        bright_white = "\033[97m"
    elif args.color_theme == 'light':
        black = "\033[97m"
        white = "\033[30m"
        bright_white = "\033[30m"
    else:
        raise ValueError(f"{red}Invalid color theme: {args.color_theme}. Options are 'light' or 'dark'.{reset}")

    print(f"\n{sky_blue}Starting DDP PyTorch Training...{reset}")

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
        print(f"{red}Warning: Device {device_type} not recognized. Defaulting to 'cpu'{reset}")

    if args.num_threads is not None:
        set_threads(args.num_threads)
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(args.num_threads)
        print(f"Using {args.num_threads} threads for PyTorch")

    suffix = '' if world_size == 1 else 'es'

    if args.interactive:
        input_queue = Queue()
        pipes = [Pipe() for _ in range(world_size)]
    else:
        input_queue = None
        pipes = None
    
    advance_epochs = 1  # Number of epochs to advance in interactive mode

    print(f"Spawning {world_size} process{suffix}...")
    running = Value('b', True)  # 'b' for boolean
    try:
        processes = mp.spawn(main,
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
                      args.lr_decay,
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
                      args.optimizer_kwargs,
                      args.use_zero,
                      args.l2_strength,
                      args.empty_cache,
                      args.decimal,
                      args.scheduler,
                      args.scheduler_kwargs,
                      args.finetune_bert,
                      args.finetune_layers,
                      args.target_score,
                      args.interactive,
                      args.mem_interval,
                      args.accumulation_steps,
                      args.freeze_bert,
                      args.dropout_rate,
                      args.chunk_size,
                      args.show_progress,
                      args.predict,
                      args.predict_file,
                      args.save_model,
                      args.save_pickle,
                      args.save_hf,
                      args.max_grad_norm,
                      args.port,
                      args.color_theme,
                      args.wandb,
                      args.wandb_project,
                      args.wandb_run,
                      args.wandb_alerts,
                      args.val_percent,
                      args.use_val_split,
                      args.eval_split,
                      args.label_template,
                      args.pos_label,
                      args.threshold,
                      args.save_plots,
                      args.model_name,
                      advance_epochs,
                      input_queue,
                      pipes,
                      running),
                nprocs=world_size,
                join=False)

        if args.interactive:

            def get_user_input(advance_epochs, rank, pipes):
                prompt = f"\n[{bright_yellow}{bold}Enter{reset}] to continue for {advance_epochs} epoch{'s' if advance_epochs > 1 else ''}, [{bright_yellow}{bold}Q{reset}]uit, [{bright_yellow}{bold}S{reset}]ave, [{bright_yellow}{bold}H{reset}]elp: "
                user_input = input(prompt).strip().lower()
                pipes[rank][0].send(user_input)
                return user_input

            while running.value:
                try:
                    rank = input_queue.get(timeout=1)
                    if rank is None:
                        print("Received None rank, exiting interactive loop.") if args.debug else None
                        break
                    print(f"Checking for messages from rank {rank}...") if args.debug else None
                    if pipes[rank][0].poll():
                        message = pipes[rank][0].recv()
                        print(f"Rank {rank} has a message: {message}...") if args.debug else None
                        if message == 'stop':
                            print(f"Stopping signal received from rank {rank}...") if args.debug else None
                            break
                        elif message.startswith('advance_epochs:'):
                            try:
                                advance_epochs = int(float(message.split(':')[1]))  # Convert to float first, then to int
                                print(f"Updated advance_epochs to {advance_epochs}") if args.debug else None
                                get_user_input(advance_epochs, rank, pipes)
                            except ValueError:
                                print(f"Error updating advance_epochs. Received invalid value: {message.split(':')[1]}")
                    else:
                        get_user_input(advance_epochs, rank, pipes)
                except Empty:
                    continue

        processes.join()  # Ensure all processes are joined and finished


    except KeyboardInterrupt:
        print(f"\n{red}KeyboardInterrupt received. Terminating all processes...{reset}")
    except Exception as e:
        print(f"{red}An error occurred during training: {str(e)}{reset}")
        traceback.print_exc()
    finally:
        cleanup_and_exit(None, args.debug, None, None)

    print("All processes finished.")

