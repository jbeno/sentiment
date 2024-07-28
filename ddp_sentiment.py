import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.", category=FutureWarning)
warnings.filterwarnings("ignore", message="TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class.", category=UserWarning, module=r"torch\._utils")
warnings.filterwarnings("ignore", message="promote has been superseded by promote_options='default'", category=FutureWarning, module=r"datasets\.table")

# Standard library imports
import argparse
import os
import time
from collections import Counter

# PyTorch imports
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

# Third-party library imports
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report
import sst
from datasets import load_dataset

# Custom utility imports
from utils import (
    setup_environment, 
    cleanup_environment, 
    prepare_device, 
    fix_random_seeds,
    convert_numeric_to_labels, 
    format_time, 
    convert_sst_label, 
    get_activation,
    set_threads
)
from torch_ddp_neural_classifier import TorchDDPNeuralClassifier


def initialize_bert_model(weights_name, device, rank, device_type, debug):
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
        print("Using different datasets for training and evaluation")
    else:
        eval_dataset = dataset
        print("Using the same dataset for training and evaluation")
    
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

def process_data(bert_tokenizer, bert_model, pooling, train, dev, device, batch_size, rank, debug):
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
            train_samples.append((f'Dev[{i}]: ', X_dev_sent[i], f' - {y_dev[i].upper()}'))
    # Process X sentences (tokenize and encode with BERT)
    X_train_tensor = bert_phi(X_train_sent, bert_tokenizer, bert_model, pooling, device, batch_size, train_samples, rank, debug).to(device)
    X_dev_tensor = bert_phi(X_dev_sent, bert_tokenizer, bert_model, pooling, device, batch_size, dev_samples, rank, debug).to(device)
    # Convert tensors to numpy arrays on CPU
    X_train = X_train_tensor.cpu().numpy()
    X_dev = X_dev_tensor.cpu().numpy()
    dist.barrier()
    if rank == 0:
        print(f"Train shape: {list(np.shape(X_train))}, Dev shape: {list(np.shape(X_dev))}")
        print(f"Data processed ({format_time(time.time() - data_process_start)})")
    return X_train, X_dev, y_train, y_dev

def bert_phi(texts, tokenizer, model, pooling, device, batch_size, sample_texts, rank, debug):
    encoding_start = time.time()
    total_batches = len(texts) // batch_size + 1
    print(f"Encoding {len(texts)} sentences with BERT...") if rank == 0 else None

    embeddings = []

    # Process and display sample texts first
    if debug and rank == 0:
        print(f"\nPooling strategy: {pooling.upper() if pooling == 'cls' else pooling.capitalize()}")
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
        print(f"Batch {(i // batch_size) + 1} of {total_batches}, Shape: {list(batch_embeddings.shape)}, Time: {format_time(time.time() - batch_start)}") if rank == 0 else None

    dist.barrier()
    print(f"Encoding completed ({format_time(time.time() - encoding_start)})") if rank == 0 else None

    return final_embeddings.to(device) 

def initialize_classifier(num_layers, hidden_dim, batch_size, epochs, lr, early_stop, hidden_activation, n_iter_no_change, tol,
                          rank, debug):
    class_init_start = time.time()
    hidden_activation = get_activation(hidden_activation)
    print(f"\nInitializing DDP Neural Classifier...") if rank == 0 else None
    print(f"Number of Layers: {num_layers}, Hidden dimension: {hidden_dim}, Hidden activation: {hidden_activation}, Batch size: {batch_size}, Max epochs: {epochs}, Early stop: {early_stop}")
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
        debug=debug)
    dist.barrier()
    if rank == 0:
        print(classifier) if debug else None
        print(f"Classifier initialized ({format_time(time.time() - class_init_start)})")
    return classifier

def evaluate_model(model, X_dev, y_dev, label_dict, world_size, device, rank, debug):
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
            preds_labels = convert_numeric_to_labels(all_preds.argmax(dim=1).cpu().numpy(), label_dict)
            print(f"Predictions: {len(preds_labels)}, True labels: {len(y_dev)}") if debug else None
            print("\nClassification report:")
            print(classification_report(y_dev, preds_labels, digits=3, zero_division=0))
            macro_f1_score = model.score(X_dev, y_dev)
            print(f"Macro F1 Score: {macro_f1_score:.2f}")
            print(f"\nEvaluation completed ({format_time(time.time() - eval_start)})")

def main(rank, world_size, device_type, backend, dataset, eval_dataset, weights_name, hidden_activation, label_dict,
         num_layers, hidden_dim, batch_size=32, epochs=10, lr=0.001, sample_percent=None, random_seed=42, early_stop=True, 
         n_iter_no_change=10, tol=1e-5, pooling='cls', debug=False):
    try:
        start_time = time.time()
        # Initialize the distributed environment
        device = prepare_device(rank, device_type)
        setup_environment(rank, world_size, backend, device, debug)
        fix_random_seeds(random_seed)

        # Initialize BERT model and tokenizer. Load, tokenize and encode data
        bert_tokenizer, bert_model = initialize_bert_model(weights_name, device, rank, device_type, debug)
        train, dev = load_data(dataset, eval_dataset, sample_percent, rank, debug)
        X_train, X_dev, y_train, y_dev = process_data(bert_tokenizer, bert_model, pooling, train, dev, device, batch_size, rank, debug)

        # Initialize and train the neural classifier
        classifier = initialize_classifier(num_layers, hidden_dim, batch_size, epochs, lr, early_stop, hidden_activation,
                                           n_iter_no_change, tol, rank, debug)
        classifier.fit(X_train, y_train, rank, world_size, device, debug)

        # Evaluate the model
        evaluate_model(classifier, X_dev, y_dev, label_dict, world_size, device, rank, debug)
        print(f"TOTAL Time: {format_time(time.time() - start_time)}") if rank == 0 else None
        dist.barrier()
    finally:
        cleanup_environment(rank, debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDP PyTorch Training")
    parser.add_argument('--dataset', type=str, default='sst_local', help='Training dataset to use: sst or dynasent')
    parser.add_argument('--eval_dataset', type=str, default=None, help='Evaluation dataset to use: sst or dynasent')
    parser.add_argument('--weights_name', type=str, default='bert-base-uncased', help='Pre-trained model name')
    parser.add_argument('--hidden_activation', type=str, default='tanh', help='Hidden activation function')
    parser.add_argument('--label_dict', type=dict, default={0: 'negative', 1: 'neutral', 2: 'positive'}, help='Label dictionary')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of hidden layers for neural classifier')
    parser.add_argument('--hidden_dim', type=int, default=300, help='Hidden dimension for neural classifier')
    parser.add_argument('--device', type=str, default=None, help="Device: auto-detected, or specify 'cuda' or 'cpu'")
    parser.add_argument('--gpus', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training on each GPU (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--sample_percent', type=float, default=None, help='Percentage of data to use for training')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--early_stop', type=bool, default=True, help='Use early stopping (default: True)')
    parser.add_argument('--n_iter_no_change', type=int, default=10, help='Number of iterations with no improvement to stop training')
    parser.add_argument('--tol', type=float, default=1e-5, help='Tolerance for early stopping')
    parser.add_argument('--pooling', type=str, default='cls', help='Pooling method for BERT embeddings')
    parser.add_argument('--num_threads', type=int, default=None, help='Number of threads for CPU training')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode (default: False)')
    args = parser.parse_args()

    print("\nStarting DDP PyTorch Training...")
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
                      args.debug),
                nprocs=world_size,
                join=True)
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
    finally:
        pass
