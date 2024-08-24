import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_model_base import TorchModelBase
import torch.optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer
import os
import time
import utils
import sys
import select
from tqdm import tqdm
from multiprocessing import Value
from colors import *
from utils import (format_time, print_state_summary, format_tolerance, get_nic_color, get_score_colors,
                   print_rank_memory_summary, convert_labels_to_tensor, convert_numeric_to_labels, tensor_to_numpy)


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, tokenizer, label_dict=None, max_length=512, device='cpu'):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        if labels is not None:
            try:
                self.labels = convert_labels_to_tensor(labels, label_dict, device)
            except Exception as e:
                print(f"Error converting labels in SentimentDataset: {str(e)}")
                print(f"First few labels: {labels[:5]}")
                print(f"Label dict: {label_dict}")
                raise
        else:
            self.labels = None

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

        if self.labels is not None:
            item['labels'] = self.labels[idx]

        return item

class BERTClassifier(nn.Module):
    def __init__(self, bert_model, pooling, hidden_dim, hidden_activation, num_layers, n_classes, dropout_rate, finetune_layers=1, rank=0):
        super().__init__()
        self.bert = bert_model


        # Remove BERT's original pooler
        self.bert.pooler = None
        # Add our custom pooling layer
        self.custom_pooling = PoolingLayer(pooling)
        self.classifier = Classifier(bert_model.config.hidden_size, hidden_dim, hidden_activation, num_layers, n_classes, dropout_rate)

        # Get the total number of layers in the BERT model
        total_bert_layers = len(self.bert.encoder.layer)

        # Freeze all layers except the specified number of final layers
        if finetune_layers == 0:
            # Freeze all BERT parameters
            for param in self.bert.parameters():
                param.requires_grad = False
        elif finetune_layers < total_bert_layers:
            modules_to_freeze = [self.bert.embeddings, *self.bert.encoder.layer[:-finetune_layers]]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        # Count trainable and non-trainable parameters
        bert_trainable_params = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        bert_non_trainable_params = sum(p.numel() for p in self.bert.parameters() if not p.requires_grad)
        
        # Count layers requiring gradients
        layers_requiring_grad = sum(any(p.requires_grad for p in layer.parameters()) for layer in self.bert.encoder.layer)
        
        if rank == 0:
            print(f"BERT's original pooler removed. Using custom pooling type: {pooling}")
            print(f"BERT has {bert_trainable_params:,} trainable parameters and {bert_non_trainable_params:,} non-trainable parameters")
            print(f"Number of BERT layers requiring gradients: {layers_requiring_grad} out of {total_bert_layers}")
            if finetune_layers == 0:
                print("All BERT layers are frozen")
            else:
                print(f"Fine-tuning the last {finetune_layers} out of {total_bert_layers} BERT layers")

    def forward(self, input_ids, attention_mask):
       # Get the last hidden states from BERT
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        # Apply our custom pooling
        pooled_output = self.custom_pooling(bert_outputs.last_hidden_state, attention_mask)
        # Pass the pooled output to the classifier
        return self.classifier(pooled_output)
    
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_activation, num_layers, n_classes, dropout_rate=0.0):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(hidden_activation)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(hidden_activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_dim, n_classes))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class PoolingLayer(nn.Module):
    def __init__(self, pooling_type='cls'):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, last_hidden_state, attention_mask):
        if self.pooling_type == 'cls':
            return last_hidden_state[:, 0, :]
        elif self.pooling_type == 'mean':
            return (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooling_type == 'max':
            return torch.max(last_hidden_state * attention_mask.unsqueeze(-1), dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_type}")
            
class TorchDDPNeuralClassifier(TorchModelBase):
    def __init__(self,
                bert_model,
                bert_tokenizer,
                finetune_bert,
                finetune_layers,
                label_dict=None,
                numeric_dict=None,
                pooling='cls',
                hidden_dim=300,
                hidden_activation=nn.Tanh(),
                num_layers=1,
                batch_size=1028,
                max_iter=1000,
                eta=0.001,
                optimizer_class=torch.optim.Adam,
                l2_strength=0,
                gradient_accumulation_steps=1,
                max_grad_norm=None,
                warm_start=False,
                early_stopping=None,
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-5,
                device=None,
                rank=None,
                debug=False,
                checkpoint_dir=None,
                checkpoint_interval=None,
                resume_from_checkpoint=False,
                target_score=None,
                interactive=False,
                response_pipe=None,
                freeze_bert=False,
                dropout_rate=0.0,
                show_progress=False,
                **optimizer_kwargs):
        """
        A flexible neural network classifier with Distributed Data Parallel (DDP) support.

        This classifier allows for a variable number of hidden layers and supports
        distributed training across multiple GPUs using PyTorch's DistributedDataParallel.

        Parameters
        ----------
        hidden_dim : int, optional (default=300)
            The dimensionality of each hidden layer.

        hidden_activation : torch.nn.Module, optional (default=nn.Tanh())
            The activation function used for the hidden layers.

        num_layers : int, optional (default=1)
            The number of hidden layers in the network.

        n_iter_no_change : int, optional (default=10)
            Number of iterations with no improvement to wait before early stopping.

        early_stopping : bool, optional (default=True)
            Whether to use early stopping to terminate training when validation scores
            stop improving.

        tol : float, optional (default=1e-5)
            Tolerance for improvement in early stopping.

        rank : int or None, optional (default=None)
            The rank of the current process in distributed training.

        debug : bool, optional (default=False)
            If True, print additional debug information during training.

        **base_kwargs
            Additional keyword arguments to be passed to the TorchModelBase constructor.

        Attributes
        ----------
        model : torch.nn.Module
            The PyTorch model representing the neural network.

        loss : torch.nn.CrossEntropyLoss
            The loss function used for training.

        classes_ : list
            The list of class labels known to the classifier.

        n_classes_ : int
            The number of classes.

        input_dim : int
            The dimensionality of the input features.

        Methods
        -------
        fit(X, y, rank, world_size, device, debug=False)
            Fit the model to the training data using distributed training.

        predict(X, device=None)
            Predict class labels for samples in X.

        predict_proba(X, device=None)
            Predict class probabilities for samples in X.

        score(X, y, device=None)
            Return the mean accuracy on the given test data and labels.

        to(device)
            Move the model to the specified device.

        Notes
        -----
        This classifier is designed to be used in a distributed setting with multiple
        GPUs. It uses PyTorch's DistributedDataParallel for efficient parallel training.
        The fit method requires additional parameters (rank, world_size, device) to
        support distributed training.

        Examples
        --------
        >>> import torch
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> model = TorchDDPFlexibleNeuralClassifier(hidden_dim=100, num_layers=2)
        >>> # Assuming X_train and y_train are your training data
        >>> model.fit(X_train, y_train, rank=0, world_size=1, device=torch.device('cuda:0'))
        >>> predictions = model.predict(X_test)
        """
        # Pass all relevant parameters to the superclass constructor
        super().__init__(
            batch_size=batch_size,
            max_iter=max_iter,
            eta=eta,
            optimizer_class=optimizer_class,
            l2_strength=l2_strength,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            warm_start=warm_start,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            device=device,
            **optimizer_kwargs
        )
        # Set additional attributes specific to this class
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.finetune_bert = finetune_bert
        self.finetune_layers = finetune_layers
        self.label_dict = label_dict
        self.numeric_dict = numeric_dict
        self.pooling = pooling
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.num_layers = num_layers
        self.rank = rank
        self.debug = debug
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.resume_from_checkpoint = resume_from_checkpoint
        self.target_score = target_score
        self.interactive = interactive
        self.response_pipe = response_pipe
        self.freeze_bert = freeze_bert
        self.dropout_rate = dropout_rate
        self.show_progress = show_progress
        
        self.loss = nn.CrossEntropyLoss(reduction="mean")

        # Update self.params to include the new parameters
        self.params += ['bert_model', 'bert_tokenizer', 'finetune_bert', 'pooling', 'hidden_dim', 'hidden_activation',
                        'num_layers', 'rank', 'debug', 'checkpoint_dir', 'checkpoint_interval', 'resume_from_checkpoint',
                        'target_score', 'interactive', 'response_pipe', 'freeze_bert', 'dropout_rate']

    def build_graph(self):
        if not hasattr(self, 'n_classes_') or self.n_classes_ is None:
            raise ValueError("n_classes_ is not set. Make sure fit() is called before building the graph.")

        if self.finetune_bert:
            model = BERTClassifier(
                self.bert_model,
                self.pooling,
                self.hidden_dim,
                self.hidden_activation,
                self.num_layers,
                self.n_classes_,
                self.dropout_rate,
                self.finetune_layers,
                self.rank
            )
            if self.freeze_bert:
                for param in model.bert.parameters():
                    param.requires_grad = False
        else:
            model = Classifier(
                self.input_dim,
                self.hidden_dim,
                self.hidden_activation,
                self.num_layers,
                self.n_classes_,
                self.dropout_rate
            )

        if self.rank == 0:
            print("\nModel Architecture Summary:")
            print(model)
            total_params = 0
            trainable_params = 0
            total_layers = 0
            trainable_layers = 0

            print("\nModel Parameters:")
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                    total_layers += 1
                    layer_params = sum(p.numel() for p in module.parameters())
                    total_params += layer_params
                    if any(p.requires_grad for p in module.parameters()):
                        trainable_layers += 1
                        trainable_params += layer_params
                        print(f"  {name}: {layer_params:,} (trainable)")
                    else:
                        print(f"  {name}: {layer_params:,} (frozen)")

            print(f"\nTotal layers: {total_layers}")
            print(f"Trainable layers: {trainable_layers}")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Percentage of trainable parameters: {trainable_params/total_params*100:.2f}%")

        return model

    def build_optimizer(self):
        print(f"\nBuilding optimizer: {self.optimizer_class.__name__} with learning rate: {self.eta}, L2 strength: {self.l2_strength}") if self.rank == 0 else None
        if self.optimizer_class == ZeroRedundancyOptimizer:
            optimizer = ZeroRedundancyOptimizer(
                self.model.parameters(),
                optimizer_class=torch.optim.Adam,
                lr=self.eta,
                weight_decay=self.l2_strength,
                **self.optimizer_kwargs
            )
        else:
            optimizer = self.optimizer_class(
                self.model.parameters(),
                lr=self.eta,
                weight_decay=self.l2_strength,
                **self.optimizer_kwargs
            )
        
        return optimizer

    def build_dataset(self, X, y=None):
        X = tensor_to_numpy(X)
        self.input_dim = X.shape[1]
        X = torch.FloatTensor(X)

        if y is None:
            dataset = torch.utils.data.TensorDataset(X)
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            
            y = torch.tensor([class2index[label] for label in y], dtype=torch.long)
            
            if self.debug:
                print(f"Rank {self.rank}: X shape: {X.shape}, y shape: {y.shape}")
            
            dataset = torch.utils.data.TensorDataset(X, y)

        return dataset

    def _update_no_improvement_count_early_stopping(self, X, y, epoch, debug):
        current_score = self.score(X, y)
        if self.best_score is None or current_score > self.best_score + self.tol:
            self.best_score = current_score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        if debug and self.rank == 0:
            print(f"Current Score: {current_score:.6f}, Best Score: {self.best_score:.6f}, Tolerance: {self.tol}, No Improvement Count: {self.no_improvement_count}")
        return current_score

    def _update_no_improvement_count_errors(self, epoch_loss, epoch, debug):
        if self.best_error is None or epoch_loss < self.best_error - self.tol:
            self.best_error = epoch_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        if debug and self.rank == 0:
            print(f"Current Loss: {epoch_loss:.6f}, Best Loss: {self.best_error:.6f}, Tolerance {self.tol}, No Improvement Count: {self.no_improvement_count}")
        return epoch_loss

    def save_model(self, directory='saves', epoch=None, optimizer=None, is_final=False):
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'params': self.__dict__
        }
        if optimizer:
            state['optimizer_state_dict'] = optimizer.state_dict()
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        if is_final:
            filename = f'final_model_{timestamp}.pth'
        else:
            filename = f'checkpoint_epoch_{epoch}_{timestamp}.pth'
        
        torch.save(state, os.path.join(directory, filename))
        print(f"Saved model: {os.path.join(directory, filename)}")
    

    def load_model(self, directory='checkpoints', filename=None, pattern='checkpoint_epoch', use_saved_params=True, rank=0, debug=False):
        if not os.path.exists(directory):
            raise ValueError(f"Directory {directory} does not exist")

        if filename is not None:
            latest_checkpoint = os.path.join(directory, filename)
        else:
            # Find the latest file if no specific file is given
            matching_files = [os.path.join(directory, d) for d in os.listdir(directory) if pattern in d]
            if not matching_files:
                raise ValueError(f"No files matching the pattern '{pattern}' found in directory: {directory}")
            latest_checkpoint = max(matching_files, key=os.path.getctime)

        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        print(f"Loaded checkpoint: {latest_checkpoint}") if rank == 0 else None
        # Get the model state dict if it exists
        model_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else None

        # Remove 'module.' prefix if exists
        if model_state_dict is not None:
            model_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
            # Loop through the dictionary and print each key and value (nicely formatted f strings)
            print(f"Retrieved model state dictionary.") if rank == 0 else None
            print_state_summary(model_state_dict) if debug else None
        else:
            print(f"No model state dictionary found in checkpoint.") if rank == 0 else None

        # Get the optimizer state dict if it exists
        optimizer_state_dict = checkpoint['optimizer_state_dict'] if 'optimizer_state_dict' in checkpoint else None
        if optimizer_state_dict is not None:
            print(f"Retrieved optimizer state dictionary.") if rank == 0 else None
            print_state_summary(optimizer_state_dict) if debug else None
        else:
            print(f"No optimizer state dictionary found in checkpoint.") if rank == 0 else None

        # Optionally update model parameters with the saved parameters
        if use_saved_params and 'params' in checkpoint:
            if debug:
                print(f"BEFORE updating model parameters:") if rank == 0 else None
                print_state_summary(self.__dict__)
            self.__dict__.update(checkpoint['params'])
            if debug:
                print(f"AFTER updating model parameters:") if rank == 0 else None
                print_state_summary(self.__dict__)

        start_epoch = checkpoint.get('epoch', 0) + 1

        return start_epoch, model_state_dict, optimizer_state_dict


    def send_stop_signal(self):
        if self.response_pipe:
            try:
                self.response_pipe.send('stop')
                print(f"Rank {self.rank} sent stop signal") if self.debug else None
            except Exception as e:
                print(f"Rank {self.rank} failed to send stop signal: {e}") if self.debug else None
        else:
            print(f"Rank {self.rank} has no response_pipe to send stop signal") if self.debug else None


    def fit(self, X, y, rank, world_size, debug=False, start_epoch=1, model_state_dict=None, optimizer_state_dict=None,
            num_workers=0, prefetch=None, empty_cache=False, decimal=6, input_queue=None, mem_interval=10):
        training_start = time.time()
        if rank == 0:
            print(f"\nFitting DDP Neural Classifier on training data...")
            print(f"Num Workers: {num_workers}, Prefetch: {prefetch}")

        # Build the dataset and dataloader
        print(f"Building dataset and dataloader...") if rank == 0 else None

        if self.finetune_bert:
            if self.bert_tokenizer is None:
                raise ValueError("bert_tokenizer is required for fine-tuning BERT")
            dataset = SentimentDataset(X, y, self.bert_tokenizer, self.label_dict, device=self.device)
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
        else:
            dataset = self.build_dataset(X, y)

        if self.device.type == 'cuda' and world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=num_workers, prefetch_factor=prefetch)
        else:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=prefetch)

        # Set the classes and label dictionaries
        if self.label_dict is None:
            self.label_dict = {label: idx for idx, label in enumerate(self.classes_)}
        if self.numeric_dict is None:
            self.numeric_dict = {idx: label for label, idx in self.label_dict.items()}

        # Print tolerance as float with as many decimal places as in the tolerance
        tol_str = format_tolerance(self.tol)        

        # Determine the early stopping strategy and split the data if necessary
        if self.early_stopping == 'score':
            if rank == 0:
                print(f"Score-based early stopping enabled (macro average F1 score against validation set). Validation fraction: {self.validation_fraction}")
                print(f"Training will stop early if the score does not improve by at least {tol_str} for {self.n_iter_no_change} iterations.")
                (X_train, y_train), (X_val, y_val) = self._build_validation_split(X, y, validation_fraction=self.validation_fraction)
                data_list = [X_train, y_train, X_val, y_val]
            else:
                data_list = [None, None, None, None]
            # Broadcast the data
            if world_size > 1:
                dist.broadcast_object_list(data_list, src=0)
                # Unpack the broadcasted data
                X_train, y_train, X_val, y_val = data_list
            if rank == 0:
                print(f"Split data into (X:{len(X_train)}, y:{len(y_train)}) Training samples, and (X:{len(X_val)}, y:{len(y_val)}) Validation samples.")
        elif self.early_stopping == 'loss':
            if rank == 0:
                print(f"Loss-based early stopping enabled. No validation set required, all data used for training.")
                print(f"Training will stop early if the loss does not improve by at least {tol_str} for {self.n_iter_no_change} iterations.")
            X_train, y_train = X, y
        else:
            if rank == 0:
                print(f"Training without early stopping. No validation set required, all data used for training.")
                print(f"Training will stop after {self.max_iter} iterations.")
            X_train, y_train = X, y

        # Check target_score compatibility
        if self.target_score is not None:
            if self.early_stopping != 'score':
                print(f"Warning: target_score is set to {self.target_score}, but early_stopping is '{self.early_stopping}'. target_score will be ignored.") if rank == 0 else None
                self.target_score = None
            else:
                print(f"Target score set to {self.target_score}. Training will stop if this validation score is reached.") if rank == 0 else None

        # Initialize the model, optimizer, and set it in training mode
        print("Initializing model, graph, optimizer...") if rank == 0 else None
        
        self.initialize()
        dist.barrier()

        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
        
        if self.device.type == 'cuda':
            self.model = DDP(self.model, device_ids=[rank])
        else:
            self.model = DDP(self.model, device_ids=None)
        
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

        if rank == 0:
            print(f"Model architecture:\n{self.model}") if debug else None
            print(f"Optimizer:\n{self.optimizer}") if debug else None
            print(f"\nStarting training loop...")

        self.model.train()

        num_digits = len(str(self.max_iter))  # Number of digits for formatting epoch output
        stop_training = torch.tensor(0, device=self.device)  # Signal to stop training in DDP setup
        skip_epochs = 0  # Counter to skip epochs without prompting in interactive mode
        target_hit = False  # Flag to indicate if the target score was reached
        
        # Set the last epoch based on the start epoch and max iterations
        if start_epoch > 1:
            last_epoch = start_epoch + self.max_iter
        else:
            last_epoch = self.max_iter

        print(f"Start epoch: {start_epoch}, Max Iterations: {self.max_iter}, Early Stop: {self.early_stopping}, Tolerance: {tol_str}, Number Iterations No Change: {self.n_iter_no_change}") if rank == 0 else None
        print_rank_memory_summary(world_size, rank, all_local=True, verbose=False) if rank == 0 else None

        # Training loop
        for epoch in range(start_epoch, last_epoch+1):
            if stop_training.item() == 1:
                break  # Exit the training loop if stop_training was triggered

            epoch_start = time.time()
            if self.device.type == 'cuda' and world_size > 1:
                sampler.set_epoch(epoch)  # Required for DistributedSampler to shuffle the data
            epoch_loss = 0.0
            batch_count = 0

            # Create a progress bar for batches
            if rank == 0 and self.show_progress:
                pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}/{last_epoch}", 
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                            leave=True)
                
            for batch in dataloader:
                # Handle finetuning BERT or just training the classifier head, which have different X inputs
                if self.finetune_bert:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = self.loss(outputs, labels)
                else:
                    X_batch, y_batch = batch
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = self.loss(outputs, y_batch)
                
                print(f"Rank {rank}: Epoch: {epoch}, Batch: {batch_count}, Size: {X_batch.size(0)}, Loss: {loss.item():.6f}, Epoch Loss: {epoch_loss:.6f}") if debug else None
                
                # Average loss across all ranks
                if world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)

                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1

                # Update progress bar
                if rank == 0 and self.show_progress:
                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / batch_count

            # Update progress bar with post-epoch processes
            if rank == 0 and self.show_progress:
                pbar.set_postfix({'loss': f'{avg_epoch_loss:.4f}', 'status': 'Post-epoch processes'})

            # Consolidate optimizer state before saving
            if self.optimizer_class == ZeroRedundancyOptimizer:
                if rank == 0 and self.show_progress:
                    pbar.set_postfix({'loss': f'{avg_epoch_loss:.4f}', 'status': 'Consolidating optimizer state'})
                self.optimizer.consolidate_state_dict()

            dist.barrier()

            # Early stopping logic
            if self.early_stopping == 'score':
                if rank == 0 and self.show_progress:
                    pbar.set_postfix({'loss': f'{avg_epoch_loss:.4f}', 'status': 'Computing validation score'})
                current_score = self._update_no_improvement_count_early_stopping(X_val, y_val, epoch, debug)
                if self.no_improvement_count >= self.n_iter_no_change:
                    stop_training = torch.tensor(1, device=self.device)
                current_score_color, best_score_color = get_score_colors(current_score, self.best_score)

                # Check if target score is reached
                if self.target_score is not None and current_score >= self.target_score:
                    target_hit = True
                    if rank == 0:
                        print(f"Reached target validation score of {self.target_score}. Stopping training.")
                    stop_training = torch.tensor(1, device=self.device)

            elif self.early_stopping == 'loss':
                if rank == 0 and self.show_progress:
                    pbar.set_postfix({'loss': f'{avg_epoch_loss:.4f}', 'status': 'Checking early stopping'})
                current_loss = self._update_no_improvement_count_errors(avg_epoch_loss, epoch, debug)
                if self.no_improvement_count >= self.n_iter_no_change:
                    stop_training = torch.tensor(1, device=self.device)
                current_loss_color, best_error_color = get_loss_colors(current_loss, self.best_error, self.no_improvement_count, self.n_iter_no_change)

            # Get the no improvement count color based on value relative to n_iter_no_change
            nic_color = get_nic_color(self.no_improvement_count, self.n_iter_no_change)

            # Close progress bar here
            if rank == 0 and self.show_progress:
                pbar.close()

            # Print epoch summary
            if self.early_stopping == 'score':
                print(f"{nic_color}█{reset} Epoch {bright_white}{bold}{epoch:{num_digits}d}{reset}: Avg Loss: {bright_white}{bold}{avg_epoch_loss:.{decimal}f}{reset} | Val Score: {current_score_color}{bold}{current_score:.{decimal}f}{reset}, Best Score: {best_score_color}{bold}{self.best_score:.{decimal}f}{reset}, Stop Count: {nic_color}{bold}{self.no_improvement_count}{reset} / {self.n_iter_no_change} | Time: {format_time(time.time() - epoch_start)}")  if rank == 0 else None
            elif self.early_stopping == 'loss':
                print(f"{nic_color}█{reset} Epoch {bright_white}{bold}{epoch:{num_digits}d}{reset}: Avg Loss: {current_loss_color}{bold}{avg_epoch_loss:.{decimal}f}{reset} | Best Loss: {best_error_color}{bold}{self.best_error:.{decimal}f}{reset}, Stop Count: {nic_color}{bold}{self.no_improvement_count}{reset} / {self.n_iter_no_change} | Time: {format_time(time.time() - epoch_start)}")  if rank == 0 else None
            else:
                print(f"Epoch {bright_white}{bold}{epoch:{num_digits}d}{reset} / {self.max_iter}: Avg Loss: {bright_white}{bold}{avg_epoch_loss:.{decimal}f}{reset} | Total Loss: {epoch_loss:.{decimal}f}, Batch Count: {batch_count} | Time: {format_time(time.time() - epoch_start)}")  if rank == 0 else None

            # Print memory summary
            if epoch % mem_interval == 0:
                print_rank_memory_summary(world_size, rank, all_local=True, verbose=False) if rank == 0 else None

            # Interactive mode
            if self.interactive and skip_epochs == 0:

                # Create a global tensor for synchronization
                global_tensor = torch.zeros(2, dtype=torch.long, device=self.device)

                while True:  # Start a loop to handle the input

                    # Synchronize all processes after each epoch
                    dist.barrier()

                    command_value = torch.tensor([0.0, 0.0], device=self.device)  # Tensor to store the command and an optional value

                    if rank == 0:
                        sys.stdout.flush()
                        if input_queue is not None:
                            input_queue.put(rank)
                        user_input = self.response_pipe.recv()
                        print(f"Rank {rank} received input: {user_input}") if debug else None
                        
                        # Parse the input
                        user_input_split = user_input.split()
                        if not user_input_split:  # If the input is empty (just Enter was pressed)
                            command_value[0] = 0  # Continue
                            command_value[1] = 0  # No epochs to skip
                        else:
                            # Check if the input is just a number
                            if user_input_split[0].isdigit():
                                command_value[0] = 0  # Continue
                                command_value[1] = int(user_input_split[0])  # Number of epochs to skip
                            else:
                                # Set command based on the first input
                                if user_input_split[0] in ['c', 'checkpoint']:
                                    command_value[0] = 10  # Change checkpoint interval
                                    if len(user_input_split) > 1:
                                        try:
                                            command_value[1] = int(user_input_split[1])  # New checkpoint interval
                                        except ValueError:
                                            print("Invalid number for checkpoint interval. Please enter an integer.")
                                            command_value[1] = -1  # Indicate invalid input
                                elif user_input_split[0] in ['s', 'save']:
                                    command_value[0] = 1  # Save
                                elif user_input_split[0] in ['q', 'quit']:
                                    command_value[0] = 2  # Quit
                                elif user_input_split[0] in ['h', 'help']:
                                    command_value[0] = 3  # Help
                                elif user_input_split[0] in ['e', 'epoch', 'epochs'] and len(user_input_split) > 1:
                                    try:
                                        command_value[0] = 4  # Set max epochs
                                        command_value[1] = float(user_input_split[1])  # Associated value for max epochs
                                    except ValueError:
                                        print("Invalid value for max epochs")
                                elif user_input_split[0] in ['t', 'tol', 'tolerance'] and len(user_input_split) > 1:
                                    try:
                                        command_value[0] = 5  # Set tolerance
                                        command_value[1] = float(user_input_split[1])  # Associated value for tolerance
                                    except ValueError:
                                        print("Invalid value for tolerance")
                                elif user_input_split[0] in ['n', 'num', 'number'] and len(user_input_split) > 1:
                                    try:
                                        command_value[0] = 6  # Set n_iter_no_change
                                        command_value[1] = float(user_input_split[1])  # Associated value for n_iter_no_change
                                    except ValueError:
                                        print("Invalid value for n_iter_no_change")
                                elif user_input_split[0] in ['x', 'exit']:
                                    command_value[0] = 7  # Exit interactive mode
                                elif user_input_split[0] in ['d', 'debug']:
                                    command_value[0] = 8  # Toggle debug mode
                                elif user_input_split[0] in ['v', 'val', 'validation', 'target'] and len(user_input_split) > 1:
                                    try:
                                        command_value[0] = 9  # Set target validation score
                                        command_value[1] = float(user_input_split[1])  # Associated value for target validation score
                                    except ValueError:
                                        print("Invalid value for target validation score")
                                elif user_input_split[0] in ['m', 'mem', 'memory']:
                                    if len(user_input_split) > 1:
                                        try:
                                            command_value[0] = 11  # Set memory interval
                                            command_value[1] = float(user_input_split[1])  # Associated value for memory interval
                                        except ValueError:
                                            print("Invalid value for memory interval")
                                    else:
                                        print("A number is required for the memory interval")
                                elif user_input_split[0] in ['g', 'gpu', 'gpus']:
                                    command_value[0] = 12  # Show GPU memory usage
                                elif user_input_split[0] in ['l', 'lr', 'learning', 'rate'] and len(user_input_split) > 1:
                                    try:
                                        command_value[0] = 13  # Set learning rate
                                        command_value[1] = float(user_input_split[1])  # Associated value for learning rate
                                    except ValueError:
                                        print("Invalid value for learning rate")

                        # Set the global tensor
                        global_tensor[0] = 1  # Signal that input is ready
                        global_tensor[1] = command_value[0]  # Store the command

                    # Broadcast the global tensor to all processes
                    dist.broadcast(global_tensor, src=0)
                    dist.broadcast(command_value, src=0)  # Broadcast the command value

                    # Process the input
                    command = int(global_tensor[1].item())
                    value = command_value[1].item()

                    # Handle the command
                    if command == 0:  # Continue
                        skip_epochs = int(value) if value > 0 else 0  # Set the skip_epochs based on the provided value
                        if skip_epochs > 0 and rank == 0:
                            print(f"Skipping {skip_epochs} epochs.")
                        break  # Exit the loop and proceed to the next epoch
                    if command == 1:  # Save
                        if rank == 0:
                            self.save_model(directory=self.checkpoint_dir, epoch=epoch, optimizer=self.optimizer, is_final=False)
                            print("Model saved.") if debug else None
                        continue  # Stay in the loop to wait for another command
                    elif command == 2:  # Quit
                        print("Quitting training...") if rank == 0 else None
                        stop_training = torch.tensor(1, device=self.device)  # Set the flag to stop training
                        break  # Exit the loop to continue to the next epoch
                    elif command == 3:  # Help
                        if rank == 0:
                            print(f"\nChoose from the following commands:")
                            print(f"[{bold}{bright_yellow}Enter{reset}]            Continue to the next epoch, for the next prompt.")
                            print(f"[{bold}{bright_yellow}#{reset}]                Skip the specified number of epochs before the next prompt. Example: '5'")
                            print(f"[{bold}{bright_yellow}C{reset}]heckpoint ____  Set the checkpoint interval. Current: {self.checkpoint_interval}. Example: 'c 5'")
                            print(f"[{bold}{bright_yellow}D{reset}]ebug            Toggle debug mode. Current: {debug}")
                            print(f"[{bold}{bright_yellow}E{reset}]pochs     ____  Set the maximum number of epochs. Current: {self.max_iter}. Example: 'e 1000'")
                            print(f"[{bold}{bright_yellow}G{reset}]PUs       ____  Show current GPU memory usage.")
                            print(f"[{bold}{bright_yellow}H{reset}]elp             Display this help message")
                            print(f"[{bold}{bright_yellow}L{reset}]earning   ____  Set the learning rate. Current: {self.eta:.{decimal}f}. Example: 'l 0.001'")
                            print(f"[{bold}{bright_yellow}M{reset}]emory     ____  Set the interval to display GPU memory usage. Example: 'm 5'")
                            print(f"[{bold}{bright_yellow}N{reset}]umber     ____  Set the number of iterations no change. Current: {self.n_iter_no_change}. Example: 'n 10'")
                            print(f"[{bold}{bright_yellow}Q{reset}]uit             Quit the training loop")
                            print(f"[{bold}{bright_yellow}S{reset}]ave             Save the model as a checkpoint with DDP wrapper")
                            print(f"[{bold}{bright_yellow}T{reset}]olerance  ____  Set the tolerance for early stopping. Current: {self.tol}. Example: 't 1e-03'")
                            print(f"[{bold}{bright_yellow}V{reset}]al Score  ____  Set the target validation score for early stopping. Current: {self.target_score}. Example: 'v 0.95'")
                            print(f"[{bold}{bright_yellow}X{reset}]it              Exit interactive mode. You won't be prompted for input again.")
                        continue  # Stay in the loop to wait for another command
                    elif command == 4:  # Set max epochs
                        self.max_iter = int(value)
                        if start_epoch > 1:
                            last_epoch = start_epoch + self.max_iter
                        else:
                            last_epoch = self.max_iter
                        print(f"Updated max epochs to: {self.max_iter}") if rank == 0 else None
                        continue  # Stay in the loop to wait for another command
                    elif command == 5:  # Set tolerance
                        self.tol = value
                        print(f"Updated tolerance to: {self.tol}") if rank == 0 else None
                        continue  # Stay in the loop to wait for another command
                    elif command == 6:  # Set n_iter_no_change
                        self.n_iter_no_change = int(value)
                        print(f"Updated number of iterations no change to: {self.n_iter_no_change}") if rank == 0 else None
                        continue  # Stay in the loop to wait for another command
                    elif command == 7:  # Exit interactive mode
                        print("Exiting interactive mode...") if rank == 0 else None
                        self.interactive = False
                        self.send_stop_signal() # Send stop signal to master process to exit the interactive loop
                        break  # Exit the loop to continue to the next epoch
                    elif command == 8:  # Toggle debug mode
                        debug = not debug
                        print(f"Debug mode set to: {debug}") if rank == 0 else None
                        continue  # Stay in the loop to wait for another command
                    elif command == 9:  # Set target validation score
                        self.target_score = value
                        print(f"Updated target validation score to: {self.target_score:.{decimal}f}") if rank == 0 else None
                        continue  # Stay in the loop to wait for another command
                    elif command == 10:  # Change checkpoint interval
                        if value > 0:
                            self.checkpoint_interval = value
                            print(f"Updated checkpoint interval to: {self.checkpoint_interval}") if rank == 0 else None
                        else:
                            print(f"Invalid value for checkpoint interval. Please enter a positive integer.") if rank == 0 else None
                        continue  # Stay in the loop to wait for another command
                    elif command == 11:  # Set memory interval
                        if value > 0:
                            mem_interval = int(value)
                            print(f"Updated memory interval to: {mem_interval}") if rank == 0 else None
                        else:
                            print(f"Invalid value for memory interval. Please enter a positive integer.") if rank == 0 else None
                        continue  # Stay in the loop to wait for another command
                    elif command == 12:  # Show GPU memory usage
                        print_rank_memory_summary(world_size, rank, all_local=True, verbose=True) if rank == 0 else None
                        continue  # Stay in the loop to wait for another command
                    elif command == 13:  # Set learning rate
                        self.eta = value
                        for param_group in self.optimizer.param_groups:
                            if 'lr' in param_group:
                                param_group['lr'] = self.eta
                        if rank == 0:
                            print(f"Updated learning rate to: {self.eta:.{decimal}f}")
                        continue  # Stay in the loop to wait for another command

                    # Reset the global tensor for the next iteration
                    if rank == 0:
                        global_tensor[0] = 0
                    dist.broadcast(global_tensor, src=0)

                    # Ensure all ranks have processed the input before continuing
                    dist.barrier()

            # Print early stopping message
            if stop_training.item() == 1:
                self.send_stop_signal() # Send stop signal to master process to exit the interactive loop
                if self.early_stopping == 'score' and target_hit:
                    print(f"Stopping early after {epoch} epochs due to reaching target validation score of {self.target_score:.{decimal}f}.") if rank == 0 else None
                elif self.early_stopping == 'score':
                    print(f"Stopping early after {epoch} epochs due to no improvement in validation score in last {self.n_iter_no_change} iterations.")  if rank == 0 else None
                elif self.early_stopping == 'loss':
                    print(f"Stopping early after {epoch} epochs due to no improvement in training loss in last {self.n_iter_no_change} iterations.") if rank == 0 else None
                else:
                    print(f"Stopping after {epoch} epochs. (Start: {start_epoch}, Last: {last_epoch}, Max: {self.max_iter})") if rank == 0 else None

            # Save a checkpoint
            if self.checkpoint_interval and (epoch) % self.checkpoint_interval == 0:
                if rank == 0:
                    self.save_model(directory=self.checkpoint_dir, epoch=epoch, optimizer=self.optimizer, is_final=False)
            
            # Decrement the skip_epochs counter
            if skip_epochs > 0:
                skip_epochs -= 1

            # Check if we've reached the last_epoch
            if epoch == last_epoch:
                stop_training = torch.tensor(1, device=self.device)

            # Broadcast stop_training to all processes
            if world_size > 1:
                dist.broadcast(stop_training, 0)

            # Empty CUDA cache after each epoch
            if empty_cache:
                torch.cuda.empty_cache()

        if self.early_stopping and self.device.type == 'cuda' and world_size > 1:
            # Broadcast best parameters to all processes
            for param in self.model.parameters():
                dist.broadcast(param.data, 0)
        
        # Save the final model
        if rank == 0:
            self.save_model(directory=self.checkpoint_dir, epoch=epoch, optimizer=self.optimizer, is_final=True)
            print(f"Training completed ({format_time(time.time() - training_start)})")

        return self

    def score(self, X, y, device=None, debug=False):
        if debug:
            print(f"Scoring on {len(X)} samples...")
            if self.finetune_bert:
                print(f"X type: {type(X)}")
                print(f"X[:5]: {X[:5]}")
            else:
                print(f"X shape: {X.shape}")
                print(f"X type: {X.dtype}")
                print(f"X[:5]: {X[:5]}")
            print(f"y[:5]: {y[:5]}")
            print(f"y type: {y.dtype}")
        preds = self.predict(X, device=device, debug=debug)
        return utils.safe_macro_f1(y, preds)

    def predict_proba(self, X, device=None, debug=False):
        try:
            preds = self._predict(X, device=device, debug=debug)
            probs = torch.softmax(preds, dim=1).cpu().numpy()
            
            if debug:
                print(f"Probability shape: {probs.shape}")
                print(f"Sample probabilities:\n{probs[:5]}")
            
            return probs
        except Exception as e:
            print(f"Error in predict_proba method: {str(e)}")
            if debug:
                print(f"Input shape: {X.shape}")
                print(f"Device: {device}")
            raise ValueError(f"Probability prediction failed: {str(e)}") from e

    def predict(self, X, device=None, debug=False):
        try:
            probs = self.predict_proba(X, device=device, debug=debug)
            numeric_preds = probs.argmax(axis=1)
            labels = convert_numeric_to_labels(numeric_preds, self.numeric_dict)
            
            if debug:
                print(f"Prediction shapes: Input: {X.shape}, Probabilities: {probs.shape}, Numeric predictions: {numeric_preds.shape}")
                print(f"Sample predictions: {labels[:5]}")
            
            return labels
        except Exception as e:
            print(f"Error in predict method: {str(e)}")
            if debug:
                print(f"Input shape: {X.shape}")
                print(f"Device: {device}")
                print(f"Numeric dict: {self.numeric_dict}")
            raise ValueError(f"Prediction failed: {str(e)}") from e

    def _predict(self, X, device=None, debug=False):
        if device is None:
            device = self.device
        self.model.to(device)
        self.model.eval()

        if self.finetune_bert:
            dataset = SentimentDataset(X, [0] * len(X), self.bert_tokenizer, label_dict=self.label_dict, device=device)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            preds = []
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    preds.append(outputs)
            preds = torch.cat(preds, dim=0)
        else:
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            X = X.to(device)
            with torch.no_grad():
                preds = self.model(X)
        return preds


    def to(self, device):
        if self.model:
            self.model = self.model.to(device)
        return self#
