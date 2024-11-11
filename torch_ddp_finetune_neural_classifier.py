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
from sklearn.model_selection import train_test_split
from transformers import ElectraConfig, AutoTokenizer
from transformers import ElectraPreTrainedModel, ElectraModel
import os
import time
import utils
import sys
import select
from tqdm import tqdm
from multiprocessing import Value
from colors import *
from utils import (format_time, print_state_summary, format_tolerance, get_nic_color, get_score_colors,
                   print_rank_memory_summary, convert_labels_to_tensor, convert_numeric_to_labels, tensor_to_numpy,
                   get_scheduler, SwishGLU)

import wandb
import torch.onnx
import io
import warnings

# Ignore some warnings related to ONNX conversion
warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean")
warnings.filterwarnings("ignore", message="Converting a tensor to a Python number")
warnings.filterwarnings("ignore", message="Converting a tensor to a Python list")

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, tokenizer, label_dict=None, max_length=512, device='cpu'):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        if labels is not None:
            self.labels = convert_labels_to_tensor(labels, label_dict, device)
        else:
            self.labels = None

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}

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

class TransformerClassifier(nn.Module):
    def __init__(self, transformer_model, pooling, hidden_dim, hidden_activation, num_layers, n_classes, dropout_rate, finetune_layers=1, rank=0):
        super().__init__()
        self.transformer = transformer_model
        
        # Remove the model's original pooler if it exists
        if hasattr(self.transformer, 'pooler'):
            self.transformer.pooler = None
        
        # Add our custom pooling layer
        self.custom_pooling = PoolingLayer(pooling)
        self.classifier = Classifier(self.transformer.config.hidden_size, hidden_dim, hidden_activation, num_layers, n_classes, dropout_rate)

        # Get the total number of layers in the transformer model
        if hasattr(self.transformer, 'encoder'):
            total_layers = len(self.transformer.encoder.layer)
        elif hasattr(self.transformer, 'layers'):
            total_layers = len(self.transformer.layers)
        else:
            total_layers = 12  # default for most models, adjust if necessary

        # Freeze all layers except the specified number of final layers
        if finetune_layers == 0:
            for param in self.transformer.parameters():
                param.requires_grad = False
        elif finetune_layers < total_layers:
            modules_to_freeze = self.transformer.embeddings
            if hasattr(self.transformer, 'encoder'):
                modules_to_freeze = [modules_to_freeze, *self.transformer.encoder.layer[:-finetune_layers]]
            elif hasattr(self.transformer, 'layers'):
                modules_to_freeze = [modules_to_freeze, *self.transformer.layers[:-finetune_layers]]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        # Count trainable and non-trainable parameters
        trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.transformer.parameters() if not p.requires_grad)

        if rank == 0:
            print(f"Transformer's original pooler removed. Using custom pooling type: {pooling}")
            print(f"Transformer has {trainable_params:,} trainable parameters and {non_trainable_params:,} non-trainable parameters")
            print(f"Fine-tuning the last {finetune_layers} out of {total_layers} Transformer layers")
    
    def forward(self, **inputs):
        outputs = self.transformer(**inputs)
        pooled_output = self.custom_pooling(outputs.last_hidden_state, inputs['attention_mask'])
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


class CustomElectraClassifier(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.electra = ElectraModel(config)
        
        # Remove the original pooler if it exists
        if hasattr(self.electra, 'pooler'):
            self.electra.pooler = None
        
        # Add your custom pooling layer
        self.pooling = PoolingLayer(pooling_type=config.pooling)
        
        # Handle custom activation functions
        activation_name = config.hidden_activation
        if activation_name == 'SwishGLU':
            hidden_activation = SwishGLU(input_dim=config.hidden_dim, output_dim=config.hidden_dim)
        else:
            activation_class = getattr(nn, activation_name)
            hidden_activation = activation_class()
        
        self.classifier = Classifier(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_dim,
            hidden_activation=hidden_activation,
            num_layers=config.num_layers,
            n_classes=config.num_labels,
            dropout_rate=config.dropout_rate
        )
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.electra(input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = self.pooling(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled_output)
        return logits
    
            
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
                use_zero=True,
                scheduler_class=None,
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
                world_size=None,
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
                advance_epochs=1,
                wandb_run = None,
                random_seed=42,
                lr_decay=1.0,
                optimizer_kwargs={},
                scheduler_kwargs={}):
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
        # Call the superclass constructor
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

        # Set additional attributes
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
        self.world_size = world_size
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
        self.advance_epochs = advance_epochs
        self.use_zero = use_zero
        self.scheduler_class = scheduler_class
        self.scheduler = None
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.wandb_run = wandb_run
        self.random_seed = random_seed
        self.lr_decay = lr_decay
        
        self.loss = nn.CrossEntropyLoss(reduction="mean")

        # Extend self.params with the new parameters
        self.params.extend([
            'finetune_bert', 'pooling', 'hidden_dim', 'hidden_activation',
            'num_layers', 'checkpoint_interval', 'target_score', 'interactive',
            'freeze_bert', 'dropout_rate', 'show_progress', 'advance_epochs',
            'use_zero', 'scheduler_class'
        ])

        # Handle optimizer_kwargs
        for k, v in optimizer_kwargs.items():
            setattr(self, k, v)
            if k not in self.params:
                self.params.append(k)

    def build_graph(self):
        if not hasattr(self, 'n_classes_') or self.n_classes_ is None:
            raise ValueError(f"{red}n_classes_ is not set. Make sure fit() is called before building the graph.{reset}")

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
            print(f"\n{sky_blue}Model Architecture Summary:{reset}")
            print(model)
            total_params = 0
            trainable_params = 0
            total_layers = 0
            trainable_layers = 0

            print(f"\n{sky_blue}Model Parameters:{reset}")
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
        # Group parameters by layer depth with different learning rates
        lr_decay = getattr(self, 'lr_decay', 1.0)  # Default to 1.0 if not set
        
        if self.finetune_bert:  # Only do layer-wise decay if we're fine-tuning BERT
            if hasattr(self.model, 'module'):
                bert = self.model.module.bert
            else:
                bert = self.model.bert
                
            # Create parameter groups with decaying learning rates
            optimizer_grouped_parameters = []
            
            # Classifier (highest learning rate)
            classifier_params = [p for n, p in self.model.named_parameters() if 'classifier' in n and p.requires_grad]
            if classifier_params:
                optimizer_grouped_parameters.append({
                    'params': classifier_params,
                    'lr': self.eta,
                    'weight_decay': self.l2_strength
                })
            
            # BERT layers
            num_layers = len(bert.encoder.layer)
            for layer_num in range(num_layers):
                layer = bert.encoder.layer[-(layer_num + 1)]  # Start from last layer
                layer_params = [p for p in layer.parameters() if p.requires_grad]
                if layer_params:
                    layer_lr = self.eta * (lr_decay ** layer_num)
                    optimizer_grouped_parameters.append({
                        'params': layer_params,
                        'lr': layer_lr,
                        'weight_decay': self.l2_strength
                    })
            
            # Embeddings (lowest learning rate)
            embedding_params = [p for p in bert.embeddings.parameters() if p.requires_grad]
            if embedding_params:
                embeddings_lr = self.eta * (lr_decay ** num_layers)
                optimizer_grouped_parameters.append({
                    'params': embedding_params,
                    'lr': embeddings_lr,
                    'weight_decay': self.l2_strength
                })
        else:
            # Standard parameter groups without layer-wise decay
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable_params:
                raise ValueError("No trainable parameters found in the model!")
                
            optimizer_grouped_parameters = [{
                'params': trainable_params,
                'lr': self.eta,
                'weight_decay': self.l2_strength
            }]

        if self.use_zero:
            # Make parameters contiguous
            for group in optimizer_grouped_parameters:
                for param in group['params']:
                    if param.requires_grad:
                        param.data = param.data.contiguous()
                        
            optimizer = ZeroRedundancyOptimizer(
                optimizer_grouped_parameters,
                optimizer_class=self.optimizer_class,
                **self.optimizer_kwargs
            )
        else:
            optimizer = self.optimizer_class(
                optimizer_grouped_parameters,
                **self.optimizer_kwargs
            )

        if self.rank == 0:
            print(f"Using optimizer: {self.optimizer_class.__name__}, Use Zero: {self.use_zero}, Base Learning Rate: {self.eta}, L2 strength: {self.l2_strength}")
            if self.finetune_bert and lr_decay != 1.0:
                print(f"Layer-wise decay factor: {lr_decay}")
            if self.optimizer_kwargs:
                print("Optimizer arguments:")
                for key, value in self.optimizer_kwargs.items():
                    print(f"- {key}: {value}")

        if self.scheduler_class is not None:
            # Set default values based on scheduler type
            if self.scheduler_class == optim.lr_scheduler.CosineAnnealingLR:
                if 'T_max' not in self.scheduler_kwargs:
                    self.scheduler_kwargs['T_max'] = self.max_iter
            elif self.scheduler_class == optim.lr_scheduler.CosineAnnealingWarmRestarts:
                if 'T_0' not in self.scheduler_kwargs:
                    self.scheduler_kwargs['T_0'] = self.max_iter // 10  # Restart every 1/10th of total epochs
            elif self.scheduler_class == optim.lr_scheduler.StepLR:
                if 'step_size' not in self.scheduler_kwargs:
                    self.scheduler_kwargs['step_size'] = self.max_iter // 3  # Step every 1/3 of total epochs
            elif self.scheduler_class == optim.lr_scheduler.MultiStepLR:
                if 'milestones' not in self.scheduler_kwargs:
                    self.scheduler_kwargs['milestones'] = [self.max_iter // 2, self.max_iter * 3 // 4]  # Steps at 1/2 and 3/4 of total epochs
            elif self.scheduler_class == optim.lr_scheduler.CyclicLR:
                if 'base_lr' not in self.scheduler_kwargs:
                    self.scheduler_kwargs['base_lr'] = self.eta / 10
                if 'max_lr' not in self.scheduler_kwargs:
                    self.scheduler_kwargs['max_lr'] = self.eta
                if 'step_size_up' not in self.scheduler_kwargs:
                    self.scheduler_kwargs['step_size_up'] = self.max_iter // 20  # 1/20th of total epochs

            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)

            if self.rank == 0:
                print(f"Using scheduler: {self.scheduler_class.__name__}")
                if self.scheduler_kwargs:
                    print("Scheduler arguments:")
                    for key, value in self.scheduler_kwargs.items():
                        print(f"- {key}: {value}")
                
        else:
            scheduler = None
        
        self.scheduler = scheduler
        
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
            print(f"Classes: {self.classes_}, Number of classes: {self.n_classes_}") if self.rank == 0 else None
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            
            y = torch.tensor([class2index[label] for label in y], dtype=torch.long)
            
            if self.debug:
                print(f"Rank {self.rank}: X shape: {X.shape}, y shape: {y.shape}")
            
            dataset = torch.utils.data.TensorDataset(X, y)

        return dataset

    @staticmethod
    def _build_validation_split(*args, validation_fraction=0.2, random_seed=42):
        """
        Split `*args` into train and dev portions for early stopping.
        We use `train_test_split`. For args of length N, then delivers
        N*2 objects, arranged as

        X1_train, X1_test, X2_train, X2_test, ..., y_train, y_test

        Parameters
        ----------
        *args: List of objects to split.

        validation_fraction: float
            Percentage of the examples to use for the dev portion. In
            `fit`, this is determined by `self.validation_fraction`.
            We give it as an argument here to facilitate unit testing.

        Returns
        -------
        Pair of tuples `train` and `dev`

        """
        if validation_fraction == 1.0:
            return args, args
        results = train_test_split(*args, test_size=validation_fraction,
                                   random_state=random_seed,
                                   shuffle=True,
                                   stratify=args[-1] if isinstance(args[-1], np.ndarray) else None)
        train = results[::2]
        dev = results[1::2]
        return train, dev

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


    def save_model(self, directory='saves', epoch=None, optimizer=None, is_final=False, save_pickle=False, save_hf=False, weights_name='bert-base-uncased'):
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Prepare saveable parameters
        saveable_params = {param: getattr(self, param) for param in self.params 
                        if param not in ['rank', 'response_pipe', 'bert_model', 'bert_tokenizer']}

        # Prepare model state dictionary
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'params': saveable_params
        }
        if optimizer:
            state['optimizer_state_dict'] = optimizer.state_dict()

        # Save as a PyTorch checkpoint
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'final_model_{timestamp}' if is_final else f'checkpoint_epoch_{epoch}_{timestamp}'
        torch.save(state, os.path.join(directory, filename + '.pth'))
        print(f"Saved model state: {os.path.join(directory, filename + '.pth')}")

        # Optionally save as a pickle file if specified
        if is_final and save_pickle:
            self.to_pickle(os.path.join(directory, filename + '.pkl'))
            print(f"Saved model pickle: {os.path.join(directory, filename + '.pkl')}")
            self.model.to(self.device)

        # Optionally save in Hugging Face format
        if save_hf:
            hf_save_dir = os.path.join(directory, f"{filename}_huggingface")
            os.makedirs(hf_save_dir, exist_ok=True)
            
            # Create a config object with your custom parameters
            config = ElectraConfig.from_pretrained(weights_name)
            config.num_labels = self.n_classes_
            config.hidden_dim = self.hidden_dim
            # Adjust config.hidden_activation
            if isinstance(self.hidden_activation, nn.Module):
                config.hidden_activation = self.hidden_activation.__class__.__name__
            else:
                config.hidden_activation = self.hidden_activation
            config.num_layers = self.num_layers
            config.dropout_rate = self.dropout_rate
            config.pooling = self.pooling  # Ensure pooling_type is included

            # Create an instance of your custom model
            model = CustomElectraClassifier(config)

            # Adjust the state dict keys
            adjusted_state_dict = {}
            for k, v in state['model_state_dict'].items():
                if k.startswith('bert.'):
                    new_key = k.replace('bert.', 'electra.')
                else:
                    new_key = k
                adjusted_state_dict[new_key] = v

            # Load the adjusted state dict
            model.load_state_dict(adjusted_state_dict)

            # Save the model and tokenizer
            model.save_pretrained(hf_save_dir)
            model.save_pretrained(hf_save_dir, safe_serialization=False)
            self.bert_tokenizer.save_pretrained(hf_save_dir)
            
            print(f"Model also saved in Hugging Face format to {hf_save_dir}")

        # Debugging information if enabled
        if self.debug and self.rank == 0:
            print("Params saved:")
            print_state_summary(saveable_params)




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
        model_state_dict = checkpoint.get('model_state_dict')
        if model_state_dict:
            # Remove 'module.' prefix if exists
            model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
            print(f"Retrieved model state dictionary.") if rank == 0 else None
            if debug:
                print_state_summary(model_state_dict)
        else:
            print(f"No model state dictionary found in checkpoint.") if rank == 0 else None

        # Get the optimizer state dict if it exists
        optimizer_state_dict = checkpoint.get('optimizer_state_dict')
        if optimizer_state_dict:
            print(f"Retrieved optimizer state dictionary.") if rank == 0 else None
            if debug:
                print_state_summary(optimizer_state_dict)
        else:
            print(f"No optimizer state dictionary found in checkpoint.") if rank == 0 else None

        # Optionally update model parameters with the saved parameters
        if use_saved_params and 'params' in checkpoint:
            saved_params = checkpoint['params']
            
            if debug and rank == 0:
                print(f"BEFORE updating model parameters:")
                print_state_summary(self.__dict__)
            
            # Update the parameters
            for key, value in saved_params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            if debug and rank == 0:
                print(f"AFTER updating model parameters:")
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

    def set_advance_epochs(self, value):
        self.advance_epochs = value
        if self.response_pipe:
            try:
                self.response_pipe.send(f'advance_epochs:{value}')
                print(f"Rank {self.rank} set advance_epochs to {value}") if self.debug else None
            except Exception as e:
                print(f"Rank {self.rank} failed to set advance_epochs: {e}") if self.debug else None
        else:
            print(f"Rank {self.rank} has no response_pipe to set advance_epochs") if self.debug else None

    def compute_accuracy(self, y_true, y_pred):
        return (y_true == y_pred).mean()

    def compute_train_metrics(self, X, y, epoch, debug=False, device=None):
        if device is None:
            device = self.device
        
        # Temporarily set to eval mode
        self.model.eval()
        
        with torch.no_grad():
            all_preds = []
            all_labels = []
            
            if self.finetune_bert:
                dataset = SentimentDataset(X, y, self.bert_tokenizer, self.label_dict, device=device)
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False
                )
                dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
                
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    
                    all_preds.append(outputs.argmax(dim=1))
                    all_labels.append(labels)
            else:
                X = tensor_to_numpy(X)
                y = np.array(y)
                if y.dtype == object:
                    y = np.array([self.label_dict[label] for label in y])
                dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False
                )
                dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
                
                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    outputs = self.model(X_batch)
                    
                    all_preds.append(outputs.argmax(dim=1))
                    all_labels.append(y_batch)
        
        # Concatenate predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        # Gather sizes from all ranks
        local_size = torch.tensor([all_preds.size(0)], device=device)
        sizes_list = [torch.zeros(1, device=device, dtype=torch.int64) for _ in range(self.world_size)]
        torch.distributed.all_gather(sizes_list, local_size)
        sizes = [int(size.item()) for size in sizes_list]
        max_size = max(sizes)

        # Pad tensors to the maximum size
        pad_size = max_size - all_preds.size(0)
        if pad_size > 0:
            all_preds = torch.cat([
                all_preds,
                torch.zeros(pad_size, dtype=all_preds.dtype, device=device)
            ])
            all_labels = torch.cat([
                all_labels,
                torch.zeros(pad_size, dtype=all_labels.dtype, device=device)
            ])
        
        # Prepare lists for gathering tensors
        gathered_preds = [torch.zeros(max_size, dtype=all_preds.dtype, device=device) for _ in range(self.world_size)]
        gathered_labels = [torch.zeros(max_size, dtype=all_labels.dtype, device=device) for _ in range(self.world_size)]
        
        # Gather predictions and labels from all ranks
        torch.distributed.all_gather(gathered_preds, all_preds)
        torch.distributed.all_gather(gathered_labels, all_labels)
        
        # Remove padding and concatenate
        all_preds = torch.cat([preds[:sizes[i]] for i, preds in enumerate(gathered_preds)])
        all_labels = torch.cat([labels[:sizes[i]] for i, labels in enumerate(gathered_labels)])
        
        # Move tensors to CPU for metric computation
        all_preds = all_preds.cpu().numpy()
        all_labels = all_labels.cpu().numpy()
        
        # Compute training metrics
        train_score = utils.safe_macro_f1(all_labels, all_preds)
        train_accuracy = self.compute_accuracy(all_labels, all_preds)
        
        # Debug statements
        if debug and self.rank == 0:
            print(f"Train Score at epoch {epoch}: F1 = {train_score:.6f}, Accuracy = {train_accuracy:.6f}")
        
        self.model.train()
    
        return train_score, train_accuracy
                

    def compute_validation_metrics(self, X, y, epoch, debug=False, device=None):
        if device is None:
            device = self.device
        self.model.eval()
        
        # Initialize tensors for loss and sample count
        val_loss = torch.tensor(0.0, device=device)
        total_samples = torch.tensor(0, device=device)
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            if self.finetune_bert:
                # Create dataset and DistributedSampler
                dataset = SentimentDataset(X, y, self.bert_tokenizer, self.label_dict, device=device)
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False
                )
                dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
                
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = self.loss(outputs, labels)
                    
                    # Accumulate loss and sample count
                    val_loss += loss * labels.size(0)
                    total_samples += labels.size(0)
                    
                    # Collect predictions and labels
                    all_preds.append(outputs.argmax(dim=1))
                    all_labels.append(labels)
            else:
                X = tensor_to_numpy(X)
                y = np.array(y)
                if y.dtype == object:
                    y = np.array([self.label_dict[label] for label in y])
                dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False
                )
                dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    outputs = self.model(X_batch)
                    loss = self.loss(outputs, y_batch)
                    
                    val_loss += loss * y_batch.size(0)
                    total_samples += y_batch.size(0)
                    
                    all_preds.append(outputs.argmax(dim=1))
                    all_labels.append(y_batch)
        
        # Concatenate predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        # Gather sizes from all ranks
        local_size = torch.tensor([all_preds.size(0)], device=device)
        sizes_list = [torch.zeros(1, device=device, dtype=torch.int64) for _ in range(self.world_size)]
        torch.distributed.all_gather(sizes_list, local_size)
        sizes = [int(size.item()) for size in sizes_list]
        max_size = max(sizes)

        # Pad tensors to the maximum size
        pad_size = max_size - all_preds.size(0)
        if pad_size > 0:
            all_preds = torch.cat([
                all_preds,
                torch.zeros(pad_size, dtype=all_preds.dtype, device=device)
            ])
            all_labels = torch.cat([
                all_labels,
                torch.zeros(pad_size, dtype=all_labels.dtype, device=device)
            ])
        
        # Prepare lists for gathering tensors
        gathered_preds = [torch.zeros(max_size, dtype=all_preds.dtype, device=device) for _ in range(self.world_size)]
        gathered_labels = [torch.zeros(max_size, dtype=all_labels.dtype, device=device) for _ in range(self.world_size)]
        
        # Gather predictions and labels from all ranks
        torch.distributed.all_gather(gathered_preds, all_preds)
        torch.distributed.all_gather(gathered_labels, all_labels)
        
        # Remove padding and concatenate
        all_preds = torch.cat([preds[:sizes[i]] for i, preds in enumerate(gathered_preds)])
        all_labels = torch.cat([labels[:sizes[i]] for i, labels in enumerate(gathered_labels)])
        
        # Reduce loss and sample count across all processes
        torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_samples, op=torch.distributed.ReduceOp.SUM)
        
        # Compute average validation loss
        avg_val_loss = val_loss.item() / total_samples.item()
        
        # Move tensors to CPU for metric computation
        all_preds = all_preds.cpu().numpy()
        all_labels = all_labels.cpu().numpy()
        
        # Compute validation metrics
        val_score = utils.safe_macro_f1(all_labels, all_preds)
        val_accuracy = self.compute_accuracy(all_labels, all_preds)
        
        # Debug statements
        if debug and self.rank == 0:
            print(f"Validation at epoch {epoch}: Loss = {avg_val_loss:.6f}, F1 = {val_score:.6f}, Accuracy = {val_accuracy:.6f}")
        
        self.model.train()
        return avg_val_loss, val_score, val_accuracy


    def fit(self, X, X_val, y, y_val, rank, world_size, debug=False, start_epoch=1, model_state_dict=None, optimizer_state_dict=None,
            num_workers=0, prefetch=None, empty_cache=False, decimal=6, input_queue=None, mem_interval=10,
            save_final_model=False, save_pickle=False, save_hf=False, save_dir='saves', weights_name='bert-base-uncased'):

        training_start = time.time()
        if rank == 0:
            print(f"\n{sky_blue}Fitting DDP Neural Classifier on training data...{reset}")
            print(f"Num Workers: {num_workers}, Prefetch: {prefetch}")

        # Print tolerance as float with as many decimal places as in the tolerance
        tol_str = format_tolerance(self.tol)        

        # Determine the early stopping strategy and split the data if necessary
        if self.early_stopping == 'score':
            if rank == 0:
                print(f"Score-based early stopping enabled (macro average F1 score against validation set). Validation fraction: {self.validation_fraction}")
                print(f"Training will stop early if the score does not improve by at least {tol_str} for {self.n_iter_no_change} iterations.")
                if X_val is None and y_val is None:
                    print(f"Splitting data into training and validation sets...")
                    (X_train, y_train), (X_val, y_val) = self._build_validation_split(X, y, validation_fraction=self.validation_fraction, random_seed=self.random_seed)
                    print(f"Split data into (X:{len(X_train)}, y:{len(y_train)}) Training samples, and (X:{len(X_val)}, y:{len(y_val)}) Validation samples.")
                else:
                    print(f"Using provided validation set for early stopping.")
                    X_train, y_train = X, y
                data_list = [X_train, y_train, X_val, y_val]
            else:
                data_list = [None, None, None, None]
            # Broadcast the data
            if world_size > 1:
                dist.broadcast_object_list(data_list, src=0)
                # Unpack the broadcasted data
                X_train, y_train, X_val, y_val = data_list
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
                print(f"{bright_yellow}Warning: target_score is set to {self.target_score}, but early_stopping is '{self.early_stopping}'. target_score will be ignored.{reset}") if rank == 0 else None
                self.target_score = None
            else:
                print(f"Target score set to {self.target_score}. Training will stop if this validation score is reached.") if rank == 0 else None

        # Build the dataset and dataloader
        print(f"Building dataset and dataloader...") if rank == 0 else None

        if self.finetune_bert:
            if self.bert_tokenizer is None:
                raise ValueError(f"{red}bert_tokenizer is required for fine-tuning BERT{reset}")
            dataset = SentimentDataset(X_train, y_train, self.bert_tokenizer, self.label_dict, device=self.device)
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
        else:
            dataset = self.build_dataset(X_train, y_train)

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


        # Initialize the model, optimizer, and set it in training mode
        print("Initializing model, graph, optimizer...") if rank == 0 else None
        
        self.initialize()
        dist.barrier()

        # if self.wandb_run is not None:
        #     print(f"Watching model with Weights & Biases...") if rank == 0 else None
        #     wandb.watch(self.model, log='all', log_freq=1)

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
            print(f"\n{sky_blue}Starting training loop...{reset}")

        self.model.train()

        num_digits = len(str(self.max_iter))  # Number of digits for formatting epoch output
        stop_training = torch.tensor(0, device=self.device)  # Signal to stop training in DDP setup
        skip_epochs = 0  # Counter to skip epochs without prompting in interactive mode
        target_hit = False  # Flag to indicate if the target score was reached
        val_loss = None  # Validation loss
        
        # Set the last epoch based on the start epoch and max iterations
        if start_epoch > 1:
            last_epoch = start_epoch + self.max_iter
        else:
            last_epoch = self.max_iter

        print(f"Start epoch: {start_epoch}, Max Iterations: {self.max_iter}, Early Stop: {self.early_stopping}, Tolerance: {tol_str}, Number Iterations No Change: {self.n_iter_no_change}") if rank == 0 else None
        
        # Training loop
        for epoch in range(start_epoch, last_epoch+1):
            epoch_start = time.time()
            if self.device.type == 'cuda' and world_size > 1:
                sampler.set_epoch(epoch)  # Required for DistributedSampler to shuffle the data
            epoch_loss = 0.0
            batch_count = 0

            # Interactive mode
            if self.interactive and skip_epochs == 0:
                # Create a global tensor for synchronization
                global_tensor = torch.zeros(2, dtype=torch.long, device=self.device)

                # Function to indicate invalid input
                def invalid_input(command_tensor, message="Invalid input. Please try again."):
                    print(f"{red}{message}{reset}")
                    command_tensor[0] = -1 
                    command_tensor[1] = -1

                while True:  # Start a loop to handle the input

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
                            command_value[1] = self.advance_epochs  # Use the advance_epochs value
                        elif user_input_split[0].isdigit():
                            # Check if the input is just a number
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
                                        invalid_input(command_value, "Invalid number for checkpoint interval. Please enter an integer.")
                            elif user_input_split[0] in ['s', 'save']:
                                command_value[0] = 1  # Save
                            elif user_input_split[0] in ['q', 'quit']:
                                command_value[0] = 2  # Quit
                            elif user_input_split[0] in ['h', 'help']:
                                command_value[0] = 3  # Help
                            elif user_input_split[0] in ['e', 'epoch', 'epochs']:
                                if len(user_input_split) > 1:
                                    try:
                                        command_value[0] = 4  # Set max epochs
                                        command_value[1] = float(user_input_split[1])  # Associated value for max epochs
                                    except ValueError:
                                        invalid_input(command_value, "Invalid value for max epochs.")
                                else:
                                    invalid_input(command_value, "A number is required for max epochs.")
                            elif user_input_split[0] in ['t', 'tol', 'tolerance']:
                                if len(user_input_split) > 1:
                                    try:
                                        command_value[0] = 5  # Set tolerance
                                        command_value[1] = float(user_input_split[1])  # Associated value for tolerance
                                    except ValueError:
                                        invalid_input(command_value, "Invalid value for tolerance.")
                                else:
                                    invalid_input(command_value, "A number is required for tolerance.")
                            elif user_input_split[0] in ['n', 'num', 'number']:
                                if len(user_input_split) > 1:
                                    try:
                                        command_value[0] = 6  # Set n_iter_no_change
                                        command_value[1] = float(user_input_split[1])  # Associated value for n_iter_no_change
                                    except ValueError:
                                        invalid_input(command_value, "Invalid value for n_iter_no_change.")
                                else:
                                    invalid_input(command_value, "A number is required for n_iter_no_change.")
                            elif user_input_split[0] in ['x', 'exit']:
                                command_value[0] = 7  # Exit interactive mode
                            elif user_input_split[0] in ['d', 'debug']:
                                command_value[0] = 8  # Toggle debug mode
                            elif user_input_split[0] in ['v', 'val', 'validation', 'target']:
                                if len(user_input_split) > 1:
                                    try:
                                        command_value[0] = 9  # Set target validation score
                                        command_value[1] = float(user_input_split[1])  # Associated value for target validation score
                                    except ValueError:
                                        invalid_input(command_value, "Invalid value for target validation score.")
                                else:
                                    invalid_input(command_value, "A number is required for the target validation score.")
                            elif user_input_split[0] in ['m', 'mem', 'memory']:
                                if len(user_input_split) > 1:
                                    try:
                                        command_value[0] = 11  # Set memory interval
                                        command_value[1] = float(user_input_split[1])  # Associated value for memory interval
                                    except ValueError:
                                        invalid_input(command_value, "Invalid value for memory interval.")
                                else:
                                    invalid_input(command_value, "A number is required for the memory interval.")
                            elif user_input_split[0] in ['g', 'gpu', 'gpus']:
                                command_value[0] = 12  # Show GPU memory usage
                            elif user_input_split[0] in ['l', 'lr', 'learning', 'rate']:
                                if len(user_input_split) > 1:
                                    try:
                                        command_value[0] = 13  # Set learning rate
                                        command_value[1] = float(user_input_split[1])  # Associated value for learning rate
                                    except ValueError:
                                        invalid_input(command_value, "Invalid value for learning rate.")
                                else:
                                    invalid_input(command_value, "A number is required for learning rate.")
                            elif user_input_split[0] in ['p', 'prog', 'progress']:
                                command_value[0] = 14  # Toggle progress bar display
                            elif user_input_split[0] in ['a', 'advance', 'adv', 'advance_epochs']:
                                if len(user_input_split) > 1:
                                    try:
                                        command_value[0] = 15  # Set advance epochs
                                        command_value[1] = int(user_input_split[1])  # Associated value for advance epochs
                                    except ValueError:
                                        invalid_input(command_value, "Invalid value for advance epochs.")
                                else:
                                    invalid_input(command_value, "A number is required for advance epochs.")
                            else:
                                invalid_input(command_value, "Invalid command. Type 'help' for a list of commands.")

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
                            self.save_model(directory=self.checkpoint_dir, epoch=epoch, optimizer=self.optimizer, is_final=False, save_pickle=save_pickle, save_hf=save_hf, weights_name=weights_name)
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
                            print(f"[{bold}{bright_yellow}A{reset}]dvance    ____  Set the number of epochs to skip. Current: {self.advance_epochs}. Example: 'a 5'")
                            print(f"[{bold}{bright_yellow}C{reset}]heckpoint ____  Set the checkpoint interval. Current: {self.checkpoint_interval}. Example: 'c 5'")
                            print(f"[{bold}{bright_yellow}D{reset}]ebug            Toggle debug mode. Current: {debug}")
                            print(f"[{bold}{bright_yellow}E{reset}]pochs     ____  Set the maximum number of epochs. Current: {self.max_iter}. Example: 'e 1000'")
                            print(f"[{bold}{bright_yellow}G{reset}]PUs       ____  Show current GPU memory usage.")
                            print(f"[{bold}{bright_yellow}H{reset}]elp             Display this help message")
                            print(f"[{bold}{bright_yellow}L{reset}]earning   ____  Set the learning rate. Current: {self.eta:.{decimal}f}. Example: 'l 0.001'")
                            print(f"[{bold}{bright_yellow}M{reset}]emory     ____  Set the interval to display GPU memory usage. Example: 'm 5'")
                            print(f"[{bold}{bright_yellow}N{reset}]umber     ____  Set the number of iterations no change. Current: {self.n_iter_no_change}. Example: 'n 10'")
                            print(f"[{bold}{bright_yellow}P{reset}]rogress         Toggle progress bar display. Current: {self.show_progress}")
                            print(f"[{bold}{bright_yellow}Q{reset}]uit             Quit the training loop")
                            print(f"[{bold}{bright_yellow}S{reset}]ave             Save the model as a checkpoint with DDP wrapper and optionally as a pickle file.")
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
                            invalid_input(command_value, "Invalid value for checkpoint interval. Please enter a positive integer.") if rank == 0 else None
                        continue  # Stay in the loop to wait for another command
                    elif command == 11:  # Set memory interval
                        if value > 0:
                            mem_interval = int(value)
                            print(f"Updated memory interval to: {mem_interval}") if rank == 0 else None
                        else:
                            invalid_input(command_value, "Invalid value for memory interval. Please enter a positive integer.") if rank == 0 else None
                        continue  # Stay in the loop to wait for another command
                    elif command == 12:  # Show GPU memory usage
                        print_rank_memory_summary(world_size, rank, all_local=True, verbose=False) if rank == 0 else None
                        continue  # Stay in the loop to wait for another command
                    elif command == 13:  # Set learning rate
                        self.eta = value
                        for param_group in self.optimizer.param_groups:
                            if 'lr' in param_group:
                                param_group['lr'] = self.eta
                        if rank == 0:
                            print(f"Updated learning rate to: {self.eta:.{decimal}f}")
                        continue  # Stay in the loop to wait for another command
                    elif command == 14:  # Toggle progress bar display
                        self.show_progress = not self.show_progress
                        if rank == 0:
                            print(f"Progress bar display set to: {self.show_progress}")
                        continue
                    elif command == 15:  # Set advance epochs
                        self.set_advance_epochs(value)
                        skip_epochs = int(value) if value > 1 else 1  # Set the skip_epochs based on the provided value
                        if rank == 0:
                            print(f"Advance epochs set to: {value}")
                        continue
                    elif command == -1:  # Invalid input
                        continue  # Stay in the loop to wait for another command

                    # Reset the global tensor for the next iteration
                    if rank == 0:
                        global_tensor[0] = 0
                    dist.broadcast(global_tensor, src=0)

                    # Ensure all ranks have processed the input before continuing
                    dist.barrier()

            if stop_training.item() == 1:
                # Consolidate optimizer state before saving
                if self.use_zero:
                    self.optimizer.consolidate_state_dict()
                break  # Exit the training loop if stop_training was triggered

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
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = self.loss(outputs, labels)
                else:
                    X_batch, y_batch = batch
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = self.loss(outputs, y_batch)
                                
                # Average loss across all ranks
                if world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)

                loss.backward()
                
                # Calculate gradient norms
                #parameters = [p for p in self.model.module.parameters() if p.grad is not None]
                parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
                grad_norms = [p.grad.data.norm(2).item() for p in parameters]
                min_norm = min(grad_norms)
                max_norm = max(grad_norms)
                mean_norm = sum(grad_norms) / len(grad_norms)
                #total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
                total_norm = torch.norm(torch.stack([p.grad.data.norm(2) for p in parameters]), 2)

                clipped = False
                if batch_count % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.max_grad_norm is not None:
                        # Apply gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.max_grad_norm)
                        
                        clipped = total_norm > self.max_grad_norm
                        
                        if clipped:
                            # Recalculate norms after clipping
                            grad_norms_after = [p.grad.data.norm(2).item() for p in parameters]
                            min_norm_after = min(grad_norms_after)
                            max_norm_after = max(grad_norms_after)
                            mean_norm_after = sum(grad_norms_after) / len(grad_norms_after)
                            total_norm_after = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Step schedulers that update per iteration
                    if isinstance(self.scheduler, (torch.optim.lr_scheduler.CyclicLR, torch.optim.lr_scheduler.OneCycleLR)):
                        self.scheduler.step()
                    elif isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                        self.scheduler.step(epoch - 1 + batch_count / len(dataloader))
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                batch_count += 1
                current_lr = self.optimizer.param_groups[0]['lr']

                if clipped:
                    grad_info = f"{bright_yellow}Clip{reset} ↓ {bright_white}{bold}{min_norm_after:.{decimal}f}{reset} ↑ {bright_white}{bold}{max_norm_after:.{decimal}f}{reset} M {bright_white}{bold}{mean_norm_after:.{decimal}f}{reset}"
                else:
                    grad_info = f"Grad ↓ {bright_white}{bold}{min_norm:.{decimal}f}{reset} ↑ {bright_white}{bold}{max_norm:.{decimal}f}{reset} M {bright_white}{bold}{mean_norm:.{decimal}f}{reset}"

                if debug:     
                    if self.finetune_bert:
                        print(f"Rank {rank}: Epoch: {epoch}, Batch: {batch_count}, Input IDs Size: {input_ids.size(0)}, Attention Mask Size: {attention_mask.size(0)}, Labels Size: {labels.size(0)}, Loss: {loss.item():.6f}, Epoch Loss: {epoch_loss:.6f}, LR: {current_lr:.2e}, {grad_info}")
                    else:
                        print(f"Rank {rank}: Epoch: {epoch}, Batch: {batch_count}, Size: {X_batch.size(0)}, Loss: {loss.item():.6f}, Epoch Loss: {epoch_loss:.6f}, LR: {current_lr:.2e}, {grad_info}")
                    
                # Update progress bar
                if rank == 0 and self.show_progress:
                    if clipped:
                        grad_info_mini = f"{bright_yellow}C:{reset}{mean_norm_after:.4f}"
                    else:
                        grad_info_mini = f"M:{mean_norm:.4f}"
                    
                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}', 'grad': grad_info_mini})

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / batch_count

            # Update progress bar with post-epoch processes
            if rank == 0 and self.show_progress:
                pbar.set_postfix({'loss': f'{avg_epoch_loss:.4f}', 'lr': f'{current_lr:.2e}', 'grad': grad_info_mini, 'status': 'Post-epoch processing...'})

            # Consolidate optimizer state before saving
            if self.use_zero:
                if rank == 0 and self.show_progress:
                    pbar.set_postfix({'loss': f'{avg_epoch_loss:.4f}', 'lr': f'{current_lr:.2e}', 'grad': grad_info_mini, 'status': 'Consolidating optimizer state...'})
                self.optimizer.consolidate_state_dict()

            dist.barrier()
            # Early stopping logic
            if self.early_stopping == 'score':

                # Compute train score and accuracy
                if rank == 0 and self.show_progress:
                    pbar.set_postfix({'loss': f'{avg_epoch_loss:.4f}', 'lr': f'{current_lr:.2e}', 'grad': grad_info_mini, 'status': 'Computing train metrics...'})
                train_score, train_accuracy = self.compute_train_metrics(X_train, y_train, epoch, debug)
                
                # Compute validation loss, score, and accuracy
                if rank == 0 and self.show_progress:
                    pbar.set_postfix({'loss': f'{avg_epoch_loss:.4f}', 'lr': f'{current_lr:.2e}', 'grad': grad_info_mini, 'status': 'Computing validation metrics...'})
                val_loss, val_score, val_accuracy = self.compute_validation_metrics(X_val, y_val, epoch, debug)

                if rank == 0:
                    current_score = val_score
                    if self.best_score is None or current_score > self.best_score + self.tol:
                        self.best_score = current_score
                        self.no_improvement_count = 0
                    else:
                        self.no_improvement_count += 1
                    
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
                    pbar.set_postfix({'loss': f'{avg_epoch_loss:.4f}', 'lr': f'{current_lr:.2e}', 'grad': grad_info_mini, 'status': 'Checking early stopping'})
                current_loss = self._update_no_improvement_count_errors(avg_epoch_loss, epoch, debug)
                if self.no_improvement_count >= self.n_iter_no_change:
                    stop_training = torch.tensor(1, device=self.device)
                current_loss_color, best_error_color = get_loss_colors(current_loss, self.best_error, self.no_improvement_count, self.n_iter_no_change)

            # Step the scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler_class, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau needs a metric to monitor
                    self.scheduler.step(avg_epoch_loss)
                else:
                    self.scheduler.step()

            # Log to wandb
            if self.wandb_run and rank == 0:
                # Prepare logging dictionary
                log_dict = {
                    "other/epoch": epoch,
                    "train/loss": avg_epoch_loss,
                    "train/macro_f1_score": train_score,
                    "train/accuracy": train_accuracy,
                    "other/learning_rate": current_lr,
                    "gradients/min_norm": min_norm,
                    "gradients/max_norm": max_norm,
                    "gradients/mean_norm": mean_norm,
                    "train/stop_count": self.no_improvement_count,
                    "other/stop_limit": self.n_iter_no_change
                }
                if clipped:
                    log_dict["gradients/clipped"] = True
                    log_dict["gradients/min_norm_after_clip"] = min_norm_after
                    log_dict["gradients/max_norm_after_clip"] = max_norm_after                    
                if self.early_stopping == 'score':
                    log_dict["validation/loss"] = val_loss
                    log_dict["validation/macro_f1_score"] = current_score
                    log_dict["validation/best_macro_f1_score"] = self.best_score
                    log_dict["validation/accuracy"] = val_accuracy
                
                # Log to wandb
                if self.wandb_run:
                    self.wandb_run.log(log_dict)

            # Get the no improvement count color based on value relative to n_iter_no_change
            nic_color = get_nic_color(self.no_improvement_count, self.n_iter_no_change)

            # Close progress bar
            if rank == 0 and self.show_progress:
                pbar.set_postfix({'loss': f'{avg_epoch_loss:.4f}', 'lr': f'{current_lr:.2e}', 'grad': grad_info_mini, 'status': 'Epoch complete'})
                pbar.close()

            # Print epoch summary
            if self.early_stopping == 'score':
                print(f"{nic_color}█{reset} Epoch {bright_white}{bold}{epoch:{num_digits}d}{reset}: "
                      f"Loss T {bright_white}{bold}{avg_epoch_loss:.{decimal}f}{reset} V {bright_white}{bold}{val_loss:.{decimal}f}{reset} | "
                      f"F1 T {bright_white}{bold}{train_score:.{decimal}f}{reset} V {current_score_color}{bold}{val_score:.{decimal}f}{reset} B {best_score_color}{bold}{self.best_score:.{decimal}f}{reset} SC {nic_color}{bold}{self.no_improvement_count}{reset}/{self.n_iter_no_change} | "
                      f"Acc T {bright_white}{bold}{train_accuracy:.{decimal}f}{reset} V {bright_white}{bold}{val_accuracy:.{decimal}f}{reset} | "
                      f"LR {bright_white}{bold}{current_lr:.2e}{reset} | {grad_info} | {format_time(time.time() - epoch_start)}")  if rank == 0 else None
            elif self.early_stopping == 'loss':
                print(f"{nic_color}█{reset} Epoch {bright_white}{bold}{epoch:{num_digits}d}{reset}: Avg Loss: {current_loss_color}{bold}{avg_epoch_loss:.{decimal}f}{reset} | Best Loss: {best_error_color}{bold}{self.best_error:.{decimal}f}{reset}, Stop Count: {nic_color}{bold}{self.no_improvement_count}{reset} / {self.n_iter_no_change} | LR: {current_lr:.2e}, {grad_info} | Time: {format_time(time.time() - epoch_start)}")  if rank == 0 else None
            else:
                print(f"Epoch {bright_white}{bold}{epoch:{num_digits}d}{reset} / {self.max_iter}: Avg Loss: {bright_white}{bold}{avg_epoch_loss:.{decimal}f}{reset} | Total Loss: {epoch_loss:10.{decimal}f} | LR: {current_lr:.2e}, {grad_info} | Batch Count: {batch_count} | Time: {format_time(time.time() - epoch_start)}")  if rank == 0 else None

            # Print memory summary
            if epoch % mem_interval == 0:
                print_rank_memory_summary(world_size, rank, all_local=True, verbose=False) if rank == 0 else None

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
                    self.save_model(directory=self.checkpoint_dir, epoch=epoch, optimizer=self.optimizer, is_final=False, save_pickle=False, save_hf=save_hf, weights_name=weights_name)
            
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
        
        # Save the wandb model in onnx format
        if self.wandb_run and save_final_model and rank == 0:
            print("Saving model in ONNX format...")
            
            # Suppress TracerWarnings
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            
            # Create a dummy input tensor
            if self.finetune_bert:
                dummy_input = (
                    torch.zeros(1, 512, dtype=torch.long, device=self.device),  # input_ids
                    torch.zeros(1, 512, dtype=torch.long, device=self.device)   # attention_mask
                )
            else:
                dummy_input = torch.zeros(1, self.input_dim, device=self.device)
            
            # Export the model to ONNX format
            if not os.path.exists(save_dir):
                print(f"Creating save directory: {save_dir}")
                os.makedirs(save_dir)
            # Create a filename timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            onnx_file_path = os.path.join(save_dir, f'model_{timestamp}.onnx')
            torch.onnx.export(self.model.module, dummy_input, onnx_file_path, opset_version=13)
            
            # Save the ONNX model to wandb
            wandb.save(onnx_file_path)
            print(f"Model saved in ONNX format to {onnx_file_path} and uploaded to Weights & Biases.")

        # Save the final model
        if rank == 0 and save_final_model:
            self.save_model(directory=self.checkpoint_dir, epoch=epoch, optimizer=self.optimizer, is_final=True, save_pickle=save_pickle, save_hf=save_hf, weights_name=weights_name)
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
        return self
