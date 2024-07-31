import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_model_base import TorchModelBase
import os
import time
import utils
from utils import format_time, print_state_summary

class TorchDDPNeuralClassifier(TorchModelBase):
    def __init__(self,
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
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.num_layers = num_layers
        self.rank = rank
        self.debug = debug
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.resume_from_checkpoint = resume_from_checkpoint
        
        self.loss = nn.CrossEntropyLoss(reduction="mean")

        # Update self.params to include the new parameters
        self.params += ['hidden_dim', 'hidden_activation', 'num_layers', 'rank', 'debug', 
                        'checkpoint_dir', 'checkpoint_interval', 'resume_from_checkpoint']

    def build_graph(self):
        layers = []
        
        # Input to first hidden layer
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(self.hidden_activation)
        
        # Additional hidden layers
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(self.hidden_activation)
        
        # Output layer
        layers.append(nn.Linear(self.hidden_dim, self.n_classes_))
        
        return nn.Sequential(*layers)

    def build_dataset(self, X, y=None):
        X = np.array(X)
        self.input_dim = X.shape[1]
        X = torch.FloatTensor(X)
        if y is None:
            dataset = torch.utils.data.TensorDataset(X)
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
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
            if debug and rank == 0:
                print(f"BEFORE updating model parameters:")
                print_state_summary(self.__dict__)
            self.__dict__.update(checkpoint['params'])
            if debug and rank == 0:
                print(f"AFTER updating model parameters:")
                print_state_summary(self.__dict__)

        start_epoch = checkpoint['epoch']+1 if 'epoch' in checkpoint else 1

        return start_epoch, model_state_dict, optimizer_state_dict

    def fit(self, X, y, rank, world_size, debug=False, start_epoch=1, model_state_dict=None, optimizer_state_dict=None):
        training_start = time.time()
        if rank == 0:
            print(f"\nFitting DDP Neural Classifier on training data...")
        # Determine the early stopping strategy and split the data if necessary
        if self.early_stopping == 'score':
            if rank == 0:
                print(f"Score-based early stopping enabled (macro average F1 score against validation set). Validation fraction: {self.validation_fraction}")
                print(f"Training will stop early if the score does not improve by at least {self.tol} for {self.n_iter_no_change} iterations.")
            (X_train, y_train), (X_val, y_val) = self._build_validation_split(X, y, validation_fraction=self.validation_fraction)
            if rank == 0:
                print(f"Split data into {len(X_train)} training samples and {len(X_val)} validation samples.")
                print(f"Split data into {len(y_train)} training labels and {len(y_val)} validation labels.")
        elif self.early_stopping == 'loss':
            if rank == 0:
                print(f"Loss-based early stopping enabled. No validation set required, all data used for training.")
                print(f"Training will stop early if the loss does not improve by at least {self.tol} for {self.n_iter_no_change} iterations.")
            X_train, y_train = X, y
        else:
            if rank == 0:
                print(f"Training without early stopping. No validation set required, all data used for training.")
                print(f"Training will stop after {self.max_iter} iterations.")
            X_train, y_train = X, y

        # Build the dataset and dataloader
        if rank == 0:
            print(f"Building dataset and dataloader...")
        dataset = self.build_dataset(X_train, y_train)
        if self.device.type == 'cuda':
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize the model, optimizer, and other training parameters
        print("Initializing model, graph, optimizer...") if rank == 0 else None
        self.initialize()
        self.model = self.build_graph()
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        if self.device.type == 'cuda':
            self.model = DDP(self.model, device_ids=[rank])
        else:
            self.model = DDP(self.model, device_ids=None)
        optimizer = self.build_optimizer()
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)

        if rank == 0:
            print(f"Model architecture:\n{self.model}") if debug else None
            print(f"Optimizer:\n{optimizer}") if debug else None
            print(f"Starting training loop...")

        self.model.train()

        stop_training = torch.tensor(0, device=self.device)
        if start_epoch > 1:
            last_epoch = start_epoch + self.max_iter
        else:
            last_epoch = self.max_iter
        print(f"Start epoch: {start_epoch}, Max Iterations: {self.max_iter}") if rank == 0 else None
        for epoch in range(start_epoch, last_epoch+1):
            epoch_start = time.time()
            if self.device.type == 'cuda':
                sampler.set_epoch(epoch)
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.loss(outputs, y_batch)
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / world_size
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Time: {format_time(time.time() - epoch_start)}") if rank == 0 else None

            if rank == 0:
                if self.checkpoint_interval and (epoch) % self.checkpoint_interval == 0:
                    print(f"Saving checkpoint at epoch {epoch}...")
                    self.save_model(directory=self.checkpoint_dir, epoch=epoch, optimizer=optimizer, is_final=False)

                # Early stopping logic
                if self.early_stopping == 'score':
                    current_score = self._update_no_improvement_count_early_stopping(X_val, y_val, epoch, debug)
                    if self.no_improvement_count >= self.n_iter_no_change:
                        print(f"Stopping early after {epoch} epochs due to no improvement in validation score in last {self.n_iter_no_change} iterations.")
                        print(f"Best score: {self.best_score:.6f}, Current score: {current_score:.6f}, No improvement count: {self.no_improvement_count}")
                        stop_training = torch.tensor(1, device=self.device)
                elif self.early_stopping == 'loss':
                    current_loss = self._update_no_improvement_count_errors(epoch_loss, epoch, debug)
                    if self.no_improvement_count >= self.n_iter_no_change:
                        print(f"Stopping early after {epoch} epochs due to no improvement in training loss in last {self.n_iter_no_change} iterations.")
                        print(f"Best loss: {self.best_error:.6f}, Current loss: {current_loss:.6f}, No improvement count: {self.no_improvement_count}")
                        stop_training = torch.tensor(1, device=self.device)

            # Broadcast stop_training to all processes
            dist.broadcast(stop_training, 0)

            if stop_training.item() == 1:
                break

        if self.early_stopping:
            # Broadcast best parameters to all processes
            for param in self.model.parameters():
                dist.broadcast(param.data, 0)

        if rank == 0:
            # Save the final model
            self.save_model(directory=self.checkpoint_dir, epoch=epoch, optimizer=optimizer, is_final=True)
            print(f"Training completed ({format_time(time.time() - training_start)})")

        return self

    def score(self, X, y, device=None, debug=False):
        if debug:
            print(f"Scoring on {len(X)} samples...")
            print(f"X shape: {X.shape}")
            print(f"X type: {X.dtype}")
            print(f"X device: {X.device}")
            print(f"X[:5]: {X[:5]}")
            print(f"y[:5]: {y[:5]}")
            print(f"y type: {y.dtype}")
        preds = self.predict(X, device=device, debug=debug)
        return utils.safe_macro_f1(y, preds)

    def predict_proba(self, X, device=None, debug=False):
        preds = self._predict(X, device=device, debug=debug)
        probs = torch.softmax(preds, dim=1).cpu().numpy()
        return probs

    def predict(self, X, device=None, debug=False):
        probs = self.predict_proba(X, device=device, debug=debug)
        return [self.classes_[i] for i in probs.argmax(axis=1)]

    def _predict(self, X, device=None, debug=False):
        if device is None:
            device = self.device
        self.model.to(device)
        self.model.eval()
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
