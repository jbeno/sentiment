import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_model_base import TorchModelBase
import time
import utils
from utils import format_time

class TorchDDPNeuralClassifier(TorchModelBase):
    def __init__(self,
                 hidden_dim=300,
                 hidden_activation=nn.Tanh(),
                 num_layers=1,
                 n_iter_no_change=10,
                 early_stopping=True,
                 tol=1e-5,
                 rank=None,
                 debug=False,
                 **base_kwargs):
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
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.num_layers = num_layers
        self.n_iter_no_change = n_iter_no_change
        self.early_stopping = early_stopping
        self.tol = tol
        self.rank = rank
        self.debug = debug
        super().__init__(**base_kwargs)
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.params += ['hidden_dim', 'hidden_activation', 'num_layers', 
                        'n_iter_no_change', 'early_stopping', 'tol']

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

    def fit(self, X, y, rank, world_size, device, debug=False):
        training_start = time.time()
        if rank == 0:
            print(f"\nFitting DDP Neural Classifier on training data...")
            print(f"Building dataset and dataloader...")
        dataset = self.build_dataset(X, y)
        if device.type == 'cuda':
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if rank == 0:
            print("Initializing model, graph, optimizer...")
        self.initialize()
        self.model = self.build_graph()
        self.model.to(device)
        if device.type == 'cuda':
            self.model = DDP(self.model, device_ids=[rank])
        else:
            self.model = DDP(self.model, device_ids=None)
        optimizer = self.build_optimizer()

        if rank == 0:
            print(f"Model architecture:\n{self.model}") if debug else None
            print(f"Starting training loop...")

        self.model.train()

        stop_training = torch.tensor(0, device=device)
        for epoch in range(self.max_iter):
            epoch_start = time.time()
            if device.type == 'cuda':
                sampler.set_epoch(epoch)
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.loss(outputs, y_batch)
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / world_size
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}, Time: {format_time(time.time() - epoch_start)}") if rank == 0 else None

            if rank == 0:
                # Early stopping logic
                if self.early_stopping:
                    self._update_no_improvement_count_early_stopping(X, y)
                    if self.no_improvement_count > self.n_iter_no_change:
                        if self.display_progress:
                            print(f"Stopping early after {epoch+1} epochs due to no improvement.")
                        stop_training = torch.tensor(1, device=device)
                else:
                    self._update_no_improvement_count_errors(epoch_loss)
                    if self.no_improvement_count > self.n_iter_no_change:
                        if self.display_progress:
                            print(f"Stopping early after {epoch+1} epochs due to no improvement.")
                        stop_training = torch.tensor(1, device=device)

            # Broadcast stop_training to all processes
            dist.broadcast(stop_training, 0)

            if stop_training.item() == 1:
                break

        if self.early_stopping:
            # Broadcast best parameters to all processes
            for param in self.model.parameters():
                dist.broadcast(param.data, 0)

        if rank == 0:
            print(f"Training completed ({format_time(time.time() - training_start)})")

        return self

    def score(self, X, y, device=None):
        preds = self.predict(X, device=device)
        return utils.safe_macro_f1(y, preds)

    def predict_proba(self, X, device=None):
        preds = self._predict(X, device=device)
        probs = torch.softmax(preds, dim=1).cpu().numpy()
        return probs

    def predict(self, X, device=None):
        probs = self.predict_proba(X, device=device)
        return [self.classes_[i] for i in probs.argmax(axis=1)]

    def _predict(self, X, device=None):
        if device is None:
            device = self.device
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(device)
            preds = self.model(X)
        return preds

    def to(self, device):
        if self.model:
            self.model = self.model.to(device)
        return self
