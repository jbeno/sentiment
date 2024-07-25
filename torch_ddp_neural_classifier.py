import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_model_base import TorchModelBase
import time
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier

class TorchDDPNeuralClassifier(TorchShallowNeuralClassifier):
    def __init__(self,
                 hidden_dim=50,
                 hidden_activation=nn.Tanh(),
                 **base_kwargs):
        """
        A model

        h = f(xW_xh + b_h)
        y = softmax(hW_hy + b_y)

        with a cross-entropy loss and f determined by `hidden_activation`.

        Parameters
        ----------
        hidden_dim : int
            Dimensionality of the hidden layer.

        hidden_activation : nn.Module
            The non-activation function used by the network for the
            hidden layer.

        **base_kwargs
            For details, see `torch_model_base.py`.

        Attributes
        ----------
        loss: nn.CrossEntropyLoss(reduction="mean")

        self.params: list
            Extends TorchModelBase.params with names for all of the
            arguments for this class to support tuning of these values
            using `sklearn.model_selection` tools.

        """
        super().__init__(hidden_dim=hidden_dim, hidden_activation=hidden_activation, **base_kwargs)
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.params += ['hidden_dim', 'hidden_activation']

    def fit(self, X, y, rank, world_size, device, debug=False):
        dataset = self.build_dataset(X, y)
        if device.type == 'cuda':
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model, optimizer, and other attributes
        self.initialize()

        # Initialize the model here, after input_dim is set
        print(f"Rank {rank} - Building model...")
        print(f"Device: {device}")
        self.model = self.build_graph()
        self.model.to(device)
        if device.type == 'cuda':
            self.model = DDP(self.model, device_ids=[rank])
            print(f"device_type: {device.type}, rank: {rank}, device_ids: {rank}")
        else:
            self.model = DDP(self.model, device_ids=None)
            print(f"device_type: {device.type}, rank: {rank}, device_ids: None")
        optimizer = self.build_optimizer()

        # Set the batch size for each GPU
        if debug:
            print(f"Rank {rank} - Starting training loop...")
        elif rank == 0:
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
            if debug:
                print(f"Rank {rank} - Epoch {epoch}, Loss: {loss.item():.6f}, Time: {time.time() - epoch_start:.2f} seconds")
            elif rank == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Time: {time.time() - epoch_start:.2f} seconds")

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

        return self
    
    def to(self, device):
        """Move the model to the specified device."""
        if self.model:
            self.model = self.model.to(device)
        return self