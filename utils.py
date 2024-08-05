from collections import Counter
import csv
import logging
import numpy as np
import pandas as pd
import random
from scipy import stats
from sklearn.base import TransformerMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import sys
import os
import signal
import decimal
import torch
import torch.distributed as dist
import torch.nn as nn
from contextlib import contextmanager
import multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from colors import *

START_SYMBOL = "<s>"
END_SYMBOL = "</s>"
UNK_SYMBOL = "$UNK"


def format_time(seconds):
    if seconds >= 1:
        hrs, secs = divmod(seconds, 3600)
        mins, secs = divmod(secs, 60)
        formatted_time = []
        if hrs > 0:
            formatted_time.append(f"{int(hrs):,}h")
        if mins > 0:
            formatted_time.append(f"{int(mins):,}m")
        if secs >= 1:
            formatted_time.append(f"{int(secs):,}s")
        else:
            millisecs = int((secs - int(secs)) * 1e3)
            formatted_time.append(f"{millisecs:,}ms")
        return ' '.join(formatted_time)
    else:
        millisecs = int(seconds * 1e3)
        return f"{millisecs:,}ms"

def format_tolerance(tolerance):
    # Convert the tolerance to a float
    tolerance_float = float(tolerance)
    
    # Use decimal module for precise representation
    d = decimal.Decimal(str(tolerance_float))
    
    # Normalize the decimal to remove any extra zeros
    normalized = d.normalize()

    return str(normalized)

def setup_environment(rank, world_size, backend, device, debug):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    print(f"Rank {rank} - Device: {device}")
    dist.barrier()
    if rank == 0:
        print(f"{world_size} process groups initialized with '{backend}' backend on {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

def signal_handler(signum, frame):
    print("\nCtrl+C received. Terminating all processes...")
    cleanup_and_exit(0, True) 

def cleanup_and_exit(rank, debug):
    # Terminate all child processes
    current_process = mp.current_process()
    if current_process.name == 'MainProcess':
        # For the main process, terminate all children
        for child in mp.active_children():
            child.terminate()
    
    # Destroy the process group
    if dist.is_initialized():
        dist.destroy_process_group()
        if debug:
            print(f"Rank {rank} - Process group destroyed")

    # Clean up CUDA resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Exit the program
    sys.exit(0)

def prepare_device(rank, device_type):
    if device_type == "cuda":
        device = torch.device('cuda', rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    return device

def gather_tensors(tensor, world_size):
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list)

def convert_labels_to_tensor(labels, label_dict, device):
    if label_dict is None:
        label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
    numeric_labels = [label_dict.get(str(label), 'unknown') for label in labels]
    return torch.tensor(numeric_labels, dtype=torch.long).to(device)

def convert_numeric_to_labels(numeric_preds, numeric_dict):
    if numeric_dict is None:
        numeric_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return [numeric_dict.get(int(pred), 'unknown') for pred in numeric_preds]

def convert_sst_label(s):
    return s.split(" ")[-1]

def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "identity":
        return nn.Identity()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unknown activation function: {activation}")

def get_optimizer(optimizer_name, device, rank, world_size):
    # Options are 'zero', 'adam', 'sgd', 'adagrad', 'rmsprop'
    # If None and device CUDA and world_size > 1, use ZeroRedundancyOptimizer, else Adam
    if optimizer_name is None:
        if device.type == 'cuda' and world_size > 1:
            print(f"Optimizer not specified. Using ZeroRedundancyOptimizer for CUDA and World Size > 1") if rank == 0 else None
            return ZeroRedundancyOptimizer
        else:
            print(f"Optimizer not specified. Using Adam") if rank == 0 else None
            return torch.optim.Adam
    if optimizer_name == "adam":
        return torch.optim.Adam
    elif optimizer_name == "sgd":
        return torch.optim.SGD
    elif optimizer_name == "adagrad":
        return torch.optim.Adagrad
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop
    elif optimizer_name == "zero":
        return ZeroRedundancyOptimizer
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_scheduler(scheduler_name, device, rank, world_size):
    # Options are 'none', 'step', 'multi_step', 'exponential', 'cosine', 'reduce_on_plateau'
    # If None and device CUDA and world_size > 1, use None, else StepLR
    if scheduler_name is None:
        if device.type == 'cuda' and world_size > 1:
            print(f"Scheduler not specified. Using None for CUDA and World Size > 1") if rank == 0 else None
            return None
        else:
            print(f"Scheduler not specified. Using StepLR") if rank == 0 else None
            return optim.lr_scheduler.StepLR
    if scheduler_name == "none":
        return None
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR
    elif scheduler_name == "multi_step":
        return optim.lr_scheduler.MultiStepLR
    elif scheduler_name == "exponential":
        return optim.lr_scheduler.ExponentialLR
    elif scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR
    elif scheduler_name == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

def set_threads(num_threads):
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)

def print_state_summary(state_dict, indent=0):
    indent_str = ' ' * indent
    if isinstance(state_dict, dict):
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                if value.size():
                    print(f"{indent_str}- {key}: Tensor of shape {list(value.size())}")
                else:
                    print(f"{indent_str}- {key}: {value.item()}")
            elif isinstance(value, dict):
                print(f"{indent_str}- {key}: Dictionary with keys {list(value.keys())}")
                print_state_summary(value, indent + 2)
            elif isinstance(value, list):
                print(f"{indent_str}- {key}: List with {len(value)} elements")
                print_state_summary(value, indent + 2)
            else:
                print(f"{indent_str}- {key}: {value}")
    elif isinstance(state_dict, list):
        for i, value in enumerate(state_dict):
            if isinstance(value, torch.Tensor):
                if value.size():
                    print(f"{indent_str}- Element {i}: Tensor of shape {list(value.size())}")
                else:
                    print(f"{indent_str}- Element {i}: {value.item()}")
            elif isinstance(value, dict):
                print(f"{indent_str}- Element {i}: Dictionary with keys {list(value.keys())}")
                print_state_summary(value, indent + 2)
            elif isinstance(value, list):
                print(f"{indent_str}- Element {i}: List with {len(value)} elements")
                print_state_summary(value, indent + 2)
            else:
                print(f"{indent_str}- Element {i}: {value}")

def print_rank_memory_summary(world_size, rank, all_local=True, max_memory=None):
    """
    Print a summary of memory usage for the current rank or all ranks.
    
    Args:
    world_size (int): Total number of ranks in the current world.
    rank (int): Current rank.
    all_local (bool): If True, print info for all ranks from local process. If False, print only current rank.
    max_memory (float): Maximum memory threshold for color coding. If None, use the current max allocated.
    """
    if all_local:
        summary_parts = []
        for i in range(world_size):
            mem_allocated = torch.cuda.memory_allocated(i) / 1e9  # Convert to GB
            mem_max_allocated = torch.cuda.max_memory_allocated(i) / 1e9  # Convert to GB
            
            if max_memory is None:
                max_memory = mem_max_allocated
            
            color = get_mem_color(mem_allocated, max_memory)
            summary_parts.append(f"Rank {bright_white}{bold}{i}{reset}: {color}{bold}{mem_allocated:.2f}/{mem_max_allocated:.2f} GB{reset}")
        
        summary = " | ".join(summary_parts)
        print(f"Memory usage: {summary}")
    else:
        device = torch.cuda.current_device()
        mem_allocated = torch.cuda.memory_allocated(device) / 1e9  # Convert to GB
        mem_max_allocated = torch.cuda.max_memory_allocated(device) / 1e9  # Convert to GB
        
        if max_memory is None:
            max_memory = mem_max_allocated
        
        color = get_mem_color(mem_allocated, max_memory)
        print(f"Rank {bright_white}{bold}{i}{reset}: {color}{bold}{mem_allocated:.2f}/{mem_max_allocated:.2f} GB{reset}")

def print_rank_memory_summary(world_size, rank, all_local=True, verbose=False):
    """
    Print a summary of memory usage for the current rank or all ranks.
    
    Args:
    world_size (int): Total number of ranks in the current world.
    rank (int): Current rank.
    all_local (bool): If True, print info for all ranks from local process. If False, print only current rank.
    """
    if all_local:
        summary_parts = []
        for i in range(world_size):
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            free_mem, total_mem = free_mem / 1e9, total_mem / 1e9  # Convert to GB
            used_mem = total_mem - free_mem
            mem_allocated = torch.cuda.memory_allocated(i) / 1e9  # Convert to GB
            mem_reserved = torch.cuda.memory_reserved(i) / 1e9  # Convert to GB
            
            color = get_mem_color(used_mem, total_mem)
            if verbose:
                summary_parts.append(f"Rank {bright_white}{bold}{i}{reset}: {color}{bold}{used_mem:.2f}{reset} / {total_mem:.2f} GB (A: {mem_allocated:.2f}, R: {mem_reserved:.2f})")
            else:
                summary_parts.append(f"Rank {bright_white}{bold}{i}{reset}: {color}{bold}{used_mem:.2f}{reset} / {total_mem:.2f} GB")
        
        summary = " | ".join(summary_parts)
        print(f"{bright_white}█{reset} Memory: {summary}")
    else:
        device = torch.cuda.current_device()
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        free_mem, total_mem = free_mem / 1e9, total_mem / 1e9  # Convert to GB
        used_mem = total_mem - free_mem
        mem_allocated = torch.cuda.memory_allocated(device) / 1e9  # Convert to GB
        mem_reserved = torch.cuda.memory_reserved(device) / 1e9  # Convert to GB
        
        color = get_mem_color(used_mem, total_mem)
        print(f"Memory used/total for Rank {bright_white}{bold}{i}{reset}: {color}{bold}{used_mem:.2f} GB{reset} / {total_mem:.2f} GB (A: {mem_allocated:.2f}, R: {mem_reserved:.2f})")


    
def glove2dict(src_filename):
    """
    GloVe vectors file reader.

    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.

    Returns
    -------
    dict
        Mapping words to their GloVe vectors as `np.array`.

    """
    # This distribution has some words with spaces, so we have to
    # assume its dimensionality and parse out the lines specially:
    if '840B.300d' in src_filename:
        line_parser = lambda line: line.rsplit(" ", 300)
    else:
        line_parser = lambda line: line.strip().split()
    data = {}
    with open(src_filename, encoding='utf8') as f:
        while True:
            try:
                line = next(f)
                line = line_parser(line)
                data[line[0]] = np.array(line[1: ], dtype=np.float64)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data


def d_tanh(z):
    """
    The derivative of np.tanh. z should be a float or np.array.

    """
    return 1.0 - z**2


def softmax(z):
    """
    Softmax activation function. z should be a float or np.array.

    """
    # Increases numerical stability:
    t = np.exp(z - np.max(z))
    return t / np.sum(t)


def relu(z):
    return np.maximum(0, z)


def d_relu(z):
    return np.where(z > 0, 1, 0)


def randvec(n=50, lower=-0.5, upper=0.5):
    """
    Returns a random vector of length `n`. `w` is ignored.

    """
    return np.array([random.uniform(lower, upper) for i in range(n)])


def randmatrix(m, n, lower=-0.5, upper=0.5):
    """
    Creates an m x n matrix of random values in [lower, upper].

    """
    return np.array([random.uniform(lower, upper) for i in range(m*n)]).reshape(m, n)


def safe_macro_f1(y, y_pred, **kwargs):
    """
    Macro-averaged F1, forcing `sklearn` to report as a multiclass
    problem even when there are just two classes. `y` is the list of
    gold labels and `y_pred` is the list of predicted labels.

    """
    return f1_score(y, y_pred, average='macro', pos_label=None)


def progress_bar(msg, verbose=True):
    """
    Simple over-writing progress bar.

    """
    if verbose:
        sys.stderr.write('\r')
        sys.stderr.write(msg)
        sys.stderr.flush()


def log_of_array_ignoring_zeros(M):
    """
    Returns an array containing the logs of the nonzero
    elements of M. Zeros are left alone since log(0) isn't
    defined.

    """
    log_M = M.copy()
    mask = log_M > 0
    log_M[mask] = np.log(log_M[mask])
    return log_M


def mcnemar(y_true, pred_a, pred_b):
    """
    McNemar's test using the chi2 distribution.

    Parameters
    ----------
    y_true : list of actual labels

    pred_a, pred_b : lists
        Predictions from the two systems being evaluated.
        Assumed to have the same length as `y_true`.

    Returns
    -------
    float, float (the test statistic and p value)

    """
    c01 = 0
    c10 = 0
    for y, a, b in zip(y_true, pred_a, pred_b):
        if a == y and b != y:
            c01 += 1
        elif a != y and b == y:
            c10 += 1
    stat = ((np.abs(c10 - c01) - 1.0)**2) / (c10 + c01)
    df = 1
    pval = stats.chi2.sf(stat, df)
    return stat, pval


def fit_classifier_with_hyperparameter_search(
        X, y, basemod, cv, param_grid, scoring='f1_macro', verbose=True):
    """
    Fit a classifier with hyperparameters set via cross-validation.

    Parameters
    ----------
    X : 2d np.array
        The matrix of features, one example per row.

    y : list
        The list of labels for rows in `X`.

    basemod : an sklearn model class instance
        This is the basic model-type we'll be optimizing.

    cv : int or an sklearn Splitter
        Number of cross-validation folds, or the object used to define
        the splits. For example, where there is a predefined train/dev
        split one wants to use, one can feed in a `PredefinedSplitter`
        instance to use that split during cross-validation.

    param_grid : dict
        A dict whose keys name appropriate parameters for `basemod` and
        whose values are lists of values to try.

    scoring : value to optimize for (default: f1_macro)
        Other options include 'accuracy' and 'f1_micro'. See
        http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    verbose : bool
        Whether to print some summary information to standard output.

    Prints
    ------
    To standard output (if `verbose=True`)
        The best parameters found.
        The best macro F1 score obtained.

    Returns
    -------
    An instance of the same class as `basemod`.
        A trained model instance, the best model found.

    """
    if isinstance(cv, int):
        cv = StratifiedShuffleSplit(n_splits=cv, test_size=0.20)
    # Find the best model within param_grid:
    crossvalidator = GridSearchCV(basemod, param_grid, cv=cv, scoring=scoring)
    crossvalidator.fit(X, y)
    # Report some information:
    if verbose:
        print("Best params: {}".format(crossvalidator.best_params_))
        print("Best score: {0:0.03f}".format(crossvalidator.best_score_))
    # Return the best model found:
    return crossvalidator.best_estimator_


def get_vocab(X, n_words=None, mincount=1):
    """
    Get the vocabulary for an RNN example matrix `X`, adding $UNK$ if
    it isn't already present.

    Parameters
    ----------
    X : list of lists of str

    n_words : int or None
        If this is `int > 0`, keep only the top `n_words` by frequency.

    mincount : int
        Only words with at least this many tokens are kept.

    Returns
    -------
    list of str

    """
    wc = Counter([w for ex in X for w in ex])
    wc = wc.most_common(n_words) if n_words else wc.items()
    if mincount > 1:
        wc = {(w, c) for w, c in wc if c >= mincount}
    vocab = {w for w, _ in wc}
    vocab.add("$UNK")
    return sorted(vocab)


def create_pretrained_embedding(
        lookup, vocab, required_tokens=('$UNK', "<s>", "</s>")):
    """
    Create an embedding matrix from a lookup and a specified vocab.
    Words from `vocab` that are not in `lookup` are given random
    representations.

    Parameters
    ----------
    lookup : dict
        Must map words to their vector representations.

    vocab : list of str
        Words to create embeddings for.

    required_tokens : tuple of str
        Tokens that must have embeddings. If they are not available
        in the look-up, they will be given random representations.

    Returns
    -------
    np.array, list
        The np.array is an embedding for `vocab` and the `list` is
        the potentially expanded version of `vocab` that came in.

    """
    dim = len(next(iter(lookup.values())))
    embedding = np.array([lookup.get(w, randvec(dim)) for w in vocab])
    for tok in required_tokens:
        if tok not in vocab:
            vocab.append(tok)
            embedding = np.vstack((embedding, randvec(dim)))
    return embedding, vocab


def fix_random_seeds(
        seed=42,
        set_system=True,
        set_torch=True,
        set_tensorflow=False,
        set_torch_cudnn=True):
    """
    Fix random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed to be set.

    set_system : bool
        Whether to set `np.random.seed(seed)` and `random.seed(seed)`

    set_tensorflow : bool
        Whether to set `tf.random.set_random_seed(seed)`

    set_torch : bool
        Whether to set `torch.manual_seed(seed)`

    set_torch_cudnn: bool
        Flag for whether to enable cudnn deterministic mode.
        Note that deterministic mode can have a performance impact,
        depending on your model.
        https://pytorch.org/docs/stable/notes/randomness.html

    Notes
    -----
    The function checks that PyTorch and TensorFlow are installed
    where the user asks to set seeds for them. If they are not
    installed, the seed-setting instruction is ignored. The intention
    is to make it easier to use this function in environments that lack
    one or both of these libraries.

    Even though the random seeds are explicitly set,
    the behavior may still not be deterministic (especially when a
    GPU is enabled), due to:

    * CUDA: There are some PyTorch functions that use CUDA functions
    that can be a source of non-determinism:
    https://pytorch.org/docs/stable/notes/randomness.html

    * PYTHONHASHSEED: On Python 3.3 and greater, hash randomization is
    turned on by default. This seed could be fixed before calling the
    python interpreter (PYTHONHASHSEED=0 python test.py). However, it
    seems impossible to set it inside the python program:
    https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program

    """
    # set system seed
    if set_system:
        np.random.seed(seed)
        random.seed(seed)

    # set torch seed
    if set_torch:
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.manual_seed(seed)

    # set torch cudnn backend
    if set_torch_cudnn:
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # set tf seed
    if set_tensorflow:
        try:
            from tensorflow.compat.v1 import set_random_seed as set_tf_seed
        except ImportError:
            from tensorflow.random import set_seed as set_tf_seed
        except ImportError:
            pass
        else:
            set_tf_seed(seed)


class DenseTransformer(TransformerMixin):
    """
    From

    http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html

    Some sklearn methods return sparse matrices that don't interact
    well with estimators that expect dense arrays or regular iterables
    as inputs. This little class helps manage that. Especially useful
    in the context of Pipelines.

    """
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
