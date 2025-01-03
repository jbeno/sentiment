# ELECTRA and GPT-4o: Cost-Effective Partners for Sentiment Analysis

This repository contains the code and datasets for the research paper "ELECTRA and GPT-4o: Cost-Effective Partners for Sentiment Analysis", which explores collaborative approaches between ELECTRA and GPT-4o models for sentiment classification. This research was conducted as the final project for the [XCS224U](https://online.stanford.edu/courses/xcs224u-natural-language-understanding) "Natural Language Understanding" course by [Stanford Engineering CGOE](https://cgoe.stanford.edu).

## Research Overview

The research investigated collaborative approaches between bidirectional transformers (ELECTRA Base/Large) and Large Language Models (GPT-4o/4o-mini) for three-way sentiment classification of reviews (negative, neutral, positive). We found that:

- Augmenting GPT-4o-mini prompts with ELECTRA predictions significantly improved performance over either model alone
- However, when GPT models were fine-tuned, including predictions decreased performance
- Including probabilities or similar examples enhanced performance for GPT-4o on challenging datasets  
- Fine-tuned GPT-4o-mini achieved nearly equivalent performance to GPT-4o at 76% lower cost
- The best approach depends on project constraints (budget, privacy concerns, available compute resources)

## Resources

### Research
- [ELECTRA and GPT-4o: Cost-Effective Partners for Sentiment Analysis](research_paper.pdf) - Research paper (PDF)

### Models 
- [ELECTRA Base Classifier for Sentiment Analysis](https://huggingface.co/jbeno/electra-base-classifier-sentiment) - Fine-tuned ELECTRA base discriminator (Hugging Face)
- [ELECTRA Large Classifier for Sentiment Analysis](https://huggingface.co/jbeno/electra-large-classifier-sentiment) - Fine-tuned ELECTRA large discriminator (Hugging Face)

### Datasets
- [Sentiment Merged Dataset](https://huggingface.co/datasets/jbeno/sentiment_merged) - Combination of DynaSent R1/R2 and SST-3 (Hugging Face)

### Code
- [jbeno/sentiment](https://github.com/jbeno/sentiment) - Primary research repository (GitHub)
- [electra-classifier](https://pypi.org/project/electra-classifier/) - Package for loading fine-tuned ELECTRA classifier models (PyPI)

## Repository Structure

```text
├── data/                            # Merged dataset files
├── electra_finetune/                # ELECTRA classifier fine-tuning logs
├── results/                         # Experiment predictions and metrics
├── statistics/                      # Statistical analysis
├── classifier.py                    # Neural classifier model with DDP
├── colors.py                        # Color display utilities
├── data_processing.ipynb            # Creation of Merged dataset
├── datawaza_funcs.py                # Subset of Datawaza library with edits 
├── finetune.py                      # Interactive classifier fine-tuning program with DDP
├── gpt_finetune_experiments.ipynb   # GPT fine-tune, baselines, and experiments with DSPy
├── requirements.txt                 # Python dependencies
├── research_paper.pdf               # Research paper in PDF format
├── sst.py                           # SST dataset loader from CS224U repo
├── statistics.ipynb                 # Statistical analysis
├── torch_model_base.py              # Base neural classiifer model from CS224U repo
└── utils.py                         # General utilities modified from CS224U repo
```

## Setup and Installation 

1. Clone the GitHub repo:
```bash
git clone https://github.com/jbeno/sentiment.git
cd sentiment
```

2. Create a new virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create .env file and add environment variables:
```bash
OPENAI_API_KEY=your-api-key   # Required for GPT experiments
WANDB_API_KEY=your-wandb-key  # Optional for experiment tracking
ARIZE_API_KEY=your-arize-key  # Optional for LLM trace tracking
```

## Using the Models

The fine-tuned ELECTRA models have been published on Hugging Face, but the fine-tuned GPT-4o/4o-mini models are stored privately by OpenAI. However, you can re-create the fine-tuned models using the code in this repo.

### ELECTRA Models

You can use the fine-tuned ELECTRA models that have been published on Hugging Face. An `electra-classifier` package was created to streamline loading of the models.

```python
# Install the package in a notebook
import sys
!{sys.executable} -m pip install electra-classifier

# Import libraries
import torch
from transformers import AutoTokenizer
from electra_classifier import ElectraClassifier

# Load tokenizer and model
model_name = "jbeno/electra-base-classifier-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ElectraClassifier.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()

# Run inference
text = "I love this restaurant!"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs)
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_label = model.config.id2label[predicted_class_id]
    print(f"Predicted label: {predicted_label}")
```

### Fine-tuning ELECTRA

The ELECTRA models can be fine-tuned using `finetune.py`, which has an interactive mode and leverages multiple GPUs through Distributed Data Parallel (DDP). This can also be used on BERT or RoBERTa, and a variety of datasets.

```bash
python ./finetune.py --dataset 'merged_local' --weights_name 'google/electra-base-discriminator' --save_data \
--save_model --save_pickle --save_preds --lr 0.00001 --epochs 100 --pooling 'mean' --dropout_rate 0.3 \
--num_layers 2 --hidden_dim 1024 --hidden_activation 'swishglu' --batch_size 32 --l2_strength 0.01 \
--checkpoint_interval 5 --use_zero --optimizer 'adamw' --scheduler 'cosine_warmup' \
--scheduler_kwargs '{"T_0":5, "T_mult":1, "eta_min":1e-7}' --decimal 6 --use_val_split \
--eval_split 'test' --early_stop 'score' --n_iter_no_change 10 --interactive
```

Here is the help for the command-line arguments:

```
$ python ./finetune.py --help
usage: finetune.py [-h] [--dataset DATASET] [--eval_dataset EVAL_DATASET] [--eval_split {validation,test}]
    [--sample_percent SAMPLE_PERCENT] [--chunk_size CHUNK_SIZE] [--label_dict LABEL_DICT]
    [--numeric_dict NUMERIC_DICT] [--label_template LABEL_TEMPLATE] [--pos_label POS_LABEL]
    [--weights_name WEIGHTS_NAME] [--pooling POOLING] [--finetune_bert] [--finetune_layers FINETUNE_LAYERS]
    [--freeze_bert] [--num_layers NUM_LAYERS] [--hidden_dim HIDDEN_DIM]
    [--hidden_activation HIDDEN_ACTIVATION] [--dropout_rate DROPOUT_RATE] [--batch_size BATCH_SIZE]
    [--accumulation_steps ACCUMULATION_STEPS] [--epochs EPOCHS] [--lr LR] [--lr_decay LR_DECAY]
    [--optimizer OPTIMIZER] [--use_zero] [--l2_strength L2_STRENGTH] [--optimizer_kwargs OPTIMIZER_KWARGS]
    [--scheduler SCHEDULER] [--scheduler_kwargs SCHEDULER_KWARGS] [--max_grad_norm MAX_GRAD_NORM]
    [--random_seed RANDOM_SEED] [--interactive] [--show_progress] [--checkpoint_dir CHECKPOINT_DIR]
    [--checkpoint_interval CHECKPOINT_INTERVAL] [--resume_from_checkpoint] [--early_stop EARLY_STOP]
    [--n_iter_no_change N_ITER_NO_CHANGE] [--tol TOL] [--target_score TARGET_SCORE]
    [--val_percent VAL_PERCENT] [--use_val_split] [--wandb] [--wandb_project WANDB_PROJECT]
    [--wandb_run WANDB_RUN] [--wandb_alerts] [--threshold THRESHOLD] [--model_name MODEL_NAME]
    [--save_data] [--save_model] [--save_pickle] [--save_hf] [--save_preds] [--save_plots]
    [--save_dir SAVE_DIR] [--data_file DATA_FILE] [--model_file MODEL_FILE] [--use_saved_params]
    [--predict] [--predict_file PREDICT_FILE] [--device DEVICE] [--gpus GPUS]
    [--num_threads NUM_THREADS] [--num_workers NUM_WORKERS] [--prefetch PREFETCH] [--empty_cache]
    [--port PORT] [--debug] [--mem_interval MEM_INTERVAL] [--decimal DECIMAL] [--color_theme COLOR_THEME]

DDP Distributed PyTorch Training for Sentiment Analysis using BERT

optional arguments:
  -h, --help            show this help message and exit

Dataset configuration:
  --dataset DATASET     Training dataset to use: 'sst', 'sst_local', 'dynasent_r1', 'dynasent_r2',
                        'mteb_tweet', 'merged_local' (default: sst_local)
  --eval_dataset EVAL_DATASET
                        (Optional) Different test dataset to use: 'sst', 'sst_local', 'dynasent_r1',
                        'dynasent_r2', 'mteb_tweet', 'merged_local' (default: None)
  --eval_split {validation,test}
                        Specify whether to evaluate with 'validation' or 'test' split (default:
                        validation)
  --sample_percent SAMPLE_PERCENT
                        Percentage of data to use for training, validation and test (default: None)
  --chunk_size CHUNK_SIZE
                        Number of dataset samples to encode in each chunk (default: None, process
                        all data at once)
  --label_dict LABEL_DICT
                        Text label dictionary, string to numeric (default: {'negative': 0,
                        'neutral': 1, 'positive': 2})
  --numeric_dict NUMERIC_DICT
                        Numeric label dictionary, numeric to string (default: {0: 'negative', 1:
                        'neutral', 2: 'positive'})
  --label_template LABEL_TEMPLATE
                        Predefined class label template with dictionary mappings: 'neg_neu_pos',
                        'bin_neu', 'bin_pos', 'bin_neg' (default: None)
  --pos_label POS_LABEL
                        Positive class label for binary classification, must be integer (default: 1)

BERT tokenizer/model configuration:
  --weights_name WEIGHTS_NAME
                        Pre-trained model/tokenizer name from a Hugging Face repo. Can be root-level
                        or namespaced (default: 'bert-base-uncased')
  --pooling POOLING     Pooling method for BERT embeddings: 'cls', 'mean', 'max' (default: 'cls')
  --finetune_bert       Whether to fine-tune BERT weights. If True, specify number of finetune_layers
                        (default: False)
  --finetune_layers FINETUNE_LAYERS
                        Number of BERT layers to fine-tune. For example: 0 to freeze all, 12 or 24 to
                        tune all, 1 to tune the last layer, etc. (default: 1)
  --freeze_bert         Whether to freeze BERT weights during training (default: False)

Classifier configuration:
  --num_layers NUM_LAYERS
                        Number of hidden layers for neural classifier (default: 1)
  --hidden_dim HIDDEN_DIM
                        Hidden dimension for neural classifier layers (default: 300)
  --hidden_activation HIDDEN_ACTIVATION
                        Hidden activation function: 'tanh', 'relu', 'sigmoid', 'leaky_relu', 'gelu',
                        'swish', 'swishglu' (default: 'tanh')
  --dropout_rate DROPOUT_RATE
                        Dropout rate for neural classifier (default: 0.0)

Training configuration:
  --batch_size BATCH_SIZE
                        Batch size for both encoding text and training classifier (default: 32)
  --accumulation_steps ACCUMULATION_STEPS
                        Number of steps to accumulate gradients before updating weights (default: 1)
  --epochs EPOCHS       Number of epochs to train (default: 100)
  --lr LR               Learning rate (default: 0.001)
  --lr_decay LR_DECAY   Learning rate decay factor, defaults to none, 0.95 is 5% per layer
                        (default: 1.0)
  --optimizer OPTIMIZER
                        Optimizer to use: 'adam', 'sgd', 'adagrad', 'rmsprop', 'zero', 'adamw'
                        (default: 'adam')
  --use_zero            Use Zero Redundancy Optimizer for efficient DDP training, with the optimizer
                        specified in --optimizer (default: False)
  --l2_strength L2_STRENGTH
                        L2 regularization strength for optimizer (default: 0.0)
  --optimizer_kwargs OPTIMIZER_KWARGS
                        Additional optimizer keyword arguments as a dictionary (default: None)
  --scheduler SCHEDULER
                        Learning rate scheduler to use: 'none', 'step', 'multi_step', 'exponential',
                        'cosine', 'reduce_on_plateau', 'cyclic' (default: None)
  --scheduler_kwargs SCHEDULER_KWARGS
                        Additional scheduler keyword arguments as a dictionary (default: None)
  --max_grad_norm MAX_GRAD_NORM
                        Maximum gradient norm for clipping (default: None)
  --random_seed RANDOM_SEED
                        Random seed (default: 42)
  --interactive         Interactive mode for training (default: False)
  --show_progress       Show progress bars for training and evaluation (default: False)

Checkpoint configuration:
  --checkpoint_dir CHECKPOINT_DIR
                        Directory to save and load checkpoints (default: checkpoints)
  --checkpoint_interval CHECKPOINT_INTERVAL
                        Checkpoint interval in epochs (default: 50)
  --resume_from_checkpoint
                        Resume training from latest checkpoint (default: False)

Early stopping:
  --early_stop EARLY_STOP
                        Early stopping method, 'score' or 'loss' (default: None)
  --n_iter_no_change N_ITER_NO_CHANGE
                        Number of iterations with no improvement to stop training (default: 5)
  --tol TOL             Tolerance for early stopping (default: 1e-5)
  --target_score TARGET_SCORE
                        Target score for early stopping (default: None)
  --val_percent VAL_PERCENT
                        Fraction of training data to use for validation (default: 0.1)
  --use_val_split       Use a validation split instead of a proportion of the train data (default:
                        False)

Weights and bias integration:
  --wandb               Use Weights and Biases for logging (default: False)
  --wandb_project WANDB_PROJECT
                        Weights and Biases project name (default: None)
  --wandb_run WANDB_RUN
                        Weights and Biases run name (default: None)
  --wandb_alerts        Enable Weights and Biases alerts (default: False)

Evaluation options:
  --threshold THRESHOLD
                        Threshold for binary classification evaluation (default: 0.5)
  --model_name MODEL_NAME
                        Model name for display in evaluation plots (default: None)

Saving options:
  --save_data           Save processed data to disk as an .npz archive (X_train, X_dev, y_train,
                        y_dev, y_dev_sent)
  --save_model          Save the final model state after training in PyTorch .pth format (default:
                        False)
  --save_pickle         Save the final model after training in pickle .pkl format (default: False)
  --save_hf             Save the final model after training in Hugging Face format (default: False)
  --save_preds          Save predictions to CSV (default: False)
  --save_plots          Save evaluation plots (default: False)
  --save_dir SAVE_DIR   Directory to save archived data, predictions, plots (default: saves)

Loading options:
  --data_file DATA_FILE
                        Filename of the processed data to load as an .npz archive (default: None)
  --model_file MODEL_FILE
                        Filename of the classifier model or checkpoint to load (default: None)
  --use_saved_params    Use saved parameters for training, if loading a model

Prediction options:
  --predict             Make predictions on a provided unlabled dataset (default: False)
  --predict_file PREDICT_FILE
                        Filename of the unlabeled dataset to make predictions on (default: None)

GPU and CPU processing:
  --device DEVICE       Device will be auto-detected, or specify 'cuda' or 'cpu' (default: None)
  --gpus GPUS           Number of GPUs to use if device is 'cuda', will be auto-detected (default:
                        None)
  --num_threads NUM_THREADS
                        Number of threads for CPU training (default: None)
  --num_workers NUM_WORKERS
                        Number of workers for DataLoader (default: 0)
  --prefetch PREFETCH   Number of batches to prefetch (default: None)
  --empty_cache         Empty CUDA cache after each batch (default: False)
  --port PORT           Port number for DDP distributed training (default: 12355)

Debugging and logging:
  --debug               Debug or verbose mode to print more details (default: False)
  --mem_interval MEM_INTERVAL
                        Memory check interval in epochs (default: 10)
  --decimal DECIMAL     Decimal places for floating point numbers (default: 4)
  --color_theme COLOR_THEME
                        Color theme for console output: 'light', 'dark' (default: 'dark')
```

### GPT Fine-Tuning and Experiments

The GPT-4o/4o-mini models were fine-tuned via OpenAI API. The data processing to produce the requried JSONL files, and the `curl` commands to interact with the API, can be found in [gpt_finetune_experiments.ipynb](gpt_finetune_experiments.ipynb). This also contains all the baselines and experimental runs using various DSPy templates. This is the main notebook for the research.

### Results and Analysis

The predictions and evaluation metrics of all experimental runs can be found under [results](results). Statistical tests can be found in [statistics.ipynb](statistics.ipynb) and [statistics](statistics).

## Dataset

The dataset is a merge of Stanford Sentiment Treebank (SST-3) and DynaSent Rounds 1 and 2. The SST-3, DynaSent R1, and DynaSent R2 datasets were randomly mixed to form a new dataset with 102,097 Train examples, 5,421 Validation examples, and 6,530 Test examples.

The dataset is available in this repo under [data/merged](data/merged), or through the [Sentiment Merged Dataset](https://huggingface.co/datasets/jbeno/sentiment_merged) on Hugging Face. You can review the data processing to create the merged dataset here in [data_processing.ipynb](data_processing.ipynb).

## Tools and Technologies

Code was created in VSCode with auto-complete assistance from GitHub CoPilot with GPT-4o. Research paper copy editing, LaTeX formatting, and some code suggestions were provided by Claude 3.5 Sonnet.

## Citation

If you use this material in your research, please cite:

```bibtex
@article{beno2024electra,
  title={ELECTRA and GPT-4o: Cost-Effective Partners for Sentiment Analysis},
  author={Beno, James P.},
  year={2024}
}
```

## License

This project is licensed under the GNU GPL v3 License - see the LICENSE file for details.

## Contact

Jim Beno - jim@jimbeno.net

## Acknowledgments

- The creators of the [ELECTRA model](https://arxiv.org/abs/2003.10555) for their foundational work
- The authors of the datasets used: [Stanford Sentiment Treebank](https://huggingface.co/datasets/stanfordnlp/sst), [DynaSent](https://huggingface.co/datasets/dynabench/dynasent)
- The Stanford CS224U course repo, which provided a starting point for this code: [cgpotts/cs224u](https://github.com/cgpotts/cs224u)
- [Stanford Engineering CGOE](https://cgoe.stanford.edu), [Chris Potts](https://stanford.edu/~cgpotts/), [Insop Song](https://profiles.stanford.edu/insop), [Petra Parikova](https://profiles.stanford.edu/petra-parikova), and the Course Facilitators of [XCS224U](https://online.stanford.edu/courses/xcs224u-natural-language-understanding)
