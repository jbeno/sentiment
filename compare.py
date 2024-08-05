
import os
import sys
import time

# PyTorch imports
import torch

# Third-party library imports
import numpy as np
import pandas as pd


def save_embeddings(embeddings, world_size, device_type, test_dir, rank):
    # Create directory if it doesn't exist
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Create filename with appropriate suffix, rank, and timestamp
    suffix = f'_{world_size}_gpu' if device_type == 'cuda' else '_1_cpu'
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    filename = f'embeddings{suffix}_rank{rank}_{timestamp}.pt'
    filepath = os.path.join(test_dir, filename)

    # Save embeddings to disk
    torch.save(embeddings, filepath)
    print(f"Embeddings saved to {filepath}")

def load_and_compare_embeddings(test_dir, device):
    # List all files in the directory
    files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]

    # Initialize lists to store file details
    filenames = []
    dimensions = []
    mad_values = []

    # Load each file and get dimensions
    embeddings_list = []
    for file in files:
        filepath = os.path.join(test_dir, file)
        if device.type == 'cpu':
            embeddings = torch.load(filepath, map_location=torch.device('cpu'))
        else:
            embeddings = torch.load(filepath)
        filenames.append(file)
        dimensions.append(list(embeddings.shape))
        embeddings_list.append(embeddings.cpu())

    # Calculate mean absolute differences (MAD) between each pair of embeddings
    for i in range(len(embeddings_list)):
        for j in range(len(embeddings_list)):
            if i != j:
                mad = torch.mean(torch.abs(embeddings_list[i] - embeddings_list[j])).item()
                mad_values.append({
                    'File 1': filenames[i],
                    'File 2': filenames[j],
                    'MAD': mad
                })

    # Create a DataFrame to display the results
    df = pd.DataFrame(mad_values)
    
    return df

comparison_df = load_and_compare_embeddings('test/train', torch.device('cpu'))
print(f"\nComparison of embeddings:")
print(comparison_df)
print()