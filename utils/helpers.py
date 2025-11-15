"""
Utility Helper Functions

This module contains general utility functions used throughout the project including:
- Random seed setting for reproducibility
- File I/O helpers
- Logging configuration
- Common data transformations
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across PyTorch, NumPy, and Python's random module.

    Args:
        seed: Random seed value (default: 42)

    Example:
        >>> set_seed(42)
        >>> # All random operations will now be deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """
    Get the best available device for PyTorch computations.

    Returns:
        Device string: 'mps' for Apple Silicon GPU, 'cuda' for NVIDIA GPU, or 'cpu'

    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
        Using device: mps
    """
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def save_results(results: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save results dictionary to a JSON file.

    Args:
        results: Dictionary containing results to save
        filepath: Path where the JSON file should be saved

    Raises:
        TypeError: If results cannot be serialized to JSON
        IOError: If file cannot be written

    Example:
        >>> results = {'mse': 0.05, 'mae': 0.12}
        >>> save_results(results, 'results/metrics.json')
    """
    filepath = Path(filepath)

    # Create parent directory if it doesn't exist
    create_dir_if_not_exists(filepath.parent)

    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"Results saved to {filepath}")
    except TypeError as e:
        raise TypeError(f"Results contain non-serializable data: {e}")
    except IOError as e:
        raise IOError(f"Failed to write file {filepath}: {e}")


def load_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load results from a JSON file.

    Args:
        filepath: Path to the JSON file to load

    Returns:
        Dictionary containing the loaded results

    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file is not valid JSON

    Example:
        >>> results = load_results('results/metrics.json')
        >>> print(results['mse'])
        0.05
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        print(f"Results loaded from {filepath}")
        return results
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {filepath}: {e.msg}", e.doc, e.pos)


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    Pretty print metrics dictionary with formatted output.

    Args:
        metrics: Dictionary of metric names and values
        model_name: Name of the model for the header (default: "Model")

    Example:
        >>> metrics = {'MSE': 0.0523, 'MAE': 0.1234, 'RMSE': 0.2287}
        >>> print_metrics(metrics, "Bayesian LSTM")
        ================================================================================
        Metrics for Bayesian LSTM
        ================================================================================
        MSE  : 0.0523
        MAE  : 0.1234
        RMSE : 0.2287
        ================================================================================
    """
    separator = "=" * 80
    print(f"\n{separator}")
    print(f"Metrics for {model_name}")
    print(separator)

    if not metrics:
        print("No metrics to display")
    else:
        # Find the longest metric name for alignment
        max_name_len = max(len(name) for name in metrics.keys())

        for name, value in metrics.items():
            # Format numbers with appropriate precision
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)

            print(f"{name:<{max_name_len}} : {formatted_value}")

    print(separator)


def create_dir_if_not_exists(path: Union[str, Path]) -> None:
    """
    Create directory if it doesn't exist (including parent directories).

    Args:
        path: Directory path to create

    Example:
        >>> create_dir_if_not_exists('results/models/bayesian')
        >>> # Directory and all parent directories are now created
    """
    path = Path(path)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {path}")
    else:
        # Only print if it's being called explicitly, not from other functions
        if not hasattr(create_dir_if_not_exists, '_silent'):
            pass  # Directory already exists, no need to notify
