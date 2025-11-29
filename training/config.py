"""
Training Configuration Module

This module defines configuration classes and default hyperparameters for model training.
Includes settings for learning rates, batch sizes, number of epochs, model architectures,
and Bayesian-specific parameters (KL divergence weights, number of samples, etc.).
"""

from typing import List


class Config:
    """
    Configuration class containing all hyperparameters for the Bayesian LSTM project.

    This class stores all configuration parameters organized by category:
    - Data parameters (tickers, dates, train/val/test splits)
    - Feature engineering parameters
    - Model architecture parameters (including Bayesian-specific settings)
    - Training hyperparameters
    - Inference parameters
    - File paths for data and results
    """

    # ==================== General Parameters ====================
    SEED = 42

    # ==================== Data Parameters ====================
    TICKERS: List[str] = ['AAPL', 'SPY', 'TSLA']
    START_DATE: str = '2018-01-01'
    END_DATE: str = '2024-10-31'
    TRAIN_END: str = '2021-12-31'
    VAL_END: str = '2022-12-31'

    # ==================== Feature Parameters ====================
    SEQ_LEN: int = 20
    FEATURES: List[str] = ['ma_return_20', 'ma_volatility_20', 'normalized_volume', 'normalized_price']
    INPUT_SIZE: int = 4

    # ==================== Model Parameters ====================
    HIDDEN_SIZE: int = 64
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.2

    # Bayesian Prior Parameters
    PRIOR_BASE_STD: float = 0.1
    VIX_MEAN: float = 20.0
    VIX_SCALE: float = 10.0
    VIX_SENSITIVITY: float = 0.5
    
    # ==================== Training Parameters ====================
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 50
    EARLY_STOP_PATIENCE: int = 10
    WEIGHT_DECAY: float = 1e-5

    # ==================== Inference Parameters ====================
    N_SAMPLES_TRAIN: int = 1  # Number of samples during training (1 for efficiency)
    N_SAMPLES_TEST: int = 100  # Number of samples during testing (for uncertainty quantification)

    # ==================== Paths ====================
    DATA_DIR: str = 'data/processed'
    MODEL_DIR: str = 'results/models'
    PRED_DIR: str = 'results/predictions'
    FIG_DIR: str = 'results/figures'

    def __repr__(self) -> str:
        """
        Return a formatted string representation of all configuration parameters.

        Returns:
            Formatted string showing all config values organized by category
        """
        separator = "=" * 80

        config_str = f"\n{separator}\n"
        config_str += "Configuration Parameters\n"
        config_str += f"{separator}\n\n"

        # Data Parameters
        config_str += "DATA PARAMETERS:\n"
        config_str += "-" * 80 + "\n"
        config_str += f"  Tickers         : {self.TICKERS}\n"
        config_str += f"  Start Date      : {self.START_DATE}\n"
        config_str += f"  End Date        : {self.END_DATE}\n"
        config_str += f"  Train End       : {self.TRAIN_END}\n"
        config_str += f"  Validation End  : {self.VAL_END}\n\n"

        # Feature Parameters
        config_str += "FEATURE PARAMETERS:\n"
        config_str += "-" * 80 + "\n"
        config_str += f"  Sequence Length : {self.SEQ_LEN}\n"
        config_str += f"  Features        : {self.FEATURES}\n"
        config_str += f"  Input Size      : {self.INPUT_SIZE}\n\n"

        # Model Parameters
        config_str += "MODEL PARAMETERS:\n"
        config_str += "-" * 80 + "\n"
        config_str += f"  Hidden Size     : {self.HIDDEN_SIZE}\n"
        config_str += f"  Num Layers      : {self.NUM_LAYERS}\n"
        config_str += f"  Dropout         : {self.DROPOUT}\n"
        config_str += f"  Prior Base Std  : {self.PRIOR_BASE_STD}\n"
        config_str += f"  VIX Mean        : {self.VIX_MEAN}\n"
        config_str += f"  VIX Scale       : {self.VIX_SCALE}\n"
        config_str += f"  VIX Sensitivity : {self.VIX_SENSITIVITY}\n\n"

        # Training Parameters
        config_str += "TRAINING PARAMETERS:\n"
        config_str += "-" * 80 + "\n"
        config_str += f"  Batch Size      : {self.BATCH_SIZE}\n"
        config_str += f"  Learning Rate   : {self.LEARNING_RATE}\n"
        config_str += f"  Num Epochs      : {self.NUM_EPOCHS}\n"
        config_str += f"  Early Stop Pat. : {self.EARLY_STOP_PATIENCE}\n"
        config_str += f"  Weight Decay    : {self.WEIGHT_DECAY}\n\n"

        # Inference Parameters
        config_str += "INFERENCE PARAMETERS:\n"
        config_str += "-" * 80 + "\n"
        config_str += f"  Train Samples   : {self.N_SAMPLES_TRAIN}\n"
        config_str += f"  Test Samples    : {self.N_SAMPLES_TEST}\n\n"

        # Paths
        config_str += "PATHS:\n"
        config_str += "-" * 80 + "\n"
        config_str += f"  Data Directory  : {self.DATA_DIR}\n"
        config_str += f"  Model Directory : {self.MODEL_DIR}\n"
        config_str += f"  Pred Directory  : {self.PRED_DIR}\n"
        config_str += f"  Fig Directory   : {self.FIG_DIR}\n"

        config_str += f"\n{separator}\n"

        return config_str
