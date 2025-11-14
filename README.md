# Bayesian LSTM for Stock Price Forecasting

## Overview

This project implements a Bayesian Long Short-Term Memory (LSTM) neural network for stock price forecasting with uncertainty quantification. Unlike traditional point-estimate forecasting models, this approach provides probabilistic predictions with confidence intervals, enabling better risk assessment and decision-making.

## Key Features

- **Bayesian Deep Learning**: Implements Bayesian neural networks with weight uncertainty using variational inference
- **Uncertainty Quantification**: Provides both epistemic (model) and aleatoric (data) uncertainty estimates
- **Baseline Comparisons**: Includes standard LSTM and moving average baselines for performance comparison
- **Comprehensive Evaluation**: Evaluates models using accuracy metrics, calibration analysis, and proper scoring rules
- **Visualization Tools**: Rich plotting capabilities for predictions, uncertainty bands, and model diagnostics

## Project Structure

```
bayesian-stock-forecast/
├── data/              # Data downloading and preprocessing
├── models/            # Model implementations (baseline and Bayesian)
├── training/          # Training loops and configuration
├── evaluation/        # Metrics, calibration, and uncertainty analysis
├── visualization/     # Plotting and visualization tools
├── experiments/       # Experiment scripts
├── utils/             # Utility functions
└── results/           # Saved models, predictions, and figures
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run baseline experiments:
```bash
python -m experiments.run_baselines
```

Run Bayesian LSTM experiments:
```bash
python -m experiments.run_bayesian
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- See requirements.txt for full dependencies
