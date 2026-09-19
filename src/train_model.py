"""
train_model.py
Model training script for NetWorld-AI.
Synthesizes baseline dataset, trains LSTMWorldModel and Autoencoder models,
and saves state dictionaries to models/ directory.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Add current project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import main as train_main


def train_models():
    """Delegates model training to main Phase 2 pipeline."""
    train_main()


if __name__ == "__main__":
    train_models()
