"""
Main Training Pipeline for NetWorld-AI (Phase 2).
Loads processed sequence dataset, trains Attention-LSTM World Model, Autoencoder, and Logistic Regression baseline.
Saves model checkpoints, loss curves, and validation metrics.
"""

import os
import sys
import json
from typing import Dict, Any, Tuple

# Ensure project root is in Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import set_seed, load_config, setup_logger
from src.generate_sample_data import generate_sample_cic_ids
from src.data import CSVLoader, WindowBuilder
from src.models import LSTMWorldModel, Autoencoder, BaselineLR

logger = setup_logger("train")


def load_or_create_dataset(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads preprocessed dataset from data/processed/processed_data.npz.
    If not found, generates synthetic sample CSV and runs Phase 1 data pipeline automatically.

    Args:
        config: Loaded YAML configuration.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            X_train, X_test, y_train, y_test
    """
    proc_path = os.path.join(config.get("paths", {}).get("data_processed", "data/processed"), "processed_data.npz")
    
    if not os.path.exists(proc_path):
        logger.info(f"Processed dataset not found at {proc_path}. Executing data preprocessing pipeline...")
        sample_path = os.path.join(config.get("paths", {}).get("data_sample", "data/sample"), "sample_cic_ids.csv")
        
        if not os.path.exists(sample_path):
            generate_sample_cic_ids(num_rows=10000, output_path=sample_path)
            
        loader = CSVLoader(sample_path)
        raw_df = loader.load_raw()
        extracted_df, feature_cols = loader.extract_features(raw_df)
        cleaned_df = loader.clean_data(extracted_df)
        scaled_df, _ = loader.normalize_features(cleaned_df, feature_cols, is_train=True)
        
        wb = WindowBuilder(scaled_df)
        X_win, y_win, episode_ids = wb.build_windows(scaled_df, feature_cols=feature_cols)
        X_seq, y_seq = wb.create_sequences(X_win, y_win)
        X_train, X_test, y_train, y_test = wb.grouped_time_split(X_seq, y_seq, episode_ids=episode_ids)
        wb.save_processed(X_train, X_test, y_train, y_test)
        logger.info("Data preprocessing complete and saved.")
    else:
        logger.info(f"Loading preprocessed dataset from {proc_path}...")
        data = np.load(proc_path)
        X_train, X_test = data["X_train"], data["X_test"]
        y_train, y_test = data["y_train"], data["y_test"]

    logger.info(f"Loaded X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_lstm_world_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[LSTMWorldModel, Dict[str, list]]:
    """
    Trains Attention-LSTM World Model with joint state forecasting + threat risk loss.

    Args:
        X_train, y_train: Training sequence features and risk labels.
        X_test, y_test: Testing sequence features and risk labels.
        config: Configuration dictionary.
        device: Torch computing device.

    Returns:
        Tuple[LSTMWorldModel, Dict[str, list]]: Trained model and loss history dictionary.
    """
    hp = config.get("hyperparameters", {})
    batch_size = hp.get("batch_size", 64)
    epochs = hp.get("epochs", 50)
    lr = hp.get("learning_rate", 0.001)
    hidden_size = hp.get("lstm_hidden", 128)
    num_layers = hp.get("num_layers", 2)
    dropout = hp.get("dropout", 0.2)

    input_size = X_train.shape[2] # Auto-detected feature size

    # Prepare next-state targets (target state is state at sequence end)
    # X_train is (N, seq_len, features); target state for state forecasting is latest step or t+1 step
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    # y_train contains integer multi-class stage labels (0..6)
    y_stage_train_t = torch.tensor(y_train, dtype=torch.long)
    # Binary risk flag: 1 if stage != 6 (BENIGN), 0 if BENIGN
    y_risk_train_t = torch.tensor((y_train != 6).astype(float), dtype=torch.float32).unsqueeze(1)
    next_state_train = X_train_t[:, -1, :]

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_stage_test_t = torch.tensor(y_test, dtype=torch.long)
    y_risk_test_t = torch.tensor((y_test != 6).astype(float), dtype=torch.float32).unsqueeze(1)
    next_state_test = X_test_t[:, -1, :]

    train_dataset = TensorDataset(X_train_t, next_state_train, y_risk_train_t, y_stage_train_t)
    test_dataset = TensorDataset(X_test_t, next_state_test, y_risk_test_t, y_stage_test_t)

    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(X_train)), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=min(batch_size, len(X_test)), shuffle=False)

    model = LSTMWorldModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "state_loss": [], "risk_loss": [], "stage_loss": []}

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    models_dir = config.get("paths", {}).get("model_path", "models")
    os.makedirs(models_dir, exist_ok=True)
    best_ckpt_path = os.path.join(models_dir, "best_lstm.pt")

    logger.info(f"Starting LSTM World Model training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_state_loss = 0.0
        running_risk_loss = 0.0
        running_stage_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for seq_in, target_state, target_risk, target_stage in pbar:
            seq_in = seq_in.to(device)
            target_state = target_state.to(device)
            target_risk = target_risk.to(device)
            target_stage = target_stage.to(device)

            optimizer.zero_grad()
            pred_state, pred_risk_logits, pred_stage_logits, _ = model(seq_in)

            state_loss = mse_loss_fn(pred_state, target_state)
            risk_loss = bce_loss_fn(pred_risk_logits, target_risk)
            stage_loss = ce_loss_fn(pred_stage_logits, target_stage)

            # Combined Loss: MSE(state) + BCE(risk) + CrossEntropy(stage)
            total_loss = state_loss + (1.5 * risk_loss) + (1.0 * stage_loss)

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += total_loss.item() * len(seq_in)
            running_state_loss += state_loss.item() * len(seq_in)
            running_risk_loss += risk_loss.item() * len(seq_in)
            running_stage_loss += stage_loss.item() * len(seq_in)

        epoch_train_loss = running_loss / len(X_train)
        epoch_state_loss = running_state_loss / len(X_train)
        epoch_risk_loss = running_risk_loss / len(X_train)
        epoch_stage_loss = running_stage_loss / len(X_train)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for seq_in, target_state, target_risk, target_stage in test_loader:
                seq_in = seq_in.to(device)
                target_state = target_state.to(device)
                target_risk = target_risk.to(device)
                target_stage = target_stage.to(device)

                pred_state, pred_risk_logits, pred_stage_logits, _ = model(seq_in)
                s_loss = mse_loss_fn(pred_state, target_state)
                r_loss = bce_loss_fn(pred_risk_logits, target_risk)
                st_loss = ce_loss_fn(pred_stage_logits, target_stage)
                val_loss = s_loss + (1.5 * r_loss) + (1.0 * st_loss)

                running_val_loss += val_loss.item() * len(seq_in)

        epoch_val_loss = running_val_loss / len(X_test)
        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["state_loss"].append(epoch_state_loss)
        history["risk_loss"].append(epoch_risk_loss)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | State Loss: {epoch_state_loss:.4f} | Risk Loss: {epoch_risk_loss:.4f}"
        )

        # Checkpoint Saving & Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": epoch_val_loss,
                "input_size": input_size,
                "hidden_size": hidden_size
            }, best_ckpt_path)
            logger.info(f"--> Saved best checkpoint to {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    # Load best checkpoint into model
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, history


def train_autoencoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    device: torch.device
) -> Autoencoder:
    """
    Trains Autoencoder strictly on BENIGN samples (y_train == 0) for novelty detection.

    Args:
        X_train, y_train: Full training dataset.
        config: Configuration dictionary.
        device: Torch computing device.

    Returns:
        Autoencoder: Trained Autoencoder.
    """
    benign_mask = y_train == 0
    X_benign = X_train[benign_mask]

    if len(X_benign) == 0:
        logger.warning("No benign samples found for Autoencoder training! Using full dataset.")
        X_benign = X_train

    input_dim = X_train.shape[2]
    # Flatten latest step or average step for autoencoder input
    X_benign_flat = X_benign[:, -1, :] # (N, features)

    tensor_data = torch.tensor(X_benign_flat, dtype=torch.float32)
    dataset = TensorDataset(tensor_data, tensor_data)
    loader = DataLoader(dataset, batch_size=min(64, len(tensor_data)), shuffle=True)

    ae_model = Autoencoder(input_dim=input_dim, latent_dim=16).to(device)
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    logger.info(f"Training Autoencoder on {len(X_benign_flat)} BENIGN samples for 30 epochs...")
    ae_model.train()

    for epoch in range(1, 31):
        total_loss = 0.0
        for batch_in, batch_target in loader:
            batch_in = batch_in.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            recon = ae_model(batch_in)
            loss = loss_fn(recon, batch_target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_in)

        avg_loss = total_loss / len(X_benign_flat)
        if epoch % 5 == 0 or epoch == 30:
            logger.info(f"Autoencoder Epoch {epoch:02d}/30 | Loss: {avg_loss:.6f}")

    models_dir = config.get("paths", {}).get("model_path", "models")
    ae_path = os.path.join(models_dir, "autoencoder.pt")
    torch.save({"model_state_dict": ae_model.state_dict(), "input_dim": input_dim}, ae_path)
    logger.info(f"Saved Autoencoder checkpoint to {ae_path}")

    return ae_model


def main() -> None:
    """Main training routine."""
    set_seed(42)
    config = load_config("config/config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using computing device: {device}")

    # Load dataset
    X_train, X_test, y_train, y_test = load_or_create_dataset(config)

    # 1. Train LSTM World Model
    lstm_model, history = train_lstm_world_model(X_train, y_train, X_test, y_test, config, device)

    # 2. Train Autoencoder on BENIGN data
    ae_model = train_autoencoder(X_train, y_train, config, device)

    # 3. Train Baseline Logistic Regression
    baseline = BaselineLR()
    baseline.train(X_train, y_train)
    models_dir = config.get("paths", {}).get("model_path", "models")
    lr_path = os.path.join(models_dir, "baseline_lr.joblib")
    joblib.dump(baseline, lr_path)
    logger.info(f"Saved BaselineLR model to {lr_path}")

    # 4. Save Training History Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss (Total)")
    plt.plot(history["val_loss"], label="Val Loss (Total)")
    plt.plot(history["state_loss"], label="State Forecast Loss (MSE)", linestyle="--")
    plt.plot(history["risk_loss"], label="Risk Prediction Loss (BCE)", linestyle=":")
    plt.title("NetWorld-AI Model Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(models_dir, "training_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training history plot to {plot_path}")

    # 5. Save Training Metrics JSON
    baseline_metrics = baseline.evaluate(X_test, y_test)
    metrics_summary = {
        "epochs_trained": len(history["train_loss"]),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "baseline_lr": baseline_metrics
    }
    json_path = os.path.join(models_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    logger.info(f"Saved training metrics summary to {json_path}")

    print("\n=======================================================")
    print("  NETWORLD-AI PHASE 2 TRAINING COMPLETE!")
    print(f"  - LSTM Checkpoint: models/best_lstm.pt")
    print(f"  - Autoencoder Checkpoint: models/autoencoder.pt")
    print(f"  - Baseline LR Model: models/baseline_lr.joblib")
    print(f"  - Metrics Summary: models/metrics.json")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
