"""
Baseline Logistic Regression Classifier for NetWorld-AI.
Serves as a simple, interpretable linear baseline model to benchmark deep learning temporal performance.
"""

from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from src.utils import setup_logger

logger = setup_logger("baseline_lr")


class BaselineLR:
    """
    Logistic Regression baseline model for threat classification.
    """

    def __init__(self, random_state: int = 42, max_iter: int = 1000) -> None:
        """
        Initializes Logistic Regression baseline.

        Args:
            random_state: Random state for reproducibility.
            max_iter: Maximum optimization iterations.
        """
        self.model = LogisticRegression(
            class_weight="balanced",
            random_state=random_state,
            max_iter=max_iter
        )
        self.is_fitted = False

    def _prepare_data(self, X: np.ndarray) -> np.ndarray:
        """
        Flattens 3D temporal sequence input (N, seq_len, features) to 2D matrix (N, seq_len * features).

        Args:
            X: Input feature array (2D or 3D).

        Returns:
            np.ndarray: Flattened 2D feature matrix.
        """
        if X.ndim == 3:
            n_samples, seq_len, n_features = X.shape
            return X.reshape(n_samples, seq_len * n_features)
        return X

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains Logistic Regression model.

        Args:
            X_train: Training feature matrix.
            y_train: Training binary labels.
        """
        X_flat = self._prepare_data(X_train)
        logger.info(f"Training Logistic Regression baseline on shape {X_flat.shape}...")
        self.model.fit(X_flat, y_train)
        self.is_fitted = True
        logger.info("Logistic Regression baseline trained successfully.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts binary threat labels.

        Args:
            X: Feature matrix.

        Returns:
            np.ndarray: Predicted binary class labels (0 or 1).
        """
        if not self.is_fitted:
            raise RuntimeError("BaselineLR is not fitted yet. Call train() first.")
        X_flat = self._prepare_data(X)
        return self.model.predict(X_flat)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts risk probabilities.

        Args:
            X: Feature matrix.

        Returns:
            np.ndarray: Risk probability scores for positive class (1).
        """
        if not self.is_fitted:
            raise RuntimeError("BaselineLR is not fitted yet. Call train() first.")
        X_flat = self._prepare_data(X)
        return self.model.predict_proba(X_flat)[:, 1]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluates model performance and returns metrics dictionary.

        Args:
            X_test: Test feature matrix.
            y_test: Test ground truth labels.

        Returns:
            Dict[str, float]: Metrics dictionary containing precision, recall, f1, fpr, and accuracy.
        """
        y_pred = self.predict(X_test)

        # Convert multi-class labels (0..6) to binary risk (0=BENIGN/6, 1=ATTACK/0..5)
        if np.max(y_test) > 1 or np.max(y_pred) > 1:
            y_true_b = (y_test != 6).astype(int)
            y_pred_b = (y_pred != 6).astype(int)
        else:
            y_true_b = y_test.astype(int)
            y_pred_b = y_pred.astype(int)
        
        prec = float(precision_score(y_true_b, y_pred_b, average="binary", zero_division=0))
        rec = float(recall_score(y_true_b, y_pred_b, average="binary", zero_division=0))
        f1 = float(f1_score(y_true_b, y_pred_b, average="binary", zero_division=0))
        acc = float(accuracy_score(y_true_b, y_pred_b))

        # False Positive Rate (FPR) = FP / (FP + TN)
        cm = confusion_matrix(y_true_b, y_pred_b, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        else:
            fpr = 0.0

        metrics = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "fpr": fpr,
            "accuracy": acc
        }
        logger.info(f"BaselineLR Evaluation: {metrics}")
        return metrics
