"""
RegimeFlow: Multi-Scale Temporal Regime Detection with Uncertainty Quantification.

This module implements a neural architecture for detecting market regime changes
in financial time series, combining multi-scale dilated convolutions with
self-attention mechanisms and Bayesian uncertainty estimation.

Key innovations:
    1. Hierarchical multi-scale temporal feature extraction
    2. Positional-aware self-attention for long-range dependencies
    3. Epistemic uncertainty quantification for risk-aware predictions
    4. Optuna-based Bayesian hyperparameter optimization

Author: Research Implementation
License: MIT
"""

# =============================================================================
# IMPORTS
# Standard library imports
# =============================================================================
import gc
import logging
import math
import random
import time
import warnings
from collections import deque
from typing import Any, List, Optional, Tuple

# =============================================================================
# Third-party imports
# =============================================================================
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# All tunable parameters consolidated here per AGENTS.md requirements.
# Modify these values to adjust default behavior without code changes.
# =============================================================================

# Reproducibility settings
RANDOM_SEED: int = 42

# Default model architecture parameters
DEFAULT_HIDDEN_DIM: int = 64
DEFAULT_NUM_HEADS: int = 4
DEFAULT_DROPOUT: np.float64 = 0.2
DEFAULT_SEQ_LEN: int = 100
DEFAULT_NUM_CLASSES: int = 2

# Default training parameters
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LEARNING_RATE: np.float64 = 1e-3
DEFAULT_WEIGHT_DECAY: np.float64 = 1e-4
DEFAULT_N_EPOCHS: int = 100

# Optuna optimization defaults
DEFAULT_N_TRIALS: int = 50
DEFAULT_TIMEOUT: int = 3600  # seconds

# Multi-scale convolution dilation rates
DEFAULT_SCALES: list[int] = [1, 2, 4, 8]

# Visualization defaults
DEFAULT_FORECAST_STEPS: int = 30

# Labeling and evaluation defaults
DEFAULT_MIN_EXPANDING_WINDOWS: int = 60
DEFAULT_MAX_HISTORY_WINDOWS: int = 500
DEFAULT_N_COMPONENTS_RANGE: tuple[int, int] = (2, 6)
DEFAULT_HOLDOUT_RATIO: np.float64 = 0.15
DEFAULT_VAL_RATIO: np.float64 = 0.15
DEFAULT_CHANGE_TOLERANCE: int = 2
DEFAULT_MIN_REGIME_DURATION: int = 5


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure module-level logging with consistent formatting.

    Args:
        level: Logging verbosity level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logger instance for this module.
    """
    logger = logging.getLogger(__name__)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s %(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


logger = configure_logging()


# =============================================================================
# DETERMINISTIC SETUP
# Ensures reproducibility across runs when RANDOM_SEED is set.
# Note: CUDA operations may still have non-determinism; set
# torch.backends.cudnn.deterministic = True for full reproducibility.
# =============================================================================


def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value for all random number generators.

    Note:
        For complete CUDA determinism, additionally set:
        - torch.backends.cudnn.deterministic = True
        - torch.backends.cudnn.benchmark = False
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.debug(f"Random seed set to {seed}")


# Initialize seeds on module load
set_seed(RANDOM_SEED)


# =============================================================================
# PRODUCTION UTILITIES
# Time series cross-validation, normalization, serialization, and inference
# pipeline components for deployment-ready model operations.
# =============================================================================


class TimeSeriesCVSplitter:
    """
    Time series aware cross-validation without lookahead bias.

    Unlike standard k-fold CV, this splitter respects temporal ordering
    to prevent data leakage from future observations into training data.
    Essential for any time series prediction task.

    Supports two strategies:
    - Expanding window: Train on all data up to split point
    - Sliding window: Train on fixed-size recent window
    """

    @staticmethod
    def expanding_window_split(
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        min_train_ratio: np.float64 = 0.3,
    ) -> list[Tuple[slice, slice]]:
        """
        Generate expanding window splits for time series cross-validation.

        Each subsequent fold uses more training data while maintaining
        temporal order. No future data ever leaks into training.

        Args:
            X: Feature array of shape (n_samples, ...).
            y: Label array of shape (n_samples,).
            n_splits: Number of CV folds to generate.
            min_train_ratio: Minimum fraction of data for first training set.

        Returns:
            List of (train_slice, test_slice) tuples for each fold.

        Example:
            >>> splits = TimeSeriesCVSplitter.expanding_window_split(X, y, n_splits=3)
            >>> for train_idx, test_idx in splits:
            ...     X_train, X_test = X[train_idx], X[test_idx]
        """
        n_samples = len(X)
        min_train_size = int(n_samples * min_train_ratio)
        remaining = n_samples - min_train_size
        fold_size = remaining // n_splits

        splits = []
        for i in range(n_splits):
            train_end = min_train_size + (i * fold_size)
            test_start = train_end
            test_end = min(train_end + fold_size, n_samples)

            if test_end <= test_start:
                continue

            train_idx = slice(0, train_end)
            test_idx = slice(test_start, test_end)

            splits.append((train_idx, test_idx))
            logger.debug(
                f"Fold {i}: train[0:{train_end}], test[{test_start}:{test_end}]"
            )

        return splits

    @staticmethod
    def sliding_window_split(
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        train_size: int = None,
        test_size: int = None,
    ) -> list[Tuple[slice, slice]]:
        """
        Generate sliding window splits with fixed train/test sizes.

        Useful when recent data is more relevant than historical data,
        or when computational resources limit training set size.

        Args:
            X: Feature array of shape (n_samples, ...).
            y: Label array of shape (n_samples,).
            n_splits: Number of CV folds.
            train_size: Fixed training window size (default: 60% of data / n_splits).
            test_size: Fixed test window size (default: 20% of data / n_splits).

        Returns:
            List of (train_slice, test_slice) tuples.
        """
        n_samples = len(X)

        if train_size is None:
            train_size = int(n_samples * 0.6) // n_splits
        if test_size is None:
            test_size = int(n_samples * 0.2) // n_splits

        step = (n_samples - train_size - test_size) // max(1, n_splits - 1)

        splits = []
        for i in range(n_splits):
            start = i * step
            train_end = start + train_size
            test_end = train_end + test_size

            if test_end > n_samples:
                break

            train_idx = slice(start, train_end)
            test_idx = slice(train_end, test_end)
            splits.append((train_idx, test_idx))

        return splits


class AdaptiveTimeSeriesNormalizer:
    """
    Robust normalization for non-stationary time series with regime shifts.

    Handles the common challenges in financial time series:
    - Non-stationarity (changing mean/variance over time)
    - Regime-dependent statistics
    - Outliers and extreme values
    - Different scales across features

    Supports multiple normalization strategies that can be selected
    based on the characteristics of the data.
    """

    def __init__(
        self, method: str = "robust", window: int = 100, clip_sigma: np.float64 = 3.0
    ):
        """
        Initialize normalizer with specified strategy.

        Args:
            method: Normalization method ('standard', 'robust', 'rolling', 'minmax').
            window: Rolling window size for 'rolling' method.
            clip_sigma: Number of standard deviations for outlier clipping.
        """
        self.method = method
        self.window = window
        self.clip_sigma = clip_sigma
        self.fitted = False

        # Statistics computed during fit
        self.mean_ = None
        self.std_ = None
        self.median_ = None
        self.iqr_ = None
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray) -> "AdaptiveTimeSeriesNormalizer":
        """
        Compute normalization statistics from training data.

        Args:
            X: Training data of shape (n_samples, n_channels, seq_length).

        Returns:
            Self for method chaining.
        """
        if self.method == "standard":
            self.mean_ = X.mean(axis=(0, 2), keepdims=True)
            self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-8

        elif self.method == "robust":
            # Use median and IQR for outlier robustness
            self.median_ = np.median(X, axis=(0, 2), keepdims=True)
            q75 = np.percentile(X, 75, axis=(0, 2), keepdims=True)
            q25 = np.percentile(X, 25, axis=(0, 2), keepdims=True)
            self.iqr_ = (q75 - q25) + 1e-8

        elif self.method == "minmax":
            self.min_ = X.min(axis=(0, 2), keepdims=True)
            self.max_ = X.max(axis=(0, 2), keepdims=True)

        elif self.method == "rolling":
            # For rolling, we just store the window size
            # Actual normalization happens per-sample in transform
            self.mean_ = X.mean(axis=(0, 2), keepdims=True)
            self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-8

        self.fitted = True
        logger.debug(f"Normalizer fitted with method='{self.method}'")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply normalization to data.

        Args:
            X: Data of shape (n_samples, n_channels, seq_length).

        Returns:
            Normalized data with same shape.
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before transform")

        if self.method == "standard":
            X_norm = (X - self.mean_) / self.std_

        elif self.method == "robust":
            X_norm = (X - self.median_) / self.iqr_

        elif self.method == "minmax":
            X_norm = (X - self.min_) / (self.max_ - self.min_ + 1e-8)

        elif self.method == "rolling":
            # Per-sample rolling normalization
            X_norm = np.zeros_like(X)
            for i in range(len(X)):
                for c in range(X.shape[1]):
                    series = X[i, c]
                    # Compute rolling statistics
                    roll_mean = np.convolve(
                        series, np.ones(self.window) / self.window, mode="same"
                    )
                    roll_std = np.array(
                        [
                            series[
                                max(0, j - self.window // 2) : j + self.window // 2 + 1
                            ].std()
                            + 1e-8
                            for j in range(len(series))
                        ]
                    )
                    X_norm[i, c] = (series - roll_mean) / roll_std
        else:
            X_norm = X

        # Clip extreme values
        if self.clip_sigma > 0:
            X_norm = np.clip(X_norm, -self.clip_sigma, self.clip_sigma)

        return X_norm

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Reverse normalization to original scale.

        Args:
            X_norm: Normalized data.

        Returns:
            Data in original scale.
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")

        if self.method == "standard":
            return X_norm * self.std_ + self.mean_
        elif self.method == "robust":
            return X_norm * self.iqr_ + self.median_
        elif self.method == "minmax":
            return X_norm * (self.max_ - self.min_) + self.min_
        else:
            return X_norm


class ModelCheckpoint:
    """
    Model serialization and checkpointing for production deployment.

    Handles saving and loading of:
    - Model weights and architecture config
    - Optimizer state for training resumption
    - Normalizer statistics
    - Training metadata and hyperparameters
    """

    @staticmethod
    def save(
        path: str,
        model: nn.Module,
        optimizer: optim.Optimizer = None,
        normalizer: AdaptiveTimeSeriesNormalizer = None,
        metadata: dict = None,
    ) -> None:
        """
        Save model checkpoint to disk.

        Args:
            path: File path for checkpoint (.pt extension recommended).
            model: PyTorch model to save.
            optimizer: Optional optimizer state for training resumption.
            normalizer: Optional fitted normalizer.
            metadata: Optional dict with hyperparameters, metrics, etc.
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": getattr(model, "input_dim", None),
                "hidden_dim": getattr(model, "hidden_dim", None),
                "num_heads": getattr(model, "num_heads", None),
                "seq_len": getattr(model, "seq_len", None),
            },
            "metadata": metadata or {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if normalizer is not None and normalizer.fitted:
            checkpoint["normalizer"] = {
                "method": normalizer.method,
                "window": normalizer.window,
                "clip_sigma": normalizer.clip_sigma,
                "mean_": normalizer.mean_,
                "std_": normalizer.std_,
                "median_": normalizer.median_,
                "iqr_": normalizer.iqr_,
                "min_": normalizer.min_,
                "max_": normalizer.max_,
            }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    @staticmethod
    def load(path: str, model_class: type = None, device: str = "cpu") -> dict:
        """
        Load model checkpoint from disk.

        Args:
            path: Path to checkpoint file.
            model_class: Optional model class for instantiation.
            device: Device to load model onto.

        Returns:
            Dict containing loaded components and metadata.
        """
        checkpoint = torch.load(path, map_location=device)

        result = {
            "model_state_dict": checkpoint["model_state_dict"],
            "model_config": checkpoint.get("model_config", {}),
            "metadata": checkpoint.get("metadata", {}),
            "timestamp": checkpoint.get("timestamp"),
        }

        if "optimizer_state_dict" in checkpoint:
            result["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]

        if "normalizer" in checkpoint:
            norm_cfg = checkpoint["normalizer"]
            normalizer = AdaptiveTimeSeriesNormalizer(
                method=norm_cfg["method"],
                window=norm_cfg["window"],
                clip_sigma=norm_cfg["clip_sigma"],
            )
            normalizer.mean_ = norm_cfg["mean_"]
            normalizer.std_ = norm_cfg["std_"]
            normalizer.median_ = norm_cfg["median_"]
            normalizer.iqr_ = norm_cfg["iqr_"]
            normalizer.min_ = norm_cfg["min_"]
            normalizer.max_ = norm_cfg["max_"]
            normalizer.fitted = True
            result["normalizer"] = normalizer

        logger.info(f"Checkpoint loaded from {path}")
        return result


class RegimeInferencePipeline:
    """
    End-to-end inference pipeline for production deployment.

    Handles the complete inference workflow:
    1. Input validation and preprocessing
    2. Normalization using fitted statistics
    3. Model inference with uncertainty
    4. Post-processing with hysteresis filtering
    5. Regime change detection with confidence thresholds

    Designed for both batch and streaming inference scenarios.
    """

    def __init__(
        self,
        model: nn.Module,
        normalizer: AdaptiveTimeSeriesNormalizer = None,
        confidence_threshold: np.float64 = 0.7,
        min_regime_duration: int = 5,
        device: str = None,
    ):
        """
        Initialize inference pipeline.

        Args:
            model: Trained regime detection model.
            normalizer: Fitted normalizer (optional, raw input if None).
            confidence_threshold: Minimum confidence for regime change.
            min_regime_duration: Minimum samples before allowing regime switch.
            device: Inference device (auto-detect if None).
        """
        self.model = model
        self.normalizer = normalizer
        self.confidence_threshold = confidence_threshold
        self.min_regime_duration = min_regime_duration

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Streaming state
        self.current_regime = None
        self.regime_duration = 0
        self.buffer = deque(maxlen=getattr(model, "seq_len", 100))

    def _validate_input(self, X: np.ndarray) -> torch.Tensor:
        """Validate and format input for model."""
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        X = np.asarray(X, dtype=np.float32)

        # Handle different input shapes
        if X.ndim == 1:
            # Single channel, single sample: (seq_len,) -> (1, 1, seq_len)
            X = X[np.newaxis, np.newaxis, :]
        elif X.ndim == 2:
            # Could be (channels, seq_len) or (samples, seq_len)
            # Assume (channels, seq_len) for single sample
            X = X[np.newaxis, :, :]
        elif X.ndim == 3:
            # Already (samples, channels, seq_len)
            pass
        else:
            raise ValueError(f"Expected 1D, 2D, or 3D input, got {X.ndim}D")

        return torch.as_tensor(X, device=self.device, dtype=torch.float32)

    def predict(self, X: np.ndarray) -> dict:
        """
        Run batch inference on input data.

        Args:
            X: Input data, flexible shape handling.

        Returns:
            Dict with predictions, probabilities, uncertainty, and metadata.
        """
        X_tensor = self._validate_input(X)

        # Normalize if normalizer provided
        if self.normalizer is not None:
            X_np = X_tensor.cpu().numpy()
            X_norm = self.normalizer.transform(X_np)
            X_tensor = torch.as_tensor(X_norm, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(X_tensor)

        probs = outputs["regime_probs"]
        confidence, predictions = torch.max(probs, dim=-1)

        # Extract uncertainty if available
        uncertainty = outputs.get("uncertainty", torch.zeros_like(confidence))
        if uncertainty.dim() > 1:
            uncertainty = uncertainty.squeeze(-1)

        return {
            "predictions": predictions.cpu().numpy(),
            "probabilities": probs.cpu().numpy(),
            "confidence": confidence.cpu().numpy(),
            "uncertainty": uncertainty.cpu().numpy(),
            "frequencies": outputs.get("dominant_frequencies", torch.zeros(1))
            .cpu()
            .numpy(),
        }

    def process_stream(self, new_point: np.float64) -> Optional[dict]:
        """
        Process single data point in streaming mode.

        Maintains internal buffer and detects regime changes
        with hysteresis to avoid spurious transitions.

        Args:
            new_point: Single new observation.

        Returns:
            Regime change event dict if transition detected, else None.
        """
        self.buffer.append(new_point)

        if len(self.buffer) < self.buffer.maxlen:
            return None

        # Format buffer for prediction
        window = np.array(self.buffer)[np.newaxis, np.newaxis, :]
        result = self.predict(window)

        predicted = int(result["predictions"][0])
        confidence = np.float64(result["confidence"][0])

        # Hysteresis: require sustained high confidence and duration
        if predicted != self.current_regime:
            if confidence >= self.confidence_threshold:
                if (
                    self.regime_duration >= self.min_regime_duration
                    or self.current_regime is None
                ):
                    event = {
                        "timestamp": time.time(),
                        "old_regime": self.current_regime,
                        "new_regime": predicted,
                        "confidence": confidence,
                        "uncertainty": np.float64(result["uncertainty"][0]),
                        "duration_in_old": self.regime_duration,
                    }
                    self.current_regime = predicted
                    self.regime_duration = 0
                    return event

        self.regime_duration += 1
        return None

    def reset_stream(self) -> None:
        """Reset streaming state for new session."""
        self.buffer.clear()
        self.current_regime = None
        self.regime_duration = 0


class RegimeModelDiagnostics:
    """
    Comprehensive model validation and diagnostics for production.

    Provides statistical validation through:
    - Bootstrap confidence intervals
    - Calibration analysis
    - Per-class performance metrics
    - Uncertainty correlation analysis
    """

    @staticmethod
    def bootstrap_validate(
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: np.float64 = 0.95,
        device: str = "cpu",
    ) -> dict:
        """
        Bootstrap validation with confidence intervals.

        Args:
            model: Trained model to validate.
            X: Validation features.
            y: Validation labels.
            n_bootstrap: Number of bootstrap iterations.
            confidence_level: Confidence level for intervals (default 95%).
            device: Computation device.

        Returns:
            Dict with accuracy stats, uncertainty stats, and confidence intervals.
        """
        model = model.to(device)
        model.eval()

        accuracies = []
        uncertainties = []

        alpha = (1 - confidence_level) / 2

        for _ in range(n_bootstrap):
            # Bootstrap sample with replacement
            idx = np.random.choice(len(X), len(X), replace=True)
            X_bs = X[idx]
            y_bs = y[idx]

            X_tensor = torch.FloatTensor(X_bs).to(device)

            with torch.no_grad():
                outputs = model(X_tensor)
                preds = torch.argmax(outputs["regime_probs"], dim=-1).cpu().numpy()

            acc = (preds == y_bs).mean()
            accuracies.append(acc)

            # Get uncertainty if available
            if "uncertainty" in outputs:
                unc = outputs["uncertainty"].cpu().numpy()
                if unc.ndim > 1:
                    unc = unc.squeeze(-1)
                uncertainties.append(unc.mean())

        acc_ci = np.percentile(accuracies, [alpha * 100, (1 - alpha) * 100])

        result = {
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "accuracy_ci": acc_ci,
            "n_bootstrap": n_bootstrap,
            "confidence_level": confidence_level,
        }

        if uncertainties:
            unc_ci = np.percentile(uncertainties, [alpha * 100, (1 - alpha) * 100])
            result["uncertainty_mean"] = np.mean(uncertainties)
            result["uncertainty_ci"] = unc_ci

        return result

    @staticmethod
    def compute_calibration(
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
        device: str = "cpu",
    ) -> dict:
        """
        Compute calibration metrics (ECE and MCE).

        Measures how well predicted probabilities match actual outcomes.
        A well-calibrated model has ECE close to 0.

        Args:
            model: Trained model.
            X: Features.
            y: True labels.
            n_bins: Number of probability bins for calibration.
            device: Computation device.

        Returns:
            Dict with ECE, MCE, and per-bin calibration data.
        """
        model = model.to(device)
        model.eval()

        X_tensor = torch.FloatTensor(X).to(device)

        with torch.no_grad():
            outputs = model(X_tensor)
            probs = outputs["regime_probs"].cpu().numpy()
            preds = np.argmax(probs, axis=-1)

        # Get confidence (max probability)
        confidences = np.max(probs, axis=-1)
        accuracies = (preds == y).astype(np.float64)

        # Bin by confidence
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        bin_data = []

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                calibration_error = abs(avg_accuracy - avg_confidence)

                ece += prop_in_bin * calibration_error
                mce = max(mce, calibration_error)

                bin_data.append(
                    {
                        "bin": i,
                        "lower": bin_lower,
                        "upper": bin_upper,
                        "count": in_bin.sum(),
                        "avg_confidence": avg_confidence,
                        "avg_accuracy": avg_accuracy,
                        "calibration_error": calibration_error,
                    }
                )

        return {
            "ece": ece,  # Expected Calibration Error
            "mce": mce,  # Maximum Calibration Error
            "n_bins": n_bins,
            "bin_data": bin_data,
        }


def load_real_data(
    csv_path: str, target_column: str = None, seq_length: int = 50, n_classes: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real-world time series data from CSV and prepare for regime detection.

    Segments continuous data into overlapping windows and generates
    regime labels based on statistical properties or provided labels.

    Args:
        csv_path: Path to CSV file with time series data.
        target_column: Column to use as primary feature (first numeric if None).
        seq_length: Length of each window segment.
        n_classes: Number of regime classes for auto-labeling.

    Returns:
        Tuple of (X, y) ready for model training.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    # Find target column
    if target_column is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in CSV")
        target_column = numeric_cols[0]
        logger.info(f"Auto-selected target column: {target_column}")

    values = df[target_column].dropna().values.astype(np.float32)

    # Create overlapping windows
    n_windows = len(values) - seq_length + 1
    X = np.zeros((n_windows, 1, seq_length), dtype=np.float32)

    for i in range(n_windows):
        X[i, 0, :] = values[i : i + seq_length]

    # Generate regime labels based on volatility quantiles
    # This creates natural regime clustering
    volatilities = np.array([X[i, 0, :].std() for i in range(n_windows)])
    vol_quantiles = np.percentile(
        volatilities, np.linspace(0, 100, n_classes + 1)[1:-1]
    )

    y = np.digitize(volatilities, vol_quantiles).astype(np.int64)

    logger.info(f"Created {n_windows} windows with {n_classes} regime classes")
    logger.info(f"Class distribution: {np.bincount(y)}")

    return X, y


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def create_synthetic_data(
    num_samples: int = 1000,
    seq_length: int = 100,
    num_channels: int = 1,
    num_classes: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic time series data for model development and testing.

    Creates distinct waveform patterns for each class to simulate different
    market regimes (trending, mean-reverting, volatile, random walk).

    Args:
        num_samples: Total number of time series to generate.
        seq_length: Length of each time series in timesteps.
        num_channels: Number of input channels (features per timestep).
        num_classes: Number of distinct regime classes to simulate.

    Returns:
        Tuple of (X, y) where:
            X: ndarray of shape (num_samples, num_channels, seq_length)
            y: ndarray of shape (num_samples,) with integer class labels

    Example:
        >>> X, y = create_synthetic_data(1000, 100, 1, 3)
        >>> print(X.shape, y.shape)
        (1000, 1, 100) (1000,)
    """
    X = np.zeros((num_samples, num_channels, seq_length))
    y = np.zeros(num_samples, dtype=int)

    for i in range(num_samples):
        class_label = i % num_classes
        y[i] = class_label

        # Generate regime-specific waveforms with additive Gaussian noise
        # Each class simulates a distinct market microstructure pattern
        t = np.linspace(0, 4 * np.pi, seq_length)
        noise = np.random.normal(0, 0.1, seq_length)

        if class_label == 0:
            # Class 0: Sinusoidal oscillation (cyclical/mean-reverting regime)
            X[i, 0] = np.sin(t) + noise
        elif class_label == 1:
            # Class 1: Square wave (trending regime with sharp reversals)
            X[i, 0] = np.sign(np.sin(t)) + noise
        elif class_label == 2:
            # Class 2: Sawtooth wave (persistent trending with sudden resets)
            X[i, 0] = (t % (2 * np.pi)) / (2 * np.pi) + noise
        else:
            # Class 3+: Random walk (diffusive/unpredictable regime)
            X[i, 0] = np.cumsum(np.random.normal(0, 0.1, seq_length))

    return X, y


def load_lng_data(
    csv_path: str = "eu_lng_snapshot.csv",
    seq_length: int = 30,
    feature_cols: list = None,
    leakage_safe: bool = False,
    min_expanding_windows: int = DEFAULT_MIN_EXPANDING_WINDOWS,
    max_history_windows: int = DEFAULT_MAX_HISTORY_WINDOWS,
    n_components_range: tuple[int, int] = DEFAULT_N_COMPONENTS_RANGE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EU LNG storage data and create regime-labeled time series windows.

    Loads real-world EU LNG storage data and assigns regime labels based on
    adaptive percentile thresholds derived from tank fullness trends and
    send-out utilization patterns.

    Regime definitions:
        - 0 (Accumulation): Rising tank levels, low send-out utilization
        - 1 (Drawdown): Falling tank levels, high send-out utilization
        - 2 (Crisis): Very low tank levels or extreme volatility
        - 3 (Stable): Balanced flow, moderate changes

    Args:
        csv_path: Path to the EU LNG snapshot CSV file.
        seq_length: Lookback window length in days (default 30).
        feature_cols: List of column names to use as features.
                     Defaults to ['tank_fullness_lng', 'sendout_utilisation'].

    Returns:
        Tuple of (X, y) where:
            X: ndarray of shape (n_windows, n_features, seq_length)
            y: ndarray of shape (n_windows,) with regime labels 0-3

    Example:
        >>> X, y = load_lng_data('eu_lng_snapshot.csv', seq_length=30)
        >>> print(f"Loaded {len(X)} windows with {X.shape[1]} features")
    """
    if feature_cols is None:
        feature_cols = ["tank_fullness_lng", "sendout_utilisation"]

    logger.info(f"Loading LNG data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Parse dates and sort
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    elif "gasDayStart" in df.columns:
        df["date"] = pd.to_datetime(df["gasDayStart"])
        df = df.sort_values("date").reset_index(drop=True)

    # Handle missing values
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Clip outliers (beyond 3 sigma)
    for col in feature_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            df[col] = df[col].clip(mean_val - 3 * std_val, mean_val + 3 * std_val)

    logger.info(f"Preprocessed {len(df)} rows, {len(feature_cols)} features")

    # Create sliding windows
    features = df[feature_cols].values
    n_samples = len(features) - seq_length + 1
    n_features = len(feature_cols)

    # Shape: (n_samples, n_features, seq_length) - channels first for Conv1D
    X = np.zeros((n_samples, n_features, seq_length), dtype=np.float32)
    for i in range(n_samples):
        X[i] = features[i : i + seq_length].T

    logger.info(f"Created {n_samples} windows of shape {X.shape}")

    # =========================================================================
    # SELF-DISCOVERING REGIME DETECTION (OPTIONALLY LEAKAGE-SAFE)
    # =========================================================================

    # Step 1: Extract rich feature set from each window
    window_features = np.zeros((n_samples, 6), dtype=np.float32)

    for i in range(n_samples):
        # Channel 0 features (primary signal - e.g., tank fullness)
        signal = X[i, 0, :]
        window_features[i, 0] = signal[-1]  # Current level
        window_features[i, 1] = (signal[-1] - signal[0]) / seq_length  # Trend
        window_features[i, 2] = np.std(signal)  # Volatility

        # Channel 1 features (secondary signal - e.g., utilization)
        if n_features > 1:
            signal2 = X[i, 1, :]
            window_features[i, 3] = signal2[-1]  # Current level
            window_features[i, 4] = (signal2[-1] - signal2[0]) / seq_length  # Trend
            window_features[i, 5] = np.std(signal2)  # Volatility

    y: np.ndarray
    regime_names: dict[int, str]
    if leakage_safe:
        y, regime_probs, regime_names = generate_regimes_expanding(
            window_features=window_features,
            min_train=min_expanding_windows,
            max_history=max_history_windows,
            n_components_range=n_components_range,
        )
    else:
        # Step 2: Normalize features for clustering
        feature_mean = window_features.mean(axis=0)
        feature_std = window_features.std(axis=0) + 1e-8
        features_normalized = (window_features - feature_mean) / feature_std

        # Step 3: Automatic regime discovery via Gaussian Mixture Model
        try:
            from sklearn.mixture import GaussianMixture
            from sklearn.preprocessing import StandardScaler

            best_bic = np.inf
            best_n_regimes = 4
            best_gmm = None

            for n_regimes in range(n_components_range[0], n_components_range[1] + 1):
                gmm = GaussianMixture(
                    n_components=n_regimes,
                    covariance_type="full",
                    n_init=3,
                    random_state=RANDOM_SEED,
                )
                gmm.fit(features_normalized)
                bic = gmm.bic(features_normalized)

                if bic < best_bic:
                    best_bic = bic
                    best_n_regimes = n_regimes
                    best_gmm = gmm

            logger.info(f"Optimal number of regimes (BIC): {best_n_regimes}")

            y = best_gmm.predict(features_normalized).astype(np.int64)
            regime_probs = best_gmm.predict_proba(features_normalized)

            regime_mean_levels = []
            for r in range(best_n_regimes):
                mask = y == r
                if mask.sum() > 0:
                    regime_mean_levels.append((r, window_features[mask, 0].mean()))
                else:
                    regime_mean_levels.append((r, 0))

            sorted_regimes = sorted(regime_mean_levels, key=lambda x: x[1])
            regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}

            y = np.array([regime_mapping[label] for label in y], dtype=np.int64)

            regime_names = {}
            for new_label in range(best_n_regimes):
                mask = y == new_label
                if mask.sum() > 0:
                    mean_level = window_features[mask, 0].mean()
                    mean_trend = window_features[mask, 1].mean()

                    if mean_level < 0.3:
                        level_name = "Low"
                    elif mean_level > 0.7:
                        level_name = "High"
                    else:
                        level_name = "Mid"

                    if mean_trend > 0.001:
                        trend_name = "Rising"
                    elif mean_trend < -0.001:
                        trend_name = "Falling"
                    else:
                        trend_name = "Stable"

                    regime_names[new_label] = f"{level_name}-{trend_name}"

        except ImportError:
            logger.warning("sklearn not available, using simple percentile clustering")

            level = window_features[:, 0]
            trend = window_features[:, 1]

            y = np.zeros(n_samples, dtype=np.int64)
            y[(level < np.percentile(level, 25))] = 0
            y[
                (level >= np.percentile(level, 25)) & (level < np.percentile(level, 75))
            ] = 1
            y[(level >= np.percentile(level, 75))] = 2

            y[(y == 1) & (trend > np.percentile(trend, 66))] = 3

            best_n_regimes = 4
            regime_names = {0: "Low", 1: "Mid-Stable", 2: "High", 3: "Mid-Rising"}

    # Log discovered regime distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Discovered {len(unique)} regimes via unsupervised clustering:")
    for u, c in zip(unique, counts):
        name = regime_names.get(u, f"Regime-{u}")
        logger.info(f"  Regime {u} ({name}): {c} samples ({100 * c / n_samples:.1f}%)")

    return X, y


def generate_regimes_expanding(
    window_features: np.ndarray,
    min_train: int = DEFAULT_MIN_EXPANDING_WINDOWS,
    max_history: int = DEFAULT_MAX_HISTORY_WINDOWS,
    n_components_range: tuple[int, int] = DEFAULT_N_COMPONENTS_RANGE,
) -> Tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """
    Leakage-safe regime labeling using an expanding window GMM.

    Fits a GMM on past windows only, freezes past labels, and never lets
    future data influence earlier labels.
    """
    from sklearn.mixture import GaussianMixture

    n_samples = len(window_features)
    labels = np.full(n_samples, -1, dtype=np.int64)
    max_components = n_components_range[1]
    probs = np.zeros((n_samples, max_components), dtype=np.float32)
    regime_names: dict[int, str] = {}

    for idx in range(n_samples):
        history_start = max(0, idx - max_history + 1)
        history = window_features[history_start : idx + 1]
        if len(history) < min_train:
            # Too little history; coarse percentile labeling
            level = window_features[idx, 0]
            trend = window_features[idx, 1]
            if level < np.percentile(history[:, 0], 25):
                labels[idx] = 0
            elif level > np.percentile(history[:, 0], 75):
                labels[idx] = 2
            else:
                labels[idx] = 1
            probs[idx, labels[idx]] = 1.0
            continue

        history_mean = history.mean(axis=0)
        history_std = history.std(axis=0) + 1e-8
        history_norm = (history - history_mean) / history_std
        current_norm = ((window_features[idx] - history_mean) / history_std).reshape(
            1, -1
        )

        best_bic = np.inf
        best_gmm = None
        best_n = None
        for n_comp in range(n_components_range[0], n_components_range[1] + 1):
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type="full",
                n_init=2,
                random_state=RANDOM_SEED,
            )
            gmm.fit(history_norm)
            bic = gmm.bic(history_norm)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_n = n_comp

        assert best_gmm is not None and best_n is not None

        raw_label = int(best_gmm.predict(current_norm)[0])
        raw_probs = best_gmm.predict_proba(current_norm)[0]

        cluster_means = []
        for c in range(best_n):
            mask = best_gmm.predict(history_norm) == c
            if mask.sum() > 0:
                cluster_means.append((c, history[mask, 0].mean()))
            else:
                cluster_means.append((c, -np.inf))
        sorted_clusters = sorted(cluster_means, key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_clusters)}

        mapped_label = mapping[raw_label]
        labels[idx] = mapped_label

        probs[idx, :best_n] = raw_probs[np.argsort([c for c, _ in sorted_clusters])]

        # Track simple names by mean level
        regime_names = {
            mapping[c]: f"Level-{rank}" for rank, (c, _) in enumerate(sorted_clusters)
        }

    return (
        labels,
        probs[:, : (max(labels) + 1 if labels.max() >= 0 else 0) or max_components],
        regime_names,
    )


def time_series_split_with_holdout(
    n_samples: int,
    val_ratio: np.float64 = DEFAULT_VAL_RATIO,
    holdout_ratio: np.float64 = DEFAULT_HOLDOUT_RATIO,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sequential train/val/holdout splits without leakage."""
    if val_ratio + holdout_ratio >= 1.0:
        raise ValueError("val_ratio + holdout_ratio must be < 1.0")
    train_end = int(n_samples * (1 - val_ratio - holdout_ratio))
    val_end = int(n_samples * (1 - holdout_ratio))
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    holdout_idx = np.arange(val_end, n_samples)
    return train_idx, val_idx, holdout_idx


def apply_sticky_regime(
    preds: np.ndarray, min_duration: int = DEFAULT_MIN_REGIME_DURATION
) -> np.ndarray:
    """Enforce minimum regime duration to reduce jitter."""
    if preds.size == 0:
        return preds
    smoothed = preds.copy()
    start = 0
    current = preds[0]
    for idx in range(1, len(preds)):
        if preds[idx] != current:
            run_length = idx - start
            if run_length < min_duration:
                smoothed[start:idx] = current
            start = idx
            current = preds[idx]
    return smoothed


def change_point_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, tolerance: int = DEFAULT_CHANGE_TOLERANCE
) -> dict[str, float]:
    """Evaluate change-point precision/recall with tolerance and latency."""

    def _changes(seq: np.ndarray) -> np.ndarray:
        return np.where(np.diff(seq) != 0)[0] + 1

    true_cp = _changes(y_true)
    pred_cp = _changes(y_pred)

    if len(pred_cp) == 0 and len(true_cp) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "latency": 0.0}

    matched = []
    latencies = []
    used_true = set()
    for pc in pred_cp:
        diffs = np.abs(true_cp - pc)
        if diffs.size == 0:
            continue
        idx = np.argmin(diffs)
        if diffs[idx] <= tolerance and idx not in used_true:
            matched.append(pc)
            latencies.append(pc - true_cp[idx])
            used_true.add(idx)

    precision = len(matched) / max(1, len(pred_cp))
    recall = len(matched) / max(1, len(true_cp))
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    latency = float(np.mean(latencies)) if latencies else np.inf

    return {"precision": precision, "recall": recall, "f1": f1, "latency": latency}


def cost_sensitive_score(
    y_true: np.ndarray, y_pred: np.ndarray, cost_matrix: Optional[np.ndarray] = None
) -> float:
    """Compute average cost given an explicit cost matrix."""
    if cost_matrix is None:
        n = max(y_true.max(), y_pred.max()) + 1
        cost_matrix = np.ones((n, n)) - np.eye(n)
    costs = cost_matrix[y_true, y_pred]
    return float(costs.mean())


def economic_utility_score(
    regime_preds: np.ndarray,
    returns: np.ndarray,
    regime_to_position: dict[int, float],
    transaction_cost: np.float64 = 0.0,
) -> float:
    """
    Simple economic utility: map regimes to positions, apply returns and costs.

    Args:
        regime_preds: Predicted regimes per timestep.
        returns: Realized returns aligned to predictions (same length).
        regime_to_position: Mapping from regime id to position size (-1..1).
        transaction_cost: Cost applied on position change (absolute delta).
    """
    positions = np.array([regime_to_position.get(int(r), 0.0) for r in regime_preds])
    pnl = positions * returns
    turnover = np.abs(np.diff(positions, prepend=positions[0]))
    pnl -= transaction_cost * turnover
    return float(pnl.sum())


def mc_dropout_predict(
    model: nn.Module,
    X: np.ndarray,
    n_samples: int = 20,
    device: Optional[torch.device] = None,
) -> dict[str, np.ndarray]:
    """
    Monte Carlo dropout predictive distribution for calibrated uncertainty.

    Runs the model with dropout enabled to estimate mean and variance of
    regime probabilities.
    """
    device = device or next(model.parameters()).device
    X_tensor = torch.FloatTensor(X).to(device)
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(X_tensor)
            preds.append(outputs["regime_probs"].cpu().numpy())
    model.eval()
    probs = np.stack(preds, axis=0)
    mean_probs = probs.mean(axis=0)
    var_probs = probs.var(axis=0)
    return {"mean_probs": mean_probs, "var_probs": var_probs}


class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for time series classification.

    Handles conversion from NumPy arrays to PyTorch tensors with
    appropriate dtype casting for model consumption.

    Attributes:
        data: Tensor of shape (N, C, T) where N=samples, C=channels, T=timesteps.
        labels: Optional tensor of shape (N,) with integer class labels.
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray = None):
        """
        Initialize dataset from NumPy arrays.

        Args:
            data: Time series array of shape (num_samples, num_channels, seq_length).
            labels: Optional class labels array of shape (num_samples,).
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx], None


# =============================================================================
# MODEL COMPONENTS
# Modular building blocks for the regime detection architecture.
# =============================================================================


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequence modeling.

    Injects positional information into the input embeddings using fixed
    sinusoidal functions at different frequencies. This enables the model
    to utilize sequence position information without learning embeddings.

    The encoding follows Vaswani et al. (2017):
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Attributes:
        pe: Buffer containing precomputed positional encodings.

    Reference:
        Vaswani et al. "Attention Is All You Need" (2017), Section 3.5
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: np.float64 = 0.1):
        """
        Initialize positional encoding layer.

        Args:
            d_model: Embedding dimension (must match input channel dimension).
            max_len: Maximum sequence length to precompute encodings for.
            dropout: Dropout probability applied after adding positional encoding.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encodings for efficiency
        # Shape: (1, max_len, d_model) for broadcasting over batch dimension
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        x = x + self.pe[:, : x.size(1)]
        return x.permute(0, 2, 1)  # Back to (batch, channels, seq_len)


class TimeAttentionWithPositions(nn.Module):
    """Attention with explicit positional encoding and sanity checks"""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        max_len: int = 1024,
        dropout: np.float64 = 0.1,
    ):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels({channels}) must be divisible by num_heads({num_heads})"
        )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Explicit positional encoding
        self.pos_encoding = PositionalEncoding(channels, max_len, dropout)

        # Attention layers
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, seq_len = x.shape
        assert channels == self.channels, (
            f"Expected {self.channels} channels, got {channels}"
        )

        # Add positional encoding
        x = self.pos_encoding(x)

        # Move channels to last dimension for attention
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention with scaling
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Apply attention
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, channels)
        out = self.proj(out)
        out = self.proj_dropout(out)

        return out.permute(0, 2, 1)


class ScaleSpecificGating(nn.Module):
    """Scale-specific gating that experts would implement"""

    def __init__(self, channels: int, num_scales: int):
        super().__init__()
        # Each scale gets its own gating network
        self.scale_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, channels // 4, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv1d(channels // 4, channels, kernel_size=1),
                    nn.Sigmoid(),
                )
                for _ in range(num_scales)
            ]
        )

    def forward(self, conv_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply scale-specific gating to each convolution output"""
        gated_outputs = []
        for conv_out, gate in zip(conv_outputs, self.scale_gates):
            # Gate based on the actual convolved output, not the input
            gate_weights = gate(conv_out)
            gated_outputs.append(conv_out * gate_weights)
        return gated_outputs


class MultiScaleResidualExpert(nn.Module):
    """Expert version with scale-specific gating and residual scaling"""

    def __init__(self, channels: int, scales: list = [1, 2, 4, 8]):
        super().__init__()
        self.scales = scales

        # Separate convolutions for each scale
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    channels, channels, kernel_size=3, padding=scale, dilation=scale
                )
                for scale in scales
            ]
        )

        # Scale-specific gating
        self.gating = ScaleSpecificGating(channels, len(scales))

        # Learnable scale weights (instead of simple average)
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1), nn.BatchNorm1d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Apply each scale convolution
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))

        # Apply scale-specific gating
        gated_outputs = self.gating(conv_outputs)

        # Weighted combination (learned weights instead of simple average)
        weighted_sum = sum(
            w * out
            for w, out in zip(F.softmax(self.scale_weights, dim=0), gated_outputs)
        )

        # Output projection
        out = self.output_proj(weighted_sum)

        # Residual connection with learnable scaling
        return F.relu(out + identity)


class LearnablePoolingExpert(nn.Module):
    """Expert pooling with channel-wise attention"""

    def __init__(self, channels: int):
        super().__init__()
        # Channel-wise attention (not just temporal)
        self.channel_attention = nn.Sequential(
            nn.Conv1d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 4, 1, 1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)

        # Channel attention
        channel_weights = self.channel_attention(x)  # (batch, channels, seq_len)
        x_channel_weighted = x * channel_weights

        # Temporal attention
        temporal_weights = self.temporal_attention(
            x_channel_weighted
        )  # (batch, 1, seq_len)

        # Weighted pooling
        weighted_sum = torch.sum(
            x_channel_weighted * temporal_weights, dim=-1, keepdim=True
        )

        return weighted_sum  # (batch, channels, 1)


class RegimeDetectionExpert(nn.Module):
    """
    Multi-scale temporal regime detection with uncertainty quantification.

    This architecture combines several key innovations for financial time series:

    1. Positional-aware self-attention for capturing long-range dependencies
       while preserving temporal ordering information (critical for regime detection).

    2. Multi-scale dilated convolutions with learned gating, enabling the model
       to simultaneously capture patterns at multiple temporal granularities
       (from tick-level noise to macro regime shifts).

    3. Dual-pathway pooling (channel + temporal attention) for adaptive
       feature aggregation that emphasizes regime-discriminative patterns.

    4. Uncertainty quantification head providing epistemic uncertainty estimates,
       essential for risk-aware decision making in production trading systems.

    Architecture:
        Input -> Conv1D projection -> Positional Attention -> Multi-Scale Conv
        -> Attention Pooling -> [Regime Head, Frequency Head, Uncertainty Head]

    References:
        - Vaswani et al. (2017) "Attention Is All You Need"
        - van den Oord et al. (2016) "WaveNet: Dilated Causal Convolutions"
        - Gal & Ghahramani (2016) "Dropout as Bayesian Approximation"
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_classes: int = 2,
        seq_len: int = 100,
        dropout: np.float64 = 0.2,
        num_heads: int = 4,
    ):
        """
        Initialize the regime detection model.

        Args:
            input_dim: Number of input channels (e.g., 1 for univariate price series).
            hidden_dim: Latent representation dimension. Must be divisible by num_heads.
            num_classes: Number of distinct market regimes to classify.
            seq_len: Expected input sequence length for positional encoding buffer.
            dropout: Dropout probability for regularization.
            num_heads: Number of parallel attention heads. hidden_dim must be divisible
                      by this value to ensure equal head dimensions.

        Raises:
            AssertionError: If hidden_dim is not divisible by num_heads.
        """
        super().__init__()

        # Validate architectural constraints upfront
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}). "
            f"Consider using hidden_dim = {(hidden_dim // num_heads) * num_heads}"
        )

        # Store parameters for input validation and serialization
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_heads = num_heads

        # --------------------------------------------------------------------
        # Stage 1: Input projection with batch normalization
        # Kernel size 7 provides sufficient receptive field for initial
        # feature extraction while maintaining temporal resolution.
        # BatchNorm stabilizes training dynamics for financial data
        # which often exhibits non-stationary statistics.
        # --------------------------------------------------------------------
        self.init_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --------------------------------------------------------------------
        # Stage 2: Temporal self-attention with explicit positional encoding
        # Unlike transformers for NLP, positional information is critical
        # for time series as it encodes temporal distance relationships.
        # --------------------------------------------------------------------
        self.temporal_attention = TimeAttentionWithPositions(
            channels=hidden_dim,
            num_heads=num_heads,
            max_len=seq_len * 2,  # Buffer for variable-length sequences
            dropout=dropout,
        )

        # --------------------------------------------------------------------
        # Stage 3: Multi-scale dilated convolutions with learned gating
        # Captures regime patterns at multiple temporal resolutions.
        # Gating mechanism allows dynamic scale selection per sample.
        # --------------------------------------------------------------------
        self.multi_scale = MultiScaleResidualExpert(hidden_dim, scales=[1, 2, 4, 8])

        # --------------------------------------------------------------------
        # Stage 4: Learnable pooling with dual channel-temporal attention
        # Replaces naive global average pooling with adaptive aggregation
        # that emphasizes regime-discriminative temporal locations.
        # --------------------------------------------------------------------
        self.pooling = LearnablePoolingExpert(hidden_dim)

        # --------------------------------------------------------------------
        # Output Heads: Interpretable multi-task outputs
        # --------------------------------------------------------------------

        # Primary classification head for regime prediction
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Auxiliary head for dominant frequency estimation
        # Outputs normalized frequencies in [0, 1] for interpretability
        # Useful for characterizing regime dynamics (trending vs mean-reverting)
        self.frequency_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # Top-3 dominant frequencies
            nn.Sigmoid(),
        )

        # Epistemic uncertainty head for confidence calibration
        # Softplus ensures positive output representing prediction variance
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns dictionary with interpretable outputs
        """
        # Input validation
        assert x.dim() == 3, (
            f"Expected 3D input (batch, channels, seq_len), got {x.dim()}D"
        )
        assert x.size(1) == self.input_dim, (
            f"Expected {self.input_dim} input channels, got {x.size(1)}"
        )

        # Project
        x = self.init_proj(x)

        # Attention with positions
        x = self.temporal_attention(x)

        # Multi-scale processing
        x = self.multi_scale(x)

        # Pooling
        pooled = self.pooling(x)  # (batch, hidden_dim, 1)
        pooled_flat = pooled.squeeze(-1)  # (batch, hidden_dim)

        # Multiple heads for interpretability
        regime_logits = self.regime_head(pooled_flat)
        frequencies = self.frequency_head(pooled_flat)  # Normalized frequencies
        uncertainty = self.uncertainty_head(pooled_flat)  # Prediction confidence

        return {
            "regime_logits": regime_logits,
            "regime_probs": F.softmax(regime_logits, dim=-1),
            "dominant_frequencies": frequencies,
            "uncertainty": uncertainty,
            "attention_output": x,  # For visualization
            "pooled_features": pooled_flat,  # For downstream tasks
        }


class RobustRegimeLoss(nn.Module):
    """Loss function that addresses expert concerns"""

    def __init__(
        self,
        alpha: np.float64 = 0.1,
        beta: np.float64 = 0.05,
        freq_weight: np.float64 = 0.01,
        unc_weight: np.float64 = 0.01,
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.alpha = alpha  # Frequency diversity weight
        self.beta = beta  # Orthogonality weight
        self.freq_weight = freq_weight
        self.unc_weight = unc_weight

    def frequency_diversity_loss(self, freq_pred: torch.Tensor) -> torch.Tensor:
        """Encourage diverse but stable frequency predictions"""
        # Batch variance (encourage diversity)
        batch_var = torch.var(freq_pred, dim=0).mean()

        # Within-sample consistency (discourage extreme values)
        sample_range = torch.max(freq_pred, dim=1)[0] - torch.min(freq_pred, dim=1)[0]
        range_penalty = F.relu(0.5 - sample_range).mean()  # Penalize if range < 0.5

        # Want high batch variance, low range penalty
        return range_penalty - batch_var  # Negative because we want to minimize this

    def orthogonality_loss(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Encourage class-separable features"""
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=features.device)

        class_means = []
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                class_means.append(features[mask].mean(dim=0))

        if len(class_means) < 2:
            return torch.tensor(0.0, device=features.device)

        # Compute cosine similarity between class means
        total_sim = 0
        count = 0
        for i in range(len(class_means)):
            for j in range(i + 1, len(class_means)):
                sim = F.cosine_similarity(
                    class_means[i].unsqueeze(0), class_means[j].unsqueeze(0)
                )
                total_sim += sim
                count += 1

        return (
            total_sim / count
            if count > 0
            else torch.tensor(0.0, device=features.device)
        )

    def forward(
        self,
        outputs: dict,
        labels: torch.Tensor,
        freq_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute multi-component loss for interpretable regime detection.

        Loss components:
        1. Cross-entropy for regime classification (primary objective)
        2. Frequency diversity to encourage varied spectral predictions
        3. Orthogonality to promote class-separable feature representations
        4. Uncertainty calibration to encourage appropriate confidence levels

        Args:
            outputs: Model output dictionary with logits, features, uncertainty.
            labels: Ground truth regime labels.
            freq_labels: Optional supervised frequency targets.

        Returns:
            dict: Individual loss components and aggregated total loss.
        """
        # Primary classification objective
        class_loss = self.ce_loss(outputs["regime_logits"], labels)

        # Auxiliary frequency prediction loss
        if freq_labels is not None:
            freq_loss = F.mse_loss(outputs["dominant_frequencies"], freq_labels)
        else:
            freq_loss = self.frequency_diversity_loss(outputs["dominant_frequencies"])

        # Feature orthogonality regularization
        ortho_loss = self.orthogonality_loss(outputs["pooled_features"], labels)

        # --------------------------------------------------------------------
        # Uncertainty calibration loss
        # Penalizes high confidence on incorrect predictions.
        # When uncertainty_head is disabled (Identity), skip this component.
        # --------------------------------------------------------------------
        with torch.no_grad():
            preds = torch.argmax(outputs["regime_logits"], dim=-1)
            correct = (preds == labels).float()

        uncertainty = outputs["uncertainty"]
        # Check if uncertainty has valid shape (batch_size, 1) from Softplus head
        # vs invalid shape from Identity passthrough
        if uncertainty.dim() == 2 and uncertainty.size(1) == 1:
            uncertainty_loss = torch.mean(uncertainty.squeeze() * (1 - correct))
            mean_uncertainty = uncertainty.mean()
        else:
            # Uncertainty head disabled - use zero loss
            uncertainty_loss = torch.tensor(0.0, device=labels.device)
            mean_uncertainty = torch.tensor(0.0, device=labels.device)

        # Weighted combination of loss components
        total_loss = (
            class_loss
            + self.freq_weight * freq_loss
            + self.beta * ortho_loss
            + self.unc_weight * uncertainty_loss
        )

        return {
            "total_loss": total_loss,
            "class_loss": class_loss,
            "freq_loss": freq_loss,
            "ortho_loss": ortho_loss,
            "uncertainty_loss": uncertainty_loss,
            "mean_uncertainty": mean_uncertainty,
            "accuracy": correct.mean(),
        }


# ========== OPTUNA HYPERPARAMETER OPTIMIZATION ==========


class OptunaRegimeOptimizer:
    """
    Expert-level hyperparameter optimization for regime detection.

    Supports parallel trial execution via joblib for significant speedup
    on multi-core systems. Each trial trains a model variant and evaluates
    on validation set.
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        input_dim=1,
        seq_len=100,
        n_classes=2,
        n_trials=100,
        timeout=3600,
        device=None,
        n_jobs=1,
        num_workers=0,
    ):
        """
        Initialize Optuna-based hyperparameter optimizer.

        Args:
            X_train, y_train: Training data arrays.
            X_val, y_val: Validation data arrays (time-based split recommended).
            input_dim: Number of input channels.
            seq_len: Sequence length per sample.
            n_classes: Number of regime classes.
            n_trials: Number of Optuna trials to run.
            timeout: Maximum optimization time in seconds.
            device: Torch device (auto-detects CUDA/MPS if None).
            n_jobs: Number of parallel jobs for trials (-1 for all cores).
            num_workers: DataLoader workers (0 for main process only).
        """
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.LongTensor(y_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.LongTensor(y_val)
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.num_workers = num_workers

        # Auto-detect best available device (CUDA > MPS > CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        logger.info(f"Using device: {self.device}, n_jobs: {n_jobs}")

        # Store best trial results
        self.best_trial = None
        self.best_model = None
        self.best_accuracy = 0.0

        # Optuna study
        self.study = None

    def create_model(
        self,
        trial,
        hidden_dim: int,
        dropout: np.float64,
        attention_heads: int,
        num_scales: int,
        use_uncertainty_head: bool,
        use_orthogonality_loss: bool,
    ):
        """
        Construct RegimeDetectionExpert with validated hyperparameters.

        The multi-scale convolutional architecture uses dilated convolutions
        to capture patterns at different temporal resolutions. Scale selection
        follows a quasi-geometric progression for efficient receptive field coverage.

        Args:
            trial: Optuna trial object (unused here, retained for interface consistency).
            hidden_dim: Latent dimension, guaranteed divisible by attention_heads.
            dropout: Regularization strength for attention and MLP layers.
            attention_heads: Number of parallel attention heads.
            num_scales: Number of dilation scales in multi-scale convolution.
            use_uncertainty_head: Whether to include epistemic uncertainty estimation.
            use_orthogonality_loss: Whether to enforce class-separable representations.

        Returns:
            RegimeDetectionExpert: Configured model on target device.

        Note:
            The divisibility constraint hidden_dim % attention_heads == 0 is
            guaranteed by the hyperparameter sampling strategy in objective().
        """
        # --------------------------------------------------------------------
        # Multi-scale dilation rates follow a quasi-geometric progression.
        # Rationale: Each scale should capture a distinct temporal granularity.
        # Scale 1: Local patterns (1-3 timesteps)
        # Scale 4: Short-term trends (4-12 timesteps)
        # Scale 8: Medium-term regime characteristics (8-24 timesteps)
        # --------------------------------------------------------------------
        scale_configs = {
            1: [1],
            2: [1, 4],
            3: [1, 2, 4],
            4: [1, 2, 4, 8],
            5: [1, 2, 3, 6, 12],
        }
        scales = scale_configs.get(num_scales, [1, 2, 4, 8])

        # Construct model with properly validated hidden_dim
        # No post-hoc modification needed since hidden_dim is already valid
        model = RegimeDetectionExpert(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_classes=self.n_classes,
            seq_len=self.seq_len,
            dropout=dropout,
            num_heads=attention_heads,  # Pass directly to constructor
        )

        # Override multi-scale if non-default configuration requested
        if scales != [1, 2, 4, 8]:
            model.multi_scale = MultiScaleResidualExpert(hidden_dim, scales=scales)

        # Optionally disable uncertainty quantification head
        if not use_uncertainty_head:
            model.uncertainty_head = nn.Identity()

        return model.to(self.device)

    def create_optimizer(self, trial, model):
        """Create optimizer with hyperparameters from trial"""
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["Adam", "AdamW", "RMSprop", "SGD"]
        )

        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:  # SGD
            momentum = trial.suggest_float("momentum", 0.8, 0.99)
            optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )

        return optimizer

    def create_scheduler(self, trial, optimizer):
        """Create learning rate scheduler"""
        scheduler_name = trial.suggest_categorical(
            "scheduler", ["ReduceLROnPlateau", "CosineAnnealing", "StepLR", "None"]
        )

        if scheduler_name == "ReduceLROnPlateau":
            factor = trial.suggest_float("reduce_factor", 0.1, 0.5)
            patience = trial.suggest_int("reduce_patience", 3, 10)
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=factor, patience=patience
            )
        elif scheduler_name == "CosineAnnealing":
            T_max = trial.suggest_int("T_max", 20, 100)
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
        elif scheduler_name == "StepLR":
            step_size = trial.suggest_int("step_size", 10, 30)
            gamma = trial.suggest_float("step_gamma", 0.1, 0.5)
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        else:
            scheduler = None

        return scheduler

    def objective(self, trial):
        """
        Optuna objective function for Bayesian hyperparameter optimization.

        The search space is carefully constructed to ensure architectural validity:
        - hidden_dim is parameterized as a multiple of attention_heads to guarantee
          divisibility, avoiding the common failure mode in multi-head attention
          where channels % num_heads != 0.

        Returns:
            np.float64: Validation accuracy on held-out temporal split.
        """
        # --------------------------------------------------------------------
        # CRITICAL FIX: Sample attention_heads FIRST, then derive hidden_dim
        # as a multiple of heads. This guarantees divisibility constraint:
        # hidden_dim = head_multiplier * attention_heads
        # Valid combinations: heads=2 -> dims in {32,64,...,256}
        #                     heads=4 -> dims in {32,64,...,256}
        #                     heads=8 -> dims in {64,128,192,256}
        # --------------------------------------------------------------------
        attention_heads = trial.suggest_categorical("attention_heads", [2, 4, 8])

        # Compute valid hidden_dim range based on attention_heads
        # Minimum: 32 (for sufficient representational capacity)
        # Maximum: 256 (to prevent overfitting on small datasets)
        min_multiplier = max(1, 32 // attention_heads)
        max_multiplier = 256 // attention_heads
        head_multiplier = trial.suggest_int(
            "head_multiplier", min_multiplier, max_multiplier
        )
        hidden_dim = head_multiplier * attention_heads

        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        num_scales = trial.suggest_int("num_scales", 1, 5)

        # Architecture choices
        use_uncertainty_head = trial.suggest_categorical(
            "use_uncertainty_head", [True, False]
        )
        use_orthogonality_loss = trial.suggest_categorical(
            "use_orthogonality_loss", [True, False]
        )

        # Loss function hyperparameters
        alpha = (
            trial.suggest_float("alpha", 0.01, 0.5, log=True)
            if use_orthogonality_loss
            else 0.0
        )
        freq_weight = trial.suggest_float("freq_weight", 0.001, 0.1, log=True)

        # Training hyperparameters - reduced epoch range for faster trials
        batch_size_suggested = trial.suggest_int("batch_size", 32, 64, step=16)
        n_epochs = trial.suggest_int("n_epochs", 15, 50)  # Reduced from 30-100

        # Create data loaders with parallel workers for faster I/O
        train_dataset = TimeSeriesDataset(self.X_train.numpy(), self.y_train.numpy())

        # CRITICAL: Cap batch size at dataset size to prevent DataLoader errors
        batch_size = min(batch_size_suggested, len(train_dataset))
        val_dataset = TimeSeriesDataset(self.X_val.numpy(), self.y_val.numpy())

        # pin_memory speeds up CPU->GPU transfer, num_workers for parallel loading
        pin_memory = self.device.type == "cuda"
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )

        # Create model
        model = self.create_model(
            trial,
            hidden_dim,
            dropout,
            attention_heads,
            num_scales,
            use_uncertainty_head,
            use_orthogonality_loss,
        )

        # Create optimizer and scheduler
        optimizer = self.create_optimizer(trial, model)
        scheduler = self.create_scheduler(trial, optimizer)

        # Create loss function
        criterion = RobustRegimeLoss(
            alpha=alpha,
            beta=0.05 if use_orthogonality_loss else 0.0,
            freq_weight=freq_weight,
            unc_weight=0.01 if use_uncertainty_head else 0.0,
        )

        # Aggressive early stopping for faster HPO trials
        # Shorter patience during search, longer patience for final training
        patience = 5
        best_val_acc = 0
        patience_counter = 0

        # Training loop with reduced overhead
        for epoch in range(n_epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss_dict = criterion(outputs, batch_y)
                loss = loss_dict["total_loss"]
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                train_loss += loss.item()

            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    preds = torch.argmax(outputs["regime_probs"], dim=-1)
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.size(0)

            val_acc = val_correct / val_total

            # Report intermediate value for pruning
            trial.report(val_acc, epoch)

            # Prune if necessary
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Update scheduler if using ReduceLROnPlateau
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    # Use training loss for ReduceLROnPlateau
                    scheduler.step(train_loss / len(train_loader))
                else:
                    scheduler.step()

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # CRITICAL: Clean up GPU memory between trials to prevent OOM
        del model, optimizer, criterion
        if scheduler is not None:
            del scheduler
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        return best_val_acc

    def optimize(self):
        """
        Run Optuna optimization with optional parallelization.

        Uses TPE sampler for efficient Bayesian optimization and MedianPruner
        for early termination of unpromising trials. Parallel execution via
        n_jobs parameter can provide significant speedup on multi-core systems.

        Returns:
            optuna.Study: Completed study with trial history.
        """
        # Aggressive pruning for faster iteration
        # n_startup_trials: random trials before pruning kicks in
        # n_warmup_steps: epochs before considering pruning
        self.study = optuna.create_study(
            direction="maximize",
            study_name="regime_detection_hyperopt",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3,  # Reduced from 10 for faster pruning
                n_warmup_steps=5,  # Reduced from 20 for earlier pruning
                interval_steps=3,  # Check more frequently
            ),
        )

        # Run optimization with parallel jobs if specified
        # n_jobs=-1 uses all available cores
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            gc_after_trial=True,
            show_progress_bar=True,
            catch=(Exception,),  # Catch exceptions to continue optimization
        )

        # Get best trial
        self.best_trial = self.study.best_trial

        print("Best trial:")
        print(f"  Value (Validation Accuracy): {self.best_trial.value:.4f}")
        print("  Params: ")
        for key, value in self.best_trial.params.items():
            print(f"    {key}: {value}")

        # Train final model with best hyperparameters
        self.train_final_model()

        return self.study

    def train_final_model(self):
        """
        Train final model using best hyperparameters from Optuna study.

        Extended training with increased epochs for production-quality model.
        Uses early stopping based on validation accuracy to prevent overfitting.

        Returns:
            RegimeDetectionExpert: Trained model with best validation performance.
        """
        print("\nTraining final model with best hyperparameters...")

        # Extract best parameters
        params = self.best_trial.params

        # Reconstruct hidden_dim from the parameterization used during optimization
        # hidden_dim = head_multiplier * attention_heads (guaranteed divisible)
        attention_heads = params["attention_heads"]
        head_multiplier = params["head_multiplier"]
        hidden_dim = head_multiplier * attention_heads

        # Create model with best parameters
        self.best_model = self.create_model(
            trial=None,
            hidden_dim=hidden_dim,
            dropout=params["dropout"],
            attention_heads=attention_heads,
            num_scales=params["num_scales"],
            use_uncertainty_head=params.get("use_uncertainty_head", True),
            use_orthogonality_loss=params.get("use_orthogonality_loss", True),
        )

        # Prepare data loaders
        batch_size = params["batch_size"]
        train_dataset = TimeSeriesDataset(self.X_train.numpy(), self.y_train.numpy())
        val_dataset = TimeSeriesDataset(self.X_val.numpy(), self.y_val.numpy())

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # Create optimizer and scheduler
        optimizer = optim.AdamW(
            self.best_model.parameters(),
            lr=params["lr"],
            weight_decay=params.get("weight_decay", 1e-4),
        )

        # Train for more epochs with best params
        n_epochs = 150  # More epochs for final training

        criterion = RobustRegimeLoss(
            alpha=params.get("alpha", 0.1),
            beta=0.05 if params.get("use_orthogonality_loss", True) else 0.0,
            freq_weight=params.get("freq_weight", 0.01),
            unc_weight=0.01 if params.get("use_uncertainty_head", True) else 0.0,
        )

        # Training loop
        best_val_acc = 0
        best_model_state = None

        for epoch in range(n_epochs):
            # Training
            self.best_model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.best_model(batch_x)
                loss_dict = criterion(outputs, batch_y)
                loss = loss_dict["total_loss"]
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.best_model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.best_model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.best_model(batch_x)
                    preds = torch.argmax(outputs["regime_probs"], dim=-1)
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.size(0)

            val_acc = val_correct / val_total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.best_model.state_dict().copy()

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3d}/{n_epochs}: "
                    f"Train Loss: {train_loss / len(train_loader):.4f}, "
                    f"Val Acc: {val_acc:.4f}"
                )

        # Load best model state
        if best_model_state is not None:
            self.best_model.load_state_dict(best_model_state)

        self.best_accuracy = best_val_acc
        print(f"\nFinal Model Validation Accuracy: {best_val_acc:.4f}")

        return self.best_model

    def get_hyperparameter_importance(self):
        """Get hyperparameter importance from Optuna study"""
        if self.study is None:
            raise ValueError("Study not yet run. Call optimize() first.")

        importance = optuna.importance.get_param_importances(self.study)

        print("Hyperparameter Importance:")
        for param, imp in importance.items():
            print(f"  {param}: {imp:.4f}")

        return importance


# Advanced ensemble approach using Optuna results
class RegimeDetectionEnsemble:
    """Ensemble of best models from Optuna optimization"""

    def __init__(self, n_models=5):
        self.n_models = n_models
        self.models = []
        self.weights = []

    def create_ensemble_from_study(
        self, study, X_train, y_train, input_dim=1, seq_len=100, n_classes=2
    ):
        """
        Create ensemble from top-performing trials in Optuna study.

        Ensemble weighting follows validation accuracy proportional weighting,
        giving higher influence to trials that achieved better generalization.

        Args:
            study: Completed Optuna study with trial results.
            X_train: Training features array.
            y_train: Training labels array.
            input_dim: Number of input channels.
            seq_len: Sequence length for model configuration.
            n_classes: Number of output classes.
        """
        # Get top trials (Pareto-optimal for multi-objective or best for single)
        top_trials = study.best_trials[: self.n_models]

        for i, trial in enumerate(top_trials):
            print(f"Training ensemble model {i + 1}/{self.n_models}...")

            # Create optimizer instance for model construction
            optimizer_helper = OptunaRegimeOptimizer(
                X_train,
                y_train,
                X_train,
                y_train,
                input_dim=input_dim,
                seq_len=seq_len,
                n_classes=n_classes,
                n_trials=0,
            )

            # Extract parameters and reconstruct hidden_dim
            params = trial.params
            attention_heads = params["attention_heads"]
            head_multiplier = params["head_multiplier"]
            hidden_dim = head_multiplier * attention_heads

            # Create model with validated parameters
            model = optimizer_helper.create_model(
                trial=None,
                hidden_dim=hidden_dim,
                dropout=params["dropout"],
                attention_heads=attention_heads,
                num_scales=params["num_scales"],
                use_uncertainty_head=params.get("use_uncertainty_head", True),
                use_orthogonality_loss=params.get("use_orthogonality_loss", True),
            )

            # Train model with abbreviated schedule
            model = self._train_single_model(model, X_train, y_train, params)

            self.models.append(model)
            self.weights.append(trial.value)  # Validation accuracy as weight

        # Normalize to probability distribution
        self.weights = np.array(self.weights) / np.sum(self.weights)

        print(f"Ensemble created with weights: {self.weights}")

    def _train_single_model(self, model, X_train, y_train, params):
        """Train a single ensemble model"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        dataset = TimeSeriesDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=params["batch_size"], shuffle=True
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params.get("weight_decay", 1e-4),
        )

        # Train for fewer epochs for ensemble
        for epoch in range(30):
            model.train()
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = nn.CrossEntropyLoss()(outputs["regime_logits"], batch_y)
                loss.backward()
                optimizer.step()

        return model

    def predict(self, X):
        """Make ensemble predictions"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X

        all_probs = []
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor.to(device))
                probs = outputs["regime_probs"]
                all_probs.append(probs.cpu() * weight)

        # Weighted average of probabilities
        ensemble_probs = torch.stack(all_probs).sum(dim=0)
        predictions = torch.argmax(ensemble_probs, dim=-1)

        # Also compute uncertainty from ensemble disagreement
        probs_tensor = torch.stack(
            [p / w for p, w in zip(all_probs, self.weights)]
        )  # Remove weight scaling
        uncertainty = torch.std(probs_tensor, dim=0).mean(dim=-1)  # Mean over classes

        return {
            "predictions": predictions,
            "probabilities": ensemble_probs,
            "uncertainty": uncertainty,
            "ensemble_agreement": 1 - uncertainty,  # Higher means more agreement
        }


# Example usage
def run_hypertuning_example():
    """Complete example of hyperparameter optimization"""
    # Generate synthetic data
    X, y = create_synthetic_data(num_samples=2000, seq_length=100, num_classes=3)

    # Time-based split (important for time series!)
    split_idx = int(0.7 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Create optimizer
    optimizer = OptunaRegimeOptimizer(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=1,
        seq_len=100,
        n_classes=3,
        n_trials=50,  # Start with fewer trials for testing
        timeout=1800,  # 30 minutes
    )

    # Run optimization
    study = optimizer.optimize()

    # Get hyperparameter importance
    importance = optimizer.get_hyperparameter_importance()

    # Train final model
    final_model = optimizer.train_final_model()

    # Create ensemble from study
    ensemble = RegimeDetectionEnsemble(n_models=3)
    ensemble.create_ensemble_from_study(
        study, X_train, y_train, input_dim=1, seq_len=100, n_classes=3
    )

    # Compare single model vs ensemble
    print("\nPerformance Comparison:")
    print(f"Single Model Accuracy: {optimizer.best_accuracy:.4f}")

    # Test ensemble
    X_test_tensor = torch.FloatTensor(X_val[:100])
    ensemble_preds = ensemble.predict(X_test_tensor)
    ensemble_acc = (
        (ensemble_preds["predictions"] == torch.LongTensor(y_val[:100])).float().mean()
    )
    print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
    print(f"Ensemble Agreement: {ensemble_preds['ensemble_agreement'].mean():.4f}")

    return study, optimizer.best_model, ensemble


# =============================================================================
# VISUALIZATION MODULE
# Clean, minimal dashboard for regime detection analysis.
# =============================================================================

# Regime color scheme and names (consistent across all plots)
REGIME_COLORS = {
    0: "#2ecc71",  # Accumulation - green
    1: "#e74c3c",  # Drawdown - red
    2: "#9b59b6",  # Crisis - purple
    3: "#3498db",  # Stable - blue
}
REGIME_NAMES = {
    0: "Accumulation",
    1: "Drawdown",
    2: "Crisis",
    3: "Stable",
}


class RegimeVisualization:
    """Minimal, clean visualization for regime detection."""

    def __init__(self):
        """Initialize with clean, minimal styling."""
        try:
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.patches import Patch

            self.sns = sns
            self.plt = plt
            self.mdates = mdates
            self.Patch = Patch
        except ImportError as e:
            raise ImportError("Install: pip install seaborn matplotlib") from e

        # Clean, minimal style
        self.sns.set_style("white")
        self.plt.rcParams.update(
            {
                "font.size": 8,
                "axes.titlesize": 9,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "figure.dpi": 150,
                "savefig.dpi": 200,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 0.5,
                "grid.linewidth": 0.3,
                "lines.linewidth": 1.0,
            }
        )

    def _get_regime_palette(self, n_classes: int) -> list:
        """Return a color palette long enough for the number of classes."""
        base_palette = [REGIME_COLORS.get(i) for i in range(n_classes)]
        fallback = self.sns.color_palette("husl", n_classes)

        palette = []
        for idx, color in enumerate(base_palette):
            palette.append(color if color is not None else fallback[idx])
        return palette

    @staticmethod
    def _calculate_regime_runs(
        preds: np.ndarray, dates: Optional[np.ndarray] = None
    ) -> list[dict[str, Any]]:
        """
        Compute contiguous regime runs with optional date alignment.

        Args:
            preds: Sequence of predicted regimes.
            dates: Optional aligned datetime values for each prediction.

        Returns:
            List of dictionaries capturing start/end, duration, and regime id.
        """
        if preds.size == 0:
            return []

        runs: list[dict[str, Any]] = []
        start_idx = 0
        current_regime = int(preds[0])

        for idx, label in enumerate(preds[1:], start=1):
            if int(label) != current_regime:
                end_idx = idx - 1
                runs.append(
                    {
                        "regime": current_regime,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "duration_steps": end_idx - start_idx + 1,
                        "start_date": pd.to_datetime(dates[start_idx])
                        if dates is not None
                        else None,
                        "end_date": pd.to_datetime(dates[end_idx])
                        if dates is not None
                        else None,
                    }
                )
                start_idx = idx
                current_regime = int(label)

        end_idx = len(preds) - 1
        runs.append(
            {
                "regime": current_regime,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "duration_steps": end_idx - start_idx + 1,
                "start_date": pd.to_datetime(dates[start_idx])
                if dates is not None
                else None,
                "end_date": pd.to_datetime(dates[end_idx])
                if dates is not None
                else None,
            }
        )
        return runs

    def plot_normalized_timeline_with_forecast(
        self,
        preds: np.ndarray,
        probs: np.ndarray,
        forecast_steps: int = 0,
        dates: Optional[np.ndarray] = None,
        save_path: str = None,
    ):
        """
        Plot normalized timeline with an optional persistence-based forecast.

        Forecast probabilities reuse the last observed distribution, making the
        horizon a persistence scenario rather than a generative forecast.
        """
        if isinstance(dates, (pd.Series, pd.Index)):
            dates = np.asarray(dates)
        if preds.size == 0 or probs.size == 0:
            fig, ax = self.plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "No predictions available", ha="center", va="center")
            ax.axis("off")
            return fig

        n_classes = probs.shape[1]
        palette = self._get_regime_palette(n_classes)

        forecast_steps = max(0, int(forecast_steps))
        history_time = np.linspace(0, 1, len(preds), endpoint=False)
        step_width = 1 / max(1, len(preds))
        forecast_time = (
            1 + np.arange(forecast_steps) * step_width
            if forecast_steps > 0
            else np.array([])
        )

        if forecast_steps > 0:
            forecast_probs = np.repeat(probs[-1][None, :], forecast_steps, axis=0)
            forecast_preds = forecast_probs.argmax(axis=1)
        else:
            forecast_probs = np.empty((0, n_classes))
            forecast_preds = np.empty(0, dtype=int)

        timeline = (
            np.concatenate([history_time, forecast_time])
            if forecast_time.size
            else history_time
        )
        all_probs = np.vstack([probs, forecast_probs]) if forecast_probs.size else probs
        all_preds = (
            np.concatenate([preds, forecast_preds]) if forecast_preds.size else preds
        )

        fig, (ax_top, ax_bottom) = self.plt.subplots(
            2,
            1,
            figsize=(8, 5),
            sharex=True,
            gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.08},
        )

        if forecast_steps > 0:
            ax_top.axvspan(
                1,
                timeline[-1] if timeline.size else 1,
                color="#f9e79f",
                alpha=0.25,
                label="Forecast horizon",
            )
            ax_top.axvline(1, color="#7f8c8d", linestyle="--", lw=1, alpha=0.7)

        ax_top.axvspan(0, 1, color="#ecf0f1", alpha=0.2, label="History")

        for regime in np.unique(all_preds):
            mask = all_preds == regime
            ax_top.scatter(
                timeline[mask],
                all_preds[mask],
                s=10,
                color=REGIME_COLORS.get(int(regime), "#7f8c8d"),
                alpha=0.85,
                label=REGIME_NAMES.get(int(regime), f"Regime {int(regime)}"),
            )

        ax_top.set_ylabel("Regime", fontsize=8)
        ax_top.set_yticks(sorted(np.unique(all_preds)))
        ax_top.set_yticklabels(
            [REGIME_NAMES.get(int(r), f"R{int(r)}") for r in np.unique(all_preds)],
            fontsize=7,
        )
        ax_top.legend(loc="upper right", fontsize=7, ncol=2, frameon=True)

        ax_bottom.stackplot(
            timeline, all_probs.T, colors=palette[:n_classes], alpha=0.8
        )
        ax_bottom.set_ylabel("Probability", fontsize=8)
        ax_bottom.set_ylim(0, 1)
        ax_bottom.axvspan(0, 1, color="#ecf0f1", alpha=0.15)
        if forecast_steps > 0:
            ax_bottom.axvspan(
                1, timeline[-1] if timeline.size else 1, color="#f9e79f", alpha=0.15
            )
        ax_bottom.set_xlabel(
            "Normalized time (0=start, 1=latest, >1=forecast)", fontsize=8
        )

        if dates is not None and len(dates) > 1:
            start_label = pd.to_datetime(dates[0]).strftime("%Y-%m-%d")
            end_label = pd.to_datetime(
                dates[min(len(dates) - 1, len(history_time) - 1)]
            ).strftime("%Y-%m-%d")
            ax_top.set_title(
                f"Regime Timeline (normalized) — {start_label} to {end_label}",
                fontsize=9,
            )
        else:
            ax_top.set_title("Regime Timeline (normalized)", fontsize=9)

        self.plt.tight_layout()
        if save_path:
            self.plt.savefig(save_path, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved normalized timeline to {save_path}")
        return fig

    def plot_regime_duration_table(
        self,
        preds: np.ndarray,
        dates: Optional[np.ndarray] = None,
        save_path: str = None,
    ):
        """
        Render a table of contiguous regimes and their durations.

        The final row highlights the current regime and how long it has persisted.
        """
        runs = self._calculate_regime_runs(preds, dates)
        fig_height = max(1.5, 0.35 * max(1, len(runs)))
        fig, ax = self.plt.subplots(figsize=(7, fig_height))
        ax.axis("off")

        if not runs:
            ax.text(0.5, 0.5, "No regime history available", ha="center", va="center")
            return fig

        rows = []
        for idx, run in enumerate(runs):
            if run["start_date"] is not None and run["end_date"] is not None:
                duration_days = (run["end_date"] - run["start_date"]).days + 1
                start_label = run["start_date"].strftime("%Y-%m-%d")
                end_label = run["end_date"].strftime("%Y-%m-%d")
                duration_label = f"{duration_days} days"
            else:
                start_label = str(run["start_idx"])
                end_label = str(run["end_idx"])
                duration_label = f"{run['duration_steps']} steps"

            rows.append(
                [
                    REGIME_NAMES.get(run["regime"], f"Regime {run['regime']}"),
                    start_label,
                    end_label,
                    duration_label,
                    "Yes" if idx == len(runs) - 1 else "",
                ]
            )

        col_labels = ["Regime", "Start", "End", "Duration", "Current"]
        table = ax.table(
            cellText=rows, colLabels=col_labels, loc="center", cellLoc="center"
        )
        table.scale(1, 1.2)
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        ax.set_title("Regime Durations", fontsize=9)

        self.plt.tight_layout()
        if save_path:
            self.plt.savefig(save_path, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved regime duration table to {save_path}")
        return fig

    def plot_optimization_history(self, study, save_path: str = None):
        """Compact optimization history plot."""
        fig, ax = self.plt.subplots(figsize=(5, 2.5))

        trials = [t for t in study.trials if t.value is not None]
        if not trials:
            return fig

        values = [t.value for t in trials]
        best_so_far = np.maximum.accumulate(values)

        ax.scatter(
            range(len(values)), values, s=15, alpha=0.5, c="#3498db", label="Trial"
        )
        ax.plot(best_so_far, c="#e74c3c", lw=1.5, label=f"Best: {max(values):.3f}")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Val Accuracy")
        ax.legend(loc="lower right", frameon=False)
        ax.set_ylim(0, 1.05)

        self.plt.tight_layout()
        if save_path:
            self.plt.savefig(save_path, bbox_inches="tight")
        return fig

    def plot_hyperparameter_importance(self, study, save_path: str = None):
        """Compact hyperparameter importance plot."""
        try:
            importance = optuna.importance.get_param_importances(study)
        except Exception:
            return None

        fig, ax = self.plt.subplots(figsize=(4, 3))

        # Top 8 parameters only
        items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8]
        params, values = zip(*items) if items else ([], [])

        # Shorten param names
        short_names = [p.replace("_", "\n") if len(p) > 12 else p for p in params]

        bars = ax.barh(range(len(values)), values, color="#3498db", height=0.6)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(short_names)
        ax.set_xlabel("Importance")
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.2 if values else 1)

        self.plt.tight_layout()
        if save_path:
            self.plt.savefig(save_path, bbox_inches="tight")
        return fig

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None
    ):
        """Compact confusion matrix."""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        n_classes = cm.shape[0]
        labels = [REGIME_NAMES.get(i, f"R{i}")[:3] for i in range(n_classes)]

        fig, ax = self.plt.subplots(figsize=(3, 2.5))

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

        # Annotate
        for i in range(n_classes):
            for j in range(n_classes):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.0%}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        self.plt.tight_layout()
        if save_path:
            self.plt.savefig(save_path, bbox_inches="tight")
        return fig

    def plot_lng_forecast(
        self,
        df: pd.DataFrame,
        model,
        seq_len: int,
        train_end_idx: int,
        save_path: str = None,
    ):
        """
        Create comprehensive LNG regime forecast visualization.

        Shows:
        - Historical data with detected regimes
        - Train/test split indicator
        - Regime probabilities as confidence bands
        - Forecast uncertainty shading
        """

        # Parse dates
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
        elif "gasDayStart" in df.columns:
            dates = pd.to_datetime(df["gasDayStart"])
        else:
            dates = pd.date_range(start="2012-01-01", periods=len(df), freq="D")

        # Get features
        tank_fullness = df["tank_fullness_lng"].values
        sendout_util = df["sendout_utilisation"].values

        # Create windows and get predictions
        n_samples = len(df) - seq_len + 1
        X = np.zeros((n_samples, 2, seq_len), dtype=np.float32)
        for i in range(n_samples):
            X[i, 0] = tank_fullness[i : i + seq_len]
            X[i, 1] = sendout_util[i : i + seq_len]

        # Normalize
        X_mean = X.mean(axis=(0, 2), keepdims=True)
        X_std = X.std(axis=(0, 2), keepdims=True) + 1e-8
        X_norm = np.clip((X - X_mean) / X_std, -3, 3)

        # Get predictions with uncertainty
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm).to(device)
            outputs = model(X_tensor)
            probs = outputs["regime_probs"].cpu().numpy()
            preds = probs.argmax(axis=1)
            uncertainty = outputs.get("uncertainty", None)
            if uncertainty is not None:
                uncertainty = uncertainty.cpu().numpy().flatten()

        # Align dates with windows (use end date of each window)
        window_dates = dates[seq_len - 1 : seq_len - 1 + n_samples].values

        # Create figure
        fig = self.plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1.5, 1, 1], hspace=0.15)

        # =====================================================================
        # Panel 1: Main time series with regime shading
        # =====================================================================
        ax1 = fig.add_subplot(gs[0])

        # Train/test split line
        train_window_end = train_end_idx - seq_len + 1
        if train_window_end > 0 and train_window_end < len(window_dates):
            ax1.axvline(
                window_dates[train_window_end],
                color="#e74c3c",
                linestyle="--",
                lw=1.5,
                alpha=0.8,
                label="Train/Test Split",
            )
            ax1.axvspan(
                window_dates[train_window_end],
                window_dates[-1],
                alpha=0.05,
                color="#e74c3c",
            )

        # Regime background shading
        for i in range(len(preds)):
            color = REGIME_COLORS.get(preds[i], "#cccccc")
            if i < len(window_dates) - 1:
                ax1.axvspan(
                    window_dates[i],
                    window_dates[i + 1],
                    alpha=0.25,
                    color=color,
                    linewidth=0,
                )

        # Plot tank fullness
        ax1.plot(
            window_dates,
            tank_fullness[seq_len - 1 : seq_len - 1 + n_samples],
            color="#2c3e50",
            lw=1.2,
            label="Tank Fullness",
            alpha=0.9,
        )

        # Secondary axis for send-out
        ax1b = ax1.twinx()
        ax1b.plot(
            window_dates,
            sendout_util[seq_len - 1 : seq_len - 1 + n_samples],
            color="#e67e22",
            lw=1.0,
            label="Send-out Util.",
            alpha=0.7,
        )
        ax1b.set_ylabel("Send-out Utilisation", fontsize=9, color="#e67e22")
        ax1b.tick_params(axis="y", labelcolor="#e67e22", labelsize=8)
        ax1b.set_ylim(0, 1)

        ax1.set_ylabel("Tank Fullness (LNG)", fontsize=9)
        ax1.set_xlim(window_dates[0], window_dates[-1])
        ax1.set_ylim(0, 1)

        # Regime legend
        handles = [
            self.Patch(
                facecolor=REGIME_COLORS[i],
                label=REGIME_NAMES[i],
                alpha=0.5,
                edgecolor="none",
            )
            for i in sorted(REGIME_COLORS.keys())
        ]
        ax1.legend(
            handles=handles,
            loc="upper left",
            frameon=True,
            ncol=4,
            fontsize=7,
            fancybox=True,
            framealpha=0.9,
        )

        ax1.set_title(
            "EU LNG Storage: Regime Detection & Forecast",
            fontsize=11,
            fontweight="bold",
        )
        ax1.tick_params(axis="x", labelbottom=False)

        # =====================================================================
        # Panel 2: Regime probabilities (confidence bands)
        # =====================================================================
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        # Stacked area for regime probabilities
        colors_stack = [REGIME_COLORS[i] for i in range(4)]
        ax2.stackplot(
            window_dates,
            probs[:, 0],
            probs[:, 1],
            probs[:, 2],
            probs[:, 3],
            colors=colors_stack,
            alpha=0.7,
            labels=["Accumulation", "Drawdown", "Crisis", "Stable"],
        )

        ax2.set_ylabel("Regime Prob.", fontsize=9)
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis="x", labelbottom=False)
        ax2.legend(loc="upper right", ncol=4, fontsize=6, frameon=True)

        # =====================================================================
        # Panel 3: Prediction confidence (1 - uncertainty)
        # =====================================================================
        ax3 = fig.add_subplot(gs[2], sharex=ax1)

        # Confidence = max probability
        confidence = probs.max(axis=1)

        # Plot confidence with fill
        ax3.fill_between(window_dates, 0, confidence, alpha=0.4, color="#3498db")
        ax3.plot(window_dates, confidence, color="#2980b9", lw=0.8)

        # Threshold line
        ax3.axhline(
            0.5, color="#e74c3c", linestyle=":", lw=1, alpha=0.7, label="50% threshold"
        )

        ax3.set_ylabel("Confidence", fontsize=9)
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis="x", labelbottom=False)

        # =====================================================================
        # Panel 4: Detected regime (discrete)
        # =====================================================================
        ax4 = fig.add_subplot(gs[3], sharex=ax1)

        # Color-code each point by regime
        for regime in range(4):
            mask = preds == regime
            if mask.any():
                ax4.scatter(
                    window_dates[mask],
                    preds[mask],
                    c=REGIME_COLORS[regime],
                    s=3,
                    alpha=0.7,
                    label=REGIME_NAMES[regime],
                )

        ax4.set_ylabel("Regime", fontsize=9)
        ax4.set_yticks([0, 1, 2, 3])
        ax4.set_yticklabels(["Acc", "Drw", "Crs", "Stb"], fontsize=7)
        ax4.set_ylim(-0.5, 3.5)
        ax4.set_xlabel("Date", fontsize=9)

        # Format x-axis dates
        import matplotlib.dates as mdates

        ax4.xaxis.set_major_locator(mdates.YearLocator())
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax4.tick_params(axis="x", rotation=0, labelsize=8)

        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, bbox_inches="tight", facecolor="white", dpi=150)
            logger.info(f"Saved LNG forecast to {save_path}")

        return fig
        if save_path:
            self.plt.savefig(save_path, bbox_inches="tight")
        return fig

    def create_unified_dashboard(
        self, study, model, X_test, y_test, features_raw=None, save_path: str = None
    ):
        """
        Create single unified dashboard with all plots.

        Layout:
        ┌─────────────────────────────────────────┐
        │     Regime Timeline (main plot)          │
        ├──────────────┬──────────────┬───────────┤
        │ Optimization │ HP Importance│ Confusion │
        └──────────────┴──────────────┴───────────┘
        """
        fig = self.plt.figure(figsize=(12, 8))

        # Grid spec for layout
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

        # Get predictions
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            outputs = model(X_tensor)
            y_pred = torch.argmax(outputs["regime_probs"], dim=-1).cpu().numpy()

        # Top row: Main regime timeline (spans all columns)
        ax_main = fig.add_subplot(gs[0, :])

        # Use last timestep of each window as feature values for plotting
        if features_raw is None:
            features_raw = X_test[:, :, -1]  # Last timestep of each window

        # Add regime-colored background
        for i in range(len(y_pred)):
            color = REGIME_COLORS.get(y_pred[i], "#cccccc")
            ax_main.axvspan(i, i + 1, alpha=0.25, color=color, linewidth=0)

        # Plot features
        if features_raw.ndim == 1:
            ax_main.plot(features_raw, color="#2c3e50", lw=0.6, alpha=0.9)
        else:
            ax_main.plot(
                features_raw[:, 0],
                color="#2c3e50",
                lw=0.6,
                label="Tank Fullness",
                alpha=0.9,
            )
            if features_raw.shape[1] > 1:
                ax2 = ax_main.twinx()
                ax2.plot(
                    features_raw[:, 1],
                    color="#e67e22",
                    lw=0.6,
                    label="Send-out",
                    alpha=0.9,
                )
                ax2.set_ylabel("Send-out Util.", fontsize=7, color="#e67e22")
                ax2.tick_params(axis="y", labelcolor="#e67e22", labelsize=6)

        ax_main.set_ylabel("Tank Fullness", fontsize=7)
        ax_main.set_xlabel("Sample Index", fontsize=7)
        ax_main.set_xlim(0, len(y_pred))

        # Accuracy annotation
        acc = (y_test == y_pred).mean()
        ax_main.set_title(
            f"EU LNG Regime Detection — Test Accuracy: {acc:.1%}", fontsize=10
        )

        # Regime legend
        handles = [
            self.Patch(facecolor=REGIME_COLORS[i], label=REGIME_NAMES[i], alpha=0.6)
            for i in sorted(REGIME_COLORS.keys())
            if i in np.unique(np.concatenate([y_test, y_pred]))
        ]
        ax_main.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            ncol=len(handles),
            fontsize=6,
        )

        # Bottom left: Optimization history
        ax_opt = fig.add_subplot(gs[1, 0])
        trials = [t for t in study.trials if t.value is not None]
        if trials:
            values = [t.value for t in trials]
            best_so_far = np.maximum.accumulate(values)
            ax_opt.scatter(range(len(values)), values, s=8, alpha=0.4, c="#3498db")
            ax_opt.plot(best_so_far, c="#e74c3c", lw=1)
            ax_opt.set_xlabel("Trial", fontsize=7)
            ax_opt.set_ylabel("Val Acc", fontsize=7)
            ax_opt.set_title("HPO Progress", fontsize=8)
            ax_opt.set_ylim(0, 1.05)

        # Bottom middle: Hyperparameter importance
        ax_hp = fig.add_subplot(gs[1, 1])
        try:
            importance = optuna.importance.get_param_importances(study)
            items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:6]
            if items:
                params, vals = zip(*items)
                short = [p[:10] for p in params]
                ax_hp.barh(range(len(vals)), vals, color="#3498db", height=0.5)
                ax_hp.set_yticks(range(len(vals)))
                ax_hp.set_yticklabels(short, fontsize=6)
                ax_hp.invert_yaxis()
                ax_hp.set_xlabel("Importance", fontsize=7)
                ax_hp.set_title("Top Hyperparams", fontsize=8)
        except Exception:
            ax_hp.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax_hp.set_title("HP Importance", fontsize=8)

        # Bottom right: Confusion matrix
        ax_cm = fig.add_subplot(gs[1, 2])
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

        n_classes = cm.shape[0]
        im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

        for i in range(n_classes):
            for j in range(n_classes):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax_cm.text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.0%}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=6,
                )

        labels = [REGIME_NAMES.get(i, f"R{i}")[:3] for i in range(n_classes)]
        ax_cm.set_xticks(range(n_classes))
        ax_cm.set_yticks(range(n_classes))
        ax_cm.set_xticklabels(labels, fontsize=6)
        ax_cm.set_yticklabels(labels, fontsize=6)
        ax_cm.set_xlabel("Predicted", fontsize=7)
        ax_cm.set_ylabel("True", fontsize=7)
        ax_cm.set_title("Confusion Matrix", fontsize=8)

        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved unified dashboard to {save_path}")

        return fig

    def create_dashboard(
        self,
        study,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        df_raw: pd.DataFrame = None,
        seq_len: int = 30,
        train_end_idx: int = None,
        save_dir: str = None,
        forecast_steps: int = DEFAULT_FORECAST_STEPS,
    ) -> None:
        """
        Generate all visualizations and save to directory.

        Creates unified dashboard plus LNG forecast if raw data provided, a
        normalized timeline with an optional forecast horizon, and a regime
        duration table summarizing persistence.
        """
        import os

        figures = {}

        # Get predictions first
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            outputs = model(X_tensor)
            predictions = torch.argmax(outputs["regime_probs"], dim=-1).cpu().numpy()
            probabilities = outputs["regime_probs"].cpu().numpy()

        window_dates = None
        if df_raw is not None:
            if "date" in df_raw.columns:
                dates = pd.to_datetime(df_raw["date"]).reset_index(drop=True)
            elif "gasDayStart" in df_raw.columns:
                dates = pd.to_datetime(df_raw["gasDayStart"]).reset_index(drop=True)
            else:
                dates = pd.date_range(start="2012-01-01", periods=len(df_raw), freq="D")

            total_windows = len(df_raw) - seq_len + 1
            aligned_dates = dates[seq_len - 1 : seq_len - 1 + total_windows]
            start_idx = train_end_idx or 0
            window_dates = np.asarray(
                aligned_dates[start_idx : start_idx + len(predictions)]
            )

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # 1. Main LNG Forecast plot (if raw data available)
        if df_raw is not None:
            fig = self.plot_lng_forecast(
                df=df_raw,
                model=model,
                seq_len=seq_len,
                train_end_idx=train_end_idx or int(0.7 * len(df_raw)),
                save_path=os.path.join(save_dir, "lng_forecast.png")
                if save_dir
                else None,
            )
            figures["lng_forecast"] = fig

        # 2. Unified dashboard
        if save_dir:
            fig = self.create_unified_dashboard(
                study,
                model,
                X_test,
                y_test,
                save_path=os.path.join(save_dir, "regime_dashboard.png"),
            )
            figures["regime_dashboard"] = fig

        # 3. Normalized timeline with optional forecast horizon
        fig = self.plot_normalized_timeline_with_forecast(
            preds=predictions,
            probs=probabilities,
            forecast_steps=forecast_steps,
            dates=window_dates,
            save_path=os.path.join(save_dir, "normalized_forecast.png")
            if save_dir
            else None,
        )
        figures["normalized_forecast"] = fig

        # 4. Regime duration summary table
        fig = self.plot_regime_duration_table(
            preds=predictions,
            dates=window_dates,
            save_path=os.path.join(save_dir, "regime_duration_table.png")
            if save_dir
            else None,
        )
        figures["regime_duration_table"] = fig

        # 5. Individual plots
        fig = self.plot_optimization_history(
            study,
            save_path=os.path.join(save_dir, "optimization_history.png")
            if save_dir
            else None,
        )
        figures["optimization_history"] = fig

        fig = self.plot_hyperparameter_importance(
            study,
            save_path=os.path.join(save_dir, "hyperparameter_importance.png")
            if save_dir
            else None,
        )
        if fig:
            figures["hyperparameter_importance"] = fig

        fig = self.plot_confusion_matrix(
            y_test,
            predictions,
            save_path=os.path.join(save_dir, "confusion_matrix.png")
            if save_dir
            else None,
        )
        figures["confusion_matrix"] = fig

        logger.info(f"Dashboard created with {len(figures)} visualizations")
        return figures


# Run the example if executed directly
if __name__ == "__main__":
    import argparse
    import multiprocessing
    import os

    # CRITICAL: Set spawn method for macOS/Windows compatibility
    # Must be called before any other multiprocessing code
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    parser = argparse.ArgumentParser(description="Regime Detection with Optuna HPO")
    parser.add_argument(
        "--data",
        type=str,
        default="synthetic",
        help="Data source: 'synthetic' or path to CSV (e.g., 'eu_lng_snapshot.csv')",
    )
    parser.add_argument("--seq-len", type=int, default=30, help="Sequence length")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna trials")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=DEFAULT_FORECAST_STEPS,
        help="Number of normalized forecast steps to display on the timeline",
    )
    args = parser.parse_args()

    print("Starting hyperparameter optimization for regime detection...")
    print("=" * 60)

    # Load data based on argument
    df_raw = None  # Will store raw dataframe for LNG visualization
    if args.data == "synthetic":
        print("Using synthetic data...")
        X, y = create_synthetic_data(
            num_samples=500, seq_length=args.seq_len, num_classes=2
        )
        n_classes = 2
    elif os.path.exists(args.data):
        print(f"Loading real data from {args.data}...")
        # Load raw dataframe for visualization
        df_raw = pd.read_csv(args.data)
        X, y = load_lng_data(csv_path=args.data, seq_length=args.seq_len)
        n_classes = len(np.unique(y))
        print(f"Loaded {len(X)} samples with {n_classes} regime classes")
    else:
        raise FileNotFoundError(f"Data file not found: {args.data}")

    # Time-based train/val/holdout split (no lookahead bias)
    train_idx, val_idx, hold_idx = time_series_split_with_holdout(len(X))
    X_train, X_val, X_hold = X[train_idx], X[val_idx], X[hold_idx]
    y_train, y_val, y_hold = y[train_idx], y[val_idx], y[hold_idx]

    # Report available CPU cores
    n_cores = multiprocessing.cpu_count()
    print(f"Detected {n_cores} CPU cores")

    input_dim = X_train.shape[1]  # Number of channels/features
    seq_len = X_train.shape[2]  # Sequence length

    print(f"\nData shape: {X_train.shape}")
    print(f"Input dim (channels): {input_dim}")
    print(f"Sequence length: {seq_len}")
    print(f"Number of classes: {n_classes}")
    print(
        f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Holdout samples: {len(X_hold)}"
    )

    optimizer = OptunaRegimeOptimizer(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=input_dim,
        seq_len=seq_len,
        n_classes=n_classes,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=1,  # Set to -1 for all cores, 1 for sequential
        num_workers=0,  # DataLoader workers (0 for macOS compatibility)
    )

    study = optimizer.optimize()
    print("\nTest completed successfully!")

    # Generate and save visualizations
    print("\nGenerating visualizations...")
    viz = RegimeVisualization()

    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Generate dashboard with all plots (including LNG forecast if real data)
    figures = viz.create_dashboard(
        study=study,
        model=optimizer.best_model,
        X_test=X_val,
        y_test=y_val,
        df_raw=df_raw,
        seq_len=args.seq_len,
        train_end_idx=len(X_train),
        save_dir=output_dir,
        forecast_steps=args.forecast_steps,
    )

    # Basic holdout evaluation with smoothing
    with torch.no_grad():
        device = next(optimizer.best_model.parameters()).device
        preds_hold = (
            torch.argmax(
                optimizer.best_model(torch.FloatTensor(X_hold).to(device))[
                    "regime_probs"
                ],
                dim=-1,
            )
            .cpu()
            .numpy()
        )
    preds_hold = apply_sticky_regime(
        preds_hold, min_duration=DEFAULT_MIN_REGIME_DURATION
    )
    cp_metrics = change_point_metrics(y_hold, preds_hold)
    print(
        f"\nHoldout change metrics: precision={cp_metrics['precision']:.3f}, "
        f"recall={cp_metrics['recall']:.3f}, "
        f"latency={cp_metrics['latency']:.2f}"
    )

    print(f"\nSaved {len(figures)} visualizations to '{output_dir}/' directory")
    print("Files created:")
    for name in figures.keys():
        print(f"  - {output_dir}/{name}.png")
