"""
 (c) Copyright 2023
 All rights reserved
 Programs written by Yasser Abduallah
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA

 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.

 Alternative transformer-based model with improved capacity for time-series classification.
 @author: Yasser Abduallah (modified)

 PyTorch implementation by: Antanas Zilinskas
"""

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import math
import json
import os
import shutil
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Enable mixed precision if available
if torch.cuda.is_available():
    print("CUDA available: enabling mixed precision training")
    use_amp = True
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("MPS (Apple Silicon) available: enabling mixed precision training")
    use_amp = True
    device = torch.device("mps")
else:
    print("GPU not available: using CPU")
    use_amp = False
    device = torch.device("cpu")

# Set mixed precision policies
if use_amp:
    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler()

# Import additional components for advanced optimizers and schedulers

# Set random seed for reproducibility across PyTorch operations


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior
    if hasattr(torch, "set_deterministic"):
        torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Call set_seed to maintain consistency between runs
set_seed(42)

# -----------------------------
# Custom TSS (True Skill Statistic) Metric
# -----------------------------


class TrueSkillStatisticMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # Convert to class indices if they're one-hot encoded
        if y_true.dim() > 1 and y_true.shape[1] > 1:
            y_true = torch.argmax(y_true, dim=1)
        if y_pred.dim() > 1 and y_pred.shape[1] > 1:
            y_pred = torch.argmax(y_pred, dim=1)

        # Cast to boolean for logical operations
        y_true_bool = y_true == 1
        y_pred_bool = y_pred == 1

        # Update confusion matrix elements
        self.true_positives += torch.logical_and(y_true_bool, y_pred_bool).sum().item()
        self.true_negatives += (
            torch.logical_and(~y_true_bool, ~y_pred_bool).sum().item()
        )
        self.false_positives += (
            torch.logical_and(~y_true_bool, y_pred_bool).sum().item()
        )
        self.false_negatives += (
            torch.logical_and(y_true_bool, ~y_pred_bool).sum().item()
        )

    def compute(self) -> float:
        """Compute the TSS (True Skill Statistic)"""
        sensitivity = self.true_positives / (
            self.true_positives + self.false_negatives + 1e-8
        )
        specificity = self.true_negatives / (
            self.true_negatives + self.false_positives + 1e-8
        )
        return sensitivity + specificity - 1.0


# -----------------------------
# Positional Encoding Layer
# -----------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, embed_dim: int):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim)
        )

        # Initialize encoding buffer
        pe = torch.zeros(1, max_len, embed_dim)

        # Apply sine to even indices and cosine to odd indices
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # Register as a buffer (not a parameter)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input tensor"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].type_as(x)


# -----------------------------
# Improved Transformer Block with Batch Normalization and Residual Connections
# -----------------------------


class TransformerBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.2
    ):
        super(TransformerBlock, self).__init__()

        # Multi-head attention layer
        self.att = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate, batch_first=True
        )

        # Feed-forward network with GELU activation - EXACT MATCH to TensorFlow
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embed_dim),
        )

        # Layer normalization and dropout
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Initialize weights using TensorFlow-like initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using TensorFlow-like initialization (Glorot/Xavier uniform)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Pure Glorot/Xavier uniform initialization
                nn.init.xavier_uniform_(m.weight)

                # TensorFlow initializes biases to zeros by default
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        # Set dropout training mode
        self.dropout1.train(training)
        self.dropout2.train(training)

        # Multi-head attention with residual connection and layer norm
        attn_output, _ = self.att(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network with residual connection and layer norm
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


# -----------------------------
# Focal Loss Implementation (TensorFlow-compatible version)
# -----------------------------


class CategoricalFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super(CategoricalFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        TensorFlow-compatible focal loss implementation for multi-class classification

        Args:
            y_pred: Prediction probabilities (batch_size, num_classes)
            y_true: One-hot encoded ground truth (batch_size, num_classes)

        Returns:
            Loss value
        """
        # Apply softmax if inputs are logits
        if not (y_pred.sum(dim=1) - 1.0).abs().max() < 1e-3:
            y_pred = F.softmax(y_pred, dim=1)

        # Clip predictions for numerical stability (matching TensorFlow's epsilon of 1e-7)
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

        # Standard categorical cross-entropy loss
        ce_loss = -torch.sum(y_true * torch.log(y_pred), dim=1)

        # Get probability of true class for each sample
        p_t = torch.sum(y_true * y_pred, dim=1)

        # Apply modulating factor with gamma
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)

        # Apply alpha weighting (matching TensorFlow implementation)
        alpha_weight = torch.sum(
            y_true * self.alpha + (1 - y_true) * (1 - self.alpha), dim=1
        )
        focal_loss = alpha_weight * modulating_factor * ce_loss

        # Return the mean loss over the batch
        return torch.mean(focal_loss)


# -----------------------------
# Improved SolarKnowledge Model Class
# -----------------------------


class SolarKnowledgeModel(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        embed_dim: int = 128,  # Exact match to TensorFlow model
        num_heads: int = 4,  # Exact match to TensorFlow model
        ff_dim: int = 256,  # Exact match to TensorFlow model
        num_transformer_blocks: int = 6,  # Changed from 4 to 6 to exactly match TensorFlow model
        dropout_rate: float = 0.2,  # Exact match to TensorFlow model
        num_classes: int = 2,
    ):
        """
        PyTorch implementation of SolarKnowledge Transformer model
        IMPORTANT: This implementation exactly matches the original TensorFlow architecture
        """
        super(SolarKnowledgeModel, self).__init__()

        # Save input dimensions
        self.timesteps, self.features = input_shape
        self.input_shape = (
            None,
            self.timesteps,
            self.features,
        )  # Mimic TensorFlow's batch-included shape

        # Project input features to embedding dimension
        self.embedding = nn.Linear(self.features, embed_dim)
        self.layernorm_input = nn.LayerNorm(embed_dim, eps=1e-6)
        self.input_dropout = nn.Dropout(dropout_rate)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            max_len=self.timesteps, embed_dim=embed_dim
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
                for _ in range(num_transformer_blocks)
            ]
        )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head - EXACT MATCH to TensorFlow
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(embed_dim, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(128, num_classes)

        # Regularization
        self.l1_regularizer = 1e-5  # Match TensorFlow value
        self.l2_regularizer = 1e-4  # Match TensorFlow value

        # Initialize weights matching TensorFlow defaults
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using TensorFlow-like initialization (Glorot/Xavier uniform)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Pure Glorot/Xavier uniform initialization
                nn.init.xavier_uniform_(m.weight)

                # TensorFlow initializes biases to zeros by default
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _l1_l2_regularization(self):
        """Calculate L1 and L2 regularization loss"""
        l1_loss = 0.0
        l2_loss = 0.0

        for name, param in self.named_parameters():
            if "weight" in name:
                l1_loss += torch.sum(torch.abs(param))
                l2_loss += torch.sum(param**2)

        return self.l1_regularizer * l1_loss + self.l2_regularizer * l2_loss

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Forward pass through the model

        Args:
            x: Input tensor of shape (batch_size, timesteps, features)
            training: Whether to apply dropout during inference (for MC dropout)

        Returns:
            Model output of shape (batch_size, num_classes)
        """
        # Set all dropout layers to training mode if requested
        if training:
            self.train()
        else:
            self.eval()
            # Keep dropout active for MC dropout if training flag is True
            self.input_dropout.train(training)
            self.dropout1.train(training)
            self.dropout2.train(training)

        batch_size = x.size(0)

        # Project to embedding dimension
        x = self.embedding(x)
        x = self.layernorm_input(x)
        x = self.input_dropout(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, training)

        # Global average pooling (transpose to get channel dimension for pooling)
        x = x.transpose(1, 2)
        x = self.global_avg_pool(x).squeeze(-1)

        # Apply dense layers with GELU activation - MATCH TensorFlow
        x = self.dropout1(x)
        x = self.dense1(x)
        x = F.gelu(x)

        x = self.dropout2(x)
        x = self.classifier(x)

        # Apply softmax activation
        return F.softmax(x, dim=1)

    # Add a save_weights method for TensorFlow API compatibility
    def save_weights(self, filepath: str):
        """Save model weights to the specified filepath"""
        torch.save(self.state_dict(), filepath)


class SolarKnowledge:
    """
    PyTorch implementation of the SolarKnowledge model with training and evaluation utilities
    """

    def __init__(self, early_stopping_patience: int = 5):  # Match TensorFlow
        self.model_name = "SolarKnowledge"
        self.model = None
        self.early_stopping_patience = early_stopping_patience
        self.input_shape = None
        self.max_grad_norm = 1.0  # Gradient clipping norm
        self.gradient_accumulation_steps = (
            1  # No gradient accumulation - match TensorFlow
        )

        # Initialize gradient scaler for mixed precision training
        if (
            use_amp
            and hasattr(torch.cuda, "amp")
            and hasattr(torch.cuda.amp, "GradScaler")
        ):
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def build_base_model(
        self,
        input_shape: Tuple[int, int],
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_transformer_blocks: int = 6,  # Changed from 4 to 6 to exactly match TensorFlow model
        dropout_rate: float = 0.2,
        num_classes: int = 2,
    ):
        """
        Build a transformer-based model for time-series classification.

        Args:
            input_shape: tuple (timesteps, features)
        """
        self.input_shape = input_shape

        # Create the PyTorch model
        self.model = SolarKnowledgeModel(
            input_shape=input_shape,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
        )

        # Move model to the appropriate device
        self.model.to(device)

        return self.model

    def summary(self):
        """Print a summary of the model architecture"""
        if self.model is not None:
            print(self.model)
            # Calculate number of parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f"Total parameters: {total_params}")
            print(f"Trainable parameters: {trainable_params}")
        else:
            print("Model is not built yet!")

    def compile(
        self,
        loss: str = "categorical_crossentropy",
        metrics: List[str] = ["accuracy"],
        learning_rate: float = 1e-4,  # Match TensorFlow's default
        weight_decay: float = 0.0,  # No weight decay in TensorFlow Adam
        use_focal_loss: bool = True,
    ):
        """
        Configure the model for training.

        Args:
            loss: Loss function to use. If use_focal_loss is True, this will be overridden.
            metrics: List of metrics to track
            learning_rate: Learning rate for the optimizer (1e-4 to match TensorFlow)
            weight_decay: Weight decay for regularization (0.0 to match TensorFlow)
            use_focal_loss: Whether to use focal loss (better for imbalanced data)
        """
        if self.model is None:
            raise ValueError("Model is not built yet. Call build_base_model first.")

        # Create Adam optimizer to match TensorFlow exactly
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-7,  # Match TensorFlow default epsilon
            betas=(0.9, 0.999),  # Adam beta parameters
        )

        # Store the learning rate for schedulers
        self.learning_rate = learning_rate

        # Set loss function
        if use_focal_loss:
            self.loss_fn = CategoricalFocalLoss(gamma=2.0, alpha=0.25)
            print("Using Categorical Focal Loss for rare event awareness")
        else:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Initialize metric trackers
        self.metrics = {}
        if "accuracy" in metrics:
            self.metrics["accuracy"] = (
                lambda preds, targets: (
                    torch.argmax(preds, dim=1) == torch.argmax(targets, dim=1)
                )
                .float()
                .mean()
                .item()
            )

        # Add TSS metric
        self.metrics["tss"] = TrueSkillStatisticMetric()

    def _prepare_batch(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convert numpy arrays to PyTorch tensors and move them to the right device"""
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
            return X_tensor, y_tensor
        return X_tensor

    def _train_epoch(self, data_loader, class_weight=None):
        """Train for one epoch and return metrics"""
        self.model.train()
        device = next(self.model.parameters()).device

        # Track metrics
        running_loss = 0.0
        running_metric_sums = {metric: 0.0 for metric in self.metrics}
        num_batches = 0

        # Convert class weights to tensor if provided
        weight_tensor = None
        if class_weight is not None:
            if isinstance(class_weight, dict):
                # Convert class weight dict to tensor format expected by PyTorch
                weight_list = [
                    class_weight.get(i, 1.0)
                    for i in range(max(class_weight.keys()) + 1)
                ]
                weight_tensor = torch.FloatTensor(weight_list).to(device)
            else:
                weight_tensor = torch.FloatTensor(class_weight).to(device)

        # Progress bar for training
        from tqdm import tqdm

        progress_bar = tqdm(data_loader, desc="Training")

        for i, (X_batch, y_batch) in enumerate(progress_bar):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass with mixed precision if available
            if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                with torch.cuda.amp.autocast():
                    outputs = self.model(X_batch)

                    # Calculate loss with optional class weights
                    if weight_tensor is not None and hasattr(self.loss_fn, "weight"):
                        self.loss_fn.weight = weight_tensor

                    loss = self.loss_fn(outputs, y_batch)
            else:
                outputs = self.model(X_batch)

                # Calculate loss with optional class weights
                if weight_tensor is not None and hasattr(self.loss_fn, "weight"):
                    self.loss_fn.weight = weight_tensor

                loss = self.loss_fn(outputs, y_batch)

            # Add L1 + L2 regularization directly to the loss (matching TensorFlow)
            loss += self.model._l1_l2_regularization()

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass - with mixed precision if available
            if (
                hasattr(torch.cuda, "amp")
                and hasattr(torch.cuda.amp, "GradScaler")
                and self.scaler is not None
            ):
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights every gradient_accumulation_steps
            if (i + 1) % self.gradient_accumulation_steps == 0:
                if (
                    hasattr(torch.cuda, "amp")
                    and hasattr(torch.cuda.amp, "GradScaler")
                    and self.scaler is not None
                ):
                    # Unscale gradients for clipping
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                # Optimizer step with mixed precision if available
                if (
                    hasattr(torch.cuda, "amp")
                    and hasattr(torch.cuda.amp, "GradScaler")
                    and self.scaler is not None
                ):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Calculate metrics (unscaled)
            with torch.no_grad():
                batch_loss = (
                    loss.item() * self.gradient_accumulation_steps
                )  # Unscale the loss
                running_loss += batch_loss

                # Calculate additional metrics
                for metric_name, metric_fn in self.metrics.items():
                    if hasattr(metric_fn, "update"):
                        # For custom metrics like TSS that have an update method
                        metric_fn.update(outputs, y_batch)
                        # Will get the actual value at the end of epoch
                        metric_value = 0.0
                    else:
                        # For simple function metrics like accuracy
                        metric_value = metric_fn(outputs, y_batch)
                        if isinstance(metric_value, torch.Tensor):
                            metric_value = metric_value.item()
                    running_metric_sums[metric_name] += metric_value

                # Update progress bar
                progress_metrics = {
                    "loss": running_loss / (num_batches + 1),
                }

                for metric_name, metric_fn in self.metrics.items():
                    if hasattr(metric_fn, "compute"):
                        # For custom metrics like TSS
                        progress_metrics[metric_name] = metric_fn.compute()
                    else:
                        # For simple metrics
                        progress_metrics[metric_name] = running_metric_sums[
                            metric_name
                        ] / (num_batches + 1)

                progress_bar.set_postfix(**progress_metrics)

            num_batches += 1

        # Calculate epoch metrics
        metrics = {"loss": running_loss / num_batches}

        for metric_name, metric_fn in self.metrics.items():
            if hasattr(metric_fn, "compute"):
                # For custom metrics like TSS that have a compute method
                metrics[metric_name] = metric_fn.compute()
            else:
                # For simple metrics, use the running average
                metrics[metric_name] = running_metric_sums[metric_name] / num_batches

        return metrics

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        epochs: int = 100,
        verbose: int = 2,
        batch_size: int = 512,
        class_weight: Optional[Dict[int, float]] = None,
        callbacks: Optional[Dict] = None,
    ):
        """
        Train the model with optional class weights for imbalanced data.

        Args:
            X_train, y_train: Training data
            X_valid, y_valid: Validation data (optional)
            epochs: Number of training epochs
            verbose: Verbosity level
            batch_size: Batch size for training
            class_weight: Optional dictionary mapping class indices to weights
            callbacks: Optional dictionary of callbacks
        """
        if self.model is None:
            raise ValueError("Model is not built. Call build_base_model first.")

        # Create a simple callback dictionary if not provided
        if callbacks is None:
            callbacks = {}

        # Create ReduceLROnPlateau scheduler to match TensorFlow's behavior
        # In TensorFlow this is: ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
        lr_scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )

        # Convert numpy arrays to PyTorch datasets
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device.type != "cpu" else False,
        )

        # Set up validation data if provided
        val_loader = None
        if X_valid is not None and y_valid is not None:
            val_dataset = TensorDataset(
                torch.tensor(X_valid, dtype=torch.float32),
                torch.tensor(y_valid, dtype=torch.float32),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if device.type != "cpu" else False,
            )

        # Initialize early stopping
        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        # Initialize history dictionary
        history = {
            "loss": [],
            "accuracy": [],
            "tss": [] if "tss" in self.metrics else None,
            "lr": [],  # Track learning rate
        }

        # If class_weight is not provided but we want to handle rare events,
        # create a default class weight that emphasizes the positive class
        if class_weight is None:
            # Default weight for imbalanced binary classification
            class_weight = {0: 1.0, 1: 10.0}
            print(f"Using default class weights: {class_weight}")

        # Training loop - matching TensorFlow behavior
        for epoch in range(epochs):
            if verbose > 0:
                print(f"\nEpoch {epoch+1}/{epochs}")

            # Train for one epoch
            train_metrics = self._train_epoch(train_loader, class_weight)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history
            for key, value in train_metrics.items():
                if key in history and history[key] is not None:
                    history[key].append(value)

            # Add learning rate to history
            history["lr"].append(current_lr)

            # Print metrics
            if verbose > 0:
                metrics_str = " - ".join(
                    [f"{k}: {v:.4f}" for k, v in train_metrics.items()]
                )
                print(f"Training: {metrics_str} - lr: {current_lr:.6f}")

            # Apply learning rate scheduler using training loss (matches TensorFlow)
            lr_scheduler.step(train_metrics["loss"])

            # Validate if validation data is provided (for monitoring only)
            if val_loader is not None:
                val_metrics = self._evaluate(val_loader)

                if verbose > 0:
                    val_metrics_str = " - ".join(
                        [f"val_{k}: {v:.4f}" for k, v in val_metrics.items()]
                    )
                    print(f"Validation: {val_metrics_str}")

            # TensorFlow early stopping monitors training loss exactly like in Keras
            current_loss = train_metrics["loss"]

            # Check if this is the best model so far (based on training loss)
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                # Save the model state
                best_model_state = {
                    k: v.cpu().detach() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if verbose > 0:
                    print(
                        f"EarlyStopping: {patience_counter}/{self.early_stopping_patience}"
                    )

                # Check if we should stop training
                if patience_counter >= self.early_stopping_patience:
                    if verbose > 0:
                        print("Early stopping triggered")
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(
                {k: v.to(device) for k, v in best_model_state.items()}
            )

        return history

    def _evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on the provided data"""
        self.model.eval()

        # Reset metrics
        for name, metric in self.metrics.items():
            if hasattr(metric, "reset"):
                metric.reset()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                # Update metrics
                total_loss += loss.item()

                # Update accuracy
                predicted = torch.argmax(outputs, dim=1)
                target_classes = torch.argmax(targets, dim=1)
                correct += (predicted == target_classes).sum().item()
                total += targets.size(0)

                # Update TSS metric
                if "tss" in self.metrics:
                    self.metrics["tss"].update(outputs, targets)

        # Compute eval metrics
        eval_metrics = {
            "loss": total_loss / len(data_loader),
            "accuracy": correct / total,
        }

        # Add TSS
        if "tss" in self.metrics:
            eval_metrics["tss"] = self.metrics["tss"].compute()

        return eval_metrics

    def predict(
        self, X_test: np.ndarray, batch_size: int = 1024, verbose: int = 0
    ) -> np.ndarray:
        """Standard prediction - no MC dropout"""
        self.model.eval()

        # Convert to PyTorch tensors
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_dataset = TensorDataset(test_tensor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device.type != "cpu" else False,
        )

        # Make predictions
        predictions = []
        with torch.no_grad():
            for (inputs,) in test_loader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu().numpy())

        # Concatenate all batch predictions
        return np.concatenate(predictions, axis=0)

    def mc_predict(
        self,
        X_test: np.ndarray,
        n_passes: int = 20,
        batch_size: int = 1024,
        verbose: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo dropout prediction - keeps dropout active during inference
        to get uncertainty estimates.

        Args:
            X_test: Input data
            n_passes: Number of forward passes with dropout
            batch_size: Batch size for prediction
            verbose: Verbosity level

        Returns:
            mean_preds: Mean of the predictions across all passes
            std_preds: Standard deviation of predictions (uncertainty)
        """
        if verbose > 0:
            print(f"Performing {n_passes} MC dropout passes...")

        # Convert to PyTorch tensors
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_dataset = TensorDataset(test_tensor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device.type != "cpu" else False,
        )

        # List to store all predictions
        all_preds = []

        # First set model to eval mode to handle BatchNorm layers correctly
        self.model.eval()

        # Selectively enable dropout layers (matching TensorFlow behavior)
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train(True)  # Only set dropout layers to training mode

        # Perform multiple forward passes with dropout enabled
        for i in range(n_passes):
            if verbose > 0 and i % 5 == 0:
                print(f"MC pass {i+1}/{n_passes}")

            # Disable gradient computation
            with torch.no_grad():
                pass_preds = []

                for (inputs,) in test_loader:
                    inputs = inputs.to(device)

                    # Forward pass with dropout enabled
                    outputs = self.model(inputs, training=True)
                    pass_preds.append(outputs.cpu().numpy())

                # Concatenate batch predictions for this pass
                pass_predictions = np.concatenate(pass_preds, axis=0)
                all_preds.append(pass_predictions)

        # Convert to numpy array (n_passes, n_samples, n_classes)
        all_preds = np.array(all_preds)

        # Calculate mean and std across passes
        mean_preds = np.mean(all_preds, axis=0)
        std_preds = np.std(all_preds, axis=0)

        return mean_preds, std_preds

    def save_weights(
        self,
        flare_class: Optional[str] = None,
        w_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """Save model weights and metadata"""
        if w_dir is None and flare_class is None:
            print("You must specify flare_class or w_dir to save the model weights.")
            exit()

        if w_dir is None:
            weight_dir = os.path.join("models", self.model_name, str(flare_class))
        else:
            weight_dir = w_dir

        if os.path.exists(weight_dir):
            shutil.rmtree(weight_dir)

        os.makedirs(weight_dir)

        if verbose:
            print("Saving model weights to directory:", weight_dir)

        weight_file = os.path.join(weight_dir, "model_weights.pt")
        torch.save(self.model.state_dict(), weight_file)

        # Generate a timestamp for this model version
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        metadata = {
            "timestamp": timestamp,
            "model_name": self.model_name,
            "flare_class": flare_class,
            "uses_focal_loss": True,
            "mc_dropout_enabled": True,
            "framework": "pytorch",
        }

        # Save metadata
        with open(os.path.join(weight_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def load_weights(
        self,
        flare_class: Optional[str] = None,
        w_dir: Optional[str] = None,
        timestamp: Optional[str] = None,
        verbose: bool = True,
    ):
        """Load model weights"""
        if w_dir is None and flare_class is None:
            print("You must specify flare_class or w_dir to load the model weights.")
            exit()

        if w_dir is None:
            weight_dir = os.path.join("models", self.model_name, str(flare_class))
        else:
            weight_dir = w_dir

        if verbose:
            print("Loading weights from model dir:", weight_dir)

        if not os.path.exists(weight_dir):
            print("Model weights directory:", weight_dir, "does not exist!")
            exit()

        if self.model is None:
            print("You must build the model first before loading weights.")
            exit()

        # If a specific timestamp is requested, try to find that version
        if timestamp:
            # Logic for loading a specific timestamped version would go here
            pass

        filepath = os.path.join(weight_dir, "model_weights.pt")
        self.model.load_state_dict(torch.load(filepath, map_location=device))

    def load_model(
        self,
        input_shape: Tuple[int, int],
        flare_class: str,
        w_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """Build model and load weights"""
        self.build_base_model(input_shape)
        self.compile()
        self.load_weights(flare_class, w_dir=w_dir, verbose=verbose)

    def get_model(self):
        """Return the underlying PyTorch model"""
        return self.model

    def update_results(self, metrics_dict: Dict):
        """Update model metadata with test results"""
        # This function can be expanded to save metrics to the model's metadata
        pass


# Add a monkey patch function to the SolarKnowledgeModel class if it doesn't have save_weights method


def add_compatibility_methods():
    """Add TensorFlow-like methods to PyTorch models for better compatibility"""
    # Check if SolarKnowledgeModel class has save_weights method
    if not hasattr(SolarKnowledgeModel, "save_weights"):

        def save_weights(self, filepath):
            """Save model weights to the specified filepath - TensorFlow compatibility method"""
            torch.save(self.state_dict(), filepath)

        # Add the method to the class
        SolarKnowledgeModel.save_weights = save_weights


# Apply compatibility patches when module is imported
add_compatibility_methods()

if __name__ == "__main__":
    # Example usage for debugging: build, compile, and show summary.
    # For example, input_shape is (timesteps, features) e.g., (100, 14)
    example_input_shape = (100, 14)
    model_instance = SolarKnowledge(early_stopping_patience=5)
    model_instance.build_base_model(example_input_shape)
    model_instance.compile(use_focal_loss=True)
    model_instance.summary()

    # Test MC dropout prediction
    X_test = np.random.random((10, 100, 14))
    mean_preds, std_preds = model_instance.mc_predict(X_test, n_passes=5, verbose=1)
    print(f"Mean predictions shape: {mean_preds.shape}")
    print(f"Std predictions shape: {std_preds.shape}")
