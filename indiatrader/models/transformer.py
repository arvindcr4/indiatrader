"""
Transformer-based models for time series forecasting (PatchTST implementation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import os
import json
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    PYTORCH_LIGHTNING_AVAILABLE = True
except ImportError:
    PYTORCH_LIGHTNING_AVAILABLE = False
    
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class PatchTST(nn.Module):
    """
    Patch Time Series Transformer for sequence-to-one prediction.
    
    As described in the paper:
    "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
    """
    
    def __init__(self, 
                input_dim: int,
                output_dim: int,
                patch_len: int = 16,
                stride: int = 8,
                d_model: int = 128,
                n_heads: int = 8,
                n_layers: int = 3,
                dropout: float = 0.1,
                activation: str = "gelu",
                context_length: int = 512):
        """
        Initialize PatchTST model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output dimensions (e.g., 1 for regression, 2+ for classification)
            patch_len: Length of each patch
            stride: Stride between patches
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
            context_length: Length of input sequence
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.context_length = context_length
        
        # Calculate number of patches
        num_patches = (context_length - patch_len) // stride + 2  # +2 for overlap handling
        self.num_patches = num_patches
        
        # Patch embedding layer
        self.patch_embedding = nn.Linear(patch_len, d_model)
        
        # Channel embedding layer (channel = each input feature)
        self.channel_embedding = nn.Parameter(torch.randn(1, input_dim, d_model))
        
        # Position embedding
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output head
        self.output_layer = nn.Sequential(
            nn.Linear(d_model * input_dim, d_model),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_model, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, context_length, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len, input_dim = x.size()
        
        # Patch extraction
        x_patch = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            x_patch.append(x[:, i:i+self.patch_len, :])
        
        # Handle case where sequence doesn't divide evenly
        if len(x_patch) < self.num_patches:
            # Add the last patch
            last_patch = x[:, -self.patch_len:, :]
            x_patch.append(last_patch)
        
        # Truncate to expected number of patches if we have too many
        x_patch = x_patch[:self.num_patches]
        
        # Stack patches
        x_patch = torch.stack(x_patch, dim=1)  # (batch_size, num_patches, patch_len, input_dim)
        
        # Reshape for patch embedding
        x_patch = x_patch.reshape(batch_size, self.num_patches, input_dim, self.patch_len)
        
        # Apply patch embedding
        x_patch = self.patch_embedding(x_patch)  # (batch_size, num_patches, input_dim, d_model)
        
        # Add channel embedding
        x_patch = x_patch + self.channel_embedding  # (batch_size, num_patches, input_dim, d_model)
        
        # Transpose to get dimensions (batch_size, num_patches, input_dim, d_model)
        x_patch = x_patch.permute(0, 1, 2, 3)
        
        # Reshape for transformer: patches as sequence, input dimensions as batch
        x_patch = x_patch.reshape(batch_size * input_dim, self.num_patches, self.d_model)
        
        # Add position embedding
        x_patch = x_patch + self.position_embedding
        
        # Apply transformer encoder
        x_patch = self.transformer_encoder(x_patch)  # (batch_size * input_dim, num_patches, d_model)
        
        # Global average pooling along patch dimension
        x_patch = x_patch.mean(dim=1)  # (batch_size * input_dim, d_model)
        
        # Reshape back to (batch_size, input_dim * d_model)
        x_patch = x_patch.reshape(batch_size, input_dim * self.d_model)
        
        # Apply output layer
        output = self.output_layer(x_patch)  # (batch_size, output_dim)
        
        return output


class PatchTSTClassifier(nn.Module):
    """
    PatchTST model for classification tasks.
    """
    
    def __init__(self, 
                input_dim: int,
                num_classes: int,
                patch_len: int = 16,
                stride: int = 8,
                d_model: int = 128,
                n_heads: int = 8,
                n_layers: int = 3,
                dropout: float = 0.1,
                activation: str = "gelu",
                context_length: int = 512):
        """
        Initialize PatchTST classifier.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of classes
            patch_len: Length of each patch
            stride: Stride between patches
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
            context_length: Length of input sequence
        """
        super().__init__()
        
        self.patch_tst = PatchTST(
            input_dim=input_dim,
            output_dim=num_classes,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            activation=activation,
            context_length=context_length
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, context_length, input_dim)
        
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        return self.patch_tst(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, context_length, input_dim)
        
        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs


class PatchTSTRegressor(nn.Module):
    """
    PatchTST model for regression tasks.
    """
    
    def __init__(self, 
                input_dim: int,
                patch_len: int = 16,
                stride: int = 8,
                d_model: int = 128,
                n_heads: int = 8,
                n_layers: int = 3,
                dropout: float = 0.1,
                activation: str = "gelu",
                context_length: int = 512):
        """
        Initialize PatchTST regressor.
        
        Args:
            input_dim: Number of input features
            patch_len: Length of each patch
            stride: Stride between patches
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
            context_length: Length of input sequence
        """
        super().__init__()
        
        self.patch_tst = PatchTST(
            input_dim=input_dim,
            output_dim=1,  # Single output for regression
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            activation=activation,
            context_length=context_length
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, context_length, input_dim)
        
        Returns:
            Predictions tensor of shape (batch_size, 1)
        """
        return self.patch_tst(x)


class PatchTSTLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for PatchTST models.
    """
    
    def __init__(self,
                model_type: str = "classifier",
                input_dim: int = 100,
                output_dim: int = 2,
                patch_len: int = 16,
                stride: int = 8,
                d_model: int = 128,
                n_heads: int = 8,
                n_layers: int = 3,
                dropout: float = 0.1,
                activation: str = "gelu",
                context_length: int = 512,
                learning_rate: float = 1e-4,
                weight_decay: float = 1e-4,
                feature_names: Optional[List[str]] = None,
                class_names: Optional[List[str]] = None):
        """
        Initialize PatchTST Lightning module.
        
        Args:
            model_type: Type of model ('classifier' or 'regressor')
            input_dim: Number of input features
            output_dim: Number of output dimensions (num_classes for classifier, 1 for regressor)
            patch_len: Length of each patch
            stride: Stride between patches
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
            context_length: Length of input sequence
            learning_rate: Learning rate
            weight_decay: Weight decay
            feature_names: Names of input features
            class_names: Names of classes (for classifier)
        """
        super().__init__()
        
        if not PYTORCH_LIGHTNING_AVAILABLE:
            raise ImportError("PyTorch Lightning is required but not installed.")
        
        self.save_hyperparameters()
        
        # Create model based on type
        if model_type == "classifier":
            self.model = PatchTSTClassifier(
                input_dim=input_dim,
                num_classes=output_dim,
                patch_len=patch_len,
                stride=stride,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                activation=activation,
                context_length=context_length
            )
        elif model_type == "regressor":
            self.model = PatchTSTRegressor(
                input_dim=input_dim,
                patch_len=patch_len,
                stride=stride,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                activation=activation,
                context_length=context_length
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Store feature names and class names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(input_dim)]
        self.class_names = class_names or [f"class_{i}" for i in range(output_dim)]
        
        # Set up model-specific parameters
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Define loss function based on model type
        if model_type == "classifier":
            self.loss_fn = nn.CrossEntropyLoss()
        else:  # regressor
            self.loss_fn = nn.MSELoss()
        
        # Metrics for validation
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, context_length, input_dim)
        
        Returns:
            Output tensor
        """
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Tuple of (x, y)
            batch_idx: Batch index
        
        Returns:
            Loss tensor
        """
        x, y = batch
        
        # Forward pass
        y_hat = self(x)
        
        # Adjust target shape for regressor
        if self.model_type == "regressor":
            y = y.view(-1, 1)
        
        # Calculate loss
        loss = self.loss_fn(y_hat, y)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """
        Validation step.
        
        Args:
            batch: Tuple of (x, y)
            batch_idx: Batch index
        
        Returns:
            Dictionary of validation metrics
        """
        x, y = batch
        
        # Forward pass
        y_hat = self(x)
        
        # Adjust target shape for regressor
        if self.model_type == "regressor":
            y = y.view(-1, 1)
        
        # Calculate loss
        loss = self.loss_fn(y_hat, y)
        
        # Calculate additional metrics based on model type
        if self.model_type == "classifier":
            # Convert logits to class predictions
            preds = torch.argmax(y_hat, dim=1)
            
            # Calculate metrics
            acc = (preds == y).float().mean()
            
            # Return metrics
            output = {
                "val_loss": loss,
                "val_acc": acc,
                "y": y,
                "y_hat": y_hat,
                "preds": preds
            }
        else:  # regressor
            # Calculate metrics
            mae = F.l1_loss(y_hat, y)
            
            # Return metrics
            output = {
                "val_loss": loss,
                "val_mae": mae,
                "y": y,
                "y_hat": y_hat
            }
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True, logger=True)
        
        if self.model_type == "classifier":
            self.log("val_acc", output["val_acc"], prog_bar=True, logger=True)
        else:
            self.log("val_mae", output["val_mae"], prog_bar=True, logger=True)
        
        self.validation_step_outputs.append(output)
        
        return output
    
    def on_validation_epoch_end(self) -> None:
        """
        Compute metrics at the end of validation epoch.
        """
        if not self.validation_step_outputs:
            return
        
        # Collect outputs
        y_list = []
        y_hat_list = []
        
        for output in self.validation_step_outputs:
            y_list.append(output["y"])
            y_hat_list.append(output["y_hat"])
        
        # Concatenate tensors
        y = torch.cat(y_list)
        y_hat = torch.cat(y_hat_list)
        
        if self.model_type == "classifier":
            # Compute metrics
            preds = torch.argmax(y_hat, dim=1)
            probs = F.softmax(y_hat, dim=1)
            
            # Calculate metrics using sklearn
            y_np = y.cpu().numpy()
            preds_np = preds.cpu().numpy()
            probs_np = probs.cpu().numpy()
            
            acc = accuracy_score(y_np, preds_np)
            prec = precision_score(y_np, preds_np, average="weighted")
            rec = recall_score(y_np, preds_np, average="weighted")
            f1 = f1_score(y_np, preds_np, average="weighted")
            
            # ROC AUC for binary classification
            if y_hat.shape[1] == 2:
                roc_auc = roc_auc_score(y_np, probs_np[:, 1])
                self.log("val_roc_auc", roc_auc, prog_bar=False, logger=True)
            
            # Log metrics
            self.log("val_acc_epoch", acc, prog_bar=False, logger=True)
            self.log("val_precision", prec, prog_bar=False, logger=True)
            self.log("val_recall", rec, prog_bar=False, logger=True)
            self.log("val_f1", f1, prog_bar=False, logger=True)
        
        else:  # regressor
            # Compute metrics
            mae = F.l1_loss(y_hat, y).item()
            mse = F.mse_loss(y_hat, y).item()
            rmse = np.sqrt(mse)
            
            # Log metrics
            self.log("val_mae_epoch", mae, prog_bar=False, logger=True)
            self.log("val_mse", mse, prog_bar=False, logger=True)
            self.log("val_rmse", rmse, prog_bar=False, logger=True)
        
        # Clear outputs
        self.validation_step_outputs = []
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """
        Test step.
        
        Args:
            batch: Tuple of (x, y)
            batch_idx: Batch index
        
        Returns:
            Dictionary of test metrics
        """
        # Reuse validation step logic
        output = self.validation_step(batch, batch_idx)
        
        # Rename metrics from val_* to test_*
        test_output = {}
        for k, v in output.items():
            if k.startswith("val_"):
                new_k = k.replace("val_", "test_")
                test_output[new_k] = v
            else:
                test_output[k] = v
        
        self.test_step_outputs.append(test_output)
        
        return test_output
    
    def on_test_epoch_end(self) -> None:
        """
        Compute metrics at the end of test epoch.
        """
        if not self.test_step_outputs:
            return
        
        # Collect outputs
        y_list = []
        y_hat_list = []
        
        for output in self.test_step_outputs:
            y_list.append(output["y"])
            y_hat_list.append(output["y_hat"])
        
        # Concatenate tensors
        y = torch.cat(y_list)
        y_hat = torch.cat(y_hat_list)
        
        if self.model_type == "classifier":
            # Compute metrics
            preds = torch.argmax(y_hat, dim=1)
            probs = F.softmax(y_hat, dim=1)
            
            # Calculate metrics using sklearn
            y_np = y.cpu().numpy()
            preds_np = preds.cpu().numpy()
            probs_np = probs.cpu().numpy()
            
            acc = accuracy_score(y_np, preds_np)
            prec = precision_score(y_np, preds_np, average="weighted")
            rec = recall_score(y_np, preds_np, average="weighted")
            f1 = f1_score(y_np, preds_np, average="weighted")
            
            # ROC AUC for binary classification
            if y_hat.shape[1] == 2:
                roc_auc = roc_auc_score(y_np, probs_np[:, 1])
                self.log("test_roc_auc", roc_auc, prog_bar=False, logger=True)
            
            # Log metrics
            self.log("test_acc_epoch", acc, prog_bar=False, logger=True)
            self.log("test_precision", prec, prog_bar=False, logger=True)
            self.log("test_recall", rec, prog_bar=False, logger=True)
            self.log("test_f1", f1, prog_bar=False, logger=True)
            
            # Record confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_np, preds_np)
            self.logger.experiment.add_figure(
                "confusion_matrix",
                self._plot_confusion_matrix(cm, self.class_names),
                self.global_step
            )
        
        else:  # regressor
            # Compute metrics
            mae = F.l1_loss(y_hat, y).item()
            mse = F.mse_loss(y_hat, y).item()
            rmse = np.sqrt(mse)
            
            # Calculate R^2
            y_mean = torch.mean(y)
            ss_tot = torch.sum((y - y_mean) ** 2)
            ss_res = torch.sum((y - y_hat) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            # Log metrics
            self.log("test_mae_epoch", mae, prog_bar=False, logger=True)
            self.log("test_mse", mse, prog_bar=False, logger=True)
            self.log("test_rmse", rmse, prog_bar=False, logger=True)
            self.log("test_r2", r2.item(), prog_bar=False, logger=True)
        
        # Clear outputs
        self.test_step_outputs = []
    
    def configure_optimizers(self) -> Dict:
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        """
        Create a matplotlib figure with a confusion matrix plot.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
        
        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")  # Use non-interactive backend
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Show all ticks and label them with the class names
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               ylabel="True label",
               xlabel="Predicted label")
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        return fig


class PatchTSTModel:
    """
    High-level interface for PatchTST models.
    """
    
    def __init__(self, 
                model_type: str = "classifier",
                input_dim: int = 100,
                output_dim: int = 2,
                patch_len: int = 16,
                stride: int = 8,
                d_model: int = 128,
                n_heads: int = 8,
                n_layers: int = 3,
                dropout: float = 0.1,
                activation: str = "gelu",
                context_length: int = 512,
                learning_rate: float = 1e-4,
                weight_decay: float = 1e-4,
                batch_size: int = 64,
                max_epochs: int = 100,
                patience: int = 10,
                feature_names: Optional[List[str]] = None,
                class_names: Optional[List[str]] = None):
        """
        Initialize PatchTST model interface.
        
        Args:
            model_type: Type of model ('classifier' or 'regressor')
            input_dim: Number of input features
            output_dim: Number of output dimensions (num_classes for classifier, 1 for regressor)
            patch_len: Length of each patch
            stride: Stride between patches
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
            context_length: Length of input sequence
            learning_rate: Learning rate
            weight_decay: Weight decay
            batch_size: Batch size
            max_epochs: Maximum number of epochs
            patience: Early stopping patience
            feature_names: Names of input features
            class_names: Names of classes (for classifier)
        """
        if not PYTORCH_LIGHTNING_AVAILABLE:
            raise ImportError("PyTorch Lightning is required but not installed.")
        
        self.model_type = model_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation
        self.context_length = context_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.feature_names = feature_names or [f"feature_{i}" for i in range(input_dim)]
        self.class_names = class_names or [f"class_{i}" for i in range(output_dim)]
        
        # Initialize model
        self.model = PatchTSTLightningModule(
            model_type=model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            activation=activation,
            context_length=context_length,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            feature_names=feature_names,
            class_names=class_names
        )
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler() if model_type == "regressor" else None
    
    def fit(self, 
           X_train: np.ndarray, 
           y_train: np.ndarray,
           X_val: Optional[np.ndarray] = None, 
           y_val: Optional[np.ndarray] = None,
           gpus: int = 1,
           seed: int = 42) -> Dict[str, Any]:
        """
        Fit the model.
        
        Args:
            X_train: Training features (samples, context_length, features)
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            gpus: Number of GPUs to use
            seed: Random seed
        
        Returns:
            Dictionary of training metrics
        """
        # Set random seed
        pl.seed_everything(seed)
        
        # Scale features
        X_train_scaled = self._scale_features(X_train, fit=True)
        
        # Scale targets for regression
        if self.model_type == "regressor":
            y_train_scaled = self._scale_targets(y_train, fit=True)
        else:
            y_train_scaled = y_train
        
        # Create validation data if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self._scale_features(X_val)
            
            if self.model_type == "regressor":
                y_val_scaled = self._scale_targets(y_val)
            else:
                y_val_scaled = y_val
            
            has_val_data = True
        else:
            has_val_data = False
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
        
        if has_val_data:
            val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        if has_val_data:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4
            )
        
        # Create callbacks
        callbacks = [
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                verbose=True
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                mode="min",
                verbose=True
            )
        ]
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            accelerator="gpu" if gpus > 0 else "cpu",
            devices=gpus if gpus > 0 else None,
            logger=True,
            log_every_n_steps=10
        )
        
        # Train model
        if has_val_data:
            trainer.fit(self.model, train_loader, val_loader)
        else:
            trainer.fit(self.model, train_loader)
        
        # Get training history
        history = {
            "train_loss": trainer.callback_metrics.get("train_loss", 0).item(),
            "val_loss": trainer.callback_metrics.get("val_loss", 0).item()
        }
        
        if self.model_type == "classifier":
            history["val_acc"] = trainer.callback_metrics.get("val_acc_epoch", 0).item()
            history["val_precision"] = trainer.callback_metrics.get("val_precision", 0).item()
            history["val_recall"] = trainer.callback_metrics.get("val_recall", 0).item()
            history["val_f1"] = trainer.callback_metrics.get("val_f1", 0).item()
            
            if self.output_dim == 2:  # Binary classification
                history["val_roc_auc"] = trainer.callback_metrics.get("val_roc_auc", 0).item()
        else:
            history["val_mae"] = trainer.callback_metrics.get("val_mae_epoch", 0).item()
            history["val_mse"] = trainer.callback_metrics.get("val_mse", 0).item()
            history["val_rmse"] = trainer.callback_metrics.get("val_rmse", 0).item()
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features (samples, context_length, features)
        
        Returns:
            Predictions
        """
        # Scale features
        X_scaled = self._scale_features(X)
        
        # Create dataset and data loader
        dataset = TimeSeriesDataset(X_scaled, np.zeros(X_scaled.shape[0]))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                x, _ = batch
                y_hat = self.model(x)
                
                if self.model_type == "classifier":
                    # Get class predictions
                    preds = torch.argmax(y_hat, dim=1)
                else:
                    # Unscale regression predictions
                    if self.target_scaler is not None:
                        preds = torch.from_numpy(
                            self.target_scaler.inverse_transform(y_hat.cpu().numpy())
                        )
                    else:
                        preds = y_hat
                
                predictions.append(preds.cpu().numpy())
        
        # Concatenate predictions
        predictions = np.concatenate(predictions, axis=0)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (for classifiers only).
        
        Args:
            X: Features (samples, context_length, features)
        
        Returns:
            Class probabilities
        """
        if self.model_type != "classifier":
            raise ValueError("predict_proba is only available for classifiers")
        
        # Scale features
        X_scaled = self._scale_features(X)
        
        # Create dataset and data loader
        dataset = TimeSeriesDataset(X_scaled, np.zeros(X_scaled.shape[0]))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        probas = []
        
        with torch.no_grad():
            for batch in loader:
                x, _ = batch
                y_hat = self.model(x)
                proba = F.softmax(y_hat, dim=1)
                probas.append(proba.cpu().numpy())
        
        # Concatenate probabilities
        probas = np.concatenate(probas, axis=0)
        
        return probas
    
    def save(self, filepath: str) -> None:
        """
        Save model and scalers.
        
        Args:
            filepath: Path to save model and scalers
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.to("cpu")
        torch.save(self.model.state_dict(), f"{filepath}.pt")
        
        # Save scalers and metadata
        metadata = {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "patch_len": self.patch_len,
            "stride": self.stride,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "activation": self.activation,
            "context_length": self.context_length,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "feature_scaler": {
                "mean": self.feature_scaler.mean_.tolist(),
                "scale": self.feature_scaler.scale_.tolist()
            }
        }
        
        if self.target_scaler is not None:
            metadata["target_scaler"] = {
                "mean": self.target_scaler.mean_.tolist(),
                "scale": self.target_scaler.scale_.tolist()
            }
        
        with open(f"{filepath}.json", "w") as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(cls, filepath: str) -> "PatchTSTModel":
        """
        Load model and scalers.
        
        Args:
            filepath: Path to load model and scalers
        
        Returns:
            Loaded model
        """
        # Load metadata
        with open(f"{filepath}.json", "r") as f:
            metadata = json.load(f)
        
        # Create model
        model = cls(
            model_type=metadata["model_type"],
            input_dim=metadata["input_dim"],
            output_dim=metadata["output_dim"],
            patch_len=metadata["patch_len"],
            stride=metadata["stride"],
            d_model=metadata["d_model"],
            n_heads=metadata["n_heads"],
            n_layers=metadata["n_layers"],
            dropout=metadata["dropout"],
            activation=metadata["activation"],
            context_length=metadata["context_length"],
            feature_names=metadata["feature_names"],
            class_names=metadata["class_names"]
        )
        
        # Restore feature scaler
        model.feature_scaler.mean_ = np.array(metadata["feature_scaler"]["mean"])
        model.feature_scaler.scale_ = np.array(metadata["feature_scaler"]["scale"])
        
        # Restore target scaler if available
        if "target_scaler" in metadata:
            model.target_scaler = StandardScaler()
            model.target_scaler.mean_ = np.array(metadata["target_scaler"]["mean"])
            model.target_scaler.scale_ = np.array(metadata["target_scaler"]["scale"])
        
        # Load model weights
        model.model.load_state_dict(torch.load(f"{filepath}.pt"))
        model.model.eval()
        
        return model
    
    def _scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Scale features.
        
        Args:
            X: Features (samples, context_length, features)
            fit: Whether to fit the scaler
        
        Returns:
            Scaled features
        """
        # Reshape to 2D
        X_2d = X.reshape(-1, X.shape[2])
        
        # Scale
        if fit:
            X_2d_scaled = self.feature_scaler.fit_transform(X_2d)
        else:
            X_2d_scaled = self.feature_scaler.transform(X_2d)
        
        # Reshape back to 3D
        X_scaled = X_2d_scaled.reshape(X.shape)
        
        return X_scaled
    
    def _scale_targets(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Scale targets (for regression only).
        
        Args:
            y: Targets
            fit: Whether to fit the scaler
        
        Returns:
            Scaled targets
        """
        if self.target_scaler is None:
            return y
        
        # Reshape to 2D if needed
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Scale
        if fit:
            y_scaled = self.target_scaler.fit_transform(y)
        else:
            y_scaled = self.target_scaler.transform(y)
        
        # Reshape back to 1D if needed
        if len(y.shape) == 1:
            y_scaled = y_scaled.flatten()
        
        return y_scaled


class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    Dataset for time series data.
    """
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            features: Features (samples, context_length, features)
            targets: Targets
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long if len(targets.shape) == 1 else torch.float32)
    
    def __len__(self) -> int:
        """
        Get dataset length.
        
        Returns:
            Dataset length
        """
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Index
        
        Returns:
            Tuple of (features, target)
        """
        return self.features[idx], self.targets[idx]


def create_sequences(data: pd.DataFrame, 
                    target_col: str, 
                    context_length: int = 20,
                    gap: int = 0,
                    horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series forecasting.
    
    Args:
        data: DataFrame with features and target
        target_col: Name of target column
        context_length: Length of input sequence
        gap: Gap between input and target
        horizon: Forecast horizon
    
    Returns:
        Tuple of (features, targets)
    """
    # Separate features and target
    features = data.drop(columns=[target_col])
    target = data[target_col]
    
    # Create sequences
    X, y = [], []
    
    for i in range(len(data) - context_length - gap - horizon + 1):
        # Input sequence
        X.append(features.iloc[i:i+context_length].values)
        
        # Target value
        y.append(target.iloc[i+context_length+gap+horizon-1])
    
    return np.array(X), np.array(y)


def run_patchtst_experiment(data: pd.DataFrame,
                          target_col: str,
                          model_type: str = "classifier",
                          test_size: float = 0.2,
                          val_size: float = 0.1,
                          context_length: int = 20,
                          gap: int = 0,
                          horizon: int = 1,
                          patch_len: int = 8,
                          stride: int = 4,
                          d_model: int = 64,
                          n_heads: int = 4,
                          n_layers: int = 2,
                          batch_size: int = 32,
                          max_epochs: int = 50,
                          save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a PatchTST experiment.
    
    Args:
        data: DataFrame with features and target
        target_col: Name of target column
        model_type: Type of model ('classifier' or 'regressor')
        test_size: Fraction of data to use for testing
        val_size: Fraction of data to use for validation
        context_length: Length of input sequence
        gap: Gap between input and target
        horizon: Forecast horizon
        patch_len: Length of each patch
        stride: Stride between patches
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        batch_size: Batch size
        max_epochs: Maximum number of epochs
        save_path: Path to save model
    
    Returns:
        Dictionary of results
    """
    # Ensure data is sorted by time
    if "timestamp" in data.columns:
        data = data.sort_values("timestamp")
    
    # Create sequences
    X, y = create_sequences(data, target_col, context_length, gap, horizon)
    
    # Split data
    n_samples = len(X)
    test_idx = int(n_samples * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    X_train, y_train = X[:val_idx], y[:val_idx]
    X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
    X_test, y_test = X[test_idx:], y[test_idx:]
    
    # Get input dimension
    input_dim = X.shape[2]
    
    # Determine output dimension based on model type
    if model_type == "classifier":
        output_dim = len(np.unique(y))
    else:
        output_dim = 1
    
    # Get feature names
    feature_names = list(data.drop(columns=[target_col]).columns)
    
    # Get class names for classifier
    if model_type == "classifier":
        if output_dim == 2:
            class_names = ["DOWN", "UP"]
        else:
            class_names = [str(i) for i in range(output_dim)]
    else:
        class_names = None
    
    # Create model
    model = PatchTSTModel(
        model_type=model_type,
        input_dim=input_dim,
        output_dim=output_dim,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        context_length=context_length,
        batch_size=batch_size,
        max_epochs=max_epochs,
        feature_names=feature_names,
        class_names=class_names
    )
    
    # Train model
    history = model.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    if model_type == "classifier":
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        # ROC AUC for binary classification
        if output_dim == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = None
        
        test_results = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
        
        if roc_auc is not None:
            test_results["roc_auc"] = roc_auc
    else:
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        test_results = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }
    
    # Save model if path is provided
    if save_path is not None:
        model.save(save_path)
    
    # Return results
    results = {
        "model": model,
        "history": history,
        "test_results": test_results,
        "config": {
            "model_type": model_type,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "context_length": context_length,
            "gap": gap,
            "horizon": horizon,
            "patch_len": patch_len,
            "stride": stride,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers
        }
    }
    
    return results