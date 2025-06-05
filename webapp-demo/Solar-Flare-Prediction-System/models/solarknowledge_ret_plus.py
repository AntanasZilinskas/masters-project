import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import subprocess
import threading
from datetime import datetime
from typing import Tuple, Dict, Optional
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

# Try to import codecarbon for CO2 tracking
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("Warning: codecarbon not available. Install with: pip install codecarbon")
except Exception as e:
    CODECARBON_AVAILABLE = False
    print(f"Warning: codecarbon failed to import: {e}")

# Device config: prefer CUDA, then MPS, then CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Enable cuDNN autotuner when using CUDA (speeds up fixed-size workloads)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    # Enable tensor-float-32 for additional throughput on Ampere/Ada GPUs
    torch.backends.cuda.matmul.allow_tf32 = True

# ----------------------------------------
# GPU Power Monitoring Utilities
# ----------------------------------------
class GPUMonitor:
    """Monitor GPU power consumption during training."""
    
    def __init__(self, interval=60):
        self.interval = interval
        self.power_readings = []
        self.monitoring = False
        self.monitor_thread = None
        
    def _monitor_power(self):
        """Background thread to monitor GPU power."""
        while self.monitoring:
            try:
                if device.type == "cuda":
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        power = float(result.stdout.strip().split('\n')[0])
                        timestamp = time.time()
                        self.power_readings.append((timestamp, power))
            except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
                pass
            time.sleep(self.interval)
    
    def start_monitoring(self):
        """Start monitoring GPU power consumption."""
        if device.type == "cuda" and not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_power, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and return power statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        if self.power_readings:
            powers = [reading[1] for reading in self.power_readings]
            return {
                "average_power_w": np.mean(powers),
                "max_power_w": np.max(powers),
                "min_power_w": np.min(powers),
                "power_readings": len(powers),
                "monitoring_duration_s": self.power_readings[-1][0] - self.power_readings[0][0] if len(self.power_readings) > 1 else 0
            }
        return None

def get_gpu_info():
    """Get GPU information."""
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)  # GB
        return {"gpu_name": gpu_name, "gpu_memory_gb": gpu_memory}
    elif device.type == "mps":
        return {"gpu_name": "Apple Silicon", "gpu_memory_gb": "Unified Memory"}
    else:
        return {"gpu_name": "CPU", "gpu_memory_gb": 0}

# ----------------------------------------
# Positional Encoding
# ----------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe)
        # Learnable scaling as in Transformer-XL
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x + self.alpha * self.pe[:, :x.size(1)].type_as(x)

# ----------------------------------------
# Transformer Block
# ----------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.att(x, x, x)
        x = self.norm1(x + self.drop1(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.drop2(ffn_out))

# ----------------------------------------
# Evidential & EVT Losses
# ----------------------------------------
def evidential_nll(y, evid):
    mu, v, a, b = torch.split(evid, 1, dim=-1)
    v = torch.clamp(v, min=1e-3)
    a = torch.clamp(a, min=1.1)
    b = torch.clamp(b, min=1e-3)
    p = torch.clamp(torch.sigmoid(mu), 1e-4, 1 - 1e-4)
    S = b * (1 + v) / a
    eps = 1e-7
    nll = - y * torch.log(p + eps) - (1 - y) * torch.log(1 - p + eps) + 0.5 * torch.log(S + eps)
    return torch.clamp(nll, min=0.0).mean()

def evt_loss(logits, gpd, pct=0.9):
    """Adaptive EVT loss using per-batch percentile as threshold."""
    try:
        if gpd is None:
            return torch.tensor(0.0, device=logits.device)

        xi, sigma = torch.split(gpd, 1, dim=-1)
        xi = torch.clamp(xi, min=-0.99, max=1.0)
        sigma = torch.clamp(sigma, min=1e-2)

        # Determine threshold u as the pct-quantile of logits in the batch
        threshold = torch.quantile(logits.detach(), pct).item()

        y = F.relu(logits - threshold)  # exceedances
        mask = (y > 0).squeeze()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        y = y[mask]
        xi = xi[mask]
        sigma = sigma[mask]

        eps = 1e-7
        z = xi * y / (sigma + eps)
        log_term = torch.log1p(torch.clamp(z, min=-0.999, max=1e6))
        term = torch.where(
            torch.abs(xi) < 1e-3,
            y / sigma + torch.log(sigma + eps),
            (1/xi + 1) * log_term + torch.log(sigma + eps)
        )

        reg = 1e-3 * (xi**2).mean() + 1e-3 * (1 / (sigma + eps)).mean()
        return torch.clamp(term.mean(), min=0.0) + reg
    except Exception:
        return torch.tensor(0.0, device=logits.device)

# ----------------------------------------
# Focal Loss
# ----------------------------------------
def focal_bce_loss(logits, targets, gamma):
    targets = targets.view(-1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    mod = (1 - p).pow(gamma) * targets + p.pow(gamma) * (1 - targets)
    return (mod * bce).mean()

# ----------------------------------------
# Binary Cross-Entropy for precursor head
# ----------------------------------------
def precursor_bce_loss(logits, targets):
    targets = targets.view(-1, 1).float()
    return F.binary_cross_entropy_with_logits(logits, targets)

# ----------------------------------------
# RET+ Model with attention bottleneck and precursor head
# ----------------------------------------
class RETPlusModel(nn.Module):
    def __init__(self, input_shape, embed_dim=128, num_heads=4, ff_dim=256, num_blocks=6, dropout=0.2,
                 use_attention_bottleneck: bool = True,
                 use_evidential: bool = True,
                 use_evt: bool = True,
                 use_precursor: bool = True):
        super().__init__()
        T, F = input_shape
        self.embedding = nn.Linear(F, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.pos = PositionalEncoding(T, embed_dim)
        self.transformers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_blocks)
        ])

        # Store ablation flags
        self.use_attention_bottleneck = use_attention_bottleneck
        self.use_evidential = use_evidential
        self.use_evt = use_evt
        self.use_precursor = use_precursor

        # Attention bottleneck (learned pooling) – optional
        if self.use_attention_bottleneck:
            self.att_pool = nn.Sequential(
                nn.Linear(embed_dim, 1),
                nn.Softmax(dim=1)
            )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.logits = nn.Linear(128, 1)

        # Optional heads
        if self.use_evidential:
            self.nig = nn.Linear(128, 4)
        else:
            self.nig = None

        if self.use_evt:
            self.gpd = nn.Linear(128, 2)
        else:
            self.gpd = None

        if self.use_precursor:
            self.precursor_head = nn.Linear(128, 1)
        else:
            self.precursor_head = None

    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.pos(x)
        for blk in self.transformers:
            x = blk(x)

        # Sequence summarisation
        if self.use_attention_bottleneck:
            attn_weights = self.att_pool(x)  # shape: [B, T, 1]
            x_bottleneck = (x * attn_weights).sum(dim=1)  # shape: [B, embed_dim]
        else:
            x_bottleneck = x.mean(dim=1)

        x = self.head(x_bottleneck)

        logits = self.logits(x)

        # Evidential outputs (optional)
        evid = None
        if self.use_evidential and self.nig is not None:
            nig_raw = self.nig(x)
            mu, logv, loga, logb = torch.split(nig_raw, 1, dim=-1)
            v = F.softplus(logv)
            a = 1 + F.softplus(loga)
            b = F.softplus(logb)
            evid = torch.cat([mu, v, a, b], dim=-1)

        # EVT outputs (optional)
        gpd_out = None
        if self.use_evt and self.gpd is not None:
            gpd_raw = self.gpd(x)
            xi_raw, log_sigma = torch.split(gpd_raw, 1, dim=-1)
            # Robust parameterisation: ξ ∈ (-0.5, +1.0); σ > 0
            xi = 1.5 * torch.tanh(xi_raw) - 0.5
            sigma = F.softplus(log_sigma) + 1e-3
            gpd_out = torch.cat([xi, sigma], dim=-1)

        # Precursor output (optional)
        precursor = None
        if self.use_precursor and self.precursor_head is not None:
            precursor = self.precursor_head(x)

        return {"logits": logits, "evid": evid, "gpd": gpd_out, "precursor": precursor}

# ----------------------------------------
# Composite Loss
# ----------------------------------------
def composite_loss(
    y_true,
    outputs,
    gamma: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 2.5,
):
    """Compute composite loss with configurable component weights.

    Args:
        y_true: Ground-truth labels tensor.
        outputs: Dict returned by the model forward pass.
        gamma: Focal-loss focusing parameter.
        weights: Dict with keys ``focal``, ``evid``, and ``evt`` specifying
            the contribution of each component. Defaults to {0.8, 0.1, 0.1}.
        threshold: EVT threshold.
    """

    if weights is None:
        weights = {"focal": 0.8, "evid": 0.1, "evt": 0.1, "prec": 0.05}

    logits = outputs["logits"]
    y_true = y_true.view(-1, 1).float()

    # Always compute focal loss
    fl = focal_bce_loss(logits, y_true, gamma)

    # Evidential component (optional)
    evid_loss = 0.0
    if weights.get("evid", 0) > 0 and outputs.get("evid", None) is not None:
        evid_loss = evidential_nll(y_true, outputs["evid"])

    # EVT component (optional)
    tail_loss = 0.0
    if weights.get("evt", 0) > 0 and outputs.get("gpd", None) is not None:
        tail_loss = evt_loss(logits, outputs["gpd"])

    # Precursor component (optional)
    prec_loss = 0.0
    if weights.get("prec", 0) > 0 and outputs.get("precursor", None) is not None:
        prec_loss = precursor_bce_loss(outputs["precursor"], y_true)

    total = (
        weights.get("focal", 0) * fl
        + weights.get("evid", 0) * evid_loss
        + weights.get("evt", 0) * tail_loss
        + weights.get("prec", 0) * prec_loss
    )

    return (
        total
        if not torch.isnan(total)
        else torch.tensor(0.0, requires_grad=True).to(logits.device)
    )

# ----------------------------------------
# Wrapper
# ----------------------------------------
class RETPlusWrapper:
    def __init__(
        self,
        input_shape: Tuple[int, int],
        early_stopping_patience: int = 10,
        use_attention_bottleneck: bool = True,
        use_evidential: bool = True,
        use_evt: bool = True,
        use_precursor: bool = True,
        loss_weights: Optional[Dict[str, float]] = None,
        *,
        compile_model: bool = False,  # 🔄 new flag – compile with TorchDynamo for speed
    ):
        """Wrapper around ``RETPlusModel`` with flexible ablation flags.

        Args:
            input_shape: Tuple specifying (timesteps, features).
            early_stopping_patience: Early-stopping window on TSS.
            use_attention_bottleneck: Enable learned attention pooling.
            use_evidential: Enable evidential (NIG) head.
            use_evt: Enable EVT (GPD) head.
            use_precursor: Enable precursor score head.
            loss_weights: Optional dict to scale composite-loss components. If
                None, defaults to {"focal":0.8,"evid":0.1,"evt":0.1}. Set a
                component's weight to 0 to exclude it during training.
        """

        self.loss_weights = (
            loss_weights
            if loss_weights is not None
            else {"focal": 0.8, "evid": 0.1, "evt": 0.1}
        )

        self.model = RETPlusModel(
            input_shape,
            use_attention_bottleneck=use_attention_bottleneck,
            use_evidential=use_evidential,
            use_evt=use_evt,
            use_precursor=use_precursor,
        ).to(device)
        # Optionally compile the model for higher throughput (PyTorch ≥ 2.0)
        if compile_model and device.type == "cuda" and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
            except Exception:
                # Fallback gracefully if compilation fails (e.g. older PyTorch)
                pass
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=1e-4, fused=True
        )
        self.early_stopping_patience = early_stopping_patience
        self.history = {"loss": [], "accuracy": [], "tss": []}
        # Buffers for later interpretability/evaluation artefacts
        self._train_data = None  # tuple (X, y) stored as NumPy arrays

    def train(
        self,
        X_train,
        y_train,
        epochs=100,
        batch_size=512,
        gamma_max=2.0,
        warmup_epochs=50,
        flare_class="M",
        time_window=24,
        *,
        in_memory_dataset: bool = False,  # 🔄 load whole dataset straight to GPU
        track_emissions: bool = True,  # 🌱 disable for cluster environments if needed
    ):
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from model_tracking import save_model_with_metadata, get_next_version

        # Convert to NumPy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Retain for later evaluation/visualisation
        self._train_data = (X_train, y_train)

        # Get GPU information
        gpu_info = get_gpu_info()
        print(f"Training on: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']} GB)")

        # Initialize lightweight monitoring (cluster-safe)
        gpu_monitor = None
        emissions_tracker = None
        
        # GPU monitoring with maximum safety
        if device.type == "cuda":
            try:
                gpu_monitor = GPUMonitor(interval=300)  # Sample every 5 minutes
                gpu_monitor.start_monitoring()
            except Exception as e:
                print(f"Note: GPU power monitoring unavailable: {e}")
                gpu_monitor = None

        # Initialize codecarbon emissions tracking (ultra-safe for clusters)
        if CODECARBON_AVAILABLE and track_emissions:
            try:
                # Create emissions directory with error handling
                emissions_dir = "models/hpo/emissions"
                try:
                    os.makedirs(emissions_dir, exist_ok=True)
                except (OSError, PermissionError) as e:
                    print(f"Warning: Cannot create emissions directory: {e}. Using current directory.")
                    emissions_dir = "."
                
                emissions_tracker = EmissionsTracker(
                    project_name=f"everest-{flare_class}-{time_window}h",
                    experiment_id=f"{flare_class}_{time_window}h_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    output_dir=emissions_dir,
                    save_to_file=True,
                    log_level="error",  # Minimal logging
                    tracking_mode="process",  # Only track this process
                    measure_power_secs=300,  # Less frequent sampling (5 min)
                    api_call_interval=300,  # Less frequent API calls
                    emissions_endpoint=None,  # Disable cloud API calls
                    save_to_logger=False,  # Disable logger output
                    save_to_api=False,  # Disable API uploads
                    offline_mode=True,  # Force offline mode
                )
                emissions_tracker.start()
                print("🌱 CO2 tracking started (cluster-safe mode)")
            except Exception as e:
                print(f"Note: CO2 tracking unavailable: {e}")
                emissions_tracker = None
                track_emissions = False  # Disable for this session

        # Training timing
        training_start_time = time.time()
        epoch_times = []
        
        # Build DataLoader – optionally keep the full dataset on the GPU
        use_amp = device.type == "cuda"
        if in_memory_dataset and device.type == "cuda":
            X_tensor = torch.tensor(X_train, dtype=torch.float16 if use_amp else torch.float32, device=device)
            y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )
        else:
            dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=(device.type == "cuda"),
                num_workers=os.cpu_count(),
                persistent_workers=True,
                prefetch_factor=4,
            )

        # AMP scaler for mixed-precision (CUDA only – avoids dtype issues on MPS/CPU)
        scaler = GradScaler(enabled=use_amp)

        best_tss = -1e8
        patience = 0
        version = get_next_version(flare_class, time_window)
        best_weights = None
        best_epoch = -1

        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            gamma = min(gamma_max, gamma_max * epoch / warmup_epochs)
            epoch_loss = 0.0
            total = 0
            correct = 0
            TP = TN = FP = FN = 0

            for X_batch, y_batch in loader:
                # Move to device
                X_batch = X_batch.to(device, non_blocking=(device.type == "cuda"))
                y_batch = y_batch.to(device, non_blocking=(device.type == "cuda"))

                self.model.train()
                self.optimizer.zero_grad()

                if use_amp:
                    # Mixed-precision path
                    with autocast(device_type=device.type):
                        # Dynamically adjust loss weights: simple 3-phase schedule
                        if epoch < 20:
                            phase_weights = {"focal": 0.9, "evid": 0.1, "evt": 0.0, "prec": 0.05}
                        elif epoch < 40:
                            phase_weights = {"focal": 0.8, "evid": 0.1, "evt": 0.1, "prec": 0.05}
                        else:
                            phase_weights = {"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.05}

                        outputs = self.model(X_batch)
                        loss = composite_loss(
                            y_batch,
                            outputs,
                            gamma=gamma,
                            weights=phase_weights,
                        )

                    if torch.isnan(loss):
                        continue

                    # Scaled backward + step
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    # Standard FP32 training path (CPU / MPS)
                    # Dynamically adjust loss weights: simple 3-phase schedule
                    if epoch < 20:
                        phase_weights = {"focal": 0.9, "evid": 0.1, "evt": 0.0, "prec": 0.05}
                    elif epoch < 40:
                        phase_weights = {"focal": 0.8, "evid": 0.1, "evt": 0.1, "prec": 0.05}
                    else:
                        phase_weights = {"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.05}

                    outputs = self.model(X_batch)
                    loss = composite_loss(
                        y_batch,
                        outputs,
                        gamma=gamma,
                        weights=phase_weights,
                    )

                    if torch.isnan(loss):
                        continue

                    loss.backward()
                    self.optimizer.step()

                # Metrics accumulation
                epoch_loss += loss.item() * X_batch.size(0)
                preds = (
                    torch.sigmoid(outputs["logits"]) > 0.5
                ).int().squeeze()
                y_true = y_batch.int().squeeze()

                TP += ((preds == 1) & (y_true == 1)).sum().item()
                TN += ((preds == 0) & (y_true == 0)).sum().item()
                FP += ((preds == 1) & (y_true == 0)).sum().item()
                FN += ((preds == 0) & (y_true == 1)).sum().item()
                correct += (preds == y_true).sum().item()
                total += y_batch.size(0)

            # Calculate epoch metrics and timing
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)
            
            avg_loss = epoch_loss / max(1, total)
            accuracy = correct / total if total > 0 else 0.0
            sensitivity = TP / (TP + FN + 1e-8)
            specificity = TN / (TN + FP + 1e-8)
            tss = sensitivity + specificity - 1.0

            self.history["loss"].append(avg_loss)
            self.history["accuracy"].append(accuracy)
            self.history["tss"].append(tss)

            print(
                f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - "
                f"acc: {accuracy:.4f} - tss: {tss:.4f} - gamma: {gamma:.2f} - "
                f"time: {epoch_duration:.1f}s"
            )

            # Early stopping check
            if tss > best_tss:
                best_tss = tss
                best_weights = self.model.state_dict()
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. "
                        f"Restoring best model from epoch {best_epoch+1}."
                    )
                    if best_weights is not None:
                        self.model.load_state_dict(best_weights)
                    break

        # Calculate total training time
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        # Stop monitoring and collect metrics
        gpu_power_stats = None
        if gpu_monitor:
            gpu_power_stats = gpu_monitor.stop_monitoring()
        
        emissions_data = None
        if emissions_tracker:
            try:
                emissions_data = emissions_tracker.stop()
                print(f"🌱 Training emissions: {emissions_data:.6f} kg CO2eq")
            except Exception as e:
                print(f"Warning: Failed to stop codecarbon tracker: {e}")

        # Store comprehensive training metrics
        self.training_metrics = {
            "total_training_time_s": total_training_time,
            "total_training_time_h": total_training_time / 3600,
            "average_epoch_time_s": np.mean(epoch_times),
            "fastest_epoch_time_s": np.min(epoch_times),
            "slowest_epoch_time_s": np.max(epoch_times),
            "epoch_times": epoch_times,
            "epochs_completed": len(epoch_times),
            "early_stopped": len(epoch_times) < epochs,
            "gpu_info": gpu_info,
            "gpu_power_stats": gpu_power_stats,
            "co2_emissions_kg": emissions_data if emissions_data else None,
            "training_samples": len(X_train),
            "batch_size": batch_size,
            "mixed_precision": use_amp,
        }

        print(f"\n📊 Training completed in {total_training_time:.1f}s ({total_training_time/3600:.2f}h)")
        print(f"   • Average epoch time: {np.mean(epoch_times):.1f}s")
        print(f"   • GPU: {gpu_info['gpu_name']}")
        if gpu_power_stats:
            print(f"   • Average GPU power: {gpu_power_stats['average_power_w']:.1f}W")
        if emissions_data:
            print(f"   • CO2 emissions: {emissions_data:.6f} kg CO2eq")

        # Save best model & metadata
        model_dir = self.save(
            version=version,
            flare_class=flare_class,
            time_window=time_window,
            # Provide evaluation data for artefact generation
            X_eval=X_train,
            y_eval=y_train,
        )
        return model_dir

    def predict_proba(self, X):
        """Return probability predictions for *either* numpy arrays or tensors.

        • If ``X`` is already a torch.Tensor we only cast dtype/transfer device.
        • If ``X`` is a NumPy array / list, convert to float32 tensor first.
        """
        self.model.eval()

        if isinstance(X, torch.Tensor):
            X_tensor = X.to(device, dtype=torch.float32)
        else:
            X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = self.model(X_tensor)["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def save_weights(self, flare_class, w_dir):
        # Dump raw PyTorch weights
        os.makedirs(w_dir, exist_ok=True)
        weights_path = os.path.join(w_dir, "model_weights.pt")
        torch.save(self.model.state_dict(), weights_path)

    def save(self, version, flare_class, time_window, X_eval=None, y_eval=None):
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from model_tracking import save_model_with_metadata

        # ---------------- Basic discrimination metrics ----------------
        metrics = {
            "accuracy": self.history["accuracy"][-1],
            "TSS": self.history["tss"][-1],
        }

        # --------------------------------------------------------------
        # Optional interpretability & uncertainty artefacts
        # --------------------------------------------------------------
        y_true = y_pred = y_scores = evid_out = evt_scores = None
        att_X = att_y_true = att_y_pred = att_y_score = None
        sample_input = None

        if (X_eval is not None) and (y_eval is not None):
            # Use a small DataLoader to avoid memory blow-up
            eval_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_eval, dtype=torch.float32),
                    torch.tensor(y_eval, dtype=torch.float32),
                ),
                batch_size=512,
                shuffle=False,
            )

            preds_list = []
            probs_list = []
            y_list = []
            evid_list = []
            gpd_list = []

            self.model.eval()
            with torch.no_grad():
                for xb, yb in eval_loader:
                    xb = xb.to(device)
                    out = self.model(xb)
                    prob = torch.sigmoid(out["logits"]).cpu().numpy().flatten()
                    pred = (prob > 0.5).astype(int)

                    preds_list.append(pred)
                    probs_list.append(prob)
                    y_list.append(yb.numpy().flatten())

                    # Optional artefacts – only append if present and not None
                    if out.get("evid") is not None:
                        evid_list.append(out["evid"].cpu().numpy())
                    if out.get("gpd") is not None:
                        gpd_list.append(out["gpd"].cpu().numpy())

            y_true = np.concatenate(y_list)
            y_scores = np.concatenate(probs_list)
            y_pred = np.concatenate(preds_list)
            evid_out = np.concatenate(evid_list) if evid_list else None
            evt_scores = np.concatenate(gpd_list) if gpd_list else None

            # Attention batch: first 10 samples
            att_X = X_eval[:10]
            att_y_true = y_true[:10]
            att_y_pred = y_pred[:10]
            att_y_score = y_scores[:10]

            sample_input = torch.tensor(att_X[:32], dtype=torch.float32).to(device) if len(X_eval) >= 32 else torch.tensor(X_eval, dtype=torch.float32).to(device)

            # Additional calibration metrics
            try:
                from sklearn.metrics import roc_auc_score, brier_score_loss

                def _ece(probs, labels, n_bins=15):
                    bins = np.linspace(0, 1, n_bins + 1)
                    ece = 0.0
                    for i in range(n_bins):
                        idx = (probs >= bins[i]) & (probs < bins[i + 1])
                        if idx.sum() == 0:
                            continue
                        acc_bin = (labels[idx] == 1).mean()
                        conf_bin = probs[idx].mean()
                        ece += abs(conf_bin - acc_bin) * idx.mean()
                    return ece

                metrics.update(
                    {
                        "ROC_AUC": float(roc_auc_score(y_true, y_scores)),
                        "Brier": float(brier_score_loss(y_true, y_scores)),
                        "ECE": float(_ece(y_scores, y_true)),
                    }
                )
            except Exception:
                pass

        model_dir = save_model_with_metadata(
            model=self,
            metrics=metrics,
            hyperparams={
                "input_shape": (10, 9),
                "embed_dim": 128,
                "num_heads": 4,
                "ff_dim": 256,
                "num_blocks": 6,
                "dropout": 0.2,
            },
            history=self.history,
            version=version,
            flare_class=flare_class,
            time_window=time_window,
            description="EVEREST model trained on SHARP data with evidential and EVT losses.",
            # Training metrics and efficiency data
            training_metrics=getattr(self, 'training_metrics', None),
            # Newly added artefacts
            y_true=y_true,
            y_pred=y_pred,
            y_scores=y_scores,
            evt_scores=evt_scores,
            sample_input=sample_input,
            att_X_batch=att_X,
            att_y_true=att_y_true,
            att_y_pred=att_y_pred,
            att_y_score=att_y_score,
            evidential_out=evid_out,
        )
        return model_dir

    def load(self, path):
        # self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.load_state_dict(torch.load(path, map_location=device), strict=False)