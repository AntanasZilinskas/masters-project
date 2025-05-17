import os
import json
from typing import Any, Dict

"""Lightweight, file-system based model tracking so training & ablations run without the optional MLflow / WandB backend.

The real paper used a richer tracking stack, but for local experiments we only
need two helpers:
  • get_next_version – returns a monotonically-increasing experiment ID string.
  • save_model_with_metadata – dumps weights & metadata to the filesystem.

Both functions mimic the signatures expected in solarknowledge_ret_plus.py so
that importing `model_tracking` never raises errors even if the full library is
absent.
"""

_BASE_DIR = os.path.join("saved_models")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def get_next_version(flare_class: str, time_window: int | str) -> str:
    """Generate a simple incremental run identifier per class/horizon."""
    tag = f"{flare_class}_{time_window}"
    path = _ensure_dir(os.path.join(_BASE_DIR, tag))
    # Count existing dirs
    existing = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    version = f"run_{len(existing)+1:03d}"
    return version

def save_model_with_metadata(
    model: Any,
    metrics: Dict[str, Any],
    hyperparams: Dict[str, Any],
    history: Dict[str, Any],
    version: str,
    flare_class: str,
    time_window: int | str,
    description: str = "",
    **artefacts,
):
    """Persist weights + JSON metadata in a local folder structure.

    This is a minimal replacement for a full-featured experiment tracker.
    """
    tag = f"{flare_class}_{time_window}"
    run_dir = _ensure_dir(os.path.join(_BASE_DIR, tag, version))

    # 1. Save weights (delegates to wrapper helper if present)
    if hasattr(model, "save_weights"):
        model.save_weights(flare_class, run_dir)

    # 2. Dump metadata JSON
    meta = {
        "version": version,
        "flare_class": flare_class,
        "time_window": time_window,
        "metrics": metrics,
        "hyperparams": hyperparams,
        "description": description,
        "history": history,
    }
    meta.update({k: "<omitted>" for k in artefacts})  # keep file small
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return run_dir 