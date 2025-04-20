# -----------------------------
# File: train_everest.py  (light wrapper around existing train script)
# -----------------------------
"""
Usage:  python train_everest.py  --specific-flare M5 --specific-window 24
This script is identical to SolarKnowledge_run_all_trainings.py but imports
EVEREST instead of SolarKnowledge and passes the same hyper‑parameters.
"""
import argparse, numpy as np, tensorflow as tf
from utils import get_training_data, data_transform, log, supported_flare_class
from model_tracking import save_model_with_metadata, get_next_version, get_latest_version
from EVEREST_model import EVEREST

def train(time_window, flare_class, auto_increment=True):
    X, y = get_training_data(time_window, flare_class)
    y = data_transform(y)
    input_shape = (X.shape[1], X.shape[2])
    model = EVEREST()
    model.build_base_model(input_shape)
    model.compile()
    version = get_next_version(flare_class, time_window) if auto_increment else "dev"
    class_counts = np.sum(y,0)
    weight = {0:1.0, 1: max(5, (len(y)/class_counts[1]))}
    hist = model.fit(X, y, epochs=100, class_weight=weight)
    metrics = {"final_training_tss": hist.history["tss"][-1],
               "final_training_loss": hist.history["loss"][-1]}
    hp = {"focal_alpha":0.25, "focal_gamma":2.0, "linear_attention":True}
    save_model_with_metadata(model, metrics, hp, hist, version, flare_class, time_window,
                             "EVEREST SHARP‑only experiment")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--specific-flare", "-f", default="M5")
    p.add_argument("--specific-window", "-w", default="24")
    a = p.parse_args()
    train(a.specific_window, a.specific_flare)