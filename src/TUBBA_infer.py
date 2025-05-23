# Inference script for Conv1D behavior model with GPU/MPS support

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from TUBBA_utils import variable_is_circular, zscore_normalize_preserve_nans
from tkinter import messagebox

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)

class Conv1DBehaviorNet(nn.Module):
    def __init__(self, input_dim, num_behaviors, hidden_channels=124, num_blocks=5, kernel_size=5):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(*[ResidualBlock1D(hidden_channels, kernel_size) for _ in range(num_blocks)])
        self.classifier = nn.Conv1d(hidden_channels, num_behaviors, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.input_proj(x)
        x = self.blocks(x)
        logits = self.classifier(x)
        return torch.sigmoid(logits.permute(0, 2, 1))  # (B, T, C)

def TUBBA_modelInference(project_json_path, video_name, video_folder):
    # Use MPS if available (Apple Silicon), otherwise CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load project
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    # Load trained model
    model_path = project.get('models', None)
    if not model_path or not os.path.isfile(model_path):
        msg = "❌ No trained model found in project file. Please run training first."
        messagebox.showerror("TUBBA Inference Error", msg)
        raise FileNotFoundError(msg)

    bundle = joblib.load(model_path)
    window_size = bundle['window_size']
    half_window = window_size // 2
    behaviors = bundle['behaviors']

    # Load video entry and features
    video = next((v for v in project['videos'] if v['name'] == video_name and v['folder'] == video_folder), None)
    if video is None:
        raise ValueError(f"Video {video_name} not found in project.")

    feature_path = os.path.join(video['folder'], video['featureFile'])
    df_features = pd.read_hdf(feature_path, key='perframes')
    X_raw = df_features.values

    for i in range(X_raw.shape[1]):
        if not variable_is_circular(X_raw[:, i]):
            X_raw[:, i] = zscore_normalize_preserve_nans(X_raw[:, i])

    imputer = bundle['imputer']
    X_full = imputer.transform(np.nan_to_num(X_raw, nan=0.0))

    # Prepare sliding windows
    segments = []
    indices = []
    for t in range(half_window, len(X_full) - half_window):
        segment = X_full[t - half_window:t + half_window + 1]
        segments.append(segment)
        indices.append(t)

    if not segments:
        return {"predictions": {b: [0]*len(X_full) for b in behaviors},
                "confidence": {b: [0.0]*len(X_full) for b in behaviors}}

    segment_tensor = torch.tensor(np.stack(segments), dtype=torch.float32).to(device)

    model = Conv1DBehaviorNet(input_dim=X_full.shape[1], num_behaviors=len(behaviors))
    model.load_state_dict(bundle['model_state'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(segment_tensor)  # (N, T, C)
        center_preds = outputs[:, window_size // 2]  # (N, C)
        probs = center_preds.cpu().numpy()

    confidence = {b: [0.0] * len(X_full) for b in behaviors}
    prediction = {b: [0] * len(X_full) for b in behaviors}

    for i, t in enumerate(indices):
        for j, b in enumerate(behaviors):
            confidence[b][t] = float(probs[i, j])
            prediction[b][t] = int(probs[i, j] >= 0.5)

    return {
        "predictions": prediction,
        "confidence": confidence
    }