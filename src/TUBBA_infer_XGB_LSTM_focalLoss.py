# Inference script for XGBoost + LSTM smoother pipeline using all behavior confidences as LSTM input

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tkinter import messagebox
from TUBBA_utils import variable_is_circular, normalize_features

class SmoothingLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=32, dropout_rate=0.3, attention=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = attention
        if attention:
            self.attn = nn.Linear(hidden_size * 2, 1)
            self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if self.attention:
            attn_weights = self.softmax(self.attn(lstm_out))  # (batch, seq_len, 1)
            context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)
        else:
            center = lstm_out[:, lstm_out.size(1) // 2]
            context = center
        context = self.dropout(context)
        return self.fc(context).squeeze(-1)  # logits

def TUBBA_modelInference(project_json_path, video_name, video_folder):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    model_path = project.get('models', None)
    if not model_path or not os.path.isfile(model_path):
        msg = "❌ No trained model found in project file. Please run training first."
        messagebox.showerror("TUBBA Inference Error", msg)
        raise FileNotFoundError(msg)

    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS (Apple Metal) backend")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS not available — falling back to CPU")

    bundle = joblib.load(model_path)
    models = bundle['models']
    behaviors = bundle['behaviors']
    window_size = bundle['window_size']
    half_window = window_size // 2

    zscore_stats = bundle['normalization'].get('zscore', {})
    minmax_stats = bundle['normalization'].get('minmax', {})
    imputer = bundle.get('imputer', None)

    video = next(
        (v for v in project['videos']
         if v['name'] == video_name and v['folder'] == video_folder),
        None
    )
    if video is None:
        raise ValueError(f"Video {video_name} not found in project.")

    feature_path = os.path.join(video['folder'], video['featureFile'])
    if not os.path.isfile(feature_path):
        raise FileNotFoundError(f"⚠️ Feature file not found: {feature_path}")

    df_features = pd.read_hdf(feature_path, key='perframes')
    X = normalize_features(df_features, zscore_stats, minmax_stats)
    if imputer is not None:
        X = imputer.transform(X)
    else:
        X = np.nan_to_num(X, nan=0.0)

    rf_conf_all = np.zeros((len(X), len(behaviors)))
    for j, b in enumerate(behaviors):
        if b in models and 'xgb' in models[b]:
            clf = models[b]['xgb']
            preds = clf.predict_proba(X)[:, 1]
            rf_conf_all[:, j] = preds

    predictions = {}
    confidence = {}

    for i, behavior in enumerate(behaviors):
        if behavior not in models or 'lstm' not in models[behavior]:
            confidence[behavior] = rf_conf_all[:, i].tolist()
            predictions[behavior] = (rf_conf_all[:, i] >= 0.5).astype(int).tolist()
            continue

        # 🚧 TEMP: output raw XGB confidence instead of LSTM
        # rf_conf = rf_conf_all[:, i]
        # confidence[behavior] = rf_conf.tolist()
        # predictions[behavior] = (rf_conf >= 0.5).astype(int).tolist()
        # continue  # skip LSTM

        segments = []
        valid_indices = []
        for t in range(half_window, len(rf_conf_all) - half_window):
            segment = rf_conf_all[t - half_window:t + half_window + 1, :]
            segments.append(segment)
            valid_indices.append(t)

        if not segments:
            confidence[behavior] = [0.0] * len(X)
            predictions[behavior] = [0] * len(X)
            continue

        segment_tensor = torch.tensor(np.stack(segments), dtype=torch.float32).to(device)

        # Load LSTM model with saved architecture config
        lstm_config = bundle.get("lstm_config", {})
        lstm_model = SmoothingLSTM(
            input_size=lstm_config.get("input_size", len(behaviors)),
            hidden_size=lstm_config.get("hidden_size", 32),
            dropout_rate=lstm_config.get("dropout", 0.3),
            attention=lstm_config.get("attention", True)
        ).to(device)
        lstm_model.load_state_dict(models[behavior]['lstm'])
        lstm_model.eval()

        with torch.no_grad():
            logits = lstm_model(segment_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

        confidence_array = np.zeros(len(X))
        prediction_array = np.zeros(len(X), dtype=int)
        for j, t in enumerate(valid_indices):
            confidence_array[t] = probs[j]
            prediction_array[t] = preds[j]

        confidence[behavior] = confidence_array.tolist()
        predictions[behavior] = prediction_array.tolist()

    return {
        "predictions": predictions,
        "confidence": confidence
    }
