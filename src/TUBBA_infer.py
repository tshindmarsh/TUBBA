# Inference script for XGBoost + LSTM smoother pipeline using all behavior confidences as LSTM input

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from TUBBA_utils import variable_is_circular, normalize_features
from sklearn.impute import SimpleImputer

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

    video = next(
        (v for v in project['videos']
         if v['name'] == video_name and v['folder'] == video_folder),
        None
    )
    if video is None:
        raise ValueError(f"Video {video_name} not found in project.")

    cache_path = os.path.join(video['folder'], 'normed_features.npy')
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"⚠️ Feature file not found: {cache_path}")

    X = np.load(cache_path)

    rf_conf_all = np.zeros((len(X), len(behaviors)))
    for j, b in enumerate(behaviors):
        if b in models and 'xgb' in models[b]:
            proba = models[b]['xgb'].predict_proba(X)
            if proba.shape[1] == 2:
                rf_conf_all[:, j] = proba[:, 1]
            else:
                rf_conf_all[:, j] = 0.0

    predictions = {}
    confidence = {}

    for i, behavior in enumerate(behaviors):
        if behavior not in models or 'lstm' not in models[behavior]:
            confidence[behavior] = rf_conf_all[:, i].tolist()
            predictions[behavior] = (rf_conf_all[:, i] >= 0.5).astype(int).tolist()
            continue

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

        # Load LSTM model with or without attention
        attention = 'attn.weight' in models[behavior]['lstm']  # backward compatibility check
        lstm_model = SmoothingLSTM(input_size=len(behaviors), attention=attention).to(device)
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

def TUBBA_modelInference_lite(project_json_path, video_name, video_folder):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    model_path = project.get('models', None)
    if not model_path or not os.path.isfile(model_path):
        msg = "❌ No trained model found in project file. Please run training first."
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

    video = next(
        (v for v in project['videos']
         if v['name'] == video_name and v['folder'] == video_folder),
        None
    )
    if video is None:
        raise ValueError(f"Video {video_name} not found in project.")

    cache_path = os.path.join(video['folder'], 'normed_features.npy')
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"⚠️ Feature file not found: {cache_path}")

    X = np.load(cache_path)

    predictions = {}
    confidence = {}

    for b in behaviors:
        if b in models and 'xgb' in models[b]:
            proba = models[b]['xgb'].predict_proba(X)
            if proba.shape[1] == 2:
                conf = proba[:, 1]
            else:
                conf = np.zeros(len(X))
        else:
            conf = np.zeros(len(X))

        predictions[b] = (conf >= 0.5).astype(int).tolist()
        confidence[b] = conf.tolist()

    return {
        "predictions": predictions,
        "confidence": confidence
    }
