# Inference script for XGBoost + LSTM smoother pipeline using all behavior confidences as LSTM input

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from tkinter import messagebox
from TUBBA_utils import variable_is_circular, normalize_features

class SmoothingLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1,
                                  batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        center = lstm_out[:, lstm_out.size(1) // 2]
        return torch.sigmoid(self.fc(center)).squeeze(-1)

def TUBBA_modelInference(project_json_path, video_name, video_folder):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    model_path = project.get('models', None)
    if not model_path or not os.path.isfile(model_path):
        msg = "❌ No trained model found in project file. Please run training first."
        messagebox.showerror("TUBBA Inference Error", msg)
        raise FileNotFoundError(msg)

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
    df_features = df_features.loc[:, ~df_features.columns.str.contains('_d1$|_d2$', regex=True)]
    X = normalize_features(df_features, zscore_stats, minmax_stats)
    if imputer is not None:
        X = imputer.transform(X)
    else:
        X = np.nan_to_num(X, nan=0.0)

    # Generate XGBoost confidences for all behaviors
    rf_conf_all = np.zeros((len(X), len(behaviors)))
    for j, b in enumerate(behaviors):
        if b in models:
            rf_conf_all[:, j] = models[b]['xgb'].predict_proba(X)[:, 1]

    predictions = {}
    confidence = {}

    for i, behavior in enumerate(behaviors):
        if behavior not in models:
            continue

        # 🚧 TEMP: output raw XGB confidence instead of LSTM
        # rf_conf = rf_conf_all[:, i]
        # confidence[behavior] = rf_conf.tolist()
        # predictions[behavior] = (rf_conf >= 0.5).astype(int).tolist()
        # continue  # skip LSTM

        segments = []
        valid_indices = []
        for t in range(half_window, len(rf_conf_all) - half_window):
            segment = rf_conf_all[t - half_window:t + half_window + 1, :]  # shape (window_size, num_behaviors)
            segments.append(segment)
            valid_indices.append(t)

        if not segments:
            confidence[behavior] = [0.0] * len(X)
            predictions[behavior] = [0] * len(X)
            continue

        segment_tensor = torch.tensor(np.stack(segments), dtype=torch.float32)

        lstm_model = SmoothingLSTM(input_size=len(behaviors))
        lstm_model.load_state_dict(models[behavior]['lstm'])
        lstm_model.eval()

        with torch.no_grad():
            probs = lstm_model(segment_tensor).numpy()
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
