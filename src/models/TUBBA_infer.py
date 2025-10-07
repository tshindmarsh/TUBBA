import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer

# Add parent directory to path to import TUBBA_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TUBBA_utils import variable_is_circular, normalize_features, ensure_normalized_features

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

    # Correct device selection for Windows (and fallback to CPU if necessary)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS (Apple Metal GPU)")
    else:
        device = torch.device("cpu")
        print("⚠️ No GPU backend found — falling back to CPU")

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

    # Ensure normalization is done for all videos
    ensure_normalized_features(project)

    cache_path = os.path.join(video['folder'], 'normed_features.npy')
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

    # Correct device selection for Windows (and fallback to CPU if necessary)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS (Apple Metal GPU)")
    else:
        device = torch.device("cpu")
        print("⚠️ No GPU backend found — falling back to CPU")

    bundle = joblib.load(model_path)
    models = bundle['models']
    behaviors = bundle['behaviors']

    video = next(
        (v for v in project['videos']
         if v['name'] == video_name and v['folder'] == video_folder),
        None
    )
    if video is None:
        raise ValueError(f"Video {video_name} not found in project.")

    # Check if all videos already have normalized features
    all_feats_normed = True
    for video in project['videos']:
        if not os.path.exists(os.path.join(video['folder'], 'normed_features.npy')):
            all_feats_normed = False
            break

    # --- Compute normalization stats across all animals ---
    if not all_feats_normed:
        print("🔍 Computing normalization stats across all animals...")
        temp_features = []
        for video in project['videos']:
            feature_path = os.path.join(video['folder'], video['featureFile'])
            if not os.path.isfile(feature_path):
                continue
            try:
                df = pd.read_hdf(feature_path, key='perframes')
                video['loadedFeatures'] = df
                temp_features.append(df)
            except:
                continue

        full_df = pd.concat(temp_features, axis=0, ignore_index=True)
        feature_names = full_df.columns.tolist()

        zscore_stats, minmax_stats = {}, {}
        for col in feature_names:
            col_data = full_df[col].values
            if variable_is_circular(col_data):
                continue
            elif 'pca' in col.lower():
                col_min = np.nanmin(col_data)
                col_max = np.nanmax(col_data)
                minmax_stats[col] = (col_min, col_max)
            else:
                col_mean = np.nanmean(col_data)
                col_std = np.nanstd(col_data)
                zscore_stats[col] = (col_mean, col_std)

        imputer = SimpleImputer(strategy='constant', fill_value=0)

        for video in project['videos']:
            cache_path = os.path.join(video['folder'], 'normed_features.npy')
            X = normalize_features(video['loadedFeatures'], zscore_stats, minmax_stats)
            X = imputer.fit_transform(X)
            np.save(cache_path, X)
            print(f"✅ Normalized features for {video['name']}")
    else:
        sample_feature_path = os.path.join(project['videos'][0]['folder'], project['videos'][0]['featureFile'])
        df = pd.read_hdf(sample_feature_path, key='perframes')
        feature_names = df.columns.tolist()
        print("Found normalized features for all videos, skipping normalization stats computation.")

    cache_path = os.path.join(video['folder'], 'normed_features.npy')
    X = np.load(cache_path)

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

def TUBBA_modelInference_MU(project_json_path, video_name, video_folder):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    model_path = project.get('models', None)
    if not model_path or not os.path.isfile(model_path):
        msg = "❌ No trained model found in project file. Please run training first."
        raise FileNotFoundError(msg)

    # Correct device selection for Windows (and fallback to CPU if necessary)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA GPU backend")
    else:
        device = torch.device("cpu")
        print("⚠️ No CUDA found — falling back to CPU")

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

    # Check if all videos already have normalized features
    all_feats_normed = True
    for video in project['videos']:
        if not os.path.exists(os.path.join(video['folder'], 'normed_features.npy')):
            all_feats_normed = False
            break

    # --- Compute normalization stats across all animals ---
    if not all_feats_normed:
        print("🔍 Computing normalization stats across all animals...")
        temp_features = []
        for video in project['videos']:
            feature_path = os.path.join(video['folder'], video['featureFile'])
            if not os.path.isfile(feature_path):
                continue
            try:
                df = pd.read_hdf(feature_path, key='perframes')
                video['loadedFeatures'] = df
                temp_features.append(df)
            except:
                continue

        full_df = pd.concat(temp_features, axis=0, ignore_index=True)
        feature_names = full_df.columns.tolist()

        zscore_stats, minmax_stats = {}, {}
        for col in feature_names:
            col_data = full_df[col].values
            if variable_is_circular(col_data):
                continue
            elif 'pca' in col.lower():
                col_min = np.nanmin(col_data)
                col_max = np.nanmax(col_data)
                minmax_stats[col] = (col_min, col_max)
            else:
                col_mean = np.nanmean(col_data)
                col_std = np.nanstd(col_data)
                zscore_stats[col] = (col_mean, col_std)

        imputer = SimpleImputer(strategy='constant', fill_value=0)

        for video in project['videos']:
            cache_path = os.path.join(video['folder'], 'normed_features.npy')
            X = normalize_features(video['loadedFeatures'], zscore_stats, minmax_stats)
            X = imputer.fit_transform(X)
            np.save(cache_path, X)
            print(f"✅ Normalized features for {video['name']}")
    else:
        sample_feature_path = os.path.join(project['videos'][0]['folder'], project['videos'][0]['featureFile'])
        df = pd.read_hdf(sample_feature_path, key='perframes')
        feature_names = df.columns.tolist()
        print("Found normalized features for all videos, skipping normalization stats computation.")

    cache_path = os.path.join(video['folder'], 'normed_features.npy')
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
