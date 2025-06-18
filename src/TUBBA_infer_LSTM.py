import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from TUBBA_utils import variable_is_circular, zscore_normalize_preserve_nans

class BehaviorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BehaviorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.2, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        center = lstm_out[:, lstm_out.size(1) // 2]
        center = self.dropout(center)
        hidden = self.relu(self.fc1(center))
        return self.fc2(hidden).squeeze(-1)

def TUBBA_modelInference(project_json_path, video_name, video_folder):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    model_path = project.get('models', None)
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError("❌ Trained model not found in project.")

    bundle = joblib.load(model_path)
    behaviors = bundle['behaviors']
    input_size = bundle['input_size']
    hidden_size = bundle['hidden_size']
    window_size = bundle['window_size']
    half_window = window_size // 2
    imputer = bundle['imputer']
    model_weights = bundle['models']

    video = next((v for v in project['videos']
                  if v['name'] == video_name and v['folder'] == video_folder), None)
    if video is None:
        raise ValueError(f"Video {video_name} not found in project.")

    feature_path = os.path.join(video['folder'], video['featureFile'])
    df_features = pd.read_hdf(feature_path, key='perframes')
    X = df_features.values

    for i in range(X.shape[1]):
        if not variable_is_circular(X[:, i]):
            X[:, i] = zscore_normalize_preserve_nans(X[:, i])
    X = imputer.transform(X)

    n_frames = len(X)
    confidence = {b: [0.0] * n_frames for b in behaviors}
    predictions = {b: [0] * n_frames for b in behaviors}

    # Precompute all segments
    segments = []
    valid_indices = []
    for t in range(half_window, n_frames - half_window):
        segment = X[t - half_window:t + half_window + 1]
        if segment.shape[0] == window_size:
            segments.append(segment)
            valid_indices.append(t)
    segment_tensor = torch.tensor(np.stack(segments), dtype=torch.float32)

    # Always use CPU to avoid MPS issues
    device = torch.device("cpu")

    batch_size = 512  # Tune this depending on available RAM

    for b_idx, behavior in enumerate(behaviors):
        if behavior not in model_weights:
            continue

        print(f"\n🧠 Running model for behavior: {behavior}")

        model = BehaviorLSTM(input_size, hidden_size).to(device)
        model.load_state_dict(model_weights[behavior])
        model.eval()

        probs = np.zeros(len(valid_indices))

        with torch.no_grad():
            for start in range(0, len(valid_indices), batch_size):
                end = min(start + batch_size, len(valid_indices))
                batch_input = segment_tensor[start:end].to(device)
                output = model(batch_input)
                probs[start:end] = torch.sigmoid(output).cpu().numpy()

        for i, t in enumerate(valid_indices):
            confidence[behavior][t] = probs[i]
            predictions[behavior][t] = int(probs[i] >= 0.5)

    return {
        "predictions": predictions,
        "confidence": confidence
    }