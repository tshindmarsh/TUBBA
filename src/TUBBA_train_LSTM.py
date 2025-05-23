# Full reworked TUBBA LSTM training script with Validation, Early Stopping, and Weight Decay

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.impute import SimpleImputer
from PyQt5.QtWidgets import QMessageBox
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

def train_TUBBAmodel(project_json_path, window_size=61, batch_size=64, epochs=100, patience=10):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    behaviors = project['behaviors']
    half_window = window_size // 2
    all_X, all_Y, all_B = [], [], []

    for video in project['videos']:
        feature_path = os.path.join(video['folder'], video['featureFile'])
        if not os.path.isfile(feature_path):
            print(f"⚠️ Feature file not found: {feature_path}")
            continue

        df_features = pd.read_hdf(feature_path, key='perframes')
        X = df_features.values

        # Normalize non-circular columns
        for i in range(X.shape[1]):
            if not variable_is_circular(X[:, i]):
                X[:, i] = zscore_normalize_preserve_nans(X[:, i])

        # Impute NaNs with zero (after normalization)
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X = imputer.fit_transform(X)

        for b_idx, behavior in enumerate(behaviors):
            annotations = video.get('annotations', {}).get(behavior, [])
            for (start, end, val) in annotations:
                if val not in [-1, 1]:
                    continue
                for t in range(start, end + 1):
                    if t - half_window < 0 or t + half_window >= len(X):
                        continue
                    segment = X[t - half_window:t + half_window + 1]
                    if segment.shape[0] == window_size:
                        all_X.append(segment)
                        all_Y.append(1 if val == 1 else 0)
                        all_B.append(b_idx)

    if not all_X:
        raise ValueError("❌ No labeled frames found for training.")

    X_tensor = torch.tensor(np.stack(all_X), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(all_Y), dtype=torch.float32)
    B_tensor = torch.tensor(np.array(all_B), dtype=torch.long)

    input_size = X_tensor.shape[2]
    hidden_size = 64
    models = {}

    for b_idx, behavior in enumerate(behaviors):
        mask = B_tensor == b_idx
        if mask.sum() == 0:
            print(f"⚠️ No training data for behavior: {behavior}")
            continue

        full_dataset = TensorDataset(X_tensor[mask], Y_tensor[mask])
        val_size = int(0.1 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = BehaviorLSTM(input_size, hidden_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        pos_count = (Y_tensor[mask] == 1).sum()
        neg_count = (Y_tensor[mask] == 0).sum()
        pos_weight = neg_count / (pos_count + 1e-6)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        print(f"\n🧠 Training model for behavior: {behavior}")
        best_val_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)

            print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        models[behavior] = model.state_dict()

    model_bundle = {
        'models': models,
        'behaviors': behaviors,
        'input_size': input_size,
        'hidden_size': hidden_size,
        'window_size': window_size,
        'imputer': imputer,
    }

    model_path = os.path.splitext(project_json_path)[0] + "_trainedModel.pkl"
    joblib.dump(model_bundle, model_path)
    print(f"\n✅ All models trained and saved to: {model_path}")
    return model_path