# Train script for Conv1D behavior classifier (supervised with class balancing and early stopping)
import time
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.impute import SimpleImputer
from TUBBA_utils import variable_is_circular, zscore_normalize_preserve_nans

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
        # x: (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.blocks(x)
        logits = self.classifier(x)  # (B, C, T)
        return logits.permute(0, 2, 1)  # (B, T, C)

def train_TUBBAmodel(project_json_path, window_size=201, batch_size=124, epochs=100, patience=10):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")
    behaviors = project['behaviors']
    half_window = window_size // 2
    imputer = SimpleImputer(strategy='constant', fill_value=0)

    all_X, all_Y = [], []

    for video in project['videos']:
        feature_path = os.path.join(video['folder'], video['featureFile'])
        if not os.path.isfile(feature_path):
            continue

        df_features = pd.read_hdf(feature_path, key='perframes')
        X_raw = df_features.values

        for i in range(X_raw.shape[1]):
            if not variable_is_circular(X_raw[:, i]):
                X_raw[:, i] = zscore_normalize_preserve_nans(X_raw[:, i])

        X_full = imputer.fit_transform(X_raw)

        Y_full = np.zeros((X_full.shape[0], len(behaviors)), dtype=float)
        valid = np.zeros_like(Y_full, dtype=bool)

        for b_idx, behavior in enumerate(behaviors):
            for (start, end, val) in video.get('annotations', {}).get(behavior, []):
                if val in [-1, 1]:
                    Y_full[start:end+1, b_idx] = 1 if val == 1 else 0
                    valid[start:end+1, b_idx] = True

        for t in range(half_window, len(X_full) - half_window):
            window = X_full[t - half_window:t + half_window + 1]
            label = Y_full[t]
            mask = valid[t]
            if np.any(mask):
                all_X.append(window)
                all_Y.append(label)

        print('Finished appending video frames for video: ', video['name'], '')

    if not all_X:
        raise ValueError("❌ No valid training data found.")

    print('Finished contructing dataset. Converting to tensors.')
    X_tensor = torch.tensor(np.stack(all_X), dtype=torch.float32)  # (N, T, F)
    Y_tensor = torch.tensor(np.stack(all_Y), dtype=torch.float32)  # (N, C)

    # Split into training and validation sets
    dataset = TensorDataset(X_tensor, Y_tensor)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=4,pin_memory=True)

    model = Conv1DBehaviorNet(input_dim=X_tensor.shape[2], num_behaviors=Y_tensor.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model = model.to(device)

    # Class weights for each behavior
    pos_counts = torch.sum(Y_tensor, dim=0)
    neg_counts = torch.tensor(Y_tensor.shape[0]) - pos_counts
    class_weights = neg_counts / (pos_counts + 1e-6)
    class_weights = class_weights.to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            t0 = time.time()

            xb, yb = xb.to(device), yb.to(device)
            t1 = time.time()

            optimizer.zero_grad()
            logits = model(xb)[:, window_size // 2]
            t2 = time.time()

            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, weight=class_weights)
            t3 = time.time()

            loss.backward()
            optimizer.step()
            t4 = time.time()

            print(
                f"to(device): {t1 - t0:.3f}s | forward: {t2 - t1:.3f}s | loss: {t3 - t2:.3f}s | backward+opt: {t4 - t3:.3f}s")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)[:, window_size // 2]
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, weight=class_weights)
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # Save model
    model_bundle = {
        'model_state': best_state,
        'behaviors': behaviors,
        'input_dim': X_tensor.shape[2],
        'window_size': window_size,
        'imputer': imputer,
    }
    model_path = os.path.splitext(project_json_path)[0] + "_trainedModel.pkl"
    joblib.dump(model_bundle, model_path)
    print(f"\n✅ Conv1D model saved to: {model_path}")
    return model_path