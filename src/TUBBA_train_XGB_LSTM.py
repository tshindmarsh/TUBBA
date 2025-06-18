import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from TUBBA_utils import variable_is_circular, normalize_features

class SmoothingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        center = lstm_out[:, lstm_out.size(1) // 2]
        return torch.sigmoid(self.fc(center)).squeeze(-1)

def train_TUBBAmodel(project_json_path, window_size=11, lstm_epochs=300):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    # --- Compute normalization stats across all animals ---
    temp_features = []
    for video in project['videos']:
        feature_path = os.path.join(video['folder'], video['featureFile'])
        if not os.path.isfile(feature_path):
            continue
        try:
            df = pd.read_hdf(feature_path, key='perframes')
            df = df.loc[:, ~df.columns.str.contains('_d1$|_d2$', regex=True)]
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

    behaviors = project['behaviors']
    half_window = window_size // 2
    model_bundle = {
        'models': {},
        'behaviors': behaviors,
        'window_size': window_size,
        'normalization': {
            'zscore': zscore_stats,
            'minmax': minmax_stats
        },
        'feature_names': feature_names
    }

    imputer = SimpleImputer(strategy='constant', fill_value=0)

    # --- Train XGB per behavior ---
    for behavior in behaviors:
        X_frames, y_frames = [], []

        for video in project['videos']:
            feature_path = os.path.join(video['folder'], video['featureFile'])
            if not os.path.isfile(feature_path):
                continue

            df = pd.read_hdf(feature_path, key='perframes')
            df = df.loc[:, ~df.columns.str.contains('_d1$|_d2$', regex=True)]
            X = normalize_features(df, zscore_stats, minmax_stats)

            y = np.full(len(X), np.nan)
            for (start, end, val) in video.get('annotations', {}).get(behavior, []):
                if val in [-1, 1]:
                    y[start:end+1] = val

            valid = ~np.isnan(y)
            X_frames.append(X[valid])
            y_frames.append((y[valid] == 1).astype(int))

        if not X_frames:
            print(f"⚠️ No training data for behavior: {behavior}")
            continue

        X_all = np.vstack(X_frames)
        y_all = np.concatenate(y_frames)
        X_all = imputer.fit_transform(X_all)

        pos_count = (y_all == 1).sum()
        neg_count = (y_all == 0).sum()
        sW = neg_count / (pos_count + 1e-6)

        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.1, stratify=y_all)

        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric=['logloss', 'auc', 'error'],
            n_estimators=500,
            max_depth=6,
            min_child_weight=5,
            gamma=0.8,
            subsample=0.8,
            colsample_bytree=0.65,
            scale_pos_weight=sW,
            learning_rate=0.02,
            early_stopping_rounds=10
        )

        print(f"🚀 Training XGBoost model for behavior: {behavior}")
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        model_bundle['models'][behavior] = {'xgb': xgb_model}

    model_bundle['imputer'] = imputer  # Save once

    # --- Train LSTM using all behavior confidences ---
    for behavior in behaviors:
        print(f"🧠 Training LSTM smoother for behavior: {behavior}")
        smoothed_X, smoothed_Y = [], []

        for video in project['videos']:
            feature_path = os.path.join(video['folder'], video['featureFile'])
            if not os.path.isfile(feature_path):
                continue

            df = pd.read_hdf(feature_path, key='perframes')
            df = df.loc[:, ~df.columns.str.contains('_d1$|_d2$', regex=True)]
            X = normalize_features(df, zscore_stats, minmax_stats)
            X = imputer.transform(X)

            # Get RF confidences for all behaviors
            conf_all = np.zeros((len(X), len(behaviors)))
            for j, b in enumerate(behaviors):
                conf_all[:, j] = model_bundle['models'][b]['xgb'].predict_proba(X)[:, 1]

            y = np.full(len(X), np.nan)
            for (start, end, val) in video.get('annotations', {}).get(behavior, []):
                if val in [-1, 1]:
                    y[start:end+1] = val

            for t in range(half_window, len(conf_all) - half_window):
                if np.isnan(y[t]):
                    continue
                window = conf_all[t - half_window:t + half_window + 1, :]  # (window, behaviors)
                smoothed_X.append(window)
                smoothed_Y.append(int(y[t] == 1))

        if not smoothed_X:
            print(f"⚠️ No smoothing data for behavior: {behavior}")
            continue

        X_tensor = torch.tensor(np.stack(smoothed_X), dtype=torch.float32)
        Y_tensor = torch.tensor(np.array(smoothed_Y), dtype=torch.float32)

        model = SmoothingLSTM(input_size=len(behaviors))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        for epoch in range(lstm_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, Y_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                print(f"  🌀 Epoch {epoch:3d} | Loss: {loss.item():.4f}")

        model_bundle['models'][behavior]['lstm'] = model.state_dict()

    model_path = os.path.splitext(project_json_path)[0] + "_trainedModel.pkl"
    joblib.dump(model_bundle, model_path)
    print(f"\n✅ All models trained and saved to: {model_path}")
    return model_path
