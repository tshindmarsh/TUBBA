import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add parent directory to path to import TUBBA_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TUBBA_utils import variable_is_circular, normalize_features


def augment_labeled_frames(X_frames, y_frames, noise_factor=0.05, copies=3,name=None):
    """Efficiently augment labeled frames by adding noise to positive and negative examples."""
    all_X = []
    all_y = []

    for X, y in zip(X_frames, y_frames):
        all_X.append(X)
        all_y.append(y)

        pos_mask = y == 1
        neg_mask = y == -1

        # Skip if nothing to augment
        if not np.any(pos_mask) and not np.any(neg_mask):
            continue

        # Compute once per video
        feature_scales = np.nanstd(X, axis=0, keepdims=True) + 1e-8
        if np.any(np.isnan(feature_scales)):
            feature_scales[np.isnan(feature_scales)] = 1

        pos_idx = np.where(pos_mask)[0]
        neg_idx = np.where(neg_mask)[0]

        for _ in range(copies):
            X_aug = X.copy()

            if len(pos_idx) > 0:
                noise = np.random.normal(0, noise_factor * feature_scales, size=(len(pos_idx), X.shape[1]))
                X_aug[pos_idx] = X[pos_idx] + noise

            if len(neg_idx) > 0:
                noise = np.random.normal(0, noise_factor * feature_scales, size=(len(neg_idx), X.shape[1]))
                X_aug[neg_idx] = X[neg_idx] + noise

            all_X.append(X_aug)
            all_y.append(y)  # Labels unchanged

    return all_X, all_y


def create_balanced_frame_samples(X_frames, y_frames, target_ratio=(1, 1, 0)):
    """Efficiently create a balanced dataset of positive, negative, and optional unlabeled frames."""
    pos_X, neg_X, unlabeled_X = [], [], []

    for X, y in zip(X_frames, y_frames):
        if np.any(y == 1):
            pos_X.append(X[y == 1])
        if np.any(y == -1):
            neg_X.append(X[y == -1])
        if target_ratio[2] > 0 and np.any(np.isnan(y)):
            unlabeled_X.append(X[np.isnan(y)])

    pos_X = np.vstack(pos_X) if pos_X else np.empty((0, X_frames[0].shape[1]))
    neg_X = np.vstack(neg_X) if neg_X else np.empty((0, X_frames[0].shape[1]))
    unlabeled_X = np.vstack(unlabeled_X) if unlabeled_X else np.empty((0, X_frames[0].shape[1]))

    n_pos, n_neg = len(pos_X), len(neg_X)
    pos_ratio, neg_ratio, unlabeled_ratio = target_ratio

    if n_pos == 0 or n_neg == 0:
        return np.array([]), np.array([])

    if n_pos / pos_ratio <= n_neg / neg_ratio:
        target_pos = n_pos
        target_neg = int(n_pos * neg_ratio / pos_ratio)
        target_unlabeled = int(n_pos * unlabeled_ratio / pos_ratio)
    else:
        target_neg = n_neg
        target_pos = int(n_neg * pos_ratio / neg_ratio)
        target_unlabeled = int(n_neg * unlabeled_ratio / neg_ratio)

    # Sample without replacement
    sampled_pos = pos_X[np.random.choice(n_pos, target_pos, replace=False)] if target_pos < n_pos else pos_X
    sampled_neg = neg_X[np.random.choice(n_neg, target_neg, replace=False)] if target_neg < n_neg else neg_X
    sampled_unlabeled = (
        unlabeled_X[np.random.choice(len(unlabeled_X), min(target_unlabeled, len(unlabeled_X)), replace=False)]
        if target_unlabeled > 0 and len(unlabeled_X) > 0 else np.empty((0, pos_X.shape[1]))
    )

    X_balanced = np.vstack([sampled_pos, sampled_neg, sampled_unlabeled])
    y_balanced = np.concatenate([
        np.ones(len(sampled_pos)),
        -np.ones(len(sampled_neg)),
        np.zeros(len(sampled_unlabeled)),
    ])

    # Shuffle
    idx = np.random.permutation(len(X_balanced))
    return X_balanced[idx], y_balanced[idx]


def generate_windows_for_all_labels(conf_all, y, window_size=101, offsets=0):
    """Efficiently generate LSTM windows for labeled frames with offsets."""
    half_window = window_size // 2
    seq_len = conf_all.shape[0]
    smoothed_X = []
    smoothed_Y = []

    labeled_idx = np.where(~np.isnan(y))[0]

    for idx in labeled_idx:
        for offset in offsets:
            center = idx + offset
            if center < half_window or center >= seq_len - half_window:
                continue  # skip out-of-bounds windows

            window = conf_all[center - half_window:center + half_window + 1, :]
            label_val = y[center]
            if label_val == 1:
                smoothed_X.append(window)
                smoothed_Y.append(1)
            elif label_val == -1:
                smoothed_X.append(window)
                smoothed_Y.append(0)

    if not smoothed_X:
        return np.empty((0, window_size, conf_all.shape[1])), np.empty((0,))

    return np.stack(smoothed_X), np.array(smoothed_Y)


class SmoothingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, dropout_rate=0.3, attention=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,  # Added depth
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = attention
        if attention:
            self.attn = nn.Linear(hidden_size * 2, 1)
            self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if self.attention:
            # Simple attention mechanism
            attn_weights = self.softmax(self.attn(lstm_out))
            context = torch.sum(lstm_out * attn_weights, dim=1)
            context = self.dropout(context)
            return self.fc(context).squeeze(-1)
        else:
            # Original center-frame approach
            center = lstm_out[:, lstm_out.size(1) // 2]
            center = self.dropout(center)
            return self.fc(center).squeeze(-1)


def train_TUBBAmodel(project_json_path, window_size=26, lstm_epochs=2500):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS (Apple Metal GPU)")
    else:
        device = torch.device("cpu")
        print("⚠️ No GPU backend found — falling back to CPU")

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

    behaviors = project['behaviors']
    half_window = window_size // 2
    model_bundle = {
        'models': {},
        'behaviors': behaviors,
        'window_size': window_size,
        'feature_names': feature_names
    }

    # --- Train XGB per behavior ---
    for behavior in behaviors:
        X_all = []
        y_all = []

        for video in project['videos']:
            cache_path = os.path.join(video['folder'], 'normed_features.npy')
            if not os.path.isfile(cache_path):
                continue

            try:
                X = np.load(cache_path)
            except Exception as e:
                print(f"❌ Failed to load {cache_path}: {e}")
                continue

            y = np.full(len(X), np.nan)
            for (start, end, val) in video.get('annotations', {}).get(behavior, []):
                if val in [-1, 1]:
                    y[start:end + 1] = val

            mask = ~np.isnan(y)
            if np.sum(mask) == 0:
                continue

            y_masked = np.where(y[mask] == -1, 0, y[mask])
            X_masked = X[mask]

            X_all.append(X_masked)
            y_all.append(y_masked)

        if not X_all:
            print(f"⚠️ No valid training data for behavior: {behavior}")
            continue

        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)

        pos_frac = np.mean(y_all == 1)
        neg_frac = 1 - pos_frac
        sample_weights = np.where(y_all == 1, 1 / (pos_frac + 1e-6), 1 / (neg_frac + 1e-6))

        if len(X_all) == 0 or pos_frac == 1 or neg_frac == 1:
             print(f"⚠️ No balanced data available for behavior: {behavior}")
             continue

        # Skipping this in favor of pos/neg weighting
            # Now use the balanced frame samples function to create a balanced dataset
            # Use target_ratio=(1, 1, 0.2) to include some unlabeled samples (0.2x the limiting class)
            # X_balanced, y_balanced = create_balanced_frame_samples(X_frames, y_frames, target_ratio=(1, 1, 0))

        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X_all, y_all, sample_weights, test_size=0.1, stratify=y_all
        )

        # Train XGBoost model
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=2000,
            max_depth=6,
            min_child_weight=5,
            gamma=0.8,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.01,
            early_stopping_rounds=20
        )

        print(f"🚀 Training XGBoost model for behavior: {behavior}")
        xgb_model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)]
        )
        model_bundle['models'][behavior] = {'xgb': xgb_model}


    # --- Train LSTM using all behavior confidences ---
    for behavior in behaviors:
        print(f"🧠 Training LSTM smoother for behavior: {behavior}")
        smoothed_X, smoothed_Y = [], []

        for video in project['videos']:
            cache_path = os.path.join(video['folder'], 'normed_features.npy')
            if not os.path.isfile(cache_path):
                continue

            X = np.load(cache_path)

            # Generate confidence scores
            conf_all = np.zeros((len(X), len(behaviors)))
            for j, b in enumerate(behaviors):
                conf_all[:, j] = model_bundle['models'][b]['xgb'].predict_proba(X)[:, 1]

            # Get raw labels with both positive and negative
            y = np.full(len(X), np.nan)
            for (start, end, val) in video.get('annotations', {}).get(behavior, []):
                if val in [-1, 1]:
                    y[start:end + 1] = val

            # Generate windows from both positive and negative labels
            video_X, video_Y = generate_windows_for_all_labels(
                conf_all, y, window_size, offsets=[0]
            )

            smoothed_X.extend(video_X)
            smoothed_Y.extend(video_Y)

        if not smoothed_X:
            print(f"⚠️ No smoothing data for behavior: {behavior}")
            continue

        X_np = np.stack(smoothed_X)
        Y_np = np.array(smoothed_Y)

        # Convert to torch tensors
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_np, Y_np, test_size=0.1, stratify=Y_np
        )

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        Y_val = torch.tensor(Y_val, dtype=torch.float32).to(device)

        model = SmoothingLSTM(input_size=len(behaviors)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, min_lr=1e-6
        )

        pos_weight_val = (len(Y_train) - Y_train.sum()) / (Y_train.sum() + 1e-6)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)

        # Early stopping setup
        patience = 50
        best_val_loss = float('inf')
        best_model_state = None
        epochs_since_improvement = 0

        for epoch in range(lstm_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, Y_train)
            loss.backward()
            optimizer.step()

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = criterion(val_logits, Y_val)

            if val_loss.item() < best_val_loss - 1e-5:
                best_val_loss = val_loss.item()
                best_model_state = model.state_dict()
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epoch % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"  🌀 Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Patience: {epochs_since_improvement} | LR: {current_lr:.6f}")
            if epochs_since_improvement >= patience:
                print(f"  ⏹️ Early stopping at epoch {epoch} — best val loss: {best_val_loss:.4f}")
                break

            scheduler.step(val_loss.item())

        model_bundle['models'][behavior]['lstm'] = best_model_state
        torch.mps.empty_cache()

    model_path = os.path.splitext(project_json_path)[0] + "_trainedModel.pkl"
    joblib.dump(model_bundle, model_path)
    print(f"\n✅ All models trained and saved to: {model_path}")
    return model_path


def train_TUBBAmodel_lite(project_json_path):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

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

    behaviors = project['behaviors']
    model_bundle = {
        'models': {},
        'behaviors': behaviors,
        'window_size': None,
        'feature_names': feature_names
    }

    for behavior in behaviors:

        print(f"🧠 Training lightweight XGB for behavior: {behavior}")

        X_all = []
        y_all = []

        for video in project['videos']:
            cache_path = os.path.join(video['folder'], 'normed_features.npy')
            if not os.path.isfile(cache_path):
                continue

            try:
                X = np.load(cache_path)
            except Exception as e:
                print(f"❌ Failed to load {cache_path}: {e}")
                continue

            y = np.full(len(X), np.nan)
            for (start, end, val) in video.get('annotations', {}).get(behavior, []):
                if val in [-1, 1]:
                    y[start:end + 1] = val

            mask = ~np.isnan(y)
            if np.sum(mask) == 0:
                continue

            y_masked = np.where(y[mask] == -1, 0, y[mask])
            X_masked = X[mask]

            X_all.append(X_masked)
            y_all.append(y_masked)

        if not X_all:
            print(f"⚠️ No valid training data for behavior: {behavior}")
            continue

        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)

        pos_frac = np.mean(y_all == 1)
        neg_frac = 1 - pos_frac
        sample_weights = np.where(y_all == 1, 1 / (pos_frac + 1e-6), 1 / (neg_frac + 1e-6))

        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X_all, y_all, sample_weights, test_size=0.1, stratify=y_all
        )

        # Train XGBoost model
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=800,
            max_depth=6,
            min_child_weight=5,
            gamma=0.8,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.01,
            early_stopping_rounds=20
        )

        xgb_model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        model_bundle['models'][behavior] = {'xgb': xgb_model}


    model_path = os.path.splitext(project_json_path)[0] + "_lightXGB.pkl"
    joblib.dump(model_bundle, model_path)
    print(f"\n✅ Lightweight XGB models trained and saved to: {model_path}")
    return model_path
