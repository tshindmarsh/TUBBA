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


def augment_labeled_frames(X_frames, y_frames, noise_factor=0.05, copies=3):
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
        feature_scales = np.std(X, axis=0, keepdims=True) + 1e-8
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


def focal_loss_objective(y_pred, dtrain, gamma=2.0, alpha=0.25):
    """Custom focal loss objective for XGBoost.

    Args:
        y_pred: Raw prediction values from XGBoost
        dtrain: XGBoost's DMatrix containing labels
        gamma: Focusing parameter that controls weight given to hard examples
        alpha: Balancing parameter for class weights

    Returns:
        grad: First order gradient (derivative of loss function)
        hess: Second order gradient (second derivative of loss function)
    """
    y_true = dtrain.get_label()

    # Convert raw predictions to probabilities using sigmoid
    sigmoid_pred = 1.0 / (1.0 + np.exp(-y_pred))

    # Compute pt (probability of true class)
    # For y=1: pt = p, for y=0: pt = 1-p
    pt = sigmoid_pred * y_true + (1 - sigmoid_pred) * (1 - y_true)

    # Compute alpha weights
    # For y=1: alpha_t = alpha, for y=0: alpha_t = 1-alpha
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)

    # Compute focal weights
    gamma = 2
    focal_weight = alpha_t * np.power(1 - pt, gamma)

    # Compute gradients and hessians for XGBoost
    # Gradient: derivative of loss w.r.t. prediction
    grad = sigmoid_pred - y_true
    grad = focal_weight * grad

    # Hessian: second derivative of loss w.r.t. prediction
    hess = sigmoid_pred * (1 - sigmoid_pred)
    hess = focal_weight * hess

    return grad, hess

class FocalLossXGBClassifier(xgb.XGBClassifier):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        self.gamma = gamma
        self.alpha = alpha

        kwargs.pop("objective", None)
        kwargs.pop("use_label_encoder", None)

        super().__init__(**kwargs)

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False, **kwargs):
        gamma = self.gamma
        alpha = self.alpha

        def objective_wrapper(y_pred, dtrain):
            return focal_loss_objective(y_pred, dtrain, gamma, alpha)

        dtrain = xgb.DMatrix(X, label=y)

        evals = []
        if eval_set:
            eval_X, eval_y = eval_set[0]
            dval = xgb.DMatrix(eval_X, label=eval_y)
            evals = [(dtrain, "train"), (dval, "val")]
        else:
            evals = [(dtrain, "train")]

        params = self.get_xgb_params()
        num_boost_round = params.pop("n_estimators", 100)
        params["disable_default_eval_metric"] = True

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            obj=objective_wrapper,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose,
        )

        self._Booster = booster
        self._n_features_in = X.shape[1]
        self._classes = np.array([0, 1])
        return self

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        pred = self._Booster.predict(dtest)
        # Return as 2D array to match scikit-learn's convention
        return np.vstack([1 - pred, pred]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    @property
    def n_classes_(self):
        return 2  # Binary classification

def generate_windows_for_all_labels(conf_all, y, window_size=101, offsets=[0]):
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


def train_TUBBAmodel(project_json_path, window_size=101, lstm_epochs=1000):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS (Apple Metal) backend")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS not available — falling back to CPU")

    # --- Compute normalization stats across all animals ---
    temp_features = []
    for video in project['videos']:
        feature_path = os.path.join(video['folder'], video['featureFile'])
        if not os.path.isfile(feature_path):
            continue
        try:
            df = pd.read_hdf(feature_path, key='perframes')
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

    print("✅ Finished computing normalization stats across all animals.")
    imputer = SimpleImputer(strategy='constant', fill_value=0)

    # --- Train XGB per behavior ---
    for behavior in behaviors:
        X_frames, y_frames = [], []

        print(f"🧠 Augmenting training data for behavior: {behavior}")

        for video in project['videos']:
            feature_path = os.path.join(video['folder'], video['featureFile'])
            if not os.path.isfile(feature_path):
                continue

            df = pd.read_hdf(feature_path, key='perframes')
            X = normalize_features(df, zscore_stats, minmax_stats)

            y = np.full(len(X), np.nan)
            for (start, end, val) in video.get('annotations', {}).get(behavior, []):
                if val in [-1, 1]:
                    y[start:end + 1] = val

            # Store frames with their labels
            X_frames.append(X)
            y_frames.append(y)

        if not X_frames:
            print(f"⚠️ No training data for behavior: {behavior}")
            continue

        # Augment data first
        X_frames, y_frames = augment_labeled_frames(X_frames, y_frames, noise_factor=0.03, copies=3)

        # Now use the balanced frame samples function to create a balanced dataset
        # Use target_ratio=(1, 1, 0.2) to include some unlabeled samples (0.2x the limiting class)
        X_balanced, y_balanced = create_balanced_frame_samples(X_frames, y_frames, target_ratio=(1, 1, 0.2))

        if len(X_balanced) == 0:
            print(f"⚠️ No balanced data available for behavior: {behavior}")
            continue

        # Convert -1 → 0, 1 → 1, and drop NaNs
        y_xgb = np.full_like(y_balanced, np.nan)
        y_xgb[y_balanced == 1] = 1
        y_xgb[y_balanced == -1] = 0

        # Drop NaN labels (i.e. unlabeled frames)
        valid_mask = ~np.isnan(y_xgb)
        X_balanced = X_balanced[valid_mask]
        y_xgb = y_xgb[valid_mask]

        # Apply imputation
        X_balanced = imputer.fit_transform(X_balanced)

        # Create train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_balanced, y_xgb, test_size=0.1, stratify=y_xgb)

        # Train XGBoost model
        xgb_model = FocalLossXGBClassifier(
            gamma=2.0,
            alpha=0.25,  # Use higher alpha for more imbalanced datasets
            n_estimators=800,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.7,
            learning_rate=0.02,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        print(f"🚀 Training XGBoost model with focal loss for behavior: {behavior}")
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
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
            X = normalize_features(df, zscore_stats, minmax_stats)
            X = imputer.transform(X)

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

        pos_weight_val = (len(Y_train) - Y_train.sum()) / (Y_train.sum() + 1e-6)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)

        # Early stopping setup
        patience = 20
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

            if epoch % 5 == 0:
                print(
                    f"  🌀 Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Patience: {epochs_since_improvement}")

            if epochs_since_improvement >= patience:
                print(f"  ⏹️ Early stopping at epoch {epoch} — best val loss: {best_val_loss:.4f}")
                break

        model_bundle['models'][behavior]['lstm'] = best_model_state
        torch.mps.empty_cache()

    model_bundle['lstm_config'] = {
        'input_size': len(behaviors),
        'hidden_size': 32,
        'dropout': 0.3,
        'attention': True
    }
    model_path = os.path.splitext(project_json_path)[0] + "_trainedModel.pkl"
    joblib.dump(model_bundle, model_path)
    print(f"\n✅ All models trained and saved to: {model_path}")
    return model_path
