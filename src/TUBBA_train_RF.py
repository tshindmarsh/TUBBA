import os
import json
import joblib
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QMessageBox
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import _MultiOutputEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from TUBBA_utils import variable_is_circular, zscore_normalize_preserve_nans, normalize_features

class CustomMultiOutputClassifier:
    def __init__(self, base_model, behavior_names):
        self.base_model = base_model
        self.behavior_names = behavior_names
        self.estimators_ = []

    def fit(self, X, Y):
        self.estimators_ = []
        for i in range(Y.shape[1]):
            y = Y[:, i]
            valid = ~np.isnan(y)
            clf = clone(self.base_model)
            clf.fit(X[valid], y[valid])
            self.estimators_.append(clf)

    def predict(self, X):
        return np.column_stack([
            clf.predict(X) for clf in self.estimators_
        ])

    def predict_proba(self, X):
        return [
            clf.predict_proba(X) for clf in self.estimators_
        ]

def train_TUBBAmodel(project_json_path):
    # Load project
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    # --- First pass: compute normalization stats across all animals ---
    feature_names = None
    temp_features = []

    for video in project['videos']:
        feature_path = os.path.join(video['folder'], video['featureFile'])
        if not os.path.isfile(feature_path):
            continue

        try:
            df_features = pd.read_hdf(feature_path, key='perframes')
            temp_features.append(df_features)
        except:
            continue

    # Stack and compute stats
    full_df = pd.concat(temp_features, axis=0, ignore_index=True)
    feature_names = full_df.columns.tolist()

    zscore_stats = {}
    minmax_stats = {}

    for col in feature_names:
        col_data = full_df[col].values
        if variable_is_circular(col_data):
            continue  # skip circular
        elif 'pca' in col.lower():
            col_min = np.nanmin(col_data)
            col_max = np.nanmax(col_data)
            minmax_stats[col] = (col_min, col_max)
        else:
            col_mean = np.nanmean(col_data)
            col_std = np.nanstd(col_data)
            zscore_stats[col] = (col_mean, col_std)

    behaviors = project['behaviors']
    all_X = []
    all_Y = []

    for video in project['videos']:
        feature_path = os.path.join(video['folder'], video['featureFile'])
        if not os.path.isfile(feature_path):
            print(f"⚠️ Feature file not found: {feature_path}")
            continue

        try:
            df_features = pd.read_hdf(feature_path, key='perframes')
            df_features = df_features.loc[:, ~df_features.columns.str.contains('_d1$|_d2$', regex=True)]
        except Exception as e:
            print(f"❌ Failed to load features from {feature_path}: {e}")
            continue

        # Normalize non-circular columns
        X = normalize_features(df_features, zscore_stats, minmax_stats)
        n_frames = X.shape[0]

        # Build binary label matrix
        Y = np.full((n_frames, len(behaviors)), np.nan)
        for b_idx, behavior in enumerate(behaviors):
            if behavior not in video['annotations']:
                continue
            for annot in video['annotations'][behavior]:
                start, end, val = annot
                if val == 1:
                    Y[start:end + 1, b_idx] = 1
                elif val == -1:
                    Y[start:end + 1, b_idx] = -1

        # Filter to frames with known labels for all behaviors
        known_mask = (Y != 0).any(axis=1)
        X = X[known_mask]
        Y = Y[known_mask]

        all_X.append(X)
        all_Y.append(Y)

    # Stack and preprocess
    X_all = np.vstack(all_X)
    Y_all = np.vstack(all_Y)

    # Check for missing labels
    validate_behavior_labels(Y_all, behaviors)

    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    X_imputed = imputer.fit_transform(X_all)

    # Train model
    base_model = RandomForestClassifier(n_estimators=100, random_state=13, n_jobs=-1, class_weight='balanced')

    model = CustomMultiOutputClassifier(base_model, behaviors)
    model.fit(X_imputed, Y_all)

    # Accuracy report
    Y_pred = model.predict(X_imputed)
    print("\n📊 Per-behavior training accuracy:")
    for i, behavior in enumerate(behaviors):
        y_true = Y_all[:, i]
        y_pred = Y_pred[:, i]
        valid = ~np.isnan(y_true)
        if np.sum(valid) == 0:
            print(f"  {behavior:20s}: ❌ No annotated frames")
        else:
            acc = accuracy_score(y_true[valid], y_pred[valid])
            print(f"  {behavior:20s}: {acc:.3f}")

    # Save model + metadata
    model_bundle = {
        'model': model,
        'imputer': imputer,
        'behaviors': behaviors,
        'feature_names': df_features.columns.tolist(),
        'normalization': {
            'zscore': zscore_stats,
            'minmax': minmax_stats
        }
    }

    base = os.path.splitext(project_json_path)[0]  # strips .json
    model_path = f"{base}_trainedModel.pkl"
    joblib.dump(model_bundle, model_path)
    print(f"\n✅ Model trained and saved to: {model_path}")

    return model_path

def validate_behavior_labels(Y_all, behaviors):
    """
    Checks that each behavior has both positive and negative training examples,
    and that at least 50 frames are labeled.

    Shows GUI warnings or errors via tkinter.messagebox.
    """
    missing_pos = []
    missing_neg = []
    low_sample = []

    for i, behavior in enumerate(behaviors):
        n_pos = np.sum(Y_all[:, i] == 1)
        n_neg = np.sum(Y_all[:, i] == -1)

        if n_pos == 0:
            missing_pos.append(behavior)
        if n_neg == 0:
            missing_neg.append(behavior)
        if n_pos + n_neg < 50:
            low_sample.append(behavior)

    # Hard fail if any missing
    if missing_pos or missing_neg:
        msg = "❌ Training data is incomplete:\n"
        if missing_pos:
            msg += f"- No positive examples for: {', '.join(missing_pos)}\n"
        if missing_neg:
            msg += f"- No negative examples for: {', '.join(missing_neg)}\n"
        QMessageBox.critical(None, "Training Error", msg)
        raise ValueError(msg)

    # Soft warning if too few labeled frames
    if low_sample:
        msg = f"⚠️ Fewer than 50 labeled frames for: {', '.join(low_sample)}"
        QMessageBox.critical(None, "Training Warning", msg)


