import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from TUBBA_utils import variable_is_circular, normalize_features
import time

def train_TUBBAmodel_lite(project_json_path):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

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
    model_bundle = {
        'models': {},
        'behaviors': behaviors,
        'window_size': None,
        'normalization': {
            'zscore': zscore_stats,
            'minmax': minmax_stats
        },
        'feature_names': feature_names
    }

    imputer = SimpleImputer(strategy='constant', fill_value=0)

    for behavior in behaviors:
        X_frames, y_frames = [], []

        print(f"🧠 Training lightweight XGB for behavior: {behavior}")

        for video in project['videos']:
            feature_path = os.path.join(video['folder'], video['featureFile'])
            if not os.path.isfile(feature_path):
                continue

            t1 = time.time()
            df = pd.read_hdf(feature_path, key='perframes')
            t2 = time.time()
            X = normalize_features(df, zscore_stats, minmax_stats)
            t3 = time.time()

            y = np.full(len(X), np.nan)
            for (start, end, val) in video.get('annotations', {}).get(behavior, []):
                if val in [-1, 1]:
                    y[start:end + 1] = val
            t4 = time.time()

            X_frames.append(X)
            y_frames.append(y)
            print(
                f"Read HDF: {t2 - t1:.3f}s | Normalize: {t3 - t2:.3f}s | Labeling: {t4 - t3:.3f}s | Total: {t4 - t1:.3f}s")

        if not X_frames:
            print(f"⚠️ No training data for behavior: {behavior}")
            continue

        X_all = np.vstack(X_frames)
        y_all = np.concatenate(y_frames)

        mask = ~np.isnan(y_all)
        X_all = X_all[mask]
        y_all = y_all[mask]

        y_xgb = np.where(y_all == -1, 0, y_all)

        if len(X_all) == 0 or np.all(y_xgb == 1) or np.all(y_xgb == 0):
            print(f"⚠️ No valid training data for behavior: {behavior}")
            continue

        X_all = imputer.fit_transform(X_all)

        pos_frac = np.mean(y_xgb == 1)
        neg_frac = 1 - pos_frac
        sample_weights = np.where(y_xgb == 1, 1 / (pos_frac + 1e-6), 1 / (neg_frac + 1e-6))

        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X_all, y_xgb, sample_weights, test_size=0.1, stratify=y_xgb
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

    model_bundle['imputer'] = imputer

    model_path = os.path.splitext(project_json_path)[0] + "_lightXGB.pkl"
    joblib.dump(model_bundle, model_path)
    print(f"\n✅ Lightweight XGB models trained and saved to: {model_path}")
    return model_path
