import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from TUBBA_utils import variable_is_circular, normalize_features

def train_TUBBAmodel(project_json_path):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    # --- Gather normalization stats ---
    temp_features = []
    for video in project['videos']:
        feature_path = os.path.join(video['folder'], video['featureFile'])
        if os.path.isfile(feature_path):
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
    imputer = SimpleImputer(strategy='constant', fill_value=0)

    model_bundle = {
        'models': {},
        'behaviors': behaviors,
        'normalization': {
            'zscore': zscore_stats,
            'minmax': minmax_stats
        },
        'feature_names': feature_names,
        'imputer': imputer
    }

    # --- Train XGBoost model per behavior ---
    for behavior in behaviors:
        X_frames, y_frames = [], []

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

            valid = ~np.isnan(y)
            X_frames.append(X[valid])
            y_frames.append((y[valid] == 1).astype(int))

        if not X_frames:
            print(f"⚠️ No training data for behavior: {behavior}")
            continue

        X_all = np.vstack(X_frames)
        y_all = np.concatenate(y_frames)
        X_all = imputer.fit_transform(X_all)

        # Balance positive/negative weights
        pos_count = (y_all == 1).sum()
        neg_count = (y_all == 0).sum()
        sW = neg_count / (pos_count + 1e-6)

        # Split for evaluation
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.1, stratify=y_all
        )

        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=600,
            max_depth=6,
            min_child_weight=5,
            gamma=0.8,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=sW,
            learning_rate=0.02,
            early_stopping_rounds=10
        )

        print(f"🚀 Training XGBoost model for behavior: {behavior}")
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        model_bundle['models'][behavior] = {'xgb': xgb_model}

    # --- Save model bundle ---
    model_path = os.path.splitext(project_json_path)[0] + "_trainedModel.pkl"
    joblib.dump(model_bundle, model_path)
    print(f"\n✅ XGBoost models trained and saved to: {model_path}")
    return model_path