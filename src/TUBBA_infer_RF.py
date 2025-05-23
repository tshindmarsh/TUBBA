import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from tkinter import messagebox
from TUBBA_utils import variable_is_circular, zscore_normalize_preserve_nans, normalize_features

def TUBBA_modelInference(project_json_path, video_name, video_folder):
    # Load project
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    # Load trained model path from project
    model_path = project.get('models', None)
    if not model_path or not os.path.isfile(model_path):
        msg = "❌ No trained model found in project file. Please run training first."
        messagebox.showerror("TUBBA Inference Error", msg)
        raise FileNotFoundError(msg)

    # Load model bundle
    model_bundle = joblib.load(model_path)
    model = model_bundle['model']  # CustomMultiOutputClassifier
    imputer = model_bundle['imputer']
    behaviors = model_bundle['behaviors']

    # Get normalization stats
    norm_stats = model_bundle.get('normalization', {})
    zscore_stats = norm_stats.get('zscore', {})
    minmax_stats = norm_stats.get('minmax', {})

    # Find the target video
    video = next(
        (v for v in project['videos']
         if v['name'] == video_name and v['folder'] == video_folder),
        None
    )
    if video is None:
        raise ValueError(f"Video {video_name} not found in project.")

    feature_path = os.path.join(video['folder'], video['featureFile'])
    if not os.path.isfile(feature_path):
        raise FileNotFoundError(f"⚠️ Feature file not found: {feature_path}")

    df_features = pd.read_hdf(feature_path, key='perframes')
    df_features = df_features.loc[:, ~df_features.columns.str.contains('_d1$|_d2$', regex=True)]
    X = normalize_features(df_features, zscore_stats, minmax_stats)

    # Impute missing values (same as in training)
    X_imputed = imputer.transform(X)

    # Run inference
    predictions = {}
    confidence = {}

    for i, behavior in enumerate(behaviors):
        clf = model.estimators_[i]
        y_pred = clf.predict(X_imputed)
        y_conf = clf.predict_proba(X_imputed)[:, 1]
        predictions[behavior] = y_pred.tolist()
        confidence[behavior] = y_conf.tolist()

    return {
        "predictions": predictions,
        "confidence": confidence
    }