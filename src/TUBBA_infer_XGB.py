import os
import json
import joblib
import numpy as np
import pandas as pd
from tkinter import messagebox
from TUBBA_utils import variable_is_circular, normalize_features

def TUBBA_modelInference(project_json_path, video_name, video_folder):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    model_path = project.get('models', None)
    if not model_path or not os.path.isfile(model_path):
        msg = "❌ No trained model found in project file. Please run training first."
        messagebox.showerror("TUBBA Inference Error", msg)
        raise FileNotFoundError(msg)

    bundle = joblib.load(model_path)
    models = bundle['models']
    behaviors = bundle['behaviors']

    zscore_stats = bundle['normalization'].get('zscore', {})
    minmax_stats = bundle['normalization'].get('minmax', {})
    imputer = bundle.get('imputer', None)

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
    X = normalize_features(df_features, zscore_stats, minmax_stats)
    if imputer is not None:
        X = imputer.transform(X)
    else:
        X = np.nan_to_num(X, nan=0.0)

    # Generate XGBoost confidences and predictions
    predictions = {}
    confidence = {}

    for behavior in behaviors:
        if behavior not in models or 'xgb' not in models[behavior]:
            confidence[behavior] = [0.0] * len(X)
            predictions[behavior] = [0] * len(X)
            continue

        xgb_model = models[behavior]['xgb']
        prob = xgb_model.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(int)

        confidence[behavior] = prob.tolist()
        predictions[behavior] = pred.tolist()

    return {
        "predictions": predictions,
        "confidence": confidence
    }