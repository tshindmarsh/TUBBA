
import os
import json
import h5py
import numpy as np
import joblib

def TUBBA_modelInference(project_json_path):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    model_path = os.path.join(os.path.dirname(project_json_path), "TUBBA_model.pkl")
    if not os.path.isfile(model_path):
        print("❌ No trained model found. Please run training first.")
        return

    data = joblib.load(model_path)
    model = data['model']
    imputer = data['imputer']
    behaviors = data['behaviors']

    for video in project['videos']:
        feature_path = os.path.join(video['folder'], video['feature_file'])
        if not os.path.isfile(feature_path):
            print(f"⚠️ Feature file not found: {feature_path}")
            continue

        with h5py.File(feature_path, 'r') as f:
            features = np.array(f['features'])

        n_frames = features.shape[0]
        X = features
        X_imputed = imputer.transform(X)

        Y_pred = model.predict(X_imputed)
        Y_conf = np.zeros_like(Y_pred, dtype=float)

        for i, est in enumerate(model.estimators_):
            proba = est.predict_proba(X_imputed)
            if proba.shape[1] > 1:
                Y_conf[:, i] = proba[:, 1]
            else:
                Y_conf[:, i] = proba[:, 0]

        inferred = {}
        for i, behavior in enumerate(behaviors):
            inferred[behavior] = {
                "predictions": Y_pred[:, i].tolist(),
                "confidence": Y_conf[:, i].tolist()
            }

        video['inferred'] = inferred
        print(f"✅ Inference complete for video: {video['name']}")

    with open(project_json_path, 'w') as f:
        json.dump(project, f, indent=2)

    print(f"\n🧠 Inference results saved to project file: {project_json_path}")
