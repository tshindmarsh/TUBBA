
import os
import json
import h5py
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

def train_TUBBAmodel(project_json_path):
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    behaviors = project['behaviors']
    all_X = []
    all_Y = []

    for video in project['videos']:
        feature_path = os.path.join(video['folder'], video['feature_file'])
        if not os.path.isfile(feature_path):
            print(f"⚠️ Feature file not found: {feature_path}")
            continue

        with h5py.File(feature_path, 'r') as f:
            features = np.array(f['features'])

        n_frames = features.shape[0]
        X = features
        Y = np.zeros((n_frames, len(behaviors)), dtype=int)

        for b_idx, behavior in enumerate(behaviors):
            if behavior not in video['annotations']:
                continue
            for annot in video['annotations'][behavior]:
                start, end, val = annot
                if val == 1:
                    Y[start:end + 1, b_idx] = 1
                elif val == -1:
                    Y[start:end + 1, b_idx] = -1

        known_mask = (Y != 0).all(axis=1)
        X = X[known_mask]
        Y = Y[known_mask]

        if len(X) == 0:
            print(f"⚠️ No valid training data in {video['name']}")
            continue

        all_X.append(X)
        all_Y.append(Y)

    X_all = np.vstack(all_X)
    Y_all = np.vstack(all_Y)

    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-999)
    X_imputed = imputer.fit_transform(X_all)

    base_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                        class_weight='balanced')
    model = MultiOutputClassifier(base_model)
    model.fit(X_imputed, Y_all)

    Y_pred = model.predict(X_imputed)
    print("\n📊 Per-behavior training accuracy:")
    for i, behavior in enumerate(behaviors):
        acc = accuracy_score(Y_all[:, i], Y_pred[:, i])
        print(f"  {behavior:20s}: {acc:.3f}")

    model_path = os.path.join(os.path.dirname(project_json_path), "TUBBA_model.pkl")
    joblib.dump({'model': model, 'imputer': imputer, 'behaviors': behaviors}, model_path)

    print(f"\n✅ Model trained and saved to: {model_path}")
