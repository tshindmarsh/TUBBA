import numpy as np
import pandas as pd
import pickle
import os
import cv2
import joblib
import json
import os

def save_inference_to_disk(video, inference_dict):
    video_name = os.path.splitext(video['name'])[0]
    out_dir = os.path.join(video['folder'], 'inference')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{video_name}_inferred.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(inference_dict, f)

    return out_path

def approximateCurvature2D(pts):
    """
    Approximate bending measure of a 2D path based on given points.

    Inputs:
    - pts: Nx2 array of point coordinates (or Nx3 if 3D)

    Outputs:
    - angleSum: sum of turning angles between points
    - pathExcess: total distance / straight-line distance
    - deviantArea: area between path and straight line
    """

    pts = np.asarray(pts)
    N, d = pts.shape
    if N < 3:
        raise ValueError("Need at least 3 points to have interior angles.")
    if d not in [2, 3]:
        raise ValueError("Points must be 2D (Nx2) or 3D (Nx3).")

    # --- Sum of interior angles ---
    angleSum = 0.0
    for i in range(1, N - 1):
        vPrev = pts[i] - pts[i - 1]
        vNext = pts[i + 1] - pts[i]
        angleSum += angle_between_vectors(vPrev, vNext)

    # --- Path excess ---
    straightDist = np.linalg.norm(pts[0] - pts[-1])
    totalDist = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))

    if straightDist < 1e-12:
        pathExcess = np.nan
    else:
        pathExcess = totalDist / straightDist

    # --- Deviant Area ---
    P1 = pts[0]
    PN = pts[-1]
    dVec = PN - P1
    len_d = np.linalg.norm(dVec)

    if len_d < 1e-12:
        deviantArea = 0
        return angleSum, pathExcess, deviantArea

    eX = dVec / len_d
    eY = np.array([-eX[1], eX[0]])  # 90 degree rotation

    Xvals = np.zeros(N)
    Yvals = np.zeros(N)

    for i in range(N):
        v = pts[i] - P1
        Xvals[i] = np.dot(v, eX) / len_d
        Yvals[i] = np.dot(v, eY) / len_d

    # Sort by X for proper integration
    sort_idx = np.argsort(Xvals)
    Xvals_sorted = Xvals[sort_idx]
    Yvals_sorted = Yvals[sort_idx]

    deviantArea = np.trapz(np.abs(Yvals_sorted), x=Xvals_sorted)

    if deviantArea > 0.5:
        deviantArea = np.nan

    return angleSum, pathExcess, deviantArea

def angle_between_vectors(a, b):
    """
    Unsigned angle between two vectors a and b, in radians.
    Always in [0, pi].
    """
    dotab = np.dot(a, b)
    normA = np.linalg.norm(a)
    normB = np.linalg.norm(b)

    if normA < 1e-12 or normB < 1e-12:
        return 0.0

    cosVal = np.clip(dotab / (normA * normB), -1.0, 1.0)
    ang = np.arccos(cosVal)
    return ang

def moving_correlation(x, y, window_size):
    N = len(x)
    corr = np.full(N, np.nan)
    halfW = int(window_size // 2)

    for i in range(N):
        l = int(max(0, i - halfW))
        r = int(min(N, i + halfW + 1))
        if (r - l) > 2:  # must have enough points
            corr_window = np.corrcoef(x[l:r], y[l:r])[0,1]
            corr[i] = corr_window
    return corr

def moving_circular_correlation(theta1, theta2, window_size):
    """
    theta1, theta2: angles in radians
    """
    sin1 = np.sin(theta1)
    cos1 = np.cos(theta1)
    sin2 = np.sin(theta2)
    cos2 = np.cos(theta2)

    sin_corr = moving_correlation(sin1, sin2, window_size)
    cos_corr = moving_correlation(cos1, cos2, window_size)

    return (sin_corr + cos_corr) / 2

def circ_dist(a, b):
    """
    Circular distance from a to b using complex exponentials, result in [-pi, pi]
    """
    return np.angle(np.exp(1j * a) / np.exp(1j * b))

def circ_var(angles):
    # Estimate circular variance: 1 - R
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    R = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
    return 1 - R

def wrapTo2Pi(angles):
    return np.mod(angles, 2 * np.pi)

def unwrapAngles_with_nans(x):
    """
    Unwrap while handling NaNs properly.
    Interpolates over NaNs, unwraps, then restores NaNs.
    """
    isnan = np.isnan(x)
    if np.all(isnan):
        return x  # all NaNs, return as is

    # Interpolate over NaNs
    x_interp = pd.Series(x).interpolate(limit_direction='both').to_numpy()

    # Unwrap interpolated signal
    x_unwrapped = np.unwrap(x_interp)

    # Restore NaNs
    x_unwrapped[isnan] = np.nan

    return x_unwrapped

def convert_store_to_table(store, maxRows):
    """
    Convert the store dictionary into a pandas DataFrame efficiently.
    """
    predictors_dict = {}

    for key, val in store.items():
        val = np.asarray(val)

        if val.ndim == 1:
            nRows, nCols = len(val), 1
            val = val.reshape(-1, 1)  # ensure 2D
        else:
            nRows, nCols = val.shape

        # Pad if necessary
        if nRows < maxRows:
            padRows = np.tile(val[-1:, :], (maxRows - nRows, 1))
            val = np.vstack([val, padRows])

        if nCols == 1:
            predictors_dict[key] = val.flatten()
        else:
            for c in range(nCols):
                predictors_dict[f"{key}_ind{c+1}"] = val[:,c]

    # Only create the DataFrame at the end
    predictors = pd.DataFrame(predictors_dict)
    return predictors

def zscore_normalize_preserve_nans(X):
    """Z-score normalize, ignoring NaNs (mean and std computed per column)."""
    if X.ndim == 1:
        mean = np.nanmean(X)
        std = np.nanstd(X)
        if std == 0 or np.isnan(std):
            std = 1
        return (X - mean) / std
    else:
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std[std == 0] = 1
        return (X - mean) / std

def variable_is_circular(values, threshold=0.1):
    finite_vals = values[np.isfinite(values)]
    if len(finite_vals) < 100:
        return False
    min_val, max_val = np.nanmin(finite_vals), np.nanmax(finite_vals)
    # Common angular bounds (degrees or radians)
    return (np.isclose(min_val, 0, atol=threshold) and
            (np.isclose(max_val, 2*np.pi, atol=threshold) or np.isclose(max_val, 360, atol=threshold))) or \
           (np.isclose(min_val, -np.pi, atol=threshold) and np.isclose(max_val, np.pi, atol=threshold))

def normalize_features(df, zscore_stats, minmax_stats):
    X = df.values.copy()
    for i, col_name in enumerate(df.columns):
        col = X[:, i]
        if np.all(np.isnan(col)) or variable_is_circular(col):
            continue
        elif col_name in minmax_stats:
            cmin, cmax = minmax_stats[col_name]
            denom = cmax - cmin if cmax > cmin else 1.0
            X[:, i] = (col - cmin) / denom
        elif col_name in zscore_stats:
            mean, std = zscore_stats[col_name]
            denom = std if std > 0 else 1.0
            X[:, i] = (col - mean) / denom
    return X

def detect_circle_region(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=40, maxRadius=150)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, 0]  # Return the first detected circle (x, y, radius)
    return None

def predictions_to_video(source_video_path, predictions_path, behavior, out_path=None):
    """Save a stitched video containing only predicted positive frames for a behavior."""
    import os
    import joblib
    import numpy as np
    import cv2

    # Load predictions from file if path is given
    if isinstance(predictions_path, str):
        predictions = joblib.load(predictions_path)["predictions"]
    else:
        predictions = predictions_path

    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {source_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    preds = np.array(predictions[behavior])
    assert len(preds) == n_frames, f"Prediction length ({len(preds)}) != video frame count ({n_frames})"

    # Generate output path if not specified
    if out_path is None:
        base, ext = os.path.splitext(source_video_path)
        out_path = f"{base}_predicted_{behavior}{ext}"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Only get the frames with predicted 1
    positive_indices = np.flatnonzero(preds == 1)

    for i in positive_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Skipped frame {i} (could not read)")
            continue
        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Saved: {out_path}")

import os
import json
import cv2

def annotations_to_video(project_json_path, behavior, out_path, target=1):
    """Efficiently creates a stitched video of annotated behavior intervals across all project videos."""
    with open(project_json_path, 'r') as f:
        project = json.load(f)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = None
    target_size = None
    initialized = False

    for video_entry in project['videos']:
        annotations = video_entry.get("annotations", {})
        if behavior not in annotations:
            continue

        video_path = os.path.join(video_entry['folder'], video_entry['name'])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Cannot open video: {video_path}")
            continue

        current_fps = cap.get(cv2.CAP_PROP_FPS)
        current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not initialized:
            target_size = (current_width, current_height)
            out_writer = cv2.VideoWriter(out_path, fourcc, current_fps, target_size)
            initialized = True

        intervals = annotations[behavior]
        for start, end, val in intervals:
            if val != target:
                continue
            for frame_idx in range(start, end + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                # Center-crop to target_size if needed
                h, w = frame.shape[:2]
                tw, th = target_size
                if (w, h) != (tw, th):
                    x_start = max((w - tw) // 2, 0)
                    y_start = max((h - th) // 2, 0)
                    frame = frame[y_start:y_start + th, x_start:x_start + tw]

                    # If crop is out of bounds (video too small), pad it
                    if frame.shape[0] != th or frame.shape[1] != tw:
                        padded = cv2.copyMakeBorder(
                            frame,
                            0, th - frame.shape[0],
                            0, tw - frame.shape[1],
                            borderType=cv2.BORDER_CONSTANT,
                            value=[0, 0, 0]
                        )
                        frame = padded

                label = os.path.basename(video_path)
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3, lineType=cv2.LINE_AA)
                out_writer.write(frame)

        cap.release()

    if out_writer:
        out_writer.release()
        print(f"✅ Saved: {out_path}")
    else:
        print(f"❌ No valid frames written for behavior '{behavior}'.")