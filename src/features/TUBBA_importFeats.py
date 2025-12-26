import os
import sys
import glob

import cv2
import pandas as pd
import numpy as np
from scipy.ndimage import label
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

# Add parent directory to path to import TUBBA_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TUBBA_utils import detect_header_rows, convert_store_to_table, circ_var, unwrapAngles_with_nans, wrapTo2Pi, variable_is_circular
import matplotlib.pyplot as plt
import warnings
import random

def tracksToFeatures(parent, spatialSR):
    vidInfo = {}

    if not os.path.isdir(parent):
        # Bad directory
        vidInfo['status'] = 0
        vidInfo['name'] = None
        vidInfo['dir'] = parent
        vidInfo['samplingRate'] = spatialSR
        vidInfo['frameRate'] = None
        print(f"Warning: Bad directory detected! Skipping {parent}")
        return vidInfo

    # Find video files
    mp4_files = glob.glob(os.path.join(parent, '*.mp4'))

    if len(mp4_files) != 1:
        vidInfo['status'] = 0
        vidInfo['name'] = None
        vidInfo['dir'] = parent
        vidInfo['samplingRate'] = spatialSR
        vidInfo['frameRate'] = None
        print(f"Warning: Multiple or no videos detected in {parent}... Skipping")
        return vidInfo
    else:
        # Load video
        vidInfo['name'] = os.path.basename(mp4_files[0])
        vidInfo['dir'] = parent
        video_path = os.path.join(parent, vidInfo['name'])
        vc = cv2.VideoCapture(video_path)
        vidInfo['frameRate'] = int(vc.get(cv2.CAP_PROP_FPS))
        vidInfo['samplingRate'] = spatialSR

        frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    # Find CSV or H5 feature files
    csv_files = [os.path.basename(f) for f in glob.glob(os.path.join(parent, '*.csv'))]
    h5_files = [os.path.basename(f) for f in glob.glob(os.path.join(parent, '*.h5'))]

    # Prioritize H5 files, then CSV files
    feature_file = None
    file_type = None

    if len(h5_files) == 1:
        feature_file = h5_files[0]
        file_type = 'h5'
        print(f"Found H5 feature file: {feature_file}")
    elif len(h5_files) > 1:
        # Multiple H5 files - try to find one with 'feature' in name
        h5_feature_files = [f for f in h5_files if 'feature' in f.lower()]
        if len(h5_feature_files) == 1:
            feature_file = h5_feature_files[0]
            file_type = 'h5'
            print(f"Found H5 feature file: {feature_file}")

    # If no H5 file found, look for CSV
    if feature_file is None:
        if len(csv_files) == 1:
            feature_file = csv_files[0]
            file_type = 'csv'
            print(f"Found CSV feature file: {feature_file}")
        elif len(csv_files) > 1:
            print(f"Warning: Multiple CSV files detected in {parent}...")
            csv_feature_files = [f for f in csv_files if 'feature' in f.lower()]
            if len(csv_feature_files) == 1:
                feature_file = csv_feature_files[0]
                file_type = 'csv'
                print(f"Found CSV feature file: {feature_file}")
            else:
                csv_feature_files = [f for f in csv_files if 'annotations' not in f.lower()]
                if len(csv_feature_files) == 1:
                    feature_file = csv_feature_files[0]
                    file_type = 'csv'
                    print(f"Found CSV feature file: {feature_file}")

    if feature_file is None:
        print(f"Error: No feature file (.csv or .h5) found in {parent}.")
        vidInfo['status'] = 0
        return vidInfo

    # Load feature data based on file type
    feature_path = os.path.join(parent, feature_file)

    if file_type == 'h5':
        print(f"Loading H5 file...")
        try:
            featureData = pd.read_hdf(feature_path, key='perframes')
            print(f"Loaded data with shape: {featureData.shape}")
        except:
            try:
                featureData = pd.read_hdf(feature_path, key='featureData')
                print(f"Loaded data with shape: {featureData.shape}")
            except Exception as e:
                print(f"Error loading H5 file: {e}")
                vidInfo['status'] = 0
                return vidInfo
    else:  # CSV
        nHeaderRows = detect_header_rows(feature_path)
        print(f"Detected {nHeaderRows} header rows")

        if nHeaderRows == 0:
            featureData = pd.read_csv(feature_path)
        elif nHeaderRows == 1:
            featureData = pd.read_csv(feature_path, header=0)
        else:
            header_indices = list(range(nHeaderRows))
            featureData = pd.read_csv(feature_path, header=header_indices)

    if featureData is None or len(featureData) == 0:
        print(f"Error: Could not load feature data from {feature_file}.")
        vidInfo['status'] = 0
        return vidInfo

    Fs = vidInfo['frameRate']

    # Give an out here in case the feature file already exists
    feature_file = os.path.join(parent, 'perframe_feats.h5')
    if os.path.exists(feature_file):
        print(f"📂 Found existing feature file at {feature_file}. Skipping feature extraction.")

        import h5py
        with h5py.File(feature_file, 'r') as f:

            # Create vidInfo based on folder info
            vidInfo['nframes'] = len(featureData)
            vidInfo['featureFile'] = 'perframe_feats.h5'
            vidInfo['status'] = 1

        return vidInfo

    # Drop columns that are monotonically increasing (likely frame indices)
    def is_monotonic_increasing(series):
        return series.is_monotonic_increasing and series.nunique() > 1

    # Identify numeric monotonically increasing columns
    monotonic_cols = [col for col in featureData.columns if
                      featureData[col].dtype in ['int64', 'int32', 'float64', 'float32']
                      and is_monotonic_increasing(featureData[col])]

    # Identify string-containing columns (object or string dtype)
    string_cols = [col for col in featureData.columns
                   if featureData[col].dtype == 'object' or featureData[col].dtype.name == 'string']

    # Drop the columns
    cols_to_drop = monotonic_cols + string_cols
    if cols_to_drop:
        featureData = featureData.drop(columns=cols_to_drop)
        print(f"Dropped columns: {cols_to_drop}")


    # Get user input on whether they want to perfrom feature expansion
    feature_expansion = prompt_feature_expansion()
    if feature_expansion == "skip":
        print("Skipping feature expansion  per user request.")
        vidInfo['status'] = 1
        vidInfo['nframes'] = len(featureData)
        vidInfo['featureFile'] = 'perframe_feats.h5'

        # Check that dimensions of featureData match the expected number of frames, or throw error
        if len(featureData) != frame_count:
            print(
                f"Error: Dimensions of {feature_file} do not match number of video frames in {vidInfo['name']}. Skipping...")
            vidInfo['status'] = 0
            return vidInfo


        feature_path = os.path.join(parent, vidInfo['featureFile'])
        featureData.to_hdf(feature_path, key='featureData', mode='w', format='table', complevel=5)

        return vidInfo

    # Extract predictors
    perframes = expandPredictors(featureData, Fs)

    # Save expanded feature set
    vidInfo['status'] = 1
    vidInfo['nframes'] = len(featureData)
    vidInfo['featureFile'] = 'perframe_feats.h5'

    # Check that dimensions of featureData match the expected number of frames, or throw error
    if len(featureData) != frame_count:
        print(
            f"Error: Dimensions of {feature_file} do not match number of video frames in {vidInfo['name']}. Skipping...")
        vidInfo['status'] = 0
        return vidInfo

    feature_path = os.path.join(parent, vidInfo['featureFile'])
    perframes.to_hdf(feature_path, key='perframes', mode='w', format='table', complevel=5)

    return vidInfo

def expandPredictors(M, Fs):
    from scipy.signal import savgol_coeffs, convolve
    dt = 1 / Fs
    store = {}

    # --- Savitzky-Golay filter coefficients ---
    polyorder = 3
    window_length = int(np.ceil(Fs / 5))
    window_length_long = int(np.ceil(Fs * 2))

    if window_length % 2 == 0:
        window_length += 1
    if window_length_long % 2 == 0:
        window_length_long += 1

    g0 = savgol_coeffs(window_length, polyorder, deriv=0)
    g1 = savgol_coeffs(window_length, polyorder, deriv=1) * (-1 / dt)
    g2 = savgol_coeffs(window_length, polyorder, deriv=2) * (1 / dt ** 2)

    g0_long = savgol_coeffs(window_length_long, polyorder, deriv=0)
    g1_long = savgol_coeffs(window_length_long, polyorder, deriv=1) * (-1 / dt)
    g2_long = savgol_coeffs(window_length_long, polyorder, deriv=2) * (1 / dt ** 2)

    # --- Detect circular variables ---
    circFields = [col for col in M.columns if variable_is_circular(M[col])]
    allFields = list(M.columns)

    # --- Derivative Computation ---
    for field in allFields:
        x = M[field].to_numpy().squeeze()
        is_circ = field in circFields

        if is_circ:
            x = unwrapAngles_with_nans(x)

        dx0 = convolve(x, g0, mode='same')
        dx1 = convolve(x, g1, mode='same')
        dx2 = convolve(x, g2, mode='same')
        dx0_long = convolve(x, g0_long, mode='same')
        dx1_long = convolve(x, g1_long, mode='same')
        dx2_long = convolve(x, g2_long, mode='same')

        if is_circ:
            dx0 = wrapTo2Pi(dx0)
            dx0_long = wrapTo2Pi(dx0_long)

        store[field] = dx0[:, None]
        store[f"{field}_d1"] = dx1[:, None]
        store[f"{field}_d2"] = dx2[:, None]
        store[f"{field}_slow"] = dx0_long[:, None]
        store[f"{field}_slow_d1"] = dx1_long[:, None]
        store[f"{field}_slow_d2"] = dx2_long[:, None]

    # --- Variance in Sliding Window ---
    winSize = int(Fs)
    if winSize % 2 == 0:
        winSize += 1
    halfW = winSize // 2

    for field in allFields:
        x = M[field].to_numpy().squeeze()
        is_circ = field in circFields

        if is_circ:
            varCentered = np.array([
                circ_var(x[max(0, j - halfW):min(len(x), j + halfW + 1)])
                for j in range(len(x))
            ])
        else:
            varCentered = np.array([
                np.var(x[max(0, j - halfW):min(len(x), j + halfW + 1)], ddof=1)
                for j in range(len(x))
            ])
        store[f"{field}_var"] = varCentered[:, None]

    # --- Pack into DataFrame ---
    perframes = pd.DataFrame({k: v.squeeze() for k, v in store.items()})
    return perframes

def clean_bodypart_tracking(data, part, frame_rate=50,
                            jump_multiplier=5, deviation_threshold=15,
                            window_size=5, max_gap_sec=3):
    """
    Cleans tracking glitches and interpolates gaps for a single bodypart.

    Parameters:
    - data: pandas DataFrame with columns like 'Nose_x', 'Nose_y'
    - part: string, e.g. 'Nose' or 'TailBase'
    - frame_rate: sampling rate (Hz)
    - jump_multiplier: threshold multiplier for segment break detection
    - deviation_threshold: spatial deviation (pixels) to flag a glitch
    - window_size: number of frames for local averaging
    - max_gap_sec: maximum gap duration (sec) allowed for interpolation

    Returns:
    - cleaned_data: pandas DataFrame with same structure, glitches removed and gaps interpolated
    - anomaly_mask: boolean array of length N, where True = flagged frame
    """
    cleaned_data = data.copy()
    x_col = f"{part}_x"
    y_col = f"{part}_y"

    points = np.column_stack([data[x_col].values, data[y_col].values])
    n_points = len(points)

    #  detect large jumps to define segments
    diffs = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    jump_threshold = jump_multiplier * np.nanstd(diffs)
    potential_breaks = np.where(diffs > jump_threshold)[0]

    if len(potential_breaks) > 0:
        all_breaks = np.concatenate([[0], potential_breaks + 1, [n_points]])
    else:
        all_breaks = np.array([0, n_points])

    segments = [(all_breaks[i], all_breaks[i + 1]) for i in range(len(all_breaks) - 1)]
    segment_durations = [end - start for start, end in segments]
    min_segment_frames = int(1 * frame_rate)

    # flag short, glitchy segments
    anomaly_mask = np.zeros(n_points, dtype=bool)
    for i, (start, end) in enumerate(segments):
        duration = end - start
        if duration >= min_segment_frames:
            continue

        # Look for good context before
        pre_mean = None
        for j in range(i - 1, -1, -1):
            if segment_durations[j] >= min_segment_frames:
                pre_start, pre_end = segments[j]
                pre_segment = points[max(pre_start, pre_end - window_size):pre_end]
                if len(pre_segment) >= 2 and not np.any(np.isnan(pre_segment)):
                    pre_mean = np.nanmean(pre_segment, axis=0)
                    break

        # Look for good context after
        post_mean = None
        for j in range(i + 1, len(segments)):
            if segment_durations[j] >= min_segment_frames:
                post_start, post_end = segments[j]
                post_segment = points[post_start:min(post_start + window_size, post_end)]
                if len(post_segment) >= 2 and not np.any(np.isnan(post_segment)):
                    post_mean = np.nanmean(post_segment, axis=0)
                    break

        if pre_mean is None or post_mean is None:
            anomaly_mask[start:end] = True
            continue

        # Compare to linear interpolation
        segment = points[start:end]
        if np.any(np.isnan(segment)):
            continue  # skip incomplete segments

        interp = np.linspace(pre_mean, post_mean, num=duration)
        deviations = np.linalg.norm(segment - interp, axis=1)
        if np.nanmean(deviations) > deviation_threshold:
            anomaly_mask[start:end] = True

    # remove anomalies
    cleaned_data.loc[anomaly_mask, x_col] = np.nan
    cleaned_data.loc[anomaly_mask, y_col] = np.nan

    # interpolate short gaps, leave long ones as NaN
    max_gap = int(max_gap_sec * frame_rate)
    long_gap_mask = np.zeros(n_points, dtype=bool)

    for col in [x_col, y_col]:
        nan_mask = cleaned_data[col].isna()
        labeled_array, num_features = label(nan_mask)

        # Interpolate short gaps
        cleaned_data[col] = cleaned_data[col].interpolate(
            method='linear', limit=max_gap, limit_direction='both'
        )

        # Reapply NaNs for long gaps
        for region_label in range(1, num_features + 1):
            region_indices = (labeled_array == region_label)
            if np.sum(region_indices) > max_gap:
                long_gap_mask |= region_indices

    for col in [x_col, y_col]:
        cleaned_data.loc[long_gap_mask, col] = np.nan

    return cleaned_data, anomaly_mask


from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt
import sys

class FeaturePrompt(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature Expansion")
        self.setStyleSheet("background-color: black; color: white;")
        self.setFixedSize(400, 150)

        self.choice = None

        label = QLabel("Perform feature expansion?\n(Smoothing, derivatives, variance)")
        label.setAlignment(Qt.AlignCenter)

        yes_btn = QPushButton("Yes")
        yes_btn.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        yes_btn.clicked.connect(lambda: self.respond("yes"))

        skip_btn = QPushButton("Skip extraction")
        skip_btn.setStyleSheet("background-color: gray; color: white; font-weight: bold;")
        skip_btn.clicked.connect(lambda: self.respond("skip"))

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(yes_btn)
        btn_layout.addWidget(skip_btn)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def respond(self, choice):
        self.choice = choice
        self.accept()

def prompt_feature_expansion():
    dialog = FeaturePrompt()
    dialog.exec_()
    return dialog.choice



