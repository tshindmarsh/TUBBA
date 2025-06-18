import os
import glob

import cv2
import pandas as pd
import numpy as np
from scipy.ndimage import label
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from TUBBA_utils import detect_circle_region, detect_header_rows, convert_store_to_table, circ_var, unwrapAngles_with_nans, wrapTo2Pi, compute_floppiness
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import warnings
import random

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")


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
        vidInfo['status'] = 0
        vidInfo['name'] = os.path.basename(mp4_files[0])
        vidInfo['dir'] = parent
        video_path = os.path.join(parent, vidInfo['name'])
        vc = cv2.VideoCapture(video_path)
        vidInfo['frameRate'] = int(vc.get(cv2.CAP_PROP_FPS))
        vidInfo['samplingRate'] = spatialSR

        frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    # Find CSV files
    csv_files = [os.path.basename(f) for f in glob.glob(os.path.join(parent, '*.csv'))]

    dlc_file = None

    if len(csv_files) == 1:
        dlc_file = csv_files[0]
        print(f"Found tracking file: {dlc_file}")
    else:
        print(f"Warning: Multiple or no csv files detected in {parent}...")

        # Try finding one with DLC in the name
        candidates = [f for f in csv_files if 'DLC' in f]
        if len(candidates) == 1:
            dlc_file = candidates[0]
            print(f"Found tracking file: {dlc_file}")
        else:
            # Try ignoring anything with 'annotations' in the name
            candidates = [f for f in csv_files if 'annotations' not in f]
            if len(candidates) == 1:
                dlc_file = candidates[0]
                print(f"Found tracking file: {dlc_file}")
            else:
                return vidInfo

    # Load tracking
    trackingData = None
    dlc_path = os.path.join(parent, dlc_file)
    nHeaderRows = detect_header_rows(dlc_path)

    print(f"Detected {nHeaderRows} header rows")

    if nHeaderRows == 0:
        # No headers detected, read normally
        trackingData_raw = pd.read_csv(dlc_path)
    elif nHeaderRows == 1:
        # Single header row
        trackingData_raw = pd.read_csv(dlc_path, header=0)
    else:
        # Multiple header rows - use list of row indices
        header_indices = list(range(nHeaderRows))[1:]  # Skip the first row
        trackingData_raw = pd.read_csv(dlc_path, header=header_indices)

    if trackingData_raw.map(lambda x: isinstance(x, str)).any().any():
        print("Warning: String values detected in tracking data. Likely multi-individual dataset. Skipping...")
        return vidInfo, None

    trackingData_raw.columns = [f"{bp}_{detail}" for bp, detail in trackingData_raw.columns]
    trackingData = trackingData_raw.copy()

    if trackingData is None:
        print(f"Error: No tracking data found, cannot continue.")
        vidInfo['status'] = 0
        return vidInfo, None

    Fs = vidInfo['frameRate']
    ts = [i / Fs for i in range(1, len(trackingData) + 1)]

    # Give an out here in case the feature file already exists
    feature_file = os.path.join(parent, 'perframe_feats.h5')
    if os.path.exists(feature_file):
        print(f"📂 Found existing feature file at {feature_file}. Skipping feature extraction.")

        import h5py
        with h5py.File(feature_file, 'r') as f:

            # Create vidInfo based on folder info
            vidInfo['nframes'] = len(trackingData)
            vidInfo['featureFile'] = 'perframe_feats.h5'
            vidInfo['status'] = 1

        return vidInfo

    # Upscale if downsampling
    if spatialSR < 1:
        tracking_cols = [col for col in trackingData.columns if '_x' in col or '_y' in col]
        trackingData[tracking_cols] *= (1 / spatialSR)

    # Process DLC CSVs
    trax = processCsv2TUBBA(trackingData, Fs)
    if trax is None:
        print(f"Error: No tracking data found, cannot continue.")
        return vidInfo

    # Extract predictors
    perframes = getPredictors(trax, Fs)

    # -- Error rate check
    errorCount = perframes.isna().sum()
    NFrames = len(perframes)
    errorRate = errorCount.mean() / NFrames

    # -- Prepare vidInfo
    vidInfo['nframes'] = len(trackingData_raw)
    vidInfo['featureFile'] = 'perframe_feats.h5'

    # -- Save perframes if acceptable
    feature_path = os.path.join(parent, vidInfo['featureFile'])

    if errorRate < 0.1:
        vidInfo['status'] = 1
        perframes.to_hdf(feature_path, key='perframes', mode='w', format='table', complevel=5)

    elif 0.05 <= errorRate < 0.5:
        vidInfo['status'] = 1
        perframes.to_hdf(feature_path, key='perframes', mode='w', format='table', complevel=5)
        print(f"⚠️Warning: High error rates in {parent}. Consider retraining DLC.")

    else:  # errorRate >= 0.2
        vidInfo['status'] = 0
        raise ValueError(f"Too few tracked points detected in {parent}! Skipping.")

    return vidInfo

def processCsv2TUBBA(data, Fs):

    # Function to check if a column is monotonically increasing
    def is_monotonic_increasing(series):
        return series.is_monotonic_increasing and series.nunique() > 1

    # Drop columns that are monotonically increasing (likely frame indices)
    monotonic_cols = [col for col in data.columns if
                      data[col].dtype in ['int64', 'int32', 'float64', 'float32'] and is_monotonic_increasing(
                          data[col])]
    if monotonic_cols:
        data = data.drop(columns=monotonic_cols)
        print("Frame count column detected and removed")

    confidenceThresh = 0.2

    # Parse columns# flatten columns to strings
    columns_flat = [str(c) for c in data.columns]
    split_columns = [c.split('_') for c in columns_flat]
    columnParts = pd.DataFrame(split_columns)
    uniqueParts = columnParts.iloc[:, -2].unique()
    print(f"Separating {len(uniqueParts)} bodyparts...")

    if columnParts.shape[1] > 2:
        uniqueInds = columnParts.iloc[:, 0].unique()
        print(f"Likely multianimal dataset detected! Skipping...")
        return None
    else:
        uniqueInds = [1]

    # Throw out low-confidence predictions
    for i in range(len(uniqueInds)):
        for j in range(len(uniqueParts)):
            if len(uniqueInds) > 1:
                coreName = f"{uniqueInds[i]}_{uniqueParts[j]}_"
            else:
                coreName = f"{uniqueParts[j]}_"

            likelihood_col = f"{coreName}likelihood"
            if likelihood_col in data.columns:
                throw = data[likelihood_col] < confidenceThresh
                if f"{coreName}x" in data.columns:
                    data.loc[throw, f"{coreName}x"] = np.nan
                if f"{coreName}y" in data.columns:
                    data.loc[throw, f"{coreName}y"] = np.nan

    # Remove likelihood columns
    lhCols = [col for col in data.columns if 'likelihood' in col]
    data = data.drop(columns=lhCols)

    # Remove jumps - filter over short missing segments (<3*Fs)
    for ind in uniqueInds:
        for bpt in uniqueParts:
            part_name = f"{ind}_{bpt}" if len(uniqueInds) > 1 else bpt

            if f"{part_name}_x" in data.columns and f"{part_name}_y" in data.columns:
                cleaned, _ = clean_bodypart_tracking(
                    data, part=part_name, frame_rate=Fs,
                    jump_multiplier=5,
                    deviation_threshold=15,
                    window_size=5,
                    max_gap_sec=3
                )
                # Replace cleaned x/y columns back into data
                data[f"{part_name}_x"] = cleaned[f"{part_name}_x"]
                data[f"{part_name}_y"] = cleaned[f"{part_name}_y"]

    # Get rough location and speed
    anis = {}
    anis['meanX'] = np.zeros((len(data), len(uniqueInds)))
    anis['meanY'] = np.zeros((len(data), len(uniqueInds)))
    anis['speed'] = np.zeros((len(data) - 1, len(uniqueInds)))

    for i in range(len(uniqueInds)):
        if len(uniqueInds) > 1:
            subData = data.filter(regex=f"^{uniqueInds[i]}_")
        else:
            subData = data

        cNames = subData.columns
        subData = subData.to_numpy()

        xcols = [k for k, col in enumerate(cNames) if '_x' in col]
        ycols = [k for k, col in enumerate(cNames) if '_y' in col]

        anis['meanX'][:, i] = np.nanmean(subData[:, xcols], axis=1)
        anis['meanY'][:, i] = np.nanmean(subData[:, ycols], axis=1)
        anis['speed'][:, i] = np.sqrt(np.diff(anis['meanX'][:, i])**2 + np.diff(anis['meanY'][:, i])**2) / (1/Fs)

    anis['bpts'] = data

    return anis

def getPredictors(M, Fs):
    from scipy.signal import savgol_coeffs, convolve

    store = {}

    # Split column names to grab individuals, body parts, and coordinates
    cpM = pd.DataFrame([col.split('_') for col in M['bpts'].columns])

    # Center of all bodyparts
    bodypart_x_cols = [col for col, part in zip(M['bpts'].columns, cpM.iloc[:, 1])
                       if ('x' in part)]
    bodypart_y_cols = [col for col, part in zip(M['bpts'].columns, cpM.iloc[:, 1])
                       if ('y' in part)]

    store['centerX'] = M['bpts'][bodypart_x_cols].mean(axis=1, skipna=True)
    store['centerY'] = M['bpts'][bodypart_y_cols].mean(axis=1, skipna=True)

    # Center speed
    store['centerSpeed'] = np.sqrt(np.diff(store['centerX']) ** 2 + np.diff(store['centerY']) ** 2)

    # --- Pairwise distances as a table ---

    # List of bodypart names
    bpts = sorted(set(col.rsplit('_', 1)[0] for col in M['bpts'].columns if '_x' in col or '_y' in col))

    # Get coordinates array of shape (nFrames, nBodyparts, 2)
    coords = np.stack([
        np.stack([M['bpts'][f'{bp}_x'], M['bpts'][f'{bp}_y']], axis=1)
        for bp in bpts
    ], axis=1)

    # Get all unique bodypart index pairs
    pair_indices = list(combinations(range(len(bpts)), 2))
    pair_names = [f'{bpts[i]}_{bpts[j]}_dist' for i, j in pair_indices]

    # Compute pairwise distances for each frame using broadcasting
    distances = np.array([
        pdist(frame, metric='euclidean') for frame in coords
    ])

    pairwise_df = pd.DataFrame(distances, columns=pair_names)
    for col_idx, name in enumerate(pair_names):
        store[name] = distances[:, col_idx][:, None]  # Shape (nFrames, 1)

    # Fill NaNs and standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pairwise_df.fillna(0))  # Or use some imputed value if needed

    # Run PCA
    pca = PCA(n_components=10)
    pcs = pca.fit_transform(X_scaled)

    # Add PCA components
    for i in range(pcs.shape[1]):
        store[f'pairwiseDist_PC{i + 1}'] = pcs[:, i][:, None]  # Shape (nFrames, 1


    # --- internal joint angles ---
    # --- Joint Angles and PCA (Local Triplets Only) ---

    # Mean position of each bodypart
    mean_coords = np.nanmean(coords, axis=0)  # shape: (nBodyparts, 2)

    # Full pairwise distance matrix
    distmat = squareform(pdist(mean_coords))  # shape: (nBodyparts, nBodyparts)

    # Build triplets (A, B, C) with B as joint and A, C from nearest neighbors
    k = 4  # Number of nearest neighbors
    triplets = []
    for b in range(len(bpts)):  # B = joint
        neighbors = np.argsort(distmat[b])[1:k + 1]  # skip self
        for a, c in combinations(neighbors, 2):
            triplets.append((a, b, c))

    # Compute angles for each triplet
    angle_features = np.full((coords.shape[0], len(triplets)), np.nan)

    for idx, (a, b, c) in enumerate(triplets):
        A = coords[:, a]
        B = coords[:, b]
        C = coords[:, c]

        v1 = A - B
        v2 = C - B

        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)
        valid = (norm1 > 0) & (norm2 > 0)

        angle = np.full(coords.shape[0], np.nan)
        cos_theta = np.einsum('ij,ij->i', v1, v2) / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle[valid] = np.arccos(cos_theta[valid])

        angle_features[:, idx] = angle

    # Require at least 90% valid triplet angles
    valid_rows = np.sum(~np.isnan(angle_features), axis=1) >= 0.9 * angle_features.shape[1]

    # Impute missing values in valid rows with column means
    imputer_angle = SimpleImputer(strategy='mean')
    angle_features_imputed = imputer_angle.fit_transform(angle_features[valid_rows])

    # Standardize and perform PCA
    scaler_angle = StandardScaler()
    angle_scaled = scaler_angle.fit_transform(angle_features_imputed)

    pca_angle = PCA(n_components=10)
    angle_pcs_valid = pca_angle.fit_transform(angle_scaled)

    # Fill result into full array
    angle_pcs = np.full((coords.shape[0], 10), np.nan)
    angle_pcs[valid_rows] = angle_pcs_valid

    # Store first 10 PCs
    for i in range(angle_pcs.shape[1]):
        store[f'angle_PC{i + 1}'] = angle_pcs[:, i][:, None]

    # --- Movement Vectors ---
    movdirBpt = pd.DataFrame()
    movmagBpt = pd.DataFrame()

    for bpt in bpts:
        x_col = f"{bpt}_x"
        y_col = f"{bpt}_y"
        if x_col in M['bpts'].columns and y_col in M['bpts'].columns:
            dx = np.diff(M['bpts'][x_col])
            dy = np.diff(M['bpts'][y_col])
            movdirBpt[bpt] = np.arctan2(dy, dx)
            movmagBpt[bpt] = np.sqrt(dx ** 2 + dy ** 2)

    # Motion coherence (using mean resultant length) - Ignore tail
    sumSin = (np.sin(movdirBpt) * movmagBpt).sum(axis=1)
    sumCos = (np.cos(movdirBpt) * movmagBpt).sum(axis=1)
    norm = movmagBpt.sum(axis=1)
    store['motionCoherence'] = np.sqrt(sumSin ** 2 + sumCos ** 2) / norm

    # PCA on movement magnitudes
    movmag_subset = movmagBpt
    valid_mov_rows = np.sum(~movmagBpt.isna().values, axis=1) >= 0.7 * movmagBpt.shape[1]

    # Impute missing values in valid rows
    imputer_mov = SimpleImputer(strategy='mean')
    movmag_imputed = imputer_mov.fit_transform(movmagBpt[valid_mov_rows])

    # Standardize
    scaler_mov = StandardScaler()
    movmag_scaled = scaler_mov.fit_transform(movmag_imputed)

    # PCA
    pca_mov = PCA(n_components=10)
    mov_pcs_valid = pca_mov.fit_transform(movmag_scaled)

    # Fill results back into full-size array
    mov_pcs = np.full((len(movmagBpt), 10), np.nan)
    mov_pcs[valid_mov_rows] = mov_pcs_valid

    # Store first 10 movement magnitude PCs
    for i in range(mov_pcs.shape[1]):
        store[f'pcaMovMag_{i + 1}'] = mov_pcs[:, i][:, None]

        # Average direction of motion
    store['movdirMn'] = np.arctan2(sumSin, sumCos)

    store['motionCoherence'] = store['motionCoherence']
    store['movdirMn'] = store['movdirMn']

    # --- Derivatives ---
    dt = 1 / Fs

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

    # Only one circular field now
    circFields = ['movdirMn']
    allFields = list(store.keys())
    nonCircFields = [f for f in allFields if f not in circFields]

    # --- Derivative Computation ---
    for field in allFields:
        x = store[field].squeeze()
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
        x = store[field].squeeze()
        is_circ = field in circFields

        if is_circ:
            varCentered = np.array([circ_var(x[max(0, j - halfW):min(len(x), j + halfW + 1)])
                                    for j in range(len(x))])
        else:
            varCentered = np.array([np.var(x[max(0, j - halfW):min(len(x), j + halfW + 1)], ddof=1)
                                    for j in range(len(x))])

        store[f"{field}_var"] = varCentered[:, None]

    # --- Pack into Table ---
    maxRows = len(M['meanX'])  # total number of frames
    perframes = convert_store_to_table(store, maxRows)
    perframes = perframes.drop(columns=['movdirMn'])  # remove circular base field if undesired

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
 # Ensure non-negative


