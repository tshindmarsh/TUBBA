import os
import glob

import cv2
import pandas as pd
import numpy as np
from scipy.ndimage import label
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.impute import SimpleImputer
from TUBBA_utils import detect_circle_region
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
        return vidInfo, None

    # Find video files
    mp4_files = glob.glob(os.path.join(parent, '*.mp4'))

    if len(mp4_files) != 1:
        vidInfo['status'] = 0
        vidInfo['name'] = None
        vidInfo['dir'] = parent
        vidInfo['samplingRate'] = spatialSR
        vidInfo['frameRate'] = None
        print(f"Warning: Multiple or no videos detected in {parent}... Skipping")
        return vidInfo, None
    else:
        # Load video
        vidInfo['name'] = os.path.basename(mp4_files[0])
        vidInfo['dir'] = parent
        video_path = os.path.join(parent, vidInfo['name'])
        vc = cv2.VideoCapture(video_path)
        vidInfo['frameRate'] = int(vc.get(cv2.CAP_PROP_FPS))
        vidInfo['samplingRate'] = spatialSR

        frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

        # Choose 10 random frame indices (ensure they're within bounds)
        random_indices = sorted(random.sample(range(frame_count), k=25))

        frames = []
        for idx in random_indices:
            vc.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = vc.read()
            if ret:
                frames.append(frame.astype(np.float32))
            else:
                print(f"⚠️ Could not read frame at index {idx}")
        vc.release()

        # Ensure we got enough frames
        if len(frames) == 0:
            raise ValueError("No valid frames were read from the video.")

        # Average the frames
        avg_frame = np.mean(frames, axis=0).astype(np.uint8)

        # Detect circle in the averaged frame
        circle = detect_circle_region(avg_frame)
        if circle is not None:
            cx, cy, radius = circle
            shelter = np.array([[cx], [cy]], dtype=np.float64)
            shelter *= (1 / spatialSR)

            print(f"Circle center: ({cx}, {cy}), radius: {radius}")

            # Plot and save
            fig, ax = plt.subplots()
            ax.imshow(avg_frame)
            ax.set_title("Detected Circle (Avg Frame)")
            ax.axis("off")

            circle_patch = plt.Circle((cx, cy), radius, color='lime', fill=False, linewidth=2)
            ax.add_patch(circle_patch)

            save_path = os.path.join(vidInfo['dir'], 'detected_Shelter.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
        else:
            shelter = np.ones((2, 1)) * -99

    # Find CSV files
    csv_files = [os.path.basename(f) for f in glob.glob(os.path.join(parent, '*.csv'))]

    if spatialSR == 1:
        cricket_files = [f for f in csv_files if 'MultiCricketTracking' in f and '_el_filtered.csv' in f]
        mouse_files = [f for f in csv_files if 'SingleMouseTracking' in f and '_filtered.csv' in f]
    else:
        cricket_files = [f for f in csv_files if 'MultiCricketTracking' in f and '_el_filtered.csv' in f and 'lowres' in f]
        mouse_files = [f for f in csv_files if 'SingleMouseTracking' in f and '_filtered.csv' in f and 'lowres' in f]

    # Load cricket and mouse data
    cricketData, mouseData = None, None

    if cricket_files:
        cricket_path = os.path.join(parent, cricket_files[0])
        cricketData_raw = pd.read_csv(cricket_path, header=[1, 2, 3])
        cricketData_raw.columns = [f"{ind}_{bp}_{detail}" for ind, bp, detail in cricketData_raw.columns]
        cricketData = cricketData_raw.copy()
    else:
        print(f"Cricket tracking file not found for {parent}")

    if mouse_files:
        mouse_path = os.path.join(parent, mouse_files[0])
        mouseData_raw = pd.read_csv(mouse_path, header=[1, 2])
        mouseData_raw.columns = [f"{bp}_{detail}" for bp, detail in mouseData_raw.columns]
        mouseData = mouseData_raw.copy()
    else:
        print(f"Mouse tracking file not found for {parent}")

    if mouseData is None:
        print(f"Error: No mouse tracking data found, cannot continue.")
        vidInfo['status'] = 0
        return vidInfo, None

    Fs = vidInfo['frameRate']
    ts = [i / Fs for i in range(1, len(mouseData) + 1)]

    # Give an out here in case the feature file already exists
    feature_file = os.path.join(parent, 'perframe_feats.h5')
    if os.path.exists(feature_file):
        print(f"📂 Found existing feature file at {feature_file}. Skipping feature extraction.")

        import h5py
        with h5py.File(feature_file, 'r') as f:
            # You don't actually need to load anything, just confirm it exists

            # Create vidInfo based on folder info
            vidInfo['nframes'] = len(mouseData_raw)
            vidInfo['featureFile'] = 'perframe_feats.h5'
            vidInfo['status'] = 1

        return vidInfo

    # Upscale if downsampling
    if spatialSR < 1:
        mouse_cols = [col for col in mouseData.columns if '_x' in col or '_y' in col]
        cricket_cols = [col for col in cricketData.columns if '_x' in col or '_y' in col] if cricketData is not None else []
        mouseData[mouse_cols] *= (1 / spatialSR)
        if cricket_cols:
            cricketData[cricket_cols] *= (1 / spatialSR)

    # Pad cricketData or mouseData if too short
    if len(mouseData) < len(cricketData):
        pad_len = len(cricketData) - len(mouseData)
        mouseData = pd.concat([mouseData, pd.DataFrame(np.nan, index=range(pad_len), columns=mouseData.columns)], axis=0)
    elif len(cricketData) < len(mouseData):
        pad_len = len(mouseData) - len(cricketData)
        cricketData = pd.concat([cricketData, pd.DataFrame(np.nan, index=range(pad_len), columns=cricketData.columns)], axis=0)

    # Process DLC CSVs
    mstrax = processDLCcsvTUBBA(mouseData, Fs)
    Ctrax = processDLCcsvTUBBA(cricketData, Fs) if cricketData is not None else None

    # Extract predictors
    perframes = getMouseCricketPredictors(mstrax, Ctrax, Fs, shelter)

    # -- Error rate check
    good_cols = [col for col in perframes.columns if 'ind' not in col and 'tail' not in col]
    errorCount = perframes[good_cols].isna().sum()
    NFrames = len(perframes)
    errorRate = errorCount.mean() / NFrames

    # -- Prepare vidInfo
    vidInfo['nframes'] = len(mouseData_raw)
    vidInfo['featureFile'] = 'perframe_feats.h5'

    # -- Save perframes if acceptable
    feature_path = os.path.join(parent, vidInfo['featureFile'])

    if errorRate < 0.05:
        vidInfo['status'] = 1
        perframes.to_hdf(feature_path, key='perframes', mode='w', format='table', complevel=5)

    elif 0.05 <= errorRate < 0.2:
        vidInfo['status'] = 1
        perframes.to_hdf(feature_path, key='perframes', mode='w', format='table', complevel=5)
        print(f"⚠️Warning: High error rates in {parent}. Consider retraining DLC.")

    else:  # errorRate >= 0.2
        vidInfo['status'] = 0
        raise ValueError(f"Too few tracked points detected in {parent}! Skipping.")

    return vidInfo

def processDLCcsvTUBBA(data, Fs):

    if 'bodyparts_coords' in data.columns:
        data = data.drop(columns=['bodyparts_coords'])
    if 'individuals_bodyparts_coords' in data.columns:
        data = data.drop(columns=['individuals_bodyparts_coords'])
    confidenceThresh = 0.2

    # Parse columns# flatten columns to strings
    columns_flat = [str(c) for c in data.columns]
    split_columns = [c.split('_') for c in columns_flat]
    columnParts = pd.DataFrame(split_columns)
    uniqueParts = columnParts.iloc[:, -2].unique()
    print(f"Separating {len(uniqueParts)} bodyparts...")

    if columnParts.shape[1] > 2:
        uniqueInds = columnParts.iloc[:, 0].unique()
        print(f"Likely multianimal dataset detected! Separating {len(uniqueInds)} individuals...")
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

def getMouseCricketPredictors(M, C, Fs, shelterLoc=None):
    from scipy.signal import savgol_coeffs, convolve
    from TUBBA_utils import (approximateCurvature2D, moving_circular_correlation, circ_var,
                             circ_dist, wrapTo2Pi, unwrapAngles_with_nans, convert_store_to_table)

    store = {}

    # Split column names to grab individuals, body parts, and coordinates
    cpM = pd.DataFrame([col.split('_') for col in M['bpts'].columns])
    cpC = pd.DataFrame([col.split('_') for col in C['bpts'].columns])

    # --- Mouse Intrinsic Predictors ---

    # Center of all bodyparts (except tail)
    bodypart_x_cols = [col for col, part in zip(M['bpts'].columns, cpM.iloc[:, 1])
                       if ('x' in part) and not any(tail in col for tail in ['Tail1', 'Tail2', 'Tail3', 'TailTip'])]
    bodypart_y_cols = [col for col, part in zip(M['bpts'].columns, cpM.iloc[:, 1])
                       if ('y' in part) and not any(tail in col for tail in ['Tail1', 'Tail2', 'Tail3', 'TailTip'])]

    centerX = M['bpts'][bodypart_x_cols].mean(axis=1, skipna=True)
    centerY = M['bpts'][bodypart_y_cols].mean(axis=1, skipna=True)

    # Euclidian distance from shelter
    store['shelterDist'] = np.sqrt((centerX - shelterLoc[0]) ** 2 + (centerY - shelterLoc[1]) ** 2)

    # Center speed
    store['centerSpeed'] = np.sqrt(np.diff(centerX) ** 2 + np.diff(centerY) ** 2)

    # Center of mouse body (Spine, Root, leg)
    body_x_cols = [col for col, part in zip(M['bpts'].columns, cpM.iloc[:, 1])
                   if ('x' in part) and any(k in col for k in ['Spine', 'Root', 'leg'])]
    body_y_cols = [col for col, part in zip(M['bpts'].columns, cpM.iloc[:, 1])
                   if ('y' in part) and any(k in col for k in ['Spine', 'Root', 'leg'])]

    bodyCenterX = M['bpts'][body_x_cols].mean(axis=1, skipna=True)
    bodyCenterY = M['bpts'][body_y_cols].mean(axis=1, skipna=True)

    # Center of mouse head (Nose, Ear, Neck)
    head_x_cols = [col for col, part in zip(M['bpts'].columns, cpM.iloc[:, 1])
                   if ('x' in part) and any(k in col for k in ['Nose', 'Ear', 'Neck'])]
    head_y_cols = [col for col, part in zip(M['bpts'].columns, cpM.iloc[:, 1])
                   if ('y' in part) and any(k in col for k in ['Nose', 'Ear', 'Neck'])]

    headCenterX = M['bpts'][head_x_cols].mean(axis=1, skipna=True)
    headCenterY = M['bpts'][head_y_cols].mean(axis=1, skipna=True)

    # Difference in speed between head and body
    headSpeed = np.sqrt(np.diff(headCenterX) ** 2 + np.diff(headCenterY) ** 2)
    bodySpeed = np.sqrt(np.diff(bodyCenterX) ** 2 + np.diff(bodyCenterY) ** 2)
    store['headVbodySpeed'] = headSpeed - bodySpeed

    # Head-Body and Tail-Body distances
    store['head2BodyDist'] = np.sqrt((headCenterX - bodyCenterX) ** 2 + (headCenterY - bodyCenterY) ** 2)
    store['tail2BodyDist'] = np.sqrt(
        (M['bpts']['TailRoot_x'] - bodyCenterX) ** 2 + (M['bpts']['TailRoot_y'] - bodyCenterY) ** 2)

    # Face direction (vector Nose -> Neck)
    noseXY = np.stack([M['bpts']['Nose_x'], M['bpts']['Nose_y']], axis=1)
    neckXY = np.stack([M['bpts']['Neck_x'], M['bpts']['Neck_y']], axis=1)
    normNose = noseXY - neckXY
    store['faceDir'] = np.arctan2(normNose[:, 1], normNose[:, 0])

    # Body direction (vector HindlegSpine -> ForelegSpine)
    fsXY = np.stack([M['bpts']['ForelegSpine_x'], M['bpts']['ForelegSpine_y']], axis=1)
    rsXY = np.stack([M['bpts']['HindlegSpine_x'], M['bpts']['HindlegSpine_y']], axis=1)
    normFs = fsXY - rsXY
    store['bodyDir'] = np.arctan2(normFs[:, 1], normFs[:, 0])

    # Head-to-body angle
    store['head2bodyAngle'] = np.arctan2(np.sin(store['faceDir'] - store['bodyDir']), np.cos(store['faceDir'] - store['bodyDir']))
    store['head2bodyAngleOffset'] = np.abs(store['head2bodyAngle'])

    # Core body size
    store['frontWidth'] = np.sqrt((M['bpts']['ForelegL_x'] - M['bpts']['ForelegR_x']) ** 2 +
                                  (M['bpts']['ForelegL_y'] - M['bpts']['ForelegR_y']) ** 2)

    store['rearWidth'] = np.sqrt((M['bpts']['HindlegL_x'] - M['bpts']['HindlegR_x']) ** 2 +
                                 (M['bpts']['HindlegL_y'] - M['bpts']['HindlegR_y']) ** 2)

    store['faceLength'] = np.sqrt((M['bpts']['Nose_x'] - M['bpts']['Neck_x']) ** 2 +
                                  (M['bpts']['Nose_y'] - M['bpts']['Neck_y']) ** 2)

    # Body Area using outline points
    outlineParts = ['Nose', 'EarL', 'Neck', 'ForelegL', 'HindlegL', 'TailRoot',
                    'HindlegR', 'ForelegR', 'Neck', 'EarR', 'Nose']

    outlineX = np.stack([M['bpts'][f"{part}_x"] for part in outlineParts], axis=1)
    outlineY = np.stack([M['bpts'][f"{part}_y"] for part in outlineParts], axis=1)

    # Polygon area (shoelace formula)
    store['bodyArea'] = 0.5 * np.abs(
        np.sum(outlineX[:, :-1] * outlineY[:, 1:] - outlineX[:, 1:] * outlineY[:, :-1], axis=1))

    # --- Movement Vectors ---

    # List of all unique body parts
    bpts = cpM.iloc[:, 0].unique()

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


    # Front vs rear leg motion
    frontMot = movmagBpt[['ForelegL', 'ForelegR']].mean(axis=1)
    rearMot = movmagBpt[['HindlegL', 'HindlegR']].mean(axis=1)
    store['frontVbackLegMotion'] = frontMot - rearMot

    # Motion coherence (using mean resultant length) - Ignore tail
    selector = ~movdirBpt.columns.str.contains('TailTip|Tail1|Tail2|Tail3')

    selectedDirs = movdirBpt.loc[:, selector]
    selectedMags = movmagBpt.loc[:, selector]

    sumSin = (np.sin(selectedDirs) * selectedMags).sum(axis=1)
    sumCos = (np.cos(selectedDirs) * selectedMags).sum(axis=1)
    norm = selectedMags.sum(axis=1)
    store['motionCoherence'] = np.sqrt(sumSin ** 2 + sumCos ** 2) / norm

    # PCA on movement magnitudes (excluding tail)
    movmag_subset = movmagBpt.loc[:, selector]

    # Drop rows with any NaNs
    valid_mov = ~movmag_subset.isna().any(axis=1)
    pca_mov = PCA(n_components=6)
    mov_pcs = np.full((len(movmag_subset), 6), np.nan)
    mov_pcs[valid_mov] = pca_mov.fit_transform(movmag_subset[valid_mov])

    # Store first 5 PCs
    for i in range(5):
        store[f'pcaMovMag_{i + 1}'] = mov_pcs[:, i]

    # Average direction of motion
    store['movdirMn'] = np.arctan2(sumSin, sumCos)

    # Select bodypoint positions (exclude tail)
    keep_cols = [col for col, part in zip(M['bpts'].columns, cpM.iloc[:, 1])
                 if ('x' in part or 'y' in part) and not any(
            tail in col for tail in ['Tail1', 'Tail2', 'Tail3', 'TailTip'])]

    # Get positions as 3D array: (n_frames, n_bodyparts, 2)
    xy_cols = sorted(keep_cols)
    coords = M['bpts'][xy_cols].to_numpy()
    n_bodyparts = len(xy_cols) // 2
    coords = coords.reshape(-1, n_bodyparts, 2)

    # Compute pairwise distances for each frame
    combos = list(combinations(range(n_bodyparts), 2))
    pairwise_dists = np.array([
        [np.linalg.norm(f[i] - f[j]) for (i, j) in combos]
        for f in coords
    ])

    # Run PCA on pairwise distances
    valid = ~np.isnan(pairwise_dists).any(axis=1)
    pca_dist = PCA(n_components=10)
    dist_pcs = np.full((coords.shape[0], 10), np.nan)
    dist_pcs[valid] = pca_dist.fit_transform(pairwise_dists[valid])

    # Store first 5 PCs
    for i in range(5):
        store[f'pcaPairDist_{i + 1}'] = dist_pcs[:, i]

    # --- Curvatures ---
    # Tail parts
    tail_x_cols = [col for col, part in zip(M['bpts'].columns, cpM.iloc[:, 1])
                   if 'x' in part and 'Tail' in col]
    tail_y_cols = [col for col, part in zip(M['bpts'].columns, cpM.iloc[:, 1])
                   if 'y' in part and 'Tail' in col]

    tailX = M['bpts'][tail_x_cols].to_numpy()
    tailY = M['bpts'][tail_y_cols].to_numpy()

    tailCurvature = []

    for i in range(tailX.shape[0]):
        pts = np.stack([tailX[i, :], tailY[i, :]], axis=1)
        try:
            _, _, deviantArea = approximateCurvature2D(pts)
        except Exception:
            deviantArea = np.nan
        tailCurvature.append(deviantArea)

    store['tailCurvature'] = np.array(tailCurvature)

    # Spine parts
    spine_x_cols = [col for col, part in zip(M['bpts'].columns, cpM.iloc[:, 1])
                    if 'x' in part and any(p in col for p in ['Spine', 'Neck', 'TailRoot'])]
    spine_y_cols = [col for col, part in zip(M['bpts'].columns, cpM.iloc[:, 1])
                    if 'y' in part and any(p in col for p in ['Spine', 'Neck', 'TailRoot'])]

    spineX = M['bpts'][spine_x_cols].to_numpy()
    spineY = M['bpts'][spine_y_cols].to_numpy()

    spineCurvature = []

    for i in range(spineX.shape[0]):
        pts = np.stack([spineX[i, :], spineY[i, :]], axis=1)
        try:
            _, _, deviantArea = approximateCurvature2D(pts)
        except Exception:
            deviantArea = np.nan
        spineCurvature.append(deviantArea)

    store['spineCurvature'] = np.array(spineCurvature)

    # --- Gait Symmetry ---
    winSize = Fs // 4
    frontL_dir = movdirBpt['ForelegL'].to_numpy()
    frontR_dir = movdirBpt['ForelegR'].to_numpy()
    hindL_dir = movdirBpt['HindlegL'].to_numpy()
    hindR_dir = movdirBpt['HindlegR'].to_numpy()

    # Front legs correlation
    store['frontlegCorr'] = moving_circular_correlation(frontL_dir, frontR_dir, winSize)

    # Hind legs correlation
    store['hindlegCorr'] = moving_circular_correlation(hindL_dir, hindR_dir, winSize)

    # Front vs Hind legs (average of forelegs vs average of hindlegs)
    front_mean_dir = np.arctan2(
        (np.sin(frontL_dir) + np.sin(frontR_dir)) / 2,
        (np.cos(frontL_dir) + np.cos(frontR_dir)) / 2
    )

    hind_mean_dir = np.arctan2(
        (np.sin(hindL_dir) + np.sin(hindR_dir)) / 2,
        (np.cos(hindL_dir) + np.cos(hindR_dir)) / 2
    )

    store['frontHindLegCorr'] = moving_circular_correlation(front_mean_dir, hind_mean_dir, winSize)

    # --- Center of Motion ---
    # Define zones
    Z1 = ['Nose']
    Z2 = ['EarL', 'EarR', 'Neck']
    Z3 = ['ForelegSpine', 'ForelegL', 'ForelegR']
    Z4 = ['MidSpine']
    Z5 = ['HindlegSpine', 'HindlegL', 'HindlegR']
    Z6 = ['TailRoot']

    # Calculate average movement magnitude in each zone
    ZoneMot = np.zeros((len(movmagBpt), 6))  # Frames x Zones

    ZoneMot[:, 0] = movmagBpt.loc[:, movmagBpt.columns.isin(Z1)].mean(axis=1)
    ZoneMot[:, 1] = movmagBpt.loc[:, movmagBpt.columns.isin(Z2)].mean(axis=1)
    ZoneMot[:, 2] = movmagBpt.loc[:, movmagBpt.columns.isin(Z3)].mean(axis=1)
    ZoneMot[:, 3] = movmagBpt.loc[:, movmagBpt.columns.isin(Z4)].mean(axis=1)
    ZoneMot[:, 4] = movmagBpt.loc[:, movmagBpt.columns.isin(Z5)].mean(axis=1)
    ZoneMot[:, 5] = movmagBpt.loc[:, movmagBpt.columns.isin(Z6)].mean(axis=1)

    # Define relative positions of zones (evenly spaced between 0 and 1)
    zoneIdx = np.linspace(0, 1, 6)

    # Compute Center of Motion: weighted average of zone indices
    numerator = (ZoneMot * zoneIdx).sum(axis=1)
    denominator = ZoneMot.sum(axis=1)

    store['comMotion'] = numerator / denominator

    # Compute pca of motion centers - PC1 = average, PC2 = asymmetry
    imputer = SimpleImputer(strategy='mean')
    ZoneMot_clean = imputer.fit_transform(ZoneMot)
    pca = PCA(n_components=2)
    zone_pca = pca.fit_transform(ZoneMot_clean)

    store['zonePC1'] = zone_pca[:, 0]
    store['zonePC2'] = zone_pca[:, 1]

    # --- Mouse-Cricket Relationships ---

    # Distance from mouse centroid to each cricket centroid
    store['MCDist_centroids'] = np.sqrt((C['meanX'] - centerX.to_numpy()[:, None]) ** 2 +
                                        (C['meanY'] - centerY.to_numpy()[:, None]) ** 2)

    # Unique crickets
    indC = cpC.iloc[:, 0].unique()

    # Distance from mouse nose to closest point on each cricket
    store['MCDist_nose2crick'] = np.full((len(centerX), len(indC)), np.nan)

    for i, cricket_id in enumerate(indC):
        xcols = [col for col, parts in zip(C['bpts'].columns, cpC.values) if parts[0] == cricket_id and parts[2] == 'x']
        ycols = [col for col, parts in zip(C['bpts'].columns, cpC.values) if parts[0] == cricket_id and parts[2] == 'y']

        if len(xcols) > 0 and len(ycols) > 0:
            dist2Nose_allBpts = np.sqrt((M['bpts']['Nose_x'].to_numpy()[:, None] - C['bpts'][xcols].to_numpy()) ** 2 +
                                        (M['bpts']['Nose_y'].to_numpy()[:, None] - C['bpts'][ycols].to_numpy()) ** 2)
            store['MCDist_nose2crick'][:, i] = np.nanmin(dist2Nose_allBpts, axis=1)

    # Distance from mouse outline to each cricket
    outlineParts = ['Nose', 'EarL', 'Neck', 'ForelegL', 'HindlegL', 'TailRoot',
                    'HindlegR', 'ForelegR', 'Neck', 'EarR', 'Nose']

    xcols_outline = [col for col, parts in zip(M['bpts'].columns, cpM.values)
                     if parts[0] in outlineParts and parts[1] == 'x']
    ycols_outline = [col for col, parts in zip(M['bpts'].columns, cpM.values)
                     if parts[0] in outlineParts and parts[1] == 'y']

    store['MCDist_outline2crick'] = np.full((len(centerX), len(indC)), np.nan)

    for i, cricket_id in enumerate(indC):
        dist2outline = np.sqrt((M['bpts'][xcols_outline].to_numpy() - C['meanX'][:, i][:, None]) ** 2 +
                               (M['bpts'][ycols_outline].to_numpy() - C['meanY'][:, i][:, None]) ** 2)
        store['MCDist_outline2crick'][:, i] = np.nanmin(dist2outline, axis=1)

    # Absolute angle from mouse centroid to each cricket centroid
    dx = C['meanX'] - centerX.to_numpy()[:, None]
    dy = C['meanY'] - centerY.to_numpy()[:, None]
    store['MCAngle_abs'] = np.arctan2(dy, dx)

    # Angle relative to mouse facing direction
    store['MCAngle_FA'] = np.full((len(centerX), len(indC)), np.nan)
    for i in range(len(indC)):
        store['MCAngle_FA'][:, i] = circ_dist(store['MCAngle_abs'][:, i], store['faceDir'])

    # Angle relative to mouse traveling direction
    store['MCAngle_travelDir'] = np.full((len(centerX) - 1, len(indC)), np.nan)
    for i in range(len(indC)):
        store['MCAngle_travelDir'][:, i] = circ_dist(store['MCAngle_abs'][:-1, i], store['movdirMn'])

    # Get minimum of all cricket relationships
    store['MCDist_nose_min'] = np.nanmin(store['MCDist_nose2crick'], axis=1)
    store['MCDist_outline_min'] = np.nanmin(store['MCDist_outline2crick'], axis=1)
    store['MCAngle_FA_min'] = np.nanmin(np.abs(store['MCAngle_FA']), axis=1)
    store['MCAngle_travelDir_min'] = np.nanmin(np.abs(store['MCAngle_travelDir']), axis=1)

    # Index of closest cricket per frame (based on nose distance)
    masked = np.ma.masked_invalid(store['MCDist_nose2crick'])
    closest_idx = np.ma.argmin(masked, axis=1)
    all_nan_mask = masked.mask.all(axis=1)
    closest_idx = closest_idx.astype('float')  # so we can insert np.nan
    closest_idx[all_nan_mask] = np.nan

    # Get closest cricket distance and angle for each frame
    frames = np.arange(len(M['bpts']))
    valid = ~np.isnan(closest_idx)

    closest_int = closest_idx[valid].astype(int)
    valid_frames = frames[valid]

    store['MCDist_nose_closest'] = np.full(len(frames), np.nan)
    store['MCDist_outline_closest'] = np.full(len(frames), np.nan)
    store['MCAngle_FA_closest'] = np.full(len(frames), np.nan)
    store['MCAngle_travelDir_closest'] = np.full(len(frames) - 1, np.nan)  # note: shorter

    store['MCDist_nose_closest'][valid_frames] = store['MCDist_nose2crick'][valid_frames, closest_int]
    store['MCDist_outline_closest'][valid_frames] = store['MCDist_outline2crick'][valid_frames, closest_int]
    store['MCAngle_FA_closest'][valid_frames] = store['MCAngle_FA'][valid_frames, closest_int]

    max_t = len(store['MCAngle_travelDir'])
    valid_frames_t = valid_frames[valid_frames < max_t]
    closest_int_t = closest_int[valid_frames < max_t]
    store['MCAngle_travelDir_closest'][valid_frames_t] = store['MCAngle_travelDir'][valid_frames_t, closest_int_t]

    # --- Derivatives ---
    dt = 1 / Fs  # timestep

    # Compute Savitzky-Golay coefficients
    polyorder = 3
    window_length = Fs/5
    window_length_long = Fs*2

    # must be odd
    if np.remainder(window_length, 2) == 0:
        window_length = window_length + 1
    if np.remainder(window_length_long, 2) == 0:
        window_length_long = window_length_long + 1

    # g matrix (each column derivative order)
    g = savgol_coeffs(window_length, polyorder, deriv=0, use='dot')  # 0th derivative (NIR smoothing, basically)
    g_first = savgol_coeffs(window_length, polyorder, deriv=1, use='dot')  # 1st derivative
    g_second = savgol_coeffs(window_length, polyorder, deriv=2, use='dot')  # 2nd derivative

    g_long = savgol_coeffs(window_length_long, polyorder,
                           deriv=0, use='dot')
    g_first_long = savgol_coeffs(window_length_long, polyorder,
                                 deriv=1, use='dot')  # 1st derivative
    g_second_long = savgol_coeffs(window_length_long, polyorder,
                                  deriv=2, use='dot')  # 2nd derivative

    # Non-circular fields
    nonCircFields = ['shelterDist', 'centerSpeed', 'head2BodyDist', 'tail2BodyDist', 'frontWidth',
                     'headVbodySpeed', 'rearWidth', 'bodyArea', 'motionCoherence', 'tailCurvature',
                     'spineCurvature', 'MCDist_centroids', 'MCDist_nose2crick', 'MCDist_outline2crick',
                     'frontHindLegCorr', 'frontlegCorr', 'hindlegCorr', 'head2bodyAngleOffset',
                     'faceLength', 'comMotion', 'frontVbackLegMotion', 'MCDist_nose_closest',
                     'MCDist_outline_closest', 'MCDist_outline_min', 'MCDist_nose_min',
                     'pcaMovMag_1', 'pcaMovMag_2', 'pcaMovMag_3', 'pcaMovMag_4', 'pcaMovMag_5',
                     'pcaPairDist_1', 'pcaPairDist_2', 'pcaPairDist_3', 'pcaPairDist_4', 'pcaPairDist_5',
                     'zonePC1', 'zonePC2']

    # Circular fields
    circFields = ['faceDir', 'bodyDir', 'movdirMn', 'MCAngle_abs',
                  'MCAngle_FA', 'MCAngle_travelDir', 'head2bodyAngle',
                  'MCAngle_FA_closest', 'MCAngle_travelDir_closest',
                  'MCAngle_FA_min', 'MCAngle_travelDir_min']

    # --- Process non-circular fields ---

    for field in nonCircFields:
        x = store[field]

        if x.ndim == 1:  # Single column
            dx0 = convolve(x, g, mode='same')
            dx1 = convolve(x, g_first * (-1 / dt), mode='same')
            dx2 = convolve(x, g_second * (1 / (dt ** 2)), mode='same')

            dx0_long = convolve(x, g_long, mode='same')
            dx1_long = convolve(x, g_first_long * (-1 / dt), mode='same')
            dx2_long = convolve(x, g_second_long * (1 / (dt ** 2)), mode='same')

            store[field] = dx0
            store[f"{field}_d1"] = dx1
            store[f"{field}_d2"] = dx2

            store[f"{field}_slow"] = dx0_long
            store[f"{field}_slow_d1"] = dx1_long
            store[f"{field}_slow_d2"] = dx2_long

        else:  # Multiple crickets (or bodyparts)
            smooth = np.zeros_like(x)
            foDer = np.zeros_like(x)
            soDer = np.zeros_like(x)

            smooth_long = np.zeros_like(x)
            foDer_long = np.zeros_like(x)
            soDer_long = np.zeros_like(x)

            for k in range(x.shape[1]):
                xk = x[:, k]
                smooth[:, k] = convolve(xk, g, mode='same')
                foDer[:, k] = convolve(xk, g_first * (-1 / dt), mode='same')
                soDer[:, k] = convolve(xk, g_second * (1 / (dt ** 2)), mode='same')

                smooth_long[:, k] = convolve(xk, g_long, mode='same')
                foDer_long[:, k] = convolve(xk, g_first_long * (-1 / dt), mode='same')
                soDer_long[:, k] = convolve(xk, g_second_long * (1 / (dt ** 2)), mode='same')

            store[field] = smooth
            store[f"{field}_d1"] = foDer
            store[f"{field}_d2"] = soDer

            store[f"{field}_slow"] = smooth_long
            store[f"{field}_slow_d1"] = foDer_long
            store[f"{field}_slow_d2"] = soDer_long

    for field in circFields:
        x = store[field]

        if x.ndim == 1:
            x = unwrapAngles_with_nans(x)
            dx0 = convolve(x, g, mode='same')
            dx1 = convolve(x, g_first * (-1 / dt), mode='same')
            dx2 = convolve(x, g_second * (1 / (dt ** 2)), mode='same')

            dx0_long = convolve(x, g_long, mode='same')
            dx1_long = convolve(x, g_first_long * (-1 / dt), mode='same')
            dx2_long = convolve(x, g_second_long * (1 / (dt ** 2)), mode='same')

            store[field] = wrapTo2Pi(dx0)
            store[f"{field}_d1"] = dx1
            store[f"{field}_d2"] = dx2

            store[f"{field}_slow"] = wrapTo2Pi(dx0_long)
            store[f"{field}_slow_d1"] = dx1_long
            store[f"{field}_slow_d2"] = dx2_long

        else:
            smooth = np.zeros_like(x)
            foDer = np.zeros_like(x)
            soDer = np.zeros_like(x)

            smooth_long = np.zeros_like(x)
            foDer_long = np.zeros_like(x)
            soDer_long = np.zeros_like(x)

            for k in range(x.shape[1]):
                xk = unwrapAngles_with_nans(x[:, k])
                smooth[:, k] = convolve(xk, g, mode='same')
                foDer[:, k] = convolve(xk, g_first * (-1 / dt), mode='same')
                soDer[:, k] = convolve(xk, g_second * (1 / (dt ** 2)), mode='same')

                smooth_long[:, k] = convolve(xk, g_long, mode='same')
                foDer_long[:, k] = convolve(xk, g_first_long * (-1 / dt), mode='same')
                soDer_long[:, k] = convolve(xk, g_second_long * (1 / (dt ** 2)), mode='same')

            store[field] = wrapTo2Pi(smooth)
            store[f"{field}_d1"] = foDer
            store[f"{field}_d2"] = soDer

            store[f"{field}_slow"] = wrapTo2Pi(smooth_long)
            store[f"{field}_slow_d1"] = foDer_long
            store[f"{field}_slow_d2"] = soDer_long

    # --- Time-binned Variance ---
    winSize = Fs
    if winSize % 2 == 0:
        winSize += 1  # make sure odd

    print(Fs)
    halfW = int(winSize // 2)

    # --- Non-circular fields variance ---
    for field in nonCircFields:
        x = store[field]

        if x.ndim == 1:
            varCentered = np.array([np.var(x[max(0, j - halfW):min(len(x), j + halfW + 1)], ddof=1)
                                    for j in range(len(x))])
        else:
            varCentered = np.zeros_like(x)
            for k in range(x.shape[1]):
                xk = x[:, k]
                varCentered[:, k] = [np.var(xk[max(0, j - halfW):min(len(xk), j + halfW + 1)], ddof=1)
                                     for j in range(len(xk))]

        store[f"{field}_var"] = varCentered

    # --- Circular fields variance ---
    for field in circFields:
        x = store[field]

        if x.ndim == 1:
            varCentered = np.array([circ_var(x[max(0, j - halfW):min(len(x), j + halfW + 1)])
                                    for j in range(len(x))])
        else:
            varCentered = np.zeros_like(x)
            for k in range(x.shape[1]):
                xk = x[:, k]
                varCentered[:, k] = [circ_var(xk[max(0, j - halfW):min(len(xk), j + halfW + 1)])
                                     for j in range(len(xk))]

        store[f"{field}_var"] = varCentered

    # --- Pack into Table ---
    maxRows = len(M['meanX'])  # total number of frames
    perframes = convert_store_to_table(store, maxRows)
    perframes = perframes.drop(columns=['faceDir','bodyDir','movdirMn'])

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

    # Step 1: detect large jumps to define segments
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

    # Step 2: flag short, glitchy segments
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

    # Step 3: remove anomalies
    cleaned_data.loc[anomaly_mask, x_col] = np.nan
    cleaned_data.loc[anomaly_mask, y_col] = np.nan

    # Step 4: interpolate short gaps, leave long ones as NaN
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


