# TUBBA: Temporal Unified Behavioral Bout Analysis

TUBBA is a modular pipeline for annotating and classifying animal behavior from video recordings. 
In part, it stems from my search for intuitive and flexible tools for rapid parallel labeling of multiple 
behavioral features across videos. TUBBA can be used as a simple annotation software, or it can be coupled with 
pose estimation (DeepLabCut, SLEAP etc.) to allow the user to leverage keypoint-based features for behavior classification.

TUBBA can use sparse behavioral labels to generate frame-by-frame likelihood estimates for multiple behaviors.
It does not force these behaviors to be fully orthogonal. At its core, TUBBA couples the XGBoost algorithm with a
relatively light-weight LSTM neural network to generate smooth predictions. TUBBA is designed to be flexible, easy to use,
and extensible for diverse behavioral neuroscience projects. TUBBA achieves state-of-the-art performance across 
multiple behaviors across multiple species!

Importantly, TUBBA works best when the user carefully design their own behavioral features to be used for 
classification as it allows the user to leverage unique high-order features in their data, such as relationships to 
other animals, or the environment. This makes TUBBA very interpretable, allowing the user to precisely 
track the features characteristic of each behavioral pattern. TUBBA thus does not require deep learning expertise to use, 
but it does require the user to think carefully about the features they want to use for classification.

TUBBA is currently in active development, and is not yet ready for public release. TUBBA is not yet able to
handle annotations from multiple interacting individual animals - if you have multiple animals in your videos, run TUBBA independently for each animal. 

TUBBA is heavily inspired by the classic JAABA software, originally developed by Kristen Branson's laboratory at Janelia farm
(Kabra, M., Robie, A., Rivera-Alba, M. et al. Nat Methods 10, 64–67 (2013). https://doi.org/10.1038/nmeth.2281), and largely 
developed with the goal of having a smooth, python-based alternative. 

https://github.com/user-attachments/assets/05a4235e-3212-4232-ada5-76ed9b93d714

---

## Installing TUBBA

TUBBA can easily be run straight from source. I recommend creating a new conda environment (miniConda on macOS).

```bash
    conda create -n tubba python=3.12
    conda activate tubba
    conda install -c conda-forge pyqt matplotlib seaborn scikit-learn h5py joblib opencv numpy pandas xgboost tqdm pytables
    cd /[path/to/TUBBA/src]
    python TUBBA.py
```

Major packages:
- `PyQt5`, `xgboost`, `torch`, `opencv-python`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `h5py`, `numpy`

---

## Launching TUBBA

To get started, navigate to the src folder of TUBBA in terminal, and simply run:

```bash
   python TUBBA.py
```

From there, you can either:
- **Create a new project** by selecting a folder of videos
- **Load an existing project** from a `.json` file that contains annotations and video paths

### Creating a new project

Creating a new project is simple. First, you should indicate whether the behavioral videos and matched keypoints have previously been downsampled (quite common during pose estimation). If the downsampled checkbox is marked, the user should enter a downsampling factor. **If no keypoints or perframe features are provided and you just want to annotate videos, check the "annotation only" checkbox, and don't worry about downsampling or preprocesessing!** 

Next, you should: 
- **Add behaviors** First add a list of behaviors (comma-separated) that they wish to track (don't worry if this is not complete - you can always add more later). 
- **Add videos** Typically, each video and its corresponding tracking files (csv format) are placed inside a separate folder. You can select multiple folders to batch-add videos to your project. However, if you are running in annotation-only mode, please provide paths to your videos directly instead. 
- **Select a feature script** Select the feature extraction script you'd like to use. These scripts are stored in the featureExtractions folder in TUBBA/src, and extract some core information about each video (frame rates, absolute and relative paths etc), and creates a table of frame-by-frame features that TUBBA can use to learn and infer the signatures of your behaviors. Please see the "feature script" section for more information. 
- **Pre-process data** Executes the feature extraction script

That's it! Press **Next** to proceed to annotation. 

---

## Annotating behaviors

Annotations are done in the gui. Each behavior defined will yield two parallel buttons, one for marking segments where the behavior is certainly occuring, and one for marking segments where the behavior is certainly NOT occuring. I encourage you to be stringent and precise in defining the frames that constitute a behavior, and to ensure that the behavior is not occuring in NOT frames. 

When a given behavioral button is toggled, all annotations will belong to this state. Segments can be initialized by pressing your "s"-key (start), and closed by pressing your "e"-key (end). Segments will immediately appear in the ethogram under the video.

Once you have created a decent number of annotations (both positive and negative, at least 10-20 segments each, ideally across multiple videos), you should attempt to train a model on the given data and run inference on your video. Odds are, the model will be coarsely correct, but will require further refinement. I recommend that you use the model's initial outputs to guide further labeling, focusing on where the model is uncertain to hone the next iteration. Proceed recursively until you are happy with your inference accuracy. 

---

##  Project Structure

Presuming you are not running in annotation mode (in which case you can place all videos in a single folder), your project structure should look something like: 

```
project_dir/
├── Data/
│   ├── Video1/
│       ├── recording.mp4
│       ├── trackletsA.csv
│       ├── trackletsB.csv
│       ├── perframe_feats.h5
│       └── inference
│           └──  recording_inferred.pkl
│   ├── Video2/...
│   └── Video3/...
├── TUBBA_project.json
├── TUBBA_project_trainedModels.pkl
```

- `*.mp4`: Original videos
- `*.h5`: Extracted features
- `annotations.json`: Manual annotations
- `inference.pkl`: Model predictions and confidence scores
- `TUBBA_project.json`: Central project file

---

## Training a Model

Typically training is done from the gui. If you prefer to run it from the command line once behaviors are annotated:

```python
from TUBBA_train import train_TUBBAmodel

train_TUBBAmodel(project_path="path/to/project", window_size=(Fs*2+1), lstm_epochs=1000)
```

This will:
- Extract training windows
- Apply jittering and class balancing
- Train an XGBoost classifier
- Train a bidirectional LSTM smoother
- Save the full model to `.pkl`

### Under the hood 

Training in TUBBA proceeds in two phases:

1. Feature Preparation and Augmentation
    •    Training data is extracted from annotated frames using a sliding window approach.
    •    Per-frame features are z-scored and NaNs imputed where needed.
    •    An optional jittering process adds Gaussian noise to positive and negative samples to improve generalization.
    •    Data is balanced to prevent bias toward the majority class, with configurable ratios of positive, negative, and unlabeled examples.

2. Model Training
    •    A multi-class XGBoost classifier is trained on the feature vectors from all videos.
    •    Predictions are then smoothed using a bidirectional LSTM, which learns temporal structure and contextual dependencies.
    •    The combined XGB + LSTM model is serialized to a .pkl file, along with normalization parameters and behavior mappings.

Training in TUBBA proceeds in two phases:

1. Feature Preparation and Augmentation
    •    Training data is extracted from annotated frames using a sliding window approach.
    •    Per-frame features are z-scored and NaNs imputed where needed.
    •    An optional jittering process adds Gaussian noise to positive and negative samples to improve generalization.
    •    Data is balanced to prevent bias toward the majority class, with configurable ratios of positive, negative, and unlabeled examples.

2. Model Training
    •    A multi-class XGBoost classifier is trained on the feature vectors from all videos.
    •    Predictions are then smoothed using a bidirectional LSTM, which learns temporal structure and contextual dependencies.
    •    The combined XGB + LSTM model is serialized to a .pkl file, along with normalization parameters and behavior mappings.
    

### What is my model picking up on?

You can export the weigtht of all of your features in the trained model to assess which ones contrubute most strongly to each behavioral category. 

```python
from TUBBA_utils import export_predictorWeights

export_predictorWeights(model_path="path/to/model.pkl", out_path='save/here/please')
```
---

## Running Inference

Typically training is done from the gui, after a model has been trained. If you prefer to run it from the command line once a model has been trained:

```python
from TUBBA_infer import TUBBA_modelInference

predictions = TUBBA_modelInference(project_json_path, video_name, video_folder)
```

Outputs:
- `predictions["binary"]`: Frame-level predictions (0/1)
- `predictions["confidence"]`: Model confidence per frame

You can also run batch inference from the gui, on all videos in the project. You can subsequently toggle between viewing inferred or annotated behaviors. 

---

## Rendering Videos of Annotated Inferred Behaviors

In order to validate your model's performance, I recommend creating stitched videos of inferred instances of each of your behaviors. 

```python
from src.TUBBA_utils import predictions_to_video

predictions_to_video(
    source_video_path="...mp4",
    predictions_path="...inference.pkl",
    behavior="Escape"
)
```
You may also want to sanity-check your annotations, in case you run into strange results: 

`annotations_to_video(project_json_path, behavior, out_path, target)`

Here, target can be set to either `1` (all positive examples of the behavior), or `-1` (all instances where behavior has been labeled as NOT occuring)

---
## Importing and Exporting Annotations

Annotations can be imported and exported to csv files in the format `[n_frames × n_behaviors]` directly from the gui. If you are importing annotations, please ensure that you don't have a column indicating frame indices (TUBBA assumes you start from zero and go through the end of the video), and that you have a header row indicating the names of the behaviors you are annotating. 

TUBBA can also export annotations made in the gui to csv files in this same format. `1` indicates positive examples, `-1` negative examples, and `0` unlabeled frames.  

---
## Feature Scripts

TUBBA supports **custom feature extraction scripts** that process video and tracking data into feature vectors used for behavior classification. These scripts live in:

    src/featureExtractions/

Each script must expose a function called `tracksToFeatures(folder, spatialSR)`. When a new project is created, TUBBA will automatically list available scripts in a dropdown so you can select the one appropriate for your dataset.

### What Feature Scripts Do

A feature script takes a **folder containing a video and tracking data**, and returns:

- A `vidInfo` dictionary describing:
  - video name ("name")
  - folder path ("dir")
  - frame count ("nFrames")
  - frame rate ("Fs")
  - spatial sampling rate ("samplingRate")
  - path to a feature file ("featureFile", see below)
  - status (True/False for whether data was successfully extracted)
  - optional metadata (e.g., detected region centers)

In addition, feature scripts place a pandas `DataFrame` of features (`[n_frames × n_features]`) inside the video directory. 

### What Are Features

In TUBBA, *features* are numerical descriptions of animal behavior derived from video or tracking data. They capture movement, posture, speed, orientation, relative positions, and more — and form the input to machine learning models that classify behaviors.

Examples of features include:
- Distance between body parts
- Speed or acceleration of the nose
- Angle between the head and tail
- PCA components of posture or motion
- Distance and angle to another animal or object

#### Why Are They Custom?

Different experiments call for different features. What matters for grooming in a mouse may not matter for courtship in a fly. TUBBA lets you define your own feature extraction pipeline tailored to:

- The **species** you study
- The **tracking data** you have (e.g., body parts, pose landmarks)
- The **behaviors** you're interested in

By writing a custom feature script, you can plug in exactly the features that will help the model learn — and ignore what you either don't think will, or don't want in your dataset. 
This flexibility makes TUBBA adaptable to any experimental setup.

### Example

Here's the required signature and expected output structure:

```python
def tracksToFeatures(folder, spatialSR):

    import pandas as pd 
    
    # Load tracking data and video
    # Compute distances, angles, speeds, PCA components, etc.

    vidInfo = {
        'name': "example.mp4",
        'dir': folder,
        'nframes': nFrames,
        'frameRate': frameRate,
        'samplingRate': spatialSR,
        'featureFile': "perframe_feats.h5",
        'status': 1  # success
    }
    
    trackingData = pd.read_csv([path/to/myDLC])
    
    # Custom function to convert tracking data to features
    perframes = getPredictors(trackingData, vidInfo['Fs'])
    
    # Place features into video folder
    feature_path = os.path.join(folder, vidInfo['featureFile'])
    perframes.to_hdf(feature_path, key='perframes', mode='w', format='table', complevel=5)

    return vidInfo
```

The `perframes` can contain raw or derived features — TUBBA typically handles normalization and missing data well. 

### Writing Your Own

To add your own script:

1. Create a new `.py` file inside `src/featureExtractions/`
2. Define a function like `myCustomFeatures(folder, spatialSR)`
3. Return the same `vidInfo` dict and `DataFrame` format
4. TUBBA will automatically pick it up in the dropdown at project creation

Use this to support alternative tracking pipelines, species, or custom features.

### 😠 But Tom, I don't want to write my own code!

All good! Sometimes we just want to plug and play. There are three potential workarounds. 

1. You can use TUBBA in annotation-only mode! If your videos are short and you don't have too many of them, hand-annotation is often the path of least resistance. TUBBA was in large-part developed to provide a quick and easy way of annotating multiple behaviors in parallel. 
2. Got keypoint tracks? Great. We provide a default `TUBBA_getFeats` script that will search your video folder for a tracking data (in csv format), and create hundreds to thousands of features from these alone that can be used to train a model. This has the added benefit of being largely environment-agnostic. These features relate to the organization of the keypoints in space, and their angular and euclidian relationships to one another. `TUBBA_getFeats` will try to filter out large jumps in raw tracking data, if they exist, and interpolate across short bouts of missing frames
3. You can pre-define features, and let TUBBA do the linking, by runnign `TUBBA_importFeats`. Just place the features you care about (perhaps the speed of your animal, distance to keypoints, joint angles etc.) into a frame-by-frame csv file (`[n_frames × n_features]`) inside the corresponding video folder. You should have a header-row providing a useful descriptor name for each column. TUBBA will ask you if you'd like it to perform a "feature expansion", in which it computes the first few derivatives of the core features and smooths them with both long and short windows to provide both instantaneous values and global context.

⚠️ **WARNING** ⚠️ TUBBA models are only as good as the features they receive. I urge you to think carefully about which features should be left out of your model to make your predictions accurate, unbiased, and generalizable. 

---

## Data Format Summary

- **Annotations**: `[start_frame, end_frame, value]` (1 = present, -1 = absent)
- **Features**: `.h5`, `[n_frames, n_features]`, may contain NaNs
- **Inference**: Pickle with predictions and confidences
- **Labels**: Multi-label possible (e.g., multiple behaviors at one frame)

---

## Dependencies

Install via:

```bash
conda create -n tubba python=3.12
conda activate tubba
pip install -r requirements.txt
conda install -c conda-forge pyqt matplotlib seaborn scikit-learn h5py joblib opencv numpy pandas xgboost tqdm pytables
```

Major packages:
- `PyQt5`, `xgboost`, `torch`, `opencv-python`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `h5py`, `numpy`

---

## 🧹 TODOs

- [ ] Enable user to load annotations from csv (frame-stamped)
- [ ] Enable user to export annotations from full gui
- [ ] Enable loading of externally defined perframe features. 
- [ ] Enable dynamic toggling between model architectures 

---

## 👋 Notes

- TUBBA is under active development.
- All suggestions, bugs, and feature requests are welcome — add them to the internal tracker or Slack thread.
