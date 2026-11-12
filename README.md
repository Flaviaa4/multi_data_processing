
# Image and face recognition pipeline



# multi_data_processing

Multimodal data preprocessing for a user identity and product recommendation system (Formative 2).

A customer is accepted in three steps:

1. **Face check:** an image model recognizes the customer from a photo.
2. **Voice check:** an audio model verifies the customer from a voice sample.
3. **Product prediction:** a tabular model predicts which product category the verified customer is likely to buy.

If either check fails, access is denied and no prediction is made.

## Team

| Member | Responsibility | Branch |
|---|---|---|
| Flavienne | Tabular data, product model, CLI app | `feature/tabular` |
| Shakira | Image processing, face model | `feature/image` |
| Placide | Audio processing, voice model | `feature/audio` |

## Setup

Tested with Python 3.14.

```bash
git clone https://github.com/Flaviaa4/multi_data_processing.git
cd multi_data_processing
python -m pip install -r requirements.txt
```

Library versions are pinned because the saved models are pickled. Loading a `.pkl` file with a different scikit-learn or xgboost version can fail or give warnings, so use the versions in `requirements.txt`.

## Repository structure

```
data/
  raw/                     original CSV files for the tabular part
  processed/               merged_dataset.csv, built by the tabular pipeline
models/
  product_model.pkl        trained product recommendation pipeline
notebooks/
  tabular_pipeline.ipynb   tabular walkthrough: cleaning, merge, EDA, features, models
src/
  tabular/product_model.py tabular pipeline, train() and predict_product()
```

## 1. Tabular data and product model

### Data

| File | Rows | Key | Contents |
|---|---|---|---|
| `data/raw/customer_social_profiles.csv` | 155 | `customer_id_new` (e.g. `A178`) | social platform, engagement score, purchase interest score, review sentiment |
| `data/raw/customer_transactions.csv` | 150 | `customer_id_legacy` (e.g. `178`) | transaction id, amount, date, product category, rating |

Both files describe the same customers (ids 100 to 199) in different id formats, and neither has one row per customer.

### Pipeline

1. **Cleaning.** Dropped 5 exact duplicate social rows (155 to 150). Created a shared integer key `customer_id` by removing the `A` prefix. Parsed dates, flagged the 10 missing ratings in `rating_missing` and filled them with the median. Standardized text values and rejected anything outside the allowed lists.
2. **Merge.** A direct join would multiply rows (213 for inner, 246 for left), so the social data is first aggregated to one row per customer: mean scores, post count, most common platform and sentiment, with ties broken alphabetically. This is then left joined onto the 150 transactions. Checks confirm 150 rows, unique transaction ids, 117 matched and 33 unmatched transactions, and 61 customers present in both files.
3. **EDA.** Target distribution, numeric distributions and outliers, a correlation heatmap and social signals by category. The classes are balanced (25 to 35 each), there are no outliers, and every correlation is weak (0.23 or below).
4. **Feature engineering.** 16 features: date (month, day of week, weekend), amount band (quartiles), customer level (transaction count, mean amount, mean rating), social (scores, post count, top platform, top sentiment, `has_social_data`) and `rating_missing`. No feature uses `product_category`, which is verified by shuffling the target and checking the features do not change.
5. **Modeling.** The train/test split holds out 20 percent of customers (`GroupShuffleSplit`), so no customer is in both sets. Each model is one sklearn `Pipeline` (median imputer, scaler, one-hot encoder) tuned with `GroupKFold`. Models: most frequent class baseline, Logistic Regression, Random Forest and XGBoost.

### Results

Test set: 25 transactions from 15 held-out customers. Chance level is about 0.20.

| Model | CV macro F1 | Test accuracy | Test macro F1 | Test log loss |
|---|---|---|---|---|
| XGBoost (selected) | 0.248 | 0.08 | 0.064 | 2.24 |
| Random Forest | 0.192 | 0.16 | 0.133 | 1.94 |
| Logistic Regression | 0.172 | 0.12 | 0.098 | 2.07 |
| Baseline (most frequent) | 0.061 | 0.16 | 0.055 | 1.66 |

The model is selected by cross-validated macro F1 on the training customers, because picking on a 25-row test set would mostly reward luck. It is then refit on all 150 transactions and saved.

No model reliably beats chance on unseen customers. The dataset is small (150 transactions, 75 customers), feature correlations are weak, and most repeat customers buy across several categories. The pipeline is sound, but the saved model should be treated as a working placeholder for the recommendation step, not a reliable recommender.

### How to run

Rebuild the dataset, retrain and print the metrics (about 40 seconds):

```bash
python src/tabular/product_model.py
```

Run the notebook from a clean kernel (about 70 seconds):

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/tabular_pipeline.ipynb
```

Predict for a customer from Python, run from the repository root:

```python
from src.tabular.product_model import predict_product

predict_product(151)     # "Sports"
predict_product("A151")  # both id formats are accepted
predict_product(999)     # raises ValueError: customer not in the dataset
```

`predict_product` loads the model once, looks up the customer's most recent transaction in `data/processed/merged_dataset.csv` and returns the category name. Customers with social data but no transactions are not in the dataset and raise `ValueError`.

## 2. Image processing and face model

This section contains  image-processing and facial-recognition
pipeline.

## Image collection

Place three images for every member in `data/images/`. The filename should
start with the member name; optional trailing digits and an optional pose
suffix are ignored when determining the member label. For example:

```text
flavia.jpeg
flavia1.jpeg
flavia3.jpeg
```

Supported formats are `.jpg`, `.jpeg`, `.png`, and `.heic`.

## Build the image model

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python scripts/build_face_model.py
```

The script:

- extracts grayscale pixel and RGB histogram features;
- creates rotate, flip, and grayscale augmentations;
- saves features to `data/image_features.csv`;
- trains a Random Forest face-recognition model;
- saves the model to `models/face_model.pkl`; and
- saves accuracy, weighted F1, and log loss to
  `models/face_metrics.json`.

## CLI integration contract

The future CLI can authenticate an image without knowing the feature
implementation:

```python
from image_pipeline import verify_face

result = verify_face("data/images/memberName_neutral.jpg")
if not result["recognized"]:
    print("Access denied")
else:
    print(result["userId"], result["confidence"])
```

`verify_face` returns `recognized`, `userId`, and `confidence`. The default
recognition threshold is `0.60`.

## 3. Audio processing and voice model

The voice check is the last step before a prediction is shown: the customer says an approval phrase, and the model must confirm that the voice belongs to a known member.

### Data

| File | Speaker | Phrase |
|---|---|---|
| `audio_pipeline/data/audio/flav1.mp4`, `flav2.mp4` | flavienne | "Yes, approve", "Confirm transaction" |
| `audio_pipeline/data/audio/plac1.mp4`, `plac2.mp4` | Placide | "Yes, approve", "Confirm transaction" |
| `audio_pipeline/data/audio/shakira 1.mp4`, `shakira 2.mp4` | shakira | "Yes, approve", "Confirm transaction" |
| `audio_pipeline/data/audio/unauthorized/unauthorized_friend_approve.mp4`, `..._confirm.mp4` | someone outside the group | both phrases, for the unauthorized test |

Each member recorded each phrase once (6 recordings). All members say the same words, so the model has to tell them apart by voice, not by what is said.

### Pipeline

1. **Loading.** Phone recordings are converted to 16 kHz mono WAV with ffmpeg, silence at the start and end is trimmed (30 dB below peak), and each clip is peak-normalised. The notebook renames `<name>1` to `<name>_approve_1` and `<name>2` to `<name>_confirm_1`.
2. **Visualisation.** A waveform and a mel spectrogram for each member and phrase (`figures/audio_waveforms_spectrograms.png`). The measured differences between voices:

    | Member | Median pitch (Hz) | Spectral centroid (Hz) | Roll-off (Hz) | RMS energy | Mean duration (s) |
    |---|---|---|---|---|---|
    | flav | 196 | 2,176 | 4,079 | 0.082 | 1.91 |
    | plac | 73 | 1,344 | 2,892 | 0.042 | 3.91 |
    | shakira | 169 | 1,123 | 1,983 | 0.173 | 1.30 |

    flav has the highest and brightest voice, plac the lowest pitch and the slowest delivery, and shakira the loudest and shortest recordings. These differences are what the features capture.
3. **Augmentation.** Six augmentations per recording (`figures/audio_augmentations.png`), giving 7 versions of each clip:

    | Augmentation | Simulates |
    |---|---|
    | `pitch_up`, `pitch_down` (±1 semitone) | a slightly higher or lower voice; kept small because pitch identifies the speaker |
    | `stretch_fast` (×1.15), `stretch_slow` (×0.87) | speaking faster or slower |
    | `noise` (white noise, 20 dB SNR) | a noisier room |
    | `gain_shift` (60% volume, up to 0.2 s delay) | speaking quieter or starting later |

4. **Features.** 50 features per clip, saved in `audio_pipeline/data/audio_features.csv` (42 rows: 6 recordings × 7 versions): MFCC 1–13 mean and standard deviation, delta-MFCC means, spectral roll-off, centroid and bandwidth, zero-crossing rate, RMS energy and duration. `clip_id` links each augmented row to its original recording.
5. **Modeling.** Logistic Regression (C = 0.1, 1, 10) and Random Forest (300 trees, leaf size 1 or 3), each a sklearn `Pipeline` with a scaler. They are evaluated by **leave one recording out**: each original recording is held out in turn, *with all its augmented copies removed from training*, so the model is never tested on a near-copy of a training clip. The model with the lowest log loss is refit on all 42 clips and saved.
6. **Accept or reject.** A classifier trained on three people always answers with one of the three, so a voice is accepted only if both checks pass:
    - the highest class probability is at least **0.55**, and
    - the clip is close to one of that member's own recordings: the nearest-neighbour distance on the MFCC features is at most a **radius of 8.13**, the largest distance between a member's held-out recording and their other recordings, plus a 10% margin.

### Results

Leave-one-recording-out, on the 6 held-out original recordings:

| Model | Accuracy | Macro F1 | Log loss | Genuine accept rate | Accuracy on augmented clips |
|---|---|---|---|---|---|
| Logistic Regression, C=10 (selected) | 1.00 | 1.00 | 0.159 | 0.67 | 0.89 |
| Logistic Regression, C=1 | 1.00 | 1.00 | 0.323 | 0.67 | 0.83 |
| Logistic Regression, C=0.1 | 0.83 | 0.82 | 0.555 | 0.50 | 0.72 |
| Random Forest, 300 trees | 1.00 | 1.00 | 0.709 | 0.17 | 0.81 |
| Random Forest, 300 trees, leaf=3 | 1.00 | 1.00 | 0.736 | 0.00 | 0.81 |

*Genuine accept rate* is the share of held-out recordings that pass both checks above. Logistic Regression gives better-calibrated probabilities than Random Forest, which is why it passes the confidence check more often.

Unauthorized attempts (`figures/voice_model_evaluation.png`, notebook section 7):

| Test | Result |
|---|---|
| Stranger says "Yes, approve" | **accepted** as shakira (confidence 0.66, distance 6.89) |
| Stranger says "Confirm transaction" | **accepted** as plac (confidence 0.71, distance 4.94) |
| Member's voice claimed as another member (3 cases) | 3 of 3 rejected |

The model identifies which member is speaking (accuracy 1.00), but the stranger check does not work yet: both stranger clips were accepted, and 2 of the 6 genuine recordings would be rejected. The cause is the data. With one recording per phrase, the only same-speaker comparison available is "approve" against "confirm", which differ a lot, so the radius has to be wide (8.13) and the stranger's distances (4.94 and 6.89) fall inside it. Six recordings are also too few for the accuracy figures to be reliable. The fix is to record each phrase 2–3 more times per member and re-run the notebook. Same-phrase comparisons then calibrate a tighter radius.

### How to run

Run `audio_pipeline/audio_pipeline.ipynb` in Google Colab: run the upload cell (`UPLOAD_NOW = True`) to add the recordings, then *Runtime → Run all*. Name stranger files `unauthorized_<name>.<ext>` and they are moved to the `unauthorized/` folder. The last cells build `voice_app_bundle.zip` and back up the model, features and figures to `My Drive/audio_pipeline/`.

Verify a voice from Python, run from the repository root (non-WAV files need ffmpeg):

```python
from audio_pipeline.src.voice import verify_voice, check_voice

verify_voice("clip.wav")                          # "flav", "plac", "shakira", or None if rejected
verify_voice("clip.wav", expected_member="flav")  # None unless the voice is flav's: use the name from the face check
check_voice("clip.wav")                           # details: best match, confidence, distance, reason
```

`verify_voice` loads `audio_pipeline/models/voice_model.pkl` once. Passing `expected_member` is what links the voice step to the face step: the voice must belong to the same person the face model recognised.

### Files

| File | Purpose |
|---|---|
| `audio_pipeline/audio_pipeline.ipynb` | full audio walkthrough: plots, augmentation, features, model comparison, unauthorized tests |
| `audio_pipeline/src/audio_processing.py` | loading, conversion, augmentation, feature extraction |
| `audio_pipeline/src/voice.py` | model comparison, training, `verify_voice()` and `check_voice()` |
| `audio_pipeline/data/audio_features.csv` | features of every recording and augmentation |
| `audio_pipeline/models/voice_model.pkl` | trained model with its thresholds |
| `audio_pipeline/models/voice_model_info.json` | members, thresholds, scores and library versions |
| `audio_pipeline/voice_app/` | copy of the model and code for the CLI app |
| `audio_pipeline/figures/` | waveforms and spectrograms, augmentations, feature overview, evaluation |

### Known issue: library versions

The model was trained in Colab with scikit-learn 1.6.1, but `requirements.txt` pins 1.9.1. It loads and predicts, but scikit-learn warns that results may be invalid. Retrain the voice model with the pinned versions (run the notebook after `pip install -r requirements.txt`) so all three models use the same environment.

Install the Python dependencies from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

FFmpeg is also required for MP4 input. On Windows, install it with:

```powershell
winget install --id Gyan.FFmpeg.Shared -e
```

Close and reopen PowerShell after installation, then verify:

```powershell
ffmpeg -version
```

## 4. CLI and transaction demo

The CLI authenticates both a face and a voice before revealing the customer's
latest transaction and product recommendation. Access is denied unless:

1. The face is recognized above the configured confidence threshold.
2. The voice is accepted by the voice model.
3. The voice identity matches the recognized face.

The sample identity-to-customer mapping is defined in `src/demo_app.py`;
replace it with the application's database mapping for production use.

Run an authorized transaction with the included Flavia recording:

```powershell
.\.venv\Scripts\python.exe scripts\secure_cli.py `
  --image data\images\flavia.jpeg `
  --voice "audio_pipeline\data\audio\flav1.mp4"
```

Run an interactive session:

```powershell
.\.venv\Scripts\python.exe scripts\secure_cli.py
```

Run the denied-access demonstration:

```powershell
.\.venv\Scripts\python.exe scripts\secure_cli.py --unauthorized-demo
```

### CLI options

| Option | Description |
|---|---|
| `--image PATH` | Image used for face authentication. |
| `--voice PATH` | Audio or video file used for voice authentication. |
| `--voice-model PATH` | Optional custom voice model; defaults to `audio_pipeline\models\voice_model.pkl`. |
| `--threshold VALUE` | Face confidence threshold; defaults to `0.60`. |
| `--unauthorized-demo` | Runs a guaranteed denied-access demonstration. |

Available sample recordings include:

```text
audio_pipeline\data\audio\flav1.mp4
audio_pipeline\data\audio\flav2.mp4
audio_pipeline\data\audio\plac1.mp4
audio_pipeline\data\audio\plac2.mp4
audio_pipeline\data\audio\shakira 1.mp4
audio_pipeline\data\audio\shakira 2.mp4
```

If PowerShell reports that `ffmpeg` is not recognized, add the FFmpeg `bin`
directory to the Windows User `Path`, reopen PowerShell, and run
`ffmpeg -version` again. The scikit-learn `InconsistentVersionWarning` shown
when loading the bundled voice model is a compatibility warning; the model
was trained with scikit-learn 1.6.1.

The same flow is available in `notebooks/secure_transaction_demo.ipynb`.
