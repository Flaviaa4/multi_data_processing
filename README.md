
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

To be completed by the audio processing owner.

## 4. CLI app

To be completed. The app will run the face check, then the voice check, and call `predict_product(user_id)` for verified customers.

