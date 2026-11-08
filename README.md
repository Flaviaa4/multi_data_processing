# Image and face recognition pipeline

This repository contains Member B's image-processing and facial-recognition
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
