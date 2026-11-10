"""Build data/image_features.csv and models/face_model.pkl."""

from image_pipeline import build_image_features_csv, train_face_model


if __name__ == "__main__":
    features = build_image_features_csv()
    metrics = train_face_model(features)
    print(f"Features written to {features}")
    print(f"Metrics: {metrics}")
