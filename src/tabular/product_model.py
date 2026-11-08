"""Tabular pipeline: clean and merge customer data, then train the product model."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
SOCIAL_PATH = ROOT / "data" / "raw" / "customer_social_profiles.csv"
TRANSACTIONS_PATH = ROOT / "data" / "raw" / "customer_transactions.csv"
MERGED_PATH = ROOT / "data" / "processed" / "merged_dataset.csv"
MODEL_PATH = ROOT / "models" / "product_model.pkl"

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SELECTION_METRIC = "macro_f1"
BASELINE = "Baseline (most frequent)"
KEY = "customer_id"
TARGET = "product_category"
UNKNOWN = "Unknown"

ALLOWED_PLATFORMS = ["Facebook", "Instagram", "LinkedIn", "TikTok", "Twitter"]
ALLOWED_SENTIMENTS = ["Negative", "Neutral", "Positive"]
ALLOWED_CATEGORIES = ["Books", "Clothing", "Electronics", "Groceries", "Sports"]

SOCIAL_SCORES = ["engagement_score", "purchase_interest_score"]
SOCIAL_CATEGORICAL = ["top_platform", "top_sentiment"]
KEY_COLUMNS = [
    KEY,
    "transaction_id",
    "purchase_amount",
    "purchase_date",
    "customer_rating",
    TARGET,
]

AMOUNT_BANDS = ["Low", "Mid-low", "Mid-high", "High"]
NUMERIC_FEATURES = [
    "purchase_amount",
    "customer_rating",
    "customer_txn_count",
    "customer_avg_amount",
    "customer_avg_rating",
    *SOCIAL_SCORES,
    "social_post_count",
]
CATEGORICAL_FEATURES = [
    "purchase_month",
    "purchase_dayofweek",
    "amount_band",
    *SOCIAL_CATEGORICAL,
]
BOOLEAN_FEATURES = ["is_weekend", "has_social_data", "rating_missing"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
ID_COLUMNS = ["transaction_id", KEY, "purchase_date"]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the raw social and transactions tables."""
    return pd.read_csv(SOCIAL_PATH), pd.read_csv(TRANSACTIONS_PATH)


def standardize_category(values: pd.Series, allowed: list[str]) -> pd.Series:
    """Strip whitespace, map any casing onto the canonical spelling, reject unknown values."""
    canonical = {value.lower(): value for value in allowed}
    standardized = values.astype(str).str.strip().str.lower().map(canonical)
    invalid = values[standardized.isna()].unique()
    if len(invalid):
        raise ValueError(f"Unexpected values in {values.name}: {list(invalid)}")
    return standardized


def clean_social(social: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates, add the integer customer_id key, standardize text columns."""
    social = social.drop_duplicates().reset_index(drop=True)
    social[KEY] = social["customer_id_new"].str.strip().str.removeprefix("A").astype(int)
    social["social_media_platform"] = standardize_category(
        social["social_media_platform"], ALLOWED_PLATFORMS
    )
    social["review_sentiment"] = standardize_category(
        social["review_sentiment"], ALLOWED_SENTIMENTS
    )
    return social.drop(columns="customer_id_new")


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add the customer_id key, parse dates, flag and fill missing ratings."""
    transactions = transactions.copy()
    transactions[KEY] = transactions["customer_id_legacy"].astype(int)
    transactions["purchase_date"] = pd.to_datetime(
        transactions["purchase_date"], format="%Y-%m-%d"
    )
    # The flag is created before filling so the model can still see which ratings were imputed.
    transactions["rating_missing"] = transactions["customer_rating"].isna()
    transactions["customer_rating"] = transactions["customer_rating"].fillna(
        transactions["customer_rating"].median()
    )
    transactions[TARGET] = standardize_category(transactions[TARGET], ALLOWED_CATEGORIES)
    return transactions.drop(columns="customer_id_legacy")


def deterministic_mode(values: pd.Series) -> str:
    """Most frequent value, with ties broken alphabetically so reruns give the same result."""
    counts = values.value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def aggregate_social(social: pd.DataFrame) -> pd.DataFrame:
    """Collapse the social table to one row per customer."""
    return (
        social.groupby(KEY)
        .agg(
            engagement_score=("engagement_score", "mean"),
            purchase_interest_score=("purchase_interest_score", "mean"),
            social_post_count=("social_media_platform", "size"),
            top_platform=("social_media_platform", deterministic_mode),
            top_sentiment=("review_sentiment", deterministic_mode),
        )
        .reset_index()
    )


def merge_social(transactions: pd.DataFrame, social_agg: pd.DataFrame) -> pd.DataFrame:
    """Left join per-customer social features onto every transaction.

    Social scores stay NaN for unmatched customers: the model pipeline imputes
    them with a median fitted on the training split only.
    """
    merged = transactions.merge(
        social_agg, on=KEY, how="left", validate="many_to_one", indicator=True
    )
    merged["has_social_data"] = merged.pop("_merge").eq("both")
    merged["social_post_count"] = merged["social_post_count"].fillna(0).astype(int)
    merged[SOCIAL_CATEGORICAL] = merged[SOCIAL_CATEGORICAL].fillna(UNKNOWN)
    return merged


def validate_merge(merged: pd.DataFrame, transactions: pd.DataFrame) -> dict[str, int]:
    """Raise if the merge changed the transaction grain, otherwise return match counts."""
    if len(merged) != len(transactions):
        raise ValueError(f"Merge changed row count: {len(transactions)} to {len(merged)}")
    if not merged["transaction_id"].is_unique:
        raise ValueError("transaction_id is no longer unique after the merge")
    null_counts = merged[KEY_COLUMNS].isna().sum()
    if null_counts.any():
        raise ValueError(f"Nulls in key columns: {null_counts[null_counts > 0].to_dict()}")
    matched = int(merged["has_social_data"].sum())
    return {"rows": len(merged), "matched": matched, "unmatched": len(merged) - matched}


def load_and_merge() -> pd.DataFrame:
    """Clean both raw files and return the validated transaction-level merge."""
    social_raw, transactions_raw = load_raw()
    transactions = clean_transactions(transactions_raw)
    merged = merge_social(transactions, aggregate_social(clean_social(social_raw)))
    validate_merge(merged, transactions)
    return merged


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Month, day of week (Monday is 0) and a weekend flag from purchase_date."""
    df = df.copy()
    df["purchase_month"] = df["purchase_date"].dt.month
    df["purchase_dayofweek"] = df["purchase_date"].dt.day_of_week
    df["is_weekend"] = df["purchase_dayofweek"].ge(5)
    return df


def add_amount_band(df: pd.DataFrame) -> pd.DataFrame:
    """Quartile band of purchase_amount, labelled Low to High."""
    df = df.copy()
    df["amount_band"] = pd.qcut(df["purchase_amount"], q=len(AMOUNT_BANDS), labels=AMOUNT_BANDS).astype(str)
    return df


def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-customer transaction count, mean amount and mean rating, repeated on each row.

    These never look at product_category, and because the split is grouped by
    customer, a test customer's aggregates come only from that customer's own rows.
    """
    df = df.copy()
    by_customer = df.groupby(KEY)
    df["customer_txn_count"] = by_customer["transaction_id"].transform("size")
    df["customer_avg_amount"] = by_customer["purchase_amount"].transform("mean")
    df["customer_avg_rating"] = by_customer["customer_rating"].transform("mean")
    return df


def build_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Add every engineered feature and return ids, features and target in a fixed order."""
    featured = add_customer_features(add_amount_band(add_date_features(merged)))
    return featured[ID_COLUMNS + FEATURES + [TARGET]]


def build_dataset() -> pd.DataFrame:
    """Rebuild the model-ready dataset from the raw files."""
    return build_features(load_and_merge())


def save_dataset(dataset: pd.DataFrame) -> None:
    MERGED_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(MERGED_PATH, index=False)
    _cached_dataset.cache_clear()


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(MERGED_PATH, parse_dates=["purchase_date"])


def encode_target(labels: pd.Series) -> np.ndarray:
    """Map category names to their index in ALLOWED_CATEGORIES (XGBoost needs integer labels)."""
    return labels.map({category: i for i, category in enumerate(ALLOWED_CATEGORIES)}).to_numpy()


def decode_target(codes: np.ndarray) -> list[str]:
    return [ALLOWED_CATEGORIES[int(code)] for code in codes]


def split_by_customer(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out 20 percent of customers, so no customer appears in both train and test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(dataset, groups=dataset[KEY]))
    return dataset.iloc[train_idx], dataset.iloc[test_idx]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("flags", "passthrough", BOOLEAN_FEATURES),
    ])


def build_candidates() -> dict[str, tuple[Pipeline, dict]]:
    """Each model wrapped with the same preprocessing, plus a small hyperparameter grid."""
    models = {
        # "prior" predicts the most frequent class like "most_frequent", but returns class
        # frequencies as probabilities, so its log loss is finite and comparable.
        BASELINE: (DummyClassifier(strategy="prior"), {}),
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
            {"model__C": [0.01, 0.1, 1.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
            {"model__max_depth": [3, 6, None], "model__min_samples_leaf": [1, 5]},
        ),
        "XGBoost": (
            XGBClassifier(
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=RANDOM_SEED,
                n_jobs=1,
            ),
            {"model__max_depth": [2, 4], "model__n_estimators": [50, 150]},
        ),
    }
    return {
        name: (Pipeline([("preprocess", build_preprocessor()), ("model", model)]), grid)
        for name, (model, grid) in models.items()
    }


def tune_models(train_df: pd.DataFrame) -> dict[str, GridSearchCV]:
    """Grid search every candidate with GroupKFold so folds never share a customer."""
    searches = {}
    for name, (pipeline, grid) in build_candidates().items():
        search = GridSearchCV(pipeline, grid, scoring="f1_macro", cv=GroupKFold(n_splits=CV_FOLDS))
        search.fit(train_df[FEATURES], encode_target(train_df[TARGET]), groups=train_df[KEY])
        searches[name] = search
    return searches


def evaluate(model: Pipeline, df: pd.DataFrame) -> dict[str, float]:
    y_true = encode_target(df[TARGET])
    y_pred = model.predict(df[FEATURES])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, model.predict_proba(df[FEATURES]), labels=range(len(ALLOWED_CATEGORIES))),
    }


def compare_models(searches: dict[str, GridSearchCV], test_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-validated macro F1 on the training customers next to test metrics, best first."""
    rows = {
        name: {
            "cv_macro_f1": search.best_score_,
            "cv_macro_f1_std": search.cv_results_["std_test_score"][search.best_index_],
            **evaluate(search.best_estimator_, test_df),
        }
        for name, search in searches.items()
    }
    return pd.DataFrame(rows).T.sort_values(f"cv_{SELECTION_METRIC}", ascending=False)


def select_best(comparison: pd.DataFrame) -> str:
    """Highest cross-validated macro F1, chosen on training folds so the small test set stays unbiased."""
    return comparison.drop(index=BASELINE)[f"cv_{SELECTION_METRIC}"].idxmax()


def fit_final_model(search: GridSearchCV, dataset: pd.DataFrame) -> Pipeline:
    """Refit the chosen configuration on every customer before saving it for the app."""
    return clone(search.best_estimator_).fit(dataset[FEATURES], encode_target(dataset[TARGET]))


def save_model(model: Pipeline) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    _cached_model.cache_clear()


def train() -> None:
    """Rebuild the merged dataset from data/raw, train, evaluate, save the model."""
    dataset = build_dataset()
    save_dataset(dataset)
    train_df, test_df = split_by_customer(dataset)
    searches = tune_models(train_df)
    comparison = compare_models(searches, test_df)
    best_name = select_best(comparison)
    save_model(fit_final_model(searches[best_name], dataset))

    print(f"Train: {len(train_df)} rows, {train_df[KEY].nunique()} customers. Test: {len(test_df)} rows, {test_df[KEY].nunique()} customers.")
    print(comparison.round(3).to_string())
    print(f"Best model by cross-validated macro F1: {best_name}, params {searches[best_name].best_params_}")
    print(f"Saved to {MODEL_PATH.relative_to(ROOT).as_posix()}")


@lru_cache(maxsize=1)
def _cached_model() -> Pipeline:
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _cached_dataset() -> pd.DataFrame:
    return load_dataset()


def predict_product(user_id: int) -> str:
    """Return the predicted product category for a customer id."""
    try:
        customer_id = int(str(user_id).strip().upper().removeprefix("A"))
    except ValueError:
        raise ValueError(f"Invalid customer id: {user_id!r}") from None

    dataset = _cached_dataset()
    history = dataset[dataset[KEY] == customer_id]
    if history.empty:
        raise ValueError(f"Customer {user_id} is not in {MERGED_PATH.name}")
    latest = history.sort_values(["purchase_date", "transaction_id"]).tail(1)
    return decode_target(_cached_model().predict(latest[FEATURES]))[0]


if __name__ == "__main__":
    train()
