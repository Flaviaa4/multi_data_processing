"""Tabular pipeline: clean and merge customer data, then train the product model."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOCIAL_PATH = ROOT / "data" / "raw" / "customer_social_profiles.csv"
TRANSACTIONS_PATH = ROOT / "data" / "raw" / "customer_transactions.csv"
MERGED_PATH = ROOT / "data" / "processed" / "merged_dataset.csv"
MODEL_PATH = ROOT / "models" / "product_model.pkl"

RANDOM_SEED = 42
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
