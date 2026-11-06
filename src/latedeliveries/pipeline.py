from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Fixed seed for repeatable experiments
RANDOM_STATE = 42

# Features available when an order is placed
NUMERIC_FEATURES = [
    "distance_km",
    "items",
    "hour",
]

CATEGORICAL_FEATURES = [
    "weather",
    "day_type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_orders(df):
    """Split orders into stratified training and test sets."""
    X = df[FEATURES].copy()
    y = df["late"].copy()

    return train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_pipeline(model):
    """Create a leak-free preprocessing and model pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
