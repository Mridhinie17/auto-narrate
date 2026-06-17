import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# 1. LOAD CSV
# ─────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)
    return df


# ─────────────────────────────────────────
# 2. BASIC EDA SUMMARY
# ─────────────────────────────────────────
def get_eda_summary(df: pd.DataFrame) -> dict:
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "null_percent": (df.isnull().mean() * 100).round(2).to_dict(),
        "numeric_summary": df.describe().round(3).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return summary


# ─────────────────────────────────────────
# 3. FULL CSV SUMMARIZER
# ─────────────────────────────────────────
def summarize_csv(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"DATASET OVERVIEW")
    lines.append(f"{'='*50}")
    lines.append(f"Rows       : {df.shape[0]}")
    lines.append(f"Columns    : {df.shape[1]}")
    lines.append(f"Duplicates : {df.duplicated().sum()}")
    lines.append(f"Total Nulls: {df.isnull().sum().sum()}")
    lines.append("")

    lines.append(f"COLUMN-WISE SUMMARY")
    lines.append(f"{'='*50}")

    for col in df.columns:
        series = df[col]
        dtype  = series.dtype
        nulls  = series.isnull().sum()
        null_p = round(nulls / len(df) * 100, 1)
        unique = series.nunique()

        lines.append(f"\n📌 {col}")
        lines.append(f"   Type    : {dtype}")
        lines.append(f"   Nulls   : {nulls} ({null_p}%)")
        lines.append(f"   Unique  : {unique}")

        if pd.api.types.is_numeric_dtype(series):
            lines.append(f"   Min     : {series.min()}")
            lines.append(f"   Max     : {series.max()}")
            lines.append(f"   Mean    : {round(series.mean(), 3)}")
            lines.append(f"   Std Dev : {round(series.std(), 3)}")
            lines.append(f"   Median  : {series.median()}")
        else:
            top_vals = series.value_counts().head(5)
            lines.append(f"   Top Values:")
            for val, cnt in top_vals.items():
                lines.append(f"      '{val}' → {cnt} times")

    return "\n".join(lines)


def summarize_column(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return f"Column '{col}' not found in dataset."

    series = df[col]
    lines  = []
    lines.append(f"COLUMN: {col}")
    lines.append(f"{'='*40}")
    lines.append(f"Data Type : {series.dtype}")
    lines.append(f"Total Rows: {len(series)}")
    lines.append(f"Nulls     : {series.isnull().sum()} ({round(series.isnull().mean()*100,1)}%)")
    lines.append(f"Unique    : {series.nunique()}")

    if pd.api.types.is_numeric_dtype(series):
        lines.append(f"\nStatistics:")
        lines.append(f"  Min        : {series.min()}")
        lines.append(f"  Max        : {series.max()}")
        lines.append(f"  Mean       : {round(series.mean(), 4)}")
        lines.append(f"  Median     : {series.median()}")
        lines.append(f"  Std Dev    : {round(series.std(), 4)}")
        lines.append(f"  25th pct   : {series.quantile(0.25)}")
        lines.append(f"  75th pct   : {series.quantile(0.75)}")
        lines.append(f"  Skewness   : {round(series.skew(), 4)}")
        lines.append(f"  Kurtosis   : {round(series.kurtosis(), 4)}")
        outliers = series[(series < series.quantile(0.01)) | (series > series.quantile(0.99))]
        lines.append(f"  Outliers (1-99%): {len(outliers)}")
    else:
        lines.append(f"\nTop 10 Values:")
        for val, cnt in series.value_counts().head(10).items():
            pct = round(cnt / len(series) * 100, 1)
            lines.append(f"  '{val}' → {cnt} ({pct}%)")

    return "\n".join(lines)


# ─────────────────────────────────────────
# 4. VALIDATE TARGET COLUMN
# ─────────────────────────────────────────
def validate_target(df: pd.DataFrame, target_col: str) -> str:
    """Returns None if valid, error message string if invalid."""
    if target_col not in df.columns:
        return f"Column '{target_col}' not found. Available: {df.columns.tolist()}"

    series = df[target_col].dropna()

    if series.nunique() < 2:
        return f"Column '{target_col}' has only {series.nunique()} unique values — cannot train on it."

    # If it is a string/object/category column, check cardinality
    if series.dtype == "object" or isinstance(series.dtype, pd.CategoricalDtype):
        pass
    else:
        # For other types, check if they can be converted to numeric
        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.isnull().mean() > 0.5:
            return (
                f"Column '{target_col}' cannot be used as a target — "
                f"more than 50% of values are non-numeric strings. "
                f"Please choose a numeric or categorical column."
            )

    return None


# ─────────────────────────────────────────
# 5. AUTO-DETECT TASK TYPE
# ─────────────────────────────────────────
def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    col = pd.to_numeric(df[target_col], errors="coerce")
    n_unique = col.nunique()
    if df[target_col].dtype == "object" or df[target_col].dtype == "bool":
        return "classification"
    if n_unique <= 10:
        return "classification"
    return "regression"


# ─────────────────────────────────────────
# 6. PREPROCESS
# ─────────────────────────────────────────
def _is_droppable_column(col_series: pd.Series) -> bool:
    s = col_series.dropna().astype(str)
    if len(s) == 0:
        return True

    sample  = s.iloc[0].strip()
    n_unique = col_series.nunique()
    n_total  = len(col_series)
    avg_len  = s.str.len().mean()

    if sample.startswith("[") or sample.startswith("{"):
        return True
    if sample.startswith("http") or sample.startswith("www") or sample.startswith("/hotel"):
        return True
    if len(sample) in (24, 32, 36) and all(c in "0123456789abcdefABCDEF-" for c in sample.replace("-", "")):
        return True
    if n_unique / max(n_total, 1) > 0.9:
        return True
    if avg_len > 60:
        return True
    if n_unique <= 1:
        return True
    try:
        pd.to_datetime(s.iloc[0])
        return True
    except Exception:
        pass
    return False


def preprocess(df: pd.DataFrame, target_col: str):
    df = df.copy()

    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])

    # We no longer drop columns with > 50% nulls per user request.

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Drop unnamed index columns
    unnamed = [c for c in X.columns if str(c).lower().startswith("unnamed")]
    X = X.drop(columns=unnamed, errors="ignore")

    # We no longer drop untrainable object columns per user request.

    # Encode remaining low-cardinality object columns
    le_map = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_map[col] = le

    # Force everything to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Fill nulls
    X = X.fillna(X.median(numeric_only=True))
    X = X.dropna(axis=1, how="all")
    X = X.fillna(0)

    # Handle target
    target_le = None
    y_numeric = pd.to_numeric(y, errors="coerce")

    if y.dtype == "object" or y.dtype == "bool":
        target_le = LabelEncoder()
        y = pd.Series(target_le.fit_transform(y.astype(str)))
    elif y_numeric.isnull().sum() == 0:
        y = y_numeric
    else:
        # Mixed target — encode as classification
        target_le = LabelEncoder()
        y = pd.Series(target_le.fit_transform(y.astype(str)))

    return X, y, le_map, target_le


# ─────────────────────────────────────────
# 7. TRAIN MODEL
# ─────────────────────────────────────────
def train_model(X, y, task_type: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=30, max_depth=10, n_jobs=-1, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=30, max_depth=10, n_jobs=-1, random_state=42)

    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────
# 8. EVALUATE MODEL
# ─────────────────────────────────────────
def evaluate_model(model, X_test, y_test, task_type: str, feature_names: list, target_le=None) -> dict:
    y_pred = model.predict(X_test)
    importances = model.feature_importances_
    feat_imp = dict(
        sorted(
            zip(feature_names, (importances * 100).round(2)),
            key=lambda x: x[1], reverse=True
        )
    )
    metrics = {"feature_importance": feat_imp, "task_type": task_type}

    if task_type == "classification":
        metrics["accuracy"] = round(accuracy_score(y_test, y_pred) * 100, 2)
        metrics["f1_score"] = round(f1_score(y_test, y_pred, average="weighted") * 100, 2)
        metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
        metrics["y_test"] = y_test.tolist() if hasattr(y_test, "tolist") else list(y_test)
        metrics["y_pred"] = y_pred.tolist()
        if target_le is not None:
            metrics["classes"] = [str(c) for c in target_le.inverse_transform(model.classes_)]
        else:
            metrics["classes"] = [str(c) for c in model.classes_]
    else:
        metrics["rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        metrics["mae"]  = round(mean_absolute_error(y_test, y_pred), 4)
        metrics["r2"]   = round(r2_score(y_test, y_pred), 4)
        metrics["y_test"] = y_test.tolist() if hasattr(y_test, "tolist") else list(y_test)
        metrics["y_pred"] = y_pred.tolist()

    return metrics


# ─────────────────────────────────────────
# 9. MASTER PIPELINE
# ─────────────────────────────────────────
def run_pipeline(filepath: str, target_col: str) -> dict:
    df = load_data(filepath)

    # Validate target first
    err = validate_target(df, target_col)
    if err:
        raise ValueError(err)

    eda       = get_eda_summary(df)
    task_type = detect_task_type(df, target_col)
    X, y, le_map, target_le = preprocess(df, target_col)
    model, X_train, X_test, y_train, y_test = train_model(X, y, task_type)
    metrics = evaluate_model(model, X_test, y_test, task_type, X.columns.tolist(), target_le)

    return {
        "eda": eda,
        "task_type": task_type,
        "metrics": metrics,
        "model": model,
        "target_col": target_col,
        "feature_names": X.columns.tolist(),
        "X_test": X_test,
        "y_test": y_test,
    }
    