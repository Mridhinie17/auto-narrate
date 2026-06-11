import os
import json
from dotenv import load_dotenv

# Try to import genai, but don't fail if not installed
try:
    # pyrefly: ignore [missing-import]
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except (ImportError, OSError, NameError):
    genai = None
    GENAI_AVAILABLE = False

load_dotenv()


# ─────────────────────────────────────────
# CONFIGURE GEMINI
# ─────────────────────────────────────────
def _get_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Check your .env file.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


# ─────────────────────────────────────────
# BUILD PROMPT
# ─────────────────────────────────────────
def _build_prompt(results: dict) -> str:
    eda      = results["eda"]
    metrics  = results["metrics"]
    task     = results["task_type"]
    target   = results["target_col"]

    # Top 5 features
    top_features = list(metrics["feature_importance"].items())[:5]
    feat_str = "\n".join([f"  - {f}: {v}%" for f, v in top_features])

    # Null info
    high_null = {k: v for k, v in eda["null_percent"].items() if v > 5}
    null_str = (
        "\n".join([f"  - {k}: {v}% missing" for k, v in high_null.items()])
        if high_null else "  - No significant missing values found."
    )

    # Metrics block
    if task == "classification":
        metrics_str = f"""
  - Accuracy     : {metrics['accuracy']}%
  - F1 Score     : {metrics['f1_score']}%
  - Classes      : {metrics.get('classes', 'N/A')}
"""
    else:
        metrics_str = f"""
  - RMSE         : {metrics['rmse']}
  - MAE          : {metrics['mae']}
  - R² Score     : {metrics['r2']}
"""

    prompt = f"""
You are an expert data scientist writing a professional model evaluation report.
Write a clear, insightful, and fluent narrative report based on the following ML pipeline results.

---
DATASET OVERVIEW:
  - Rows         : {eda['rows']}
  - Columns      : {eda['columns']}
  - Target       : {target}
  - Task Type    : {task.upper()}
  - Duplicate Rows: {eda['duplicate_rows']}

MISSING VALUES (columns with > 5% missing):
{null_str}

MODEL PERFORMANCE:
{metrics_str}

TOP PREDICTIVE FEATURES:
{feat_str}
---

Write the report in the following structure. Use plain English — no markdown headers, no bullet points. Write in flowing paragraphs like a professional data science report.

Structure:
1. Dataset Summary — describe the dataset size, quality, and any data issues.
2. Task & Model — briefly mention what task was detected and what model was used.
3. Model Performance — explain the metrics in plain English. What does this accuracy/RMSE mean in practice?
4. Key Drivers — explain which features matter most and what that implies about the data.
5. Recommendations — suggest 2–3 actionable next steps to improve the model or data quality.

Keep the tone professional but readable. Avoid jargon overload. Write as if presenting to a non-technical stakeholder.
"""
    return prompt.strip()


# ─────────────────────────────────────────
# GENERATE NARRATIVE
# ─────────────────────────────────────────
def generate_narrative(results: dict) -> str:
    try:
        model = _get_model()
        prompt = _build_prompt(results)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[NLG Error] Could not generate narrative: {str(e)}"


# ─────────────────────────────────────────
# FALLBACK (no API key / offline mode)
# ─────────────────────────────────────────
def generate_fallback_narrative(results: dict) -> str:
    eda     = results["eda"]
    metrics = results["metrics"]
    task    = results["task_type"]
    target  = results["target_col"]
    top_feat = list(metrics["feature_importance"].items())[:3]

    if task == "classification":
        perf = f"The model achieved {metrics['accuracy']}% accuracy and a weighted F1 score of {metrics['f1_score']}%."
    else:
        perf = f"The model achieved an R² of {metrics['r2']}, RMSE of {metrics['rmse']}, and MAE of {metrics['mae']}."

    feat_text = ", ".join([f"{f} ({v}%)" for f, v in top_feat])

    return (
        f"The dataset contains {eda['rows']} rows and {eda['columns']} columns, "
        f"with '{target}' as the prediction target. "
        f"AutoNarrate detected this as a {task} task and trained a Random Forest model. "
        f"{perf} "
        f"The most influential features were {feat_text}. "
        f"Consider further feature engineering and hyperparameter tuning to improve results."
    )
