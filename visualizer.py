# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Gradio
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import matplotlib.patches as mpatches
import numpy as np
import io
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ─────────────────────────────────────────
# SHARED STYLE
# ─────────────────────────────────────────
PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]
BG = "#F8FAFC"
GRID_COLOR = "#E2E8F0"

def _base_fig(figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax

def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor=BG)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


# ─────────────────────────────────────────
# 1. FEATURE IMPORTANCE BAR CHART
# ─────────────────────────────────────────
def plot_feature_importance(feature_importance: dict, top_n: int = 10) -> Image.Image:
    items = list(feature_importance.items())[:top_n]
    features = [i[0] for i in items][::-1]
    values   = [i[1] for i in items][::-1]

    fig, ax = _base_fig(figsize=(8, max(4, len(features) * 0.5)))
    bars = ax.barh(features, values, color=PALETTE[0], edgecolor="white",
                   linewidth=0.5, zorder=3, height=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=9,
                color="#374151", fontweight="bold")

    ax.set_xlabel("Importance (%)", fontsize=10, color="#374151")
    ax.set_title("Feature Importance", fontsize=13, fontweight="bold",
                 color="#1E293B", pad=12)
    ax.tick_params(colors="#374151", labelsize=9)
    ax.set_xlim(0, max(values) * 1.18)
    fig.tight_layout()
    return _fig_to_pil(fig)


# ─────────────────────────────────────────
# 2. CONFUSION MATRIX (classification)
# ─────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, classes=None) -> Image.Image:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    ax.set_facecolor(BG)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")

    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold",
                 color="#1E293B", pad=12)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.tight_layout()
    return _fig_to_pil(fig)


# ─────────────────────────────────────────
# 3. ACTUAL vs PREDICTED (regression)
# ─────────────────────────────────────────
def plot_actual_vs_predicted(y_test, y_pred) -> Image.Image:
    fig, ax = _base_fig(figsize=(7, 5))
    ax.scatter(y_test, y_pred, color=PALETTE[0], alpha=0.55,
               edgecolors="white", linewidths=0.4, s=40, zorder=3)

    mn = min(min(y_test), min(y_pred))
    mx = max(max(y_test), max(y_pred))
    ax.plot([mn, mx], [mn, mx], color=PALETTE[3], linewidth=1.5,
            linestyle="--", label="Perfect Fit", zorder=4)

    ax.set_xlabel("Actual Values", fontsize=10, color="#374151")
    ax.set_ylabel("Predicted Values", fontsize=10, color="#374151")
    ax.set_title("Actual vs Predicted", fontsize=13, fontweight="bold",
                 color="#1E293B", pad=12)
    ax.legend(fontsize=9)
    ax.tick_params(colors="#374151", labelsize=9)
    fig.tight_layout()
    return _fig_to_pil(fig)


# ─────────────────────────────────────────
# 4. RESIDUAL PLOT (regression)
# ─────────────────────────────────────────
def plot_residuals(y_test, y_pred) -> Image.Image:
    residuals = np.array(y_test) - np.array(y_pred)
    fig, ax = _base_fig(figsize=(7, 4))
    ax.scatter(y_pred, residuals, color=PALETTE[1], alpha=0.55,
               edgecolors="white", linewidths=0.4, s=40, zorder=3)
    ax.axhline(0, color=PALETTE[3], linewidth=1.5, linestyle="--", zorder=4)
    ax.set_xlabel("Predicted Values", fontsize=10, color="#374151")
    ax.set_ylabel("Residuals", fontsize=10, color="#374151")
    ax.set_title("Residual Plot", fontsize=13, fontweight="bold",
                 color="#1E293B", pad=12)
    ax.tick_params(colors="#374151", labelsize=9)
    fig.tight_layout()
    return _fig_to_pil(fig)


# ─────────────────────────────────────────
# 5. NULL HEATMAP
# ─────────────────────────────────────────
def plot_null_heatmap(null_percent: dict) -> Image.Image:
    cols = list(null_percent.keys())
    vals = list(null_percent.values())

    fig, ax = _base_fig(figsize=(8, max(3, len(cols) * 0.45)))
    colors = [PALETTE[3] if v > 20 else PALETTE[2] if v > 5 else PALETTE[1] for v in vals]
    bars = ax.barh(cols[::-1], vals[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.5, zorder=3, height=0.6)

    for bar, val in zip(bars, vals[::-1]):
        if val > 0:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8.5, color="#374151")

    ax.set_xlabel("Missing (%)", fontsize=10, color="#374151")
    ax.set_title("Missing Values per Column", fontsize=13, fontweight="bold",
                 color="#1E293B", pad=12)
    ax.set_xlim(0, max(vals + [1]) * 1.2)
    ax.tick_params(colors="#374151", labelsize=8.5)

    patches = [
        mpatches.Patch(color=PALETTE[1], label="< 5% (OK)"),
        mpatches.Patch(color=PALETTE[2], label="5–20% (Moderate)"),
        mpatches.Patch(color=PALETTE[3], label="> 20% (High)"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    fig.tight_layout()
    return _fig_to_pil(fig)


# ─────────────────────────────────────────
# 6. MASTER: GENERATE ALL CHARTS
# ─────────────────────────────────────────
def generate_all_charts(results: dict) -> dict:
    metrics     = results["metrics"]
    task_type   = results["task_type"]
    eda         = results["eda"]
    y_test      = metrics["y_test"]
    y_pred      = metrics["y_pred"]

    charts = {}
    charts["feature_importance"] = plot_feature_importance(metrics["feature_importance"])
    charts["null_heatmap"]       = plot_null_heatmap(eda["null_percent"])

    if task_type == "classification":
        charts["confusion_matrix"] = plot_confusion_matrix(
            y_test, y_pred, classes=metrics.get("classes")
        )
    else:
        charts["actual_vs_pred"] = plot_actual_vs_predicted(y_test, y_pred)
        charts["residuals"]      = plot_residuals(y_test, y_pred)

    return charts
