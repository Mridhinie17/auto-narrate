import os
import io
import textwrap
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
REPORT_BG      = (248, 250, 252)
ACCENT         = (37, 99, 235)       # blue
TEXT_DARK      = (30, 41, 59)
TEXT_MID       = (71, 85, 105)
TEXT_LIGHT     = (148, 163, 184)
WHITE          = (255, 255, 255)
SUCCESS        = (16, 185, 129)
WARNING        = (245, 158, 11)
DANGER         = (239, 68, 68)

PAGE_W         = 900
PADDING        = 48
FONT_PATH      = None   # uses PIL default; swap for TTF path if available


def _font(size=14, bold=False):
    # Try a few common font names/paths
    font_names = []
    if bold:
        font_names = [
            "Arial Bold", "Arial-Bold", "Arial-BoldMT", "Helvetica-Bold", 
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "Arial", "Helvetica"
        ]
    else:
        font_names = [
            "Arial", "Helvetica", "ArialMT",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        
    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
            
    return ImageFont.load_default()


# ─────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────
def _draw_rect(draw, x0, y0, x1, y1, fill, radius=8):
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill)


def _wrap_text(text, width=85):
    return "\n".join(textwrap.wrap(text, width=width))


def _paste_image(canvas, img: Image.Image, x, y, max_w=None):
    if max_w and img.width > max_w:
        ratio = max_w / img.width
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    canvas.paste(img, (x, y))
    return img.height


# ─────────────────────────────────────────
# SECTION BUILDERS
# ─────────────────────────────────────────
def _header_block(draw, y, target_col, task_type, timestamp):
    # Blue bar
    draw.rectangle([0, y, PAGE_W, y + 80], fill=ACCENT)
    draw.text((PADDING, y + 14), "AutoNarrate", font=_font(26, bold=True), fill=WHITE)
    draw.text((PADDING, y + 46), "ML Insight Report Generator", font=_font(12), fill=(186, 210, 255))
    draw.text((PAGE_W - PADDING - 200, y + 20), f"Target: {target_col}", font=_font(11, bold=True), fill=WHITE)
    draw.text((PAGE_W - PADDING - 200, y + 38), f"Task: {task_type.upper()}", font=_font(11), fill=(186, 210, 255))
    draw.text((PAGE_W - PADDING - 200, y + 56), timestamp, font=_font(10), fill=(186, 210, 255))
    return y + 80


def _section_title(draw, y, title):
    draw.text((PADDING, y), title, font=_font(14, bold=True), fill=ACCENT)
    draw.line([(PADDING, y + 20), (PAGE_W - PADDING, y + 20)], fill=(*ACCENT, 80), width=1)
    return y + 30


def _metric_card(draw, canvas, x, y, label, value, color=None):
    color = color or ACCENT
    W, H = 180, 72
    _draw_rect(draw, x, y, x + W, y + H, fill=WHITE)
    draw.rectangle([x, y, x + 4, y + H], fill=color)
    draw.text((x + 14, y + 12), str(value), font=_font(20, bold=True), fill=color)
    draw.text((x + 14, y + 44), label, font=_font(10), fill=TEXT_MID)
    return W


def _narrative_block(draw, y, text, width=85):
    wrapped = _wrap_text(text, width)
    lines = wrapped.split("\n")
    for line in lines:
        draw.text((PADDING, y), line, font=_font(12), fill=TEXT_DARK)
        y += 20
    return y


# ─────────────────────────────────────────
# MASTER REPORT BUILDER
# ─────────────────────────────────────────
def build_report(results: dict, charts: dict, narrative: str) -> Image.Image:
    eda      = results["eda"]
    metrics  = results["metrics"]
    task     = results["task_type"]
    target   = results["target_col"]
    timestamp = datetime.now().strftime("%d %b %Y  %H:%M")

    # Estimate canvas height
    chart_heights = sum(c.height for c in charts.values()) + len(charts) * 40
    canvas_h = 1800 + chart_heights
    canvas = Image.new("RGB", (PAGE_W, canvas_h), REPORT_BG)
    draw = ImageDraw.Draw(canvas)

    y = 0

    # ── HEADER ──────────────────────────────
    y = _header_block(draw, y, target, task, timestamp)
    y += 28

    # ── DATASET SUMMARY ─────────────────────
    y = _section_title(draw, y, "Dataset Overview")
    stats = [
        ("Rows",      eda["rows"],       ACCENT),
        ("Columns",   eda["columns"],    SUCCESS),
        ("Duplicates",eda["duplicate_rows"], WARNING if eda["duplicate_rows"] > 0 else SUCCESS),
    ]
    cx = PADDING
    for label, val, color in stats:
        _metric_card(draw, canvas, cx, y, label, val, color)
        cx += 200
    y += 90

    # ── MODEL PERFORMANCE ───────────────────
    y = _section_title(draw, y, "Model Performance")
    if task == "classification":
        perf_stats = [
            ("Accuracy",  f"{metrics['accuracy']}%",  ACCENT),
            ("F1 Score",  f"{metrics['f1_score']}%",  SUCCESS),
        ]
    else:
        perf_stats = [
            ("R² Score",  str(metrics["r2"]),  ACCENT),
            ("RMSE",      str(metrics["rmse"]), WARNING),
            ("MAE",       str(metrics["mae"]),  SUCCESS),
        ]
    cx = PADDING
    for label, val, color in perf_stats:
        _metric_card(draw, canvas, cx, y, label, val, color)
        cx += 200
    y += 90

    # ── TOP FEATURES ────────────────────────
    y = _section_title(draw, y, "Top Predictive Features")
    top5 = list(metrics["feature_importance"].items())[:5]
    for feat, imp in top5:
        bar_w = int((imp / 100) * (PAGE_W - PADDING * 2 - 120))
        draw.text((PADDING, y + 3), feat, font=_font(11), fill=TEXT_DARK)
        _draw_rect(draw, PADDING + 140, y, PADDING + 140 + bar_w, y + 18,
                   fill=(*ACCENT, 200), radius=4)
        draw.text((PADDING + 140 + bar_w + 8, y + 2), f"{imp}%",
                  font=_font(10, bold=True), fill=ACCENT)
        y += 28
    y += 16

    # ── NARRATIVE ───────────────────────────
    y = _section_title(draw, y, "AI-Generated Narrative Report")
    _draw_rect(draw, PADDING - 8, y - 8, PAGE_W - PADDING + 8, y + len(narrative.split("\n")) * 22 + 200,
               fill=WHITE)
    y = _narrative_block(draw, y, narrative, width=90)
    y += 32

    # ── CHARTS ──────────────────────────────
    y = _section_title(draw, y, "Visualizations")
    chart_labels = {
        "feature_importance": "Feature Importance",
        "null_heatmap":       "Missing Values",
        "confusion_matrix":   "Confusion Matrix",
        "actual_vs_pred":     "Actual vs Predicted",
        "residuals":          "Residual Plot",
    }
    for key, chart_img in charts.items():
        label = chart_labels.get(key, key)
        draw.text((PADDING, y), label, font=_font(12, bold=True), fill=TEXT_MID)
        y += 20
        h = _paste_image(canvas, chart_img, PADDING, y, max_w=PAGE_W - PADDING * 2)
        y += h + 30

    # ── FOOTER ──────────────────────────────
    draw.rectangle([0, y + 10, PAGE_W, y + 48], fill=ACCENT)
    draw.text((PADDING, y + 18), f"Generated by AutoNarrate  •  {timestamp}",
              font=_font(10), fill=WHITE)

    # Crop to actual content
    canvas = canvas.crop((0, 0, PAGE_W, y + 58))
    return canvas


# ─────────────────────────────────────────
# SAVE REPORT
# ─────────────────────────────────────────
def save_report(report_img: Image.Image, output_path: str = "autonarrate_report.png") -> str:
    report_img.save(output_path, format="PNG", dpi=(150, 150))
    return output_path
