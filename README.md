---
title: autonarrate
app_file: app.py
sdk: gradio
sdk_version: 6.17.3
---
# AutoNarrate — ML Insight Report Generator

Upload any CSV dataset → AutoNarrate auto-detects the task type, trains a Random Forest model, generates visualizations, and produces a **plain-English narrative report** explaining what the model found — powered by Gemini AI.

---

## Features

- Auto-detects Classification vs Regression from your target column
- Trains a Random Forest model with preprocessing (encoding, null handling)
- Generates charts: feature importance, confusion matrix / actual vs predicted, residuals, missing value heatmap
- Gemini AI writes a professional narrative report in plain English
- Fallback narrative (no API key needed) for offline use
- Downloadable full-page report as PNG
- Clean Gradio UI — upload, click, done

---

## Demo Flow

```
Upload CSV  →  Select target column  →  Click Run
     ↓
AutoNarrate detects task type (classification / regression)
     ↓
Trains Random Forest  →  Evaluates  →  Generates charts
     ↓
Gemini writes plain-English narrative
     ↓
Full report: metrics + charts + narrative (downloadable PNG)
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/autonarrate.git
cd autonarrate
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key

```bash
cp .env.example .env
```

Edit `.env`:
```
GEMINI_API_KEY=your_key_here
```

Get a free key at: https://aistudio.google.com/apikey

### 5. Run

```bash
python app.py
```

Open http://localhost:7860 in your browser.

---

## Project Structure

```
autonarrate/
├── app.py              # Gradio UI — entry point
├── ml_engine.py        # Data loading, preprocessing, training, evaluation
├── visualizer.py       # Chart generation (matplotlib)
├── narrate.py          # Gemini API call → NLG narrative
├── report_builder.py   # Assembles full report image (Pillow)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Supported Task Types

| Task | Auto-detected when |
|---|---|
| Classification | Target has ≤ 10 unique values or is string/bool |
| Regression | Target is numeric with > 10 unique values |

---

## Tech Stack

| Component | Library |
|---|---|
| UI | Gradio 4.x |
| ML | scikit-learn (Random Forest) |
| NLG | Google Gemini 1.5 Flash |
| Visualizations | matplotlib |
| Report layout | Pillow (PIL) |
| Data | pandas, numpy |

---

## Notes

- Works without a Gemini API key — fallback narrative is generated locally
- First run may take a few seconds for model training depending on dataset size
- Large datasets (> 100k rows) will be slower — consider sampling

---

## License

MIT License
