# pyrefly: ignore [missing-import]
import gradio as gr
import pandas as pd
import os
import tempfile

from ml_engine import run_pipeline, summarize_csv, summarize_column
from visualizer import generate_all_charts
from narrate import generate_narrative, generate_fallback_narrative
from report_builder import build_report, save_report


def load_columns(csv_file):
    empty = gr.Dropdown(choices=[]), gr.Dropdown(choices=[])
    if csv_file is None:
        return empty
    try:
        path = csv_file if isinstance(csv_file, str) else csv_file.name
        df = pd.read_csv(path, nrows=5)
        cols = df.columns.tolist()
        return (
            gr.Dropdown(choices=cols, value=cols[-1]),
            gr.Dropdown(choices=["(All Columns)"] + cols, value="(All Columns)")
        )
    except Exception:
        return empty


def run_summary(csv_file, col_choice):
    if csv_file is None:
        return "Please upload a CSV file first."
    try:
        path = csv_file if isinstance(csv_file, str) else csv_file.name
        df = pd.read_csv(path)
        if not col_choice or col_choice == "(All Columns)":
            return summarize_csv(df)
        return summarize_column(df, col_choice)
    except Exception as ex:
        return f"Error: {str(ex)}"


def run_autonarrate(csv_file, target_col, use_gemini):
    if csv_file is None:
        return None, None, None, None, None, "Please upload a CSV file."
    if not target_col:
        return None, None, None, None, None, "Please select a target column."
    try:
        path = csv_file if isinstance(csv_file, str) else csv_file.name
        results = run_pipeline(path, target_col.strip())
        charts = generate_all_charts(results)
        if use_gemini:
            narrative = generate_narrative(results)
        else:
            narrative = generate_fallback_narrative(results)
        report_img = build_report(results, charts, narrative)
        report_path = os.path.join(tempfile.gettempdir(), "autonarrate_report.png")
        save_report(report_img, report_path)
        metrics = results["metrics"]
        task = results["task_type"]
        eda = results["eda"]
        if task == "classification":
            metrics_text = (
                "**Task:** Classification\n\n"
                f"**Accuracy:** {metrics['accuracy']}%\n\n"
                f"**F1 Score:** {metrics['f1_score']}%\n\n"
                f"**Rows:** {eda['rows']}  |  **Cols:** {eda['columns']}\n\n"
                f"**Duplicates:** {eda['duplicate_rows']}"
            )
        else:
            metrics_text = (
                "**Task:** Regression\n\n"
                f"**R2:** {metrics['r2']}\n\n"
                f"**RMSE:** {metrics['rmse']}\n\n"
                f"**MAE:** {metrics['mae']}\n\n"
                f"**Rows:** {eda['rows']}  |  **Cols:** {eda['columns']}\n\n"
                f"**Duplicates:** {eda['duplicate_rows']}"
            )
        return (
            narrative,
            metrics_text,
            list(charts.values()),
            report_img,
            report_path,
            "Done! Report generated successfully."
        )
    except ValueError as ve:
        return None, None, None, None, None, f"Warning: {str(ve)}"
    except Exception as ex:
        import traceback
        return None, None, None, None, None, f"Error: {str(ex)}\n\n{traceback.format_exc()}"


CSS = """
body, .gradio-container {
    background: #0f172a;
}
.gradio-container {
    max-width: 1200px;
    margin: 0 auto;
}
.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 24px;
    text-align: center;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 8px 0;
}
.hero p {
    color: #93c5fd;
    font-size: 1rem;
    margin: 0;
}
.footer {
    text-align: center;
    color: #475569;
    font-size: 0.8rem;
    padding: 20px 0 8px 0;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
"""


with gr.Blocks(title="AutoNarrate", css=CSS) as demo:

    gr.HTML("""
        <div class="hero">
            <h1>AutoNarrate</h1>
            <p>Upload any CSV, explore your data, train a model, and get a plain-English AI insight report</p>
        </div>
    """)

    csv_input = gr.File(
        label="Upload CSV Dataset",
        file_types=[".csv"]
    )

    gr.HTML("<div style='height:16px'></div>")

    with gr.Tabs():

        with gr.Tab("Data Explorer"):
            gr.Markdown("Summarize your entire dataset or drill into any individual column.")
            with gr.Row():
                col_dropdown = gr.Dropdown(
                    choices=[],
                    label="Column to Inspect",
                    value="(All Columns)",
                    scale=3
                )
                explore_btn = gr.Button("Summarize", variant="secondary", scale=1)
            summary_out = gr.Textbox(
                label="Dataset Summary",
                lines=32,
                interactive=False,
                placeholder="Upload a CSV and click Summarize to explore your data..."
            )

        with gr.Tab("Run Model and Report"):
            gr.Markdown("Select a target column, run the pipeline, and get your AI report.")
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    gr.Markdown("### Configuration")
                    target_dropdown = gr.Dropdown(
                        choices=[],
                        label="Target Column",
                        interactive=True
                    )
                    use_gemini = gr.Checkbox(
                        label="Use Gemini AI narrative",
                        value=True
                    )
                    run_btn = gr.Button("Run AutoNarrate", variant="primary")
                    status_box = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=2
                    )
                    gr.Markdown("### Metrics")
                    metrics_out = gr.Markdown(value="Run the model to see metrics.")

                with gr.Column(scale=2):
                    gr.Markdown("### AI Narrative")
                    narrative_out = gr.Textbox(
                        label="Narrative Report",
                        lines=10,
                        interactive=False,
                        placeholder="Your plain-English report will appear here..."
                    )
                    gr.Markdown("### Charts")
                    charts_gallery = gr.Gallery(
                        label="Visualizations",
                        columns=2,
                        height=400,
                        object_fit="contain"
                    )
                    gr.Markdown("### Full Report")
                    report_preview = gr.Image(
                        label="Report Preview",
                        type="pil"
                    )
                    report_download = gr.File(label="Download Report PNG")

    gr.HTML("<div class='footer'>AutoNarrate &nbsp;•&nbsp; scikit-learn + Gemini AI + Gradio</div>")

    csv_input.change(
        fn=load_columns,
        inputs=[csv_input],
        outputs=[target_dropdown, col_dropdown]
    )
    explore_btn.click(
        fn=run_summary,
        inputs=[csv_input, col_dropdown],
        outputs=[summary_out]
    )
    run_btn.click(
        fn=run_autonarrate,
        inputs=[csv_input, target_dropdown, use_gemini],
        outputs=[narrative_out, metrics_out, charts_gallery, report_preview, report_download, status_box]
    )


if __name__ == "__main__":
    demo.launch() 
    