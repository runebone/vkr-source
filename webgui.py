import gradio as gr
import tempfile
import shutil
import os
from pathlib import Path
from main import main as generate_markup
from apply import annotate_pdf

MARKUP_TYPE_MAP = {
    "Сырая": 0,
    "Первичная": 1,
    "Уточненная": 2,
    "Объединенная": 3,
}

def process_file(pdf_file, markup_type_label):
    pdf_path = pdf_file.name
    stem = Path(pdf_path).stem
    json_path = f"{stem}.markup.json"
    annotated_path = f"{stem}.annotated.pdf"

    markup_type = MARKUP_TYPE_MAP[markup_type_label]

    # Генерация JSON
    generate_markup(str(pdf_path), str(json_path), markup_type)

    # Применение к PDF
    annotate_pdf(str(pdf_path), str(json_path), str(annotated_path))

    return str(json_path), str(annotated_path)

with gr.Blocks() as demo:
    gr.Markdown("# Разметка PDF-документа")

    with gr.Row():
        pdf_input = gr.File(label="Загрузите PDF-документ", file_types=[".pdf"])
        markup_type_input = gr.Dropdown(
            choices=list(MARKUP_TYPE_MAP.keys()),
            value="Сырая",
            label="Тип разметки"
        )

    with gr.Row():
        run_button = gr.Button("Разметить")

    with gr.Row():
        json_output = gr.File(label="JSON разметка", interactive=False)
        pdf_output = gr.File(label="Размеченный PDF", interactive=False)

    run_button.click(
        fn=process_file,
        inputs=[pdf_input, markup_type_input],
        outputs=[json_output, pdf_output]
    )

demo.launch()
