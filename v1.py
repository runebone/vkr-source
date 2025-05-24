import gradio as gr
import fitz  # PyMuPDF
import numpy as np
# import cv2
from PIL import Image

def pdf_to_image(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    if page_number < 1 or page_number > page_count:
        raise ValueError(f"Номер страницы должен быть от 1 до {page_count}")
    page = doc.load_page(page_number - 1)  # пока только первая страница
    matrix = fitz.Matrix(3, 3)  # увеличение в 3 раза по каждой оси
    dl = page.get_displaylist()
    pix = dl.get_pixmap(matrix=matrix)
    img = Image.frombytes("RGB", (pix.w, pix.h), pix.samples)
    return img, page_count

def process_pdf(file, page_number):
    pdf_path = file.name
    img, total_pages = pdf_to_image(pdf_path, page_number)
    img_np = np.array(img)
    # analyzed_img = analyze_image(img_np)  # возвращает np.array с отрисованной разметкой
    return Image.fromarray(img_np), f"Страниц в документе: {total_pages}"

with gr.Blocks(title="PDF Segmenter") as demo:
    gr.Markdown("## Загрузите PDF-документ, введите номер страницы и получите автоматическую разметку")

    state_img_np = gr.State()  # Храним текущую страницу в numpy
    scanline_y = gr.State(0)  # текущая строка y

    with gr.Row():
        # scanline_output = gr.Image(type="pil", label="Сканирующая строка (2 пикселя)", height=20)
        scanline_output = gr.Image(type="pil")
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="PDF-документ", file_types=['.pdf'])
            page_info = gr.Textbox(label="Информация о документе")
            page_number = gr.Number(value=1, label="Номер страницы", precision=0)
            step = gr.Number(2, label="Шаг")
            btn_up = gr.Button("Вверх", elem_id="btn-up")
            btn_down = gr.Button("Вниз", elem_id="btn-down")
        with gr.Column():
            output_image = gr.Image(type="pil", label="Результат разметки")

    def update(file, page):
        try:
            img, info = process_pdf(file, int(page))
            img_np = np.array(img)
            return img, info, img_np
            # return process_pdf(file, int(page))
        except Exception as e:
            return None, f"Ошибка: {e}"

    def extract_scanline(img_np, evt: gr.SelectData):
        _, y = evt.index
        y = int(y)
        if y + 2 > img_np.shape[0]:
            return None, None  # за границей

        # 1. Извлекаем строку
        scanline = img_np[y:y+2, :, :]

        # 2. Копируем изображение
        highlighted_img = img_np.copy()

        # 3. Цвет и прозрачность
        overlay_color = np.array([0, 255, 0], dtype=np.float32)  # красный
        alpha = 0.5  # степень прозрачности (0 = прозрачный, 1 = сплошной)

        for dy in [0, 1]:  # захватываем 3 строки: выше, текущую, ниже
            yy = y + dy
            if 0 <= yy < highlighted_img.shape[0]:
                highlighted_img[yy] = (
                    alpha * overlay_color + (1 - alpha) * highlighted_img[yy]
                )

        # 4. Возвращаем uint8 изображение
        highlighted_img = np.clip(highlighted_img, 0, 255).astype(np.uint8)

        return Image.fromarray(scanline), Image.fromarray(highlighted_img), y
    
    def move_scanline(img_np, y, direction):
        y_new = y + direction
        if y_new < 0 or y_new + 2 > img_np.shape[0]:
            return None, None, y  # не выходим за границу

        # Извлекаем строку
        scanline = img_np[y_new:y_new+2, :, :]

        # Подсветка
        highlighted_img = img_np.astype(np.float32).copy()
        overlay_color = np.array([55, 200, 0], dtype=np.float32)
        alpha = 0.5
        for dy in [-1, 0, 1]:
            yy = y_new + dy
            if 0 <= yy < highlighted_img.shape[0]:
                highlighted_img[yy] = (
                    alpha * overlay_color + (1 - alpha) * highlighted_img[yy]
                )
        highlighted_img = np.clip(highlighted_img, 0, 255).astype(np.uint8)

        return Image.fromarray(scanline), Image.fromarray(highlighted_img), y_new

    def move_up_wrapper(img_np, scanline_y, step):
        return move_scanline(img_np, scanline_y, -step)

    def move_down_wrapper(img_np, scanline_y, step):
        return move_scanline(img_np, scanline_y, step)

    file_input.change(fn=update, inputs=[file_input, page_number], outputs=[output_image, page_info, state_img_np])
    page_number.change(fn=update, inputs=[file_input, page_number], outputs=[output_image, page_info, state_img_np])
    output_image.select(
        fn=extract_scanline,
        inputs=state_img_np,
        outputs=[scanline_output, output_image, scanline_y]
    )

    btn_up.click(fn=move_up_wrapper, inputs=[state_img_np, scanline_y, step],
                 outputs=[scanline_output, output_image, scanline_y])
    btn_down.click(fn=move_down_wrapper, inputs=[state_img_np, scanline_y, step],
                   outputs=[scanline_output, output_image, scanline_y])

demo.launch()
