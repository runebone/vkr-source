import gradio as gr
import fitz  # PyMuPDF
import numpy as np
# import cv2
from PIL import Image
from tempfile import NamedTemporaryFile

# from your_analysis_module import analyze_image  # Эта функция возвращает изображение с разметкой

def pdf_to_image(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    if page_number < 1 or page_number > page_count:
        raise ValueError(f"Номер страницы должен быть от 1 до {page_count}")
    page = doc.load_page(page_number - 1)  # пока только первая страница
    matrix = fitz.Matrix(3, 3)  # увеличение в 3 раза по каждой оси
    dl = page.get_displaylist()
    pix = dl.get_pixmap(matrix=matrix)
    # dpi = 600
    # pix.set_dpi(dpi, dpi)
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
    with gr.Row():
        file_input = gr.File(label="PDF-документ", file_types=['.pdf'])
        output_image = gr.Image(type="pil", label="Результат разметки", height=800)
    with gr.Row():
        page_info = gr.Textbox(label="Информация о документе")
    with gr.Row():
        page_number = gr.Number(value=1, label="Номер страницы", precision=0)
    with gr.Row():
        scanline_output = gr.Image(type="pil", label="Сканирующая строка (2 пикселя)", height=300)

    state_img_np = gr.State()  # Храним текущую страницу в numpy

    def update(file, page):
        try:
            img, info = process_pdf(file, int(page))
            img_np = np.array(img)
            return img, info, img_np
            # return process_pdf(file, int(page))
        except Exception as e:
            return None, f"Ошибка: {e}"

    def extract_scanline(img_np, evt: gr.SelectData):
        x, y = evt.index
        y = int(y)
        if y + 2 > img_np.shape[0]:
            return None, None  # за границей

        # 1. Извлекаем строку
        scanline = img_np[y:y+2, :, :]

        # # 2. Копируем изображение
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

        # # 3. Рисуем горизонтальную полоску (красным)
        # color = [0, 255, 0]
        # highlighted_img[max(0, y-1)] = color
        # highlighted_img[y] = color
        # if y+1 < highlighted_img.shape[0]:
        #     highlighted_img[y+1] = color

        return Image.fromarray(scanline), Image.fromarray(highlighted_img)
        # x, y = evt.index
        # y = int(y)
        # if y + 2 > img_np.shape[0]:
        #     return None  # за границей изображения
        # scanline = img_np[y:y+2, :, :]
        # return Image.fromarray(scanline)

    # file_input.change(fn=update, inputs=[file_input, page_number], outputs=[output_image, page_info])
    # page_number.change(fn=update, inputs=[file_input, page_number], outputs=[output_image, page_info])

    file_input.change(fn=update, inputs=[file_input, page_number], outputs=[output_image, page_info, state_img_np])
    page_number.change(fn=update, inputs=[file_input, page_number], outputs=[output_image, page_info, state_img_np])
    # output_image.select(fn=extract_scanline, inputs=state_img_np, outputs=scanline_output)
    output_image.select(
        fn=extract_scanline,
        inputs=state_img_np,
        outputs=[scanline_output, output_image]  # теперь обновляем оба
    )




demo.launch()
