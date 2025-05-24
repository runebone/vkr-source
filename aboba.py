import logging
from typing import Tuple, Optional, List

import gradio as gr
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

# Настройки
SCALE_FACTOR = 3
SCANLINE_HEIGHT = 2
DEFAULT_STEP = 2
OVERLAY_COLOR_SELECT = np.array([0, 255, 0], dtype=np.float32)
OVERLAY_COLOR_MOVE = np.array([0, 255, 0], dtype=np.float32)
OVERLAY_ALPHA = 0.5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pdf_to_image(pdf_path: str, page_number: int) -> Tuple[Image.Image, int]:
    """
    Конвертирует указанную страницу PDF в изображение.

    :param pdf_path: путь до PDF-файла
    :param page_number: номер страницы (1-based)
    :return: кортеж (PIL.Image, общее число страниц)
    :raises ValueError: если номер страницы вне диапазона
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    if not (1 <= page_number <= total_pages):
        raise ValueError(f"Номер страницы должен быть от 1 до {total_pages}")

    page = doc.load_page(page_number - 1)
    matrix = fitz.Matrix(SCALE_FACTOR, SCALE_FACTOR)
    pix = page.get_displaylist().get_pixmap(matrix=matrix)
    image = Image.frombytes("RGB", (pix.w, pix.h), pix.samples)
    return image, total_pages


def process_pdf(file_obj, page_number: int) -> Tuple[Image.Image, str]:
    """
    Обрабатывает PDF-файл: конвертирует страницу в изображение и возвращает
    PIL-изображение и инфо по числу страниц.
    """
    img, total = pdf_to_image(file_obj.name, page_number)
    info = f"Страниц в документе: {total}"
    return img, info


def highlight_lines(
    img_np: np.ndarray,
    y: int,
    overlay_color: np.ndarray,
    height: int = SCANLINE_HEIGHT
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Выделяет scanline и возвращает саму строку и изображение с наложением.

    :param img_np: исходное изображение в виде массива
    :param y: координата начала scanline
    :param overlay_color: цвет заливки (float32)
    :param height: высота scanline
    :return: кортеж (scanline_np, highlighted_image_np)
    """
    h, w, _ = img_np.shape
    if y < 0 or y + height > h:
        raise IndexError("Scanline выходит за границы изображения")

    # вырезаем строку
    scanline_np = img_np[y : y + height].copy()

    # готовим изображение с наложением
    highlighted = img_np.astype(np.float32).copy()
    for dy in range(-1, height):
        yy = y + dy
        if 0 <= yy < h:
            highlighted[yy] = (
                OVERLAY_ALPHA * overlay_color + (1 - OVERLAY_ALPHA) * highlighted[yy]
            )
    highlighted = np.clip(highlighted, 0, 255).astype(np.uint8)
    return scanline_np, highlighted


def extract_scanline(
    img_np: np.ndarray, evt: gr.SelectData
) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[int]]:
    """
    Callback для клика по изображению: извлекает scanline по координате y.
    """
    _, y = evt.index
    y = int(y)
    try:
        scanline_np, highlighted_np = highlight_lines(
            img_np, y, OVERLAY_COLOR_SELECT
        )
    except IndexError:
        return None, None, None

    return (
        Image.fromarray(scanline_np),
        Image.fromarray(highlighted_np),
        y,
    )


def move_scanline(
    img_np: np.ndarray, current_y: int, step: int
) -> Tuple[Optional[Image.Image], Optional[Image.Image], int]:
    """
    Смещает scanline на step пикселей вверх или вниз и подсвечивает.
    """
    new_y = current_y + step
    try:
        scanline_np, highlighted_np = highlight_lines(
            img_np, new_y, OVERLAY_COLOR_MOVE
        )
    except IndexError:
        # удерживаем предыдущую позицию, если вышли за границы
        return None, None, current_y

    return (
        Image.fromarray(scanline_np),
        Image.fromarray(highlighted_np),
        new_y,
    )


def move_up(
    img_np: np.ndarray,
    current_y: int,
    step: int
) -> Tuple[Optional[Image.Image], Optional[Image.Image], int]:
    """
    Смещает scanline вверх на заданное число пикселей.
    """
    return move_scanline(img_np, current_y, -step)


def compute_margins(
    pdf_path: str,
    start_page: int,
    end_page: int,
    scale: int = 2,
    white_threshold: int = 250
) -> List[Tuple[int, int]]:
    """
    Для каждой страницы с номера start_page по end_page (1-based) вычисляет
    ширину левого и правого полей в пикселях.

    :param pdf_path: путь к PDF-файлу
    :param start_page: начальная страница (1-based)
    :param end_page: конечная страница (1-based)
    :param scale: во сколько раз увеличивать рендер (по умолчанию 2)
    :param white_threshold: порог яркости (0–255), выше которого пиксель считается «белым»
    :return: список кортежей (left_margin_px, right_margin_px) по страницам
    :raises ValueError: если указанные номера страниц вне диапазона
    """
    doc = fitz.open(pdf_path)
    total = doc.page_count
    if not (1 <= start_page <= total) or not (1 <= end_page <= total):
        raise ValueError(f"Номер страницы должен быть от 1 до {total}")
    if start_page > end_page:
        raise ValueError("start_page не может быть больше end_page")

    margins: List[Tuple[int, int]] = []

    for pno in range(start_page - 1, end_page):
        page = doc.load_page(pno)
        mat = fitz.Matrix(scale, scale)
        pix = page.get_displaylist().get_pixmap(matrix=mat, alpha=False)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8)
        img_np = img_np.reshape(pix.h, pix.w, pix.n)[:, :, :3]  # RGB

        # перевести в градации серого
        gray = np.dot(img_np[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        # бинаризация: True = пиксель не «белый»
        content = gray < white_threshold

        # найти колонки с любыми «контентными» пикселями
        cols_any = content.any(axis=0)
        if not cols_any.any():
            # полностью белая страница
            left = right = pix.w
        else:
            first = np.argmax(cols_any)
            last = pix.w - 1 - np.argmax(cols_any[::-1])
            left = int(first)
            right = pix.w - 1 - last

        margins.append((left, right))

    return margins


def gradio_interface() -> None:
    with gr.Blocks(title="PDF Segmenter") as demo:
        gr.Markdown(
            "## Загрузите PDF-документ, введите номер страницы и получите автоматическую разметку"
        )

        state_img_np = gr.State()
        scanline_y = gr.State(0)

        output_image = gr.Image(type="pil", label="Результат разметки")
        scanline_output = gr.Image(type="pil")
        page_info = gr.Textbox(label="Информация о документе")
        margins_info = gr.Textbox(label="Поля (левое и правое) по страницам")

        file_input = gr.File(label="PDF-документ", file_types=[".pdf"])
        page_number = gr.Number(value=1, label="Номер страницы", precision=0)
        step_input = gr.Number(value=DEFAULT_STEP, label="Шаг", precision=0)
        btn_up = gr.Button("Вверх")
        btn_down = gr.Button("Вниз")

        # def update_ui(file, page):
        #     try:
        #         img, info = process_pdf(file, int(page))
        #         return img, info, np.array(img)
        #     except Exception as e:
        #         logger.exception("Ошибка при обработке PDF")
        #         return None, f"Ошибка: {e}", None

        def update_ui(
            file_obj,
            page: int
        ) -> Tuple[Optional[Image.Image], str, Optional[np.ndarray], str]:
            """
            При выборе файла или страницы:
            - конвертирует страницу в изображение
            - возвращает изображение, информацию о страницах,
              массив для scanline и информацию о полях
            """
            if file_obj is None:
                return None, "", None, ""

            # получаем картинку и количество страниц
            try:
                img, total = pdf_to_image(file_obj.name, int(page))
            except Exception as e:
                return None, f"Ошибка: {e}", None, ""

            # вычисляем поля всего документа
            try:
                margins = compute_margins(file_obj.name, 1, total, scale=2, white_threshold=250)
                # форматируем: "1: лев=50, прав=40; 2: лев=48, прав=42; ..."
                margins_info = "; ".join(
                    f"{i+1}: лев={l}, прав={r}" for i, (l, r) in enumerate(margins)
                )
            except Exception as e:
                margins_info = f"Не удалось посчитать поля: {e}"

            info = f"Страниц в документе: {total}"
            return img, info, np.array(img), margins_info

        file_input.change(
            fn=update_ui,
            inputs=[file_input, page_number],
            outputs=[output_image, page_info, state_img_np, margins_info],
            # outputs=[output_image, page_info, state_img_np],
        )
        page_number.change(
            fn=update_ui,
            inputs=[file_input, page_number],
            outputs=[output_image, page_info, state_img_np, margins_info],
            # outputs=[output_image, page_info, state_img_np],
        )
        output_image.select(
            fn=extract_scanline,
            inputs=[state_img_np],
            outputs=[scanline_output, output_image, scanline_y],
        )
        btn_up.click(
            fn=move_up,
            inputs=[state_img_np, scanline_y, step_input],
            outputs=[scanline_output, output_image, scanline_y],
        )
        btn_down.click(
            fn=move_scanline,
            inputs=[state_img_np, scanline_y, step_input],
            outputs=[scanline_output, output_image, scanline_y],
        )

        demo.launch()


if __name__ == "__main__":
    gradio_interface()
