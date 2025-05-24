import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import gradio as gr
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

# Настройки приложения
autolog = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Цвета перекраски
COLOR_CONTENT = np.array([0, 255, 0], dtype=np.float32)  # зеленый
COLOR_MARGIN = np.array([255, 0, 0], dtype=np.float32)   # красный

@dataclass(frozen=True)
class Config:
    SCALE_FACTOR: int = 3
    SCANLINE_HEIGHT: int = 2
    DEFAULT_STEP: int = 2
    OVERLAY_ALPHA: float = 0.5
    AUTOCALC_PAGES: int = 10
    WHITE_THRESHOLD: int = 250
    GRAY_WEIGHTS: Tuple[float, float, float] = (0.299, 0.587, 0.114)


def pdf_to_image(pdf_path: str, page_number: int) -> Tuple[Image.Image, int]:
    """
    Преобразует страницу PDF в PIL.Image и возвращает её и общее число страниц.
    """
    doc = fitz.open(pdf_path)
    total = doc.page_count
    if page_number < 1 or page_number > total:
        raise ValueError(f"Page number must be between 1 and {total}")
    page = doc.load_page(page_number - 1)
    matrix = fitz.Matrix(Config.SCALE_FACTOR, Config.SCALE_FACTOR)
    pix = page.get_displaylist().get_pixmap(matrix=matrix)
    img = Image.frombytes("RGB", (pix.w, pix.h), pix.samples)
    return img, total


def compute_margins(
    pdf_path: str,
    start_page: int,
    end_page: int,
    scale: int = Config.SCALE_FACTOR,
    white_threshold: int = Config.WHITE_THRESHOLD,
) -> List[Tuple[int, int]]:
    """
    Вычисляет левое и правое поля для диапазона страниц.
    """
    doc = fitz.open(pdf_path)
    total = doc.page_count
    if not (1 <= start_page <= end_page <= total):
        raise ValueError(f"Pages must be between 1 and {total}, and start <= end")
    margins: List[Tuple[int, int]] = []
    for idx in range(start_page - 1, end_page):
        page = doc.load_page(idx)
        mat = fitz.Matrix(scale, scale)
        pix = page.get_displaylist().get_pixmap(matrix=mat, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        img = arr.reshape(pix.h, pix.w, pix.n)[..., :3]
        gray = np.dot(img, Config.GRAY_WEIGHTS).astype(np.uint8)
        content_mask = gray < white_threshold
        cols_any = content_mask.any(axis=0)
        if not cols_any.any():
            left = right = pix.w
        else:
            first = int(np.argmax(cols_any))
            last = int(pix.w - 1 - np.argmax(cols_any[::-1]))
            left, right = first, pix.w - 1 - last
        margins.append((left, right))
    return margins


def overlay_scanline(
    img: np.ndarray,
    y: int,
    margins: Tuple[int, int],
    height: int = Config.SCANLINE_HEIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создает срез сканлайна и изображение с перекраской:
    - внутренняя область зеленая,
    - поля красные.
    """
    h, w, _ = img.shape
    left, right = margins
    if y < 0 or y + height > h:
        raise IndexError("Scanline out of image bounds")
    scan_slice = img[y : y + height].copy()
    overlayed = img.astype(np.float32).copy()
    xs = np.arange(w)
    mask_margin = (xs < left) | (xs >= w - right)
    mask_content = ~mask_margin
    for row in range(y, y + height):
        region = overlayed[row]
        region[mask_margin] = (
            Config.OVERLAY_ALPHA * COLOR_MARGIN
            + (1 - Config.OVERLAY_ALPHA) * region[mask_margin]
        )
        region[mask_content] = (
            Config.OVERLAY_ALPHA * COLOR_CONTENT
            + (1 - Config.OVERLAY_ALPHA) * region[mask_content]
        )
    return np.clip(overlayed, 0, 255).astype(np.uint8), scan_slice


def extract_scanline(
    img: np.ndarray,
    evt: gr.SelectData,
    margins: Tuple[int, int]
) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[int]]:
    """
    Обработчик клика: извлекает scanline и подсвечивает.
    """
    _, y = evt.index
    y = int(y)
    try:
        highlighted_arr, scan_arr = overlay_scanline(img, y, margins)
    except IndexError:
        return None, None, None
    return Image.fromarray(scan_arr), Image.fromarray(highlighted_arr), y


def move_scanline(
    img: np.ndarray,
    current_y: int,
    step: int,
    margins: Tuple[int, int]
) -> Tuple[Optional[Image.Image], Optional[Image.Image], int]:
    """
    Сдвигает scanline и подсвечивает.
    """
    new_y = current_y + step
    try:
        highlighted_arr, scan_arr = overlay_scanline(img, new_y, margins)
        return Image.fromarray(scan_arr), Image.fromarray(highlighted_arr), new_y
    except IndexError:
        return None, None, current_y


def prev_page(page: int) -> int:
    """Переход к предыдущей странице."""
    return max(page - 1, 1)


def next_page(page: int, total: int) -> int:
    """Переход к следующей странице."""
    return min(page + 1, total)


def first_page() -> int:
    """Переход к первой странице."""
    return 1


def last_page(total: int) -> int:
    """Переход к последней странице."""
    return total


def save_page_margins(
    file_obj,
    page_number: int,
    saved: List[Tuple[int, int, int]]
) -> Tuple[List[Tuple[int, int, int]], str]:
    """
    Сохраняет вычисленные поля для текущей страницы.
    """
    if not file_obj:
        return saved, ""
    try:
        left, right = compute_margins(file_obj.name, page_number, page_number)[0]
    except Exception as e:
        return saved, f"Error computing margins: {e}"
    entry = (page_number, left, right)
    saved = [entry]
    info = f"Page {page_number}: left={left}, right={right}"
    return saved, info


def process_pdf(
    file_obj,
    page_number: int
) -> Tuple[Optional[Image.Image], str, Optional[np.ndarray]]:
    """
    Обрабатывает выбор страницы: возвращает изображение, информацию и массив для состояния.
    """
    if not file_obj:
        return None, "", None
    try:
        img, total = pdf_to_image(file_obj.name, page_number)
    except Exception as e:
        return None, f"Error: {e}", None
    info = f"Total pages: {total}"
    return img, info, np.array(img)


def on_file_change(
    file_obj,
    page_number: int
) -> Tuple[
    Optional[Image.Image], str, Optional[np.ndarray],
    List[Tuple[int, int, int]], str,
    Tuple[int, int], str, int
]:
    """
    При загрузке нового файла: сбрасывает состояния и автоподсчет полей.
    """
    if not file_obj:
        return None, "", None, [], "", (0, 0), "", 0
    try:
        img, total = pdf_to_image(file_obj.name, page_number)
    except Exception as e:
        return None, f"Error: {e}", None, [], "", (0, 0), "", 0
    last = min(total, Config.AUTOCALC_PAGES)
    try:
        margins_list = compute_margins(file_obj.name, 1, last)
        most_common = max(set(margins_list), key=margins_list.count)
    except Exception:
        most_common = (0, 0)
    auto_info = f"Auto margins (first {last}): left={most_common[0]}, right={most_common[1]}"
    return (
        img,
        f"Total pages: {total}",
        np.array(img),
        [],
        "",
        most_common,
        auto_info,
        total,
    )


def gradio_interface() -> None:
    with gr.Blocks(title="PDF Segmenter") as demo:
        gr.Markdown("## Upload PDF, select page and adjust scanline with margins")
        state_img = gr.State()
        state_y = gr.State(0)
        state_saved = gr.State()
        state_default_margins = gr.State((0, 0))
        state_total = gr.State(0)

        with gr.Row():
            with gr.Column():
                page_info = gr.Textbox(label="Document info")
                saved_info = gr.Textbox(label="Saved margins")
                auto_info = gr.Textbox(label="Auto margins")

                file_input = gr.File(label="PDF file", file_types=[".pdf"])
                page_num = gr.Number(value=1, label="Page number", precision=0)
                with gr.Row():
                    btn_first = gr.Button("First")
                    btn_prev = gr.Button("Prev")
                    btn_next = gr.Button("Next")
                    btn_last = gr.Button("Last")
                btn_save = gr.Button("Save margins")
                step = gr.Number(value=Config.DEFAULT_STEP, label="Step", precision=0)
                btn_up = gr.Button("Up")
                btn_down = gr.Button("Down")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Preview")

        scanline_img = gr.Image(type="pil", label="Scanline")

        file_input.change(
            fn=on_file_change,
            inputs=[file_input, page_num],
            outputs=[
                output_image, page_info, state_img,
                state_saved, saved_info,
                state_default_margins, auto_info,
                state_total,
            ],
        )
        page_num.change(
            fn=process_pdf,
            inputs=[file_input, page_num],
            outputs=[output_image, page_info, state_img],
        )
        output_image.select(
            fn=extract_scanline,
            inputs=[state_img, state_default_margins],
            outputs=[scanline_img, output_image, state_y],
        )
        btn_up.click(
            fn=lambda img, y, s, m: move_scanline(img, y, -int(s), m),
            inputs=[state_img, state_y, step, state_default_margins],
            outputs=[scanline_img, output_image, state_y],
        )
        btn_down.click(
            fn=lambda img, y, s, m: move_scanline(img, y, int(s), m),
            inputs=[state_img, state_y, step, state_default_margins],
            outputs=[scanline_img, output_image, state_y],
        )
        btn_save.click(
            fn=save_page_margins,
            inputs=[file_input, page_num, state_saved],
            outputs=[state_saved, saved_info],
        )
        btn_prev.click(
            fn=lambda p: prev_page(int(p)),
            inputs=[page_num], outputs=[page_num]
        )
        btn_next.click(
            fn=lambda p, t: next_page(int(p), int(t)),
            inputs=[page_num, state_total], outputs=[page_num]
        )
        btn_first.click(
            fn=first_page,
            inputs=[], outputs=[page_num]
        )
        btn_last.click(
            fn=last_page,
            inputs=[state_total], outputs=[page_num]
        )

        demo.launch()

if __name__ == "__main__":
    gradio_interface()
