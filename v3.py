import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import gradio as gr
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from ef import extract_line_features

# Настройки приложения
autolog = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Цвета перекраски
COLOR_CONTENT = np.array([0, 255, 0], dtype=np.float32)  # зеленый
COLOR_MARGIN = np.array([255, 0, 0], dtype=np.float32)   # красный

@dataclass(frozen=True)
class Config:
    SCALE_FACTOR: int = 3
    SCANLINE_HEIGHT: int = 1
    DEFAULT_STEP: int = 1
    OVERLAY_ALPHA: float = 0.5
    AUTOCALC_PAGES: int = 10
    WHITE_THRESHOLD: int = 250
    GRAY_WEIGHTS: Tuple[float, float, float] = (0.299, 0.587, 0.114)


def pdf_to_image(pdf_path: str, page_number: int) -> Tuple[Image.Image, int]:
    """Преобразуем страницу PDF в PIL.Image и возвращаем её вместе с числом страниц"""
    doc = fitz.open(pdf_path)
    total = doc.page_count
    if not 1 <= page_number <= total:
        raise ValueError(f"Page number must be between 1 and {total}")
    page = doc.load_page(page_number - 1)
    pix = page.get_displaylist().get_pixmap(matrix=fitz.Matrix(Config.SCALE_FACTOR, Config.SCALE_FACTOR))
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
    """Возвращаем scan_slice и overlayed изображение с подсветкой"""
    h, w, _ = img.shape
    if y < 0 or y + height > h:
        raise IndexError("Scanline out of bounds")
    left, right = margins
    xs = np.arange(w)
    mask_margin = (xs < left) | (xs >= w - right)
    mask_content = ~mask_margin
    overlay = img.astype(np.float32).copy()
    for row in range(y, y + height):
        overlay[row, mask_margin] = Config.OVERLAY_ALPHA * COLOR_MARGIN + (1 - Config.OVERLAY_ALPHA) * overlay[row, mask_margin]
        overlay[row, mask_content] = Config.OVERLAY_ALPHA * COLOR_CONTENT + (1 - Config.OVERLAY_ALPHA) * overlay[row, mask_content]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    slice_img = img[y : y + height]

    print(extract_line_features(slice_img))

    return slice_img, overlay


def extract_scanline(
    img: np.ndarray,
    evt: gr.SelectData,
    saved_margins: Tuple[int, int]
) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[int], Optional[str]]:
    """Обработчик клика: возвращаем scan_slice, overlayed и y"""
    _, y = evt.index
    try:
        slice_np, ov_np = overlay_scanline(img, int(y), saved_margins)
    except IndexError:
        return None, None, None, None
    return Image.fromarray(slice_np), Image.fromarray(ov_np), int(y), str(y)


def move_scanline(
    img: np.ndarray,
    y: int,
    step: int,
    saved_margins: Tuple[int, int]
) -> Tuple[Optional[Image.Image], Optional[Image.Image], int, str]:
    """Сдвигаем сканлайн по step и возвращаем slice, overlay и новое y"""
    new_y = y + step
    try:
        slice_np, ov_np = overlay_scanline(img, new_y, saved_margins)
        return Image.fromarray(slice_np), Image.fromarray(ov_np), new_y, str(new_y)
    except IndexError:
        return None, None, y, str(y)


def page_change(
    file_obj,
    page: int
) -> Tuple[Optional[Image.Image], Optional[np.ndarray], int]:
    if not file_obj:
        return None, None, 0
    try:
        img_pil, total = pdf_to_image(file_obj.name, page)
    except Exception:
        return None, None, 0
    img_arr = np.array(img_pil)
    return img_pil, img_arr, total

def file_upload(
    file_obj,
    page: int
) -> Tuple[Optional[Image.Image], str, Optional[np.ndarray], Tuple[int, int], str, int]:
    img_pil, img_arr, total = page_change(file_obj, page)
    end_page = min(page + Config.AUTOCALC_PAGES, total)
    candidates = compute_margins(file_obj.name, page, end_page)
    auto = max(set(candidates), key=candidates.count)
    saved = auto
    info = f"Total pages: {total}"
    auto_info = f"Auto margins (pages {page}-{end_page}): left={auto[0]}, right={auto[1]}"
    return img_pil, info, img_arr, saved, auto_info, total


def reset_margins(
    file_obj,
    page: int,
    total: int
) -> Tuple[Tuple[int, int], str, str]:
    """Сбрасываем saved_margins по auto расчету от page до page+Config.AUTOCALC_PAGES"""
    if not file_obj:
        return (0, 0), "", ""
    end_page = min(page + Config.AUTOCALC_PAGES, total)
    candidates = compute_margins(file_obj.name, page, end_page)
    auto = max(set(candidates), key=candidates.count)
    text = f"Reset margins: left={auto[0]}, right={auto[1]}"
    auto_info = f"Auto margins (pages {page}-{end_page}): left={auto[0]}, right={auto[1]}"
    return auto, text, auto_info


def save_page_margins(
    file_obj,
    page_number: int,
    saved:Tuple[int, int]
) -> Tuple[Tuple[int, int], str]:
    """
    Сохраняет вычисленные поля для текущей страницы.
    """
    if not file_obj:
        return saved, ""
    try:
        left, right = compute_margins(file_obj.name, page_number, page_number)[0]
    except Exception as e:
        return saved, f"Error computing margins: {e}"
    saved = (left, right)
    info = f"Saved margins for page {page_number}: left={left}, right={right}"
    return saved, info


def gradio_interface() -> None:
    with gr.Blocks(title="PDF Segmenter") as demo:
        gr.Markdown("## PDF Segmenter: highlight margins and scanline")
        state_img = gr.State()
        state_y = gr.State(0)
        state_saved = gr.State((0, 0))
        state_total = gr.State(0)

        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="PDF file", file_types=[".pdf"])
                page_num = gr.Number(value=1, label="Page number", precision=0)
                page_info = gr.Textbox(label="Document info")
                saved_info = gr.Textbox(label="Saved margins")
                auto_info = gr.Textbox(label="Auto margins")
                y_info = gr.Textbox(label="Scanline Y")
                with gr.Row():
                    btn_first = gr.Button("First")
                    btn_prev = gr.Button("Prev")
                    btn_next = gr.Button("Next")
                    btn_last = gr.Button("Last")
                btn_save = gr.Button("Save margins")
                btn_reset = gr.Button("Reset margins")
                step = gr.Number(value=Config.DEFAULT_STEP, label="Step", precision=0)
                btn_up = gr.Button("Up")
                btn_down = gr.Button("Down")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Preview")
        with gr.Row():
            scanline_img = gr.Image(type="pil", label="Scanline")

        file_input.change(
            fn=file_upload,
            inputs=[file_input, page_num],
            outputs=[output_image, page_info, state_img, state_saved, auto_info, state_total]
        )
        page_num.change(
            fn=page_change,
            inputs=[file_input, page_num],
            outputs=[output_image, state_img, state_total]
        )
        btn_first.click(lambda: 1, [], [page_num])
        btn_prev.click(lambda p: max(p-1,1), [page_num], [page_num])
        btn_next.click(lambda p,t: min(p+1,t), [page_num,state_total], [page_num])
        btn_last.click(lambda t: t, [state_total], [page_num])
        btn_reset.click(
            fn=reset_margins,
            inputs=[file_input, page_num, state_total],
            outputs=[state_saved, saved_info, auto_info]
        )
        btn_save.click(
            fn=save_page_margins,
            inputs=[file_input, page_num, state_saved],
            outputs=[state_saved, saved_info]
        )
        output_image.select(
            fn=extract_scanline,
            inputs=[state_img, state_saved],
            outputs=[scanline_img, output_image, state_y, y_info]
        )
        btn_up.click(
            fn=lambda img, y, s, m: move_scanline(img, y, -int(s), m),
            inputs=[state_img, state_y, step, state_saved],
            outputs=[scanline_img, output_image, state_y, y_info]
        )
        btn_down.click(
            fn=lambda img, y, s, m: move_scanline(img, y, int(s), m),
            inputs=[state_img, state_y, step, state_saved],
            outputs=[scanline_img, output_image, state_y, y_info]
        )

        demo.launch()

if __name__ == "__main__":
    gradio_interface()
