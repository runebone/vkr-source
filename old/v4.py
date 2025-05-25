import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional

import gradio as gr
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Overlay colors
COLOR_CONTENT = np.array([0, 255, 0], dtype=np.float32)
COLOR_MARGIN = np.array([255, 0, 0], dtype=np.float32)

@dataclass(frozen=True)
class Config:
    SCALE: int = 3
    SCANLINE_HEIGHT: int = 1
    DEFAULT_STEP: int = 1
    OVERLAY_ALPHA: float = 0.5
    AUTOCALC_PAGES: int = 10
    WHITE_THRESHOLD: int = 250
    GRAY_WEIGHTS: Tuple[float, float, float] = (0.299, 0.587, 0.114)


def pdf_to_image(path: str, page_num: int) -> Tuple[Image.Image, int]:
    """Convert a PDF page to a PIL image and return it with total page count."""
    doc = fitz.open(path)
    total_pages = doc.page_count
    if not (1 <= page_num <= total_pages):
        raise ValueError(f"Page number must be between 1 and {total_pages}")
    page = doc.load_page(page_num - 1)
    matrix = fitz.Matrix(Config.SCALE, Config.SCALE)
    pix = page.get_displaylist().get_pixmap(matrix=matrix)
    img = Image.frombytes("RGB", (pix.w, pix.h), pix.samples)
    return img, total_pages


def compute_margins(path: str, start: int, end: int) -> List[Tuple[int, int]]:
    """Compute left/right margins for a range of pages."""
    doc = fitz.open(path)
    total = doc.page_count
    if not (1 <= start <= end <= total):
        raise ValueError(f"Pages must be between 1 and {total}")
    margins: List[Tuple[int, int]] = []
    for i in range(start - 1, end):
        page = doc.load_page(i)
        pix = page.get_displaylist().get_pixmap(matrix=fitz.Matrix(Config.SCALE, Config.SCALE), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3]
        gray = np.dot(arr, Config.GRAY_WEIGHTS).astype(np.uint8)
        content_mask = gray < Config.WHITE_THRESHOLD
        cols = content_mask.any(axis=0)
        if not cols.any():
            margins.append((pix.w, pix.w))
        else:
            left = int(np.argmax(cols))
            right = int(pix.w - 1 - np.argmax(cols[::-1]))
            margins.append((left, right))
    return margins


def overlay_scanline(arr: np.ndarray, y: int, margins: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return the scan slice and an overlayed image highlighting the scanline."""
    h, w, _ = arr.shape
    if y < 0 or y + Config.SCANLINE_HEIGHT > h:
        raise IndexError("Scanline out of bounds")
    left, right = margins
    xs = np.arange(w)
    mask_margin = (xs < left) | (xs > right)
    mask_content = ~mask_margin
    overlay = arr.astype(np.float32).copy()
    for row in range(y, y + Config.SCANLINE_HEIGHT):
        overlay[row, mask_margin] = (
            Config.OVERLAY_ALPHA * COLOR_MARGIN
            + (1 - Config.OVERLAY_ALPHA) * overlay[row, mask_margin]
        )
        overlay[row, mask_content] = (
            Config.OVERLAY_ALPHA * COLOR_CONTENT
            + (1 - Config.OVERLAY_ALPHA) * overlay[row, mask_content]
        )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    slice_arr = arr[y : y + Config.SCANLINE_HEIGHT]
    return slice_arr, overlay


def load_page(
    file_obj,
    page_num: float,
) -> Tuple[
    Optional[Image.Image],
    str,
    Optional[np.ndarray],
    Tuple[int, int],
    str,
    int,
    Optional[Image.Image],
    int,
]:
    """Load PDF page, compute margins, and apply default scanline at y=0."""
    if not file_obj:
        return None, "", None, (0, 0), "", 0, None, 0
    path = file_obj.name
    try:
        pil_img, total = pdf_to_image(path, int(page_num))
    except Exception as e:
        logger.error(e)
        return None, "", None, (0, 0), "", 0, None, 0
    arr = np.array(pil_img)
    end_page = min(int(page_num) + Config.AUTOCALC_PAGES, total)
    candidates = compute_margins(path, int(page_num), end_page)
    auto = max(set(candidates), key=candidates.count)
    try:
        slice_arr, ov_arr = overlay_scanline(arr, 0, auto)
        slice_img = Image.fromarray(slice_arr)
        ov_img = Image.fromarray(ov_arr)
    except Exception as e:
        logger.warning("Failed to overlay scanline: %s", e)
        slice_img = None
        ov_img = pil_img
    info = f"Total pages: {total}"
    auto_info = f"Auto margins ({int(page_num)}-{end_page}): left={auto[0]}, right={auto[1]}"
    return ov_img, info, arr, auto, auto_info, total, slice_img, 0


def update_scanline(
    arr: Optional[np.ndarray],
    y: float,
    margins: Tuple[int, int],
) -> Tuple[Optional[Image.Image], Optional[Image.Image], int]:
    """Update the scanline at a given y-position."""
    if arr is None:
        return None, None, 0
    try:
        slice_arr, ov_arr = overlay_scanline(arr, int(y), margins)
        return Image.fromarray(slice_arr), Image.fromarray(ov_arr), int(y)
    except IndexError:
        return None, None, 0


def reset_margins(
    file_obj,
    page_num: float,
    total: int,
) -> Tuple[Tuple[int, int], str, str]:
    """Reset margins to auto-calculated values."""
    if not file_obj:
        return (0, 0), "", ""
    path = file_obj.name
    end_page = min(int(page_num) + Config.AUTOCALC_PAGES, total)
    candidates = compute_margins(path, int(page_num), end_page)
    auto = max(set(candidates), key=candidates.count)
    text = f"Reset margins: left={auto[0]}, right={auto[1]}"
    auto_info = f"Auto margins ({int(page_num)}-{end_page}): left={auto[0]}, right={auto[1]}"
    return auto, text, auto_info


def save_page_margins(
    file_obj,
    page_num: float,
    margins: Tuple[int, int],
) -> Tuple[Tuple[int, int], str]:
    """Save current margins for this page."""
    if not file_obj:
        return margins, ""
    path = file_obj.name
    try:
        m = compute_margins(path, int(page_num), int(page_num))[0]
        return m, f"Saved margins for page {int(page_num)}: left={m[0]}, right={m[1]}"
    except Exception as e:
        return margins, f"Error computing margins: {e}"


def extract_scanline(evt: gr.SelectData) -> int:
    return int(evt.index[1])


def move_scanline(y: int, step: int) -> int:
    return y + step


def build_interface() -> None:
    """Construct and launch the Gradio interface."""
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
                y_input = gr.Number(value=0, label="Scanline Y", precision=0)
                page_info = gr.Textbox(label="Document info")
                saved_info = gr.Textbox(label="Saved margins")
                auto_info = gr.Textbox(label="Auto margins")
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

        # Load or change page
        for trigger in (file_input.change, page_num.change):
            trigger(
                fn=load_page,
                inputs=[file_input, page_num],
                outputs=[
                    output_image,
                    page_info,
                    state_img,
                    state_saved,
                    auto_info,
                    state_total,
                    scanline_img,
                    state_y,
                ],
            )

        # Manual Y input updates scanline
        y_input.change(
            fn=update_scanline,
            inputs=[state_img, y_input, state_saved],
            outputs=[scanline_img, output_image, state_y],
        )

        output_image.select(
            fn=extract_scanline,
            inputs=[],
            outputs=[y_input]
        )
        btn_up.click(
            fn=lambda y, s: move_scanline(y, -int(s)),
            inputs=[state_y, step],
            outputs=[y_input]
        )
        btn_down.click(
            fn=lambda y, s: move_scanline(y, int(s)),
            inputs=[state_y, step],
            outputs=[y_input]
        )


        # Page navigation
        btn_first.click(lambda: 1, [], [page_num])
        btn_prev.click(lambda p: max(1, int(p) - 1), [page_num], [page_num])
        btn_next.click(lambda p, t: min(int(p) + 1, int(t)), [page_num, state_total], [page_num])
        btn_last.click(lambda t: t, [state_total], [page_num])

        # Margin controls
        btn_reset.click(
            fn=reset_margins,
            inputs=[file_input, page_num, state_total],
            outputs=[state_saved, saved_info, auto_info],
        )
        btn_save.click(
            fn=save_page_margins,
            inputs=[file_input, page_num, state_saved],
            outputs=[state_saved, saved_info],
        )

        demo.launch()


if __name__ == "__main__":
    build_interface()
