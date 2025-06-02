import json
from PIL import Image
import numpy as np
import fitz
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from multiprocessing import Manager
import threading

from fast import segdoc as sd

def page_to_image(pdf_path: str, page_index: int, scale_factor: float = 3) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    pix = page.get_displaylist().get_pixmap(matrix=fitz.Matrix(scale_factor, scale_factor))
    img = Image.frombytes("RGB", (pix.w, pix.h), pix.samples)
    doc.close()
    return img

def process_pages(pdf_path, page_range, markup_type, queue=None):
    """Функция для обработки поддиапазона страниц."""
    partial_results = []
    for i in page_range:
        image = page_to_image(pdf_path, i)
        image_np = np.array(image)
        markup = sd(image_np, markup_type)
        assert markup is not None
        del image, image_np
        partial_results.append({
            "page": i + 1,
            "segments": [
                {"y_start": s[0], "y_end": s[1], "label": s[2]} for s in markup
            ]
        })
        if queue:
            queue.put(1)
    return partial_results

def chunk_indices(total, chunks):
    """Делит диапазон страниц на N примерно равных кусков."""
    avg = total // chunks
    remainder = total % chunks
    indices = []
    start = 0
    for i in range(chunks):
        end = start + avg + (1 if i < remainder else 0)
        indices.append(list(range(start, end)))
        start = end
    return indices

def main(pdf_path, json_path, markup_type, num_workers=8):
    # markup_type = 0 -> raw
    # markup_type = 1 -> primary
    # markup_type = 2 -> specified
    # markup_type = 3 -> merged

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()

    chunks = chunk_indices(total_pages, num_workers)

    results = []

    manager = Manager()
    queue = manager.Queue()

    def tqdm_updater():
        pbar = tqdm(total=total_pages, desc="Pages processed")
        for _ in range(total_pages):
            queue.get()
            pbar.update(1)
        pbar.close()

    # Запускаем tqdm в отдельном потоке
    thread = threading.Thread(target=tqdm_updater)
    thread.start()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_pages, pdf_path, chunk, markup_type,
                                   queue) for chunk in chunks]
        for future in futures:
            results.extend(future.result())

    thread.join()  # Дождаться завершения прогресс-бара

    results.sort(key=lambda x: x["page"])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segment PDF pages and export markup to JSON.")
    parser.add_argument("pdf_path", type=str, help="Path to the input PDF file")
    parser.add_argument("json_path", type=str, help="Path to the output JSON file")
    parser.add_argument(
        "markup_type",
        type=int,
        choices=[0, 1, 2, 3],
        help="Markup type: 0 - raw, 1 - primary, 2 - specified, 3 - merged"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=8,
        help="Number of worker processes (default: 8)"
    )

    args = parser.parse_args()

    markup = main(
        pdf_path=args.pdf_path,
        json_path=args.json_path,
        markup_type=args.markup_type,
        num_workers=args.workers
    )
