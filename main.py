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

def main(pdf_path, json_path, markup_type, num_workers=8, page_indices=None):
    # markup_type = 0 -> raw
    # markup_type = 1 -> primary
    # markup_type = 2 -> specified
    # markup_type = 3 -> merged

    results = []

    if page_indices is None:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        page_indices = list(range(total_pages))
        doc.close()

    chunks = chunk_indices(len(page_indices), num_workers)
    pages_chunks = [ [page_indices[i] for i in chunk] for chunk in chunks ]

    manager = Manager()
    queue = manager.Queue()
    stop_signal = manager.Event()

    def tqdm_updater(pbar, queue, total_pages, stop_signal):
        processed = 0
        while processed < total_pages:
            try:
                queue.get(timeout=0.1)
                pbar.update(1)
                processed += 1
            except Exception:
                if stop_signal.is_set():
                    break
        pbar.close()

    # Запускаем tqdm в отдельном потоке
    pbar = tqdm(total=len(page_indices), desc="Pages processed")
    thread = threading.Thread(target=tqdm_updater, args=(pbar, queue, len(page_indices), stop_signal))
    thread.start()

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_pages, pdf_path, chunk, markup_type,
                                       queue) for chunk in pages_chunks]
            for future in futures:
                results.extend(future.result())
    finally:
        stop_signal.set()
        thread.join()

    results.sort(key=lambda x: x["page"])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results

def parse_page_ranges(pages_str, max_page):
    """Преобразует строку диапазонов страниц в список индексов (0-based)."""
    pages = set()
    for part in pages_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            pages.update(range(start - 1, min(end, max_page)))  # -1 т.к. 0-based
        else:
            page = int(part)
            if 1 <= page <= max_page:
                pages.add(page - 1)
    return sorted(pages)

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
    parser.add_argument(
        "-p", "--pages",
        type=str,
        default=None,
        help="Page ranges to process, e.g., '1-3,5,7-9'"
    )

    args = parser.parse_args()

    # Открываем PDF, чтобы узнать число страниц
    doc = fitz.open(args.pdf_path)
    total_pages = doc.page_count
    doc.close()

    if args.pages:
        pages_to_process = parse_page_ranges(args.pages, total_pages)
    else:
        pages_to_process = list(range(total_pages))

    markup = main(
        pdf_path=args.pdf_path,
        json_path=args.json_path,
        markup_type=args.markup_type,
        num_workers=args.workers,
        page_indices=pages_to_process
    )
