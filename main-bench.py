from PIL import Image
import numpy as np
import fitz
from concurrent.futures import ProcessPoolExecutor
from memory_profiler import memory_usage

from fast import segdoc as sd
import time

def page_to_image(pdf_path: str, page_index: int, scale_factor: float = 3) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    pix = page.get_displaylist().get_pixmap(matrix=fitz.Matrix(scale_factor, scale_factor))
    img = Image.frombytes("RGB", (pix.w, pix.h), pix.samples)
    doc.close()
    return img

def process_pages(pdf_path, page_range, markup_type):
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

def main(pdf_path, markup_type, num_workers=8, page_indices=None):
    results = []

    if page_indices is None:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        page_indices = list(range(total_pages))
        doc.close()

    chunks = chunk_indices(len(page_indices), num_workers)
    pages_chunks = [ [page_indices[i] for i in chunk] for chunk in chunks ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_pages, pdf_path, chunk, markup_type) for chunk in pages_chunks]
        for future in futures:
            results.extend(future.result())

    results.sort(key=lambda x: x["page"])

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

def benchmarked_main(pdf_path, markup_type, num_workers, page_indices):
    def runner():
        return main(pdf_path, markup_type, num_workers, page_indices)

    start_time = time.time()

    mem_usage, result = memory_usage(
        (runner, (), {}),
        interval=0.1,
        timeout=None,
        include_children=True,
        max_usage=True,
        retval=True
    )

    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    print(f"Max memory usage: {mem_usage:.2f} MiB")

    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segment PDF pages and export markup to JSON.")
    parser.add_argument("pdf_path", type=str, help="Path to the input PDF file")
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

    markup = benchmarked_main(
        pdf_path=args.pdf_path,
        markup_type=args.markup_type,
        num_workers=args.workers,
        page_indices=pages_to_process
    )
