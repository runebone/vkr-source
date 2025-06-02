import json
from PIL import Image
import numpy as np
import fitz
from concurrent.futures import ProcessPoolExecutor

from logic import extract_line_features, merge, segment_document_raw
from fast import segment_document

def page_to_image(pdf_path: str, page_index: int, scale_factor: float = 3) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    pix = page.get_displaylist().get_pixmap(matrix=fitz.Matrix(scale_factor, scale_factor))
    img = Image.frombytes("RGB", (pix.w, pix.h), pix.samples)
    doc.close()
    return img

def x(sl):
    return extract_line_features(sl, 0, None)

def sd(image):
    # markup = segment_document(image, x, True, 2)
    markup = segment_document(image, x, False, 4)
    # markup = segment_document_raw(image, x)
    return markup

def process_pages(pdf_path, page_range):
    """Функция для обработки поддиапазона страниц."""
    partial_results = []
    for i in page_range:
        image = page_to_image(pdf_path, i)
        image_np = np.array(image)
        markup = sd(image_np)
        markup = merge(markup)
        del image, image_np
        partial_results.append({
            "page": i + 1,
            "segments": [
                {"y_start": s[0], "y_end": s[1], "label": s[2]} for s in markup
            ]
        })
        print(i)
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

def main(pdf_path, num_workers=8):
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()

    chunks = chunk_indices(total_pages, num_workers)

    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_pages, pdf_path, chunk) for chunk in chunks]
        for future in futures:
            results.extend(future.result())

    results.sort(key=lambda x: x["page"])

    with open("sex.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results

if __name__ == "__main__":
    # markup = main("/home/rukost/index.pdf")
    markup = main("/home/rukost/University/vkr/index.pdf")
