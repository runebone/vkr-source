import json
import fitz  # PyMuPDF
import sys

from states import Class, ClassNames, State, StateNames

LABEL_COLORS = {
    StateNames[State.BACKGROUND]:        (  1,   1,   1),
    StateNames[State.UNDEFINED]:         (  1,   1,   0),
    StateNames[State.FEW_TEXT]:          (  0,   1,   0),
    StateNames[State.MANY_TEXT]:         (  1,   0,   1),
    StateNames[State.COLOR]:             (  0,   1,   1),
    StateNames[State.MEDIUM_BLACK_LINE]: (  1,   0,   0),
    StateNames[State.LONG_BLACK_LINE]:   (  0,   0,   1),
    ClassNames[Class.UNDEFINED]:         (  1,   1,   0),
    ClassNames[Class.BACKGROUND]:        (  1,   1,   1),
    ClassNames[Class.TEXT]:              (  1,   0,   1),
    ClassNames[Class.TABLE]:             (  0,   0,   1),
    ClassNames[Class.CODE]:              (0.5,   0,   1),
    ClassNames[Class.DIAGRAM]:           (  1,   0,   0),
    ClassNames[Class.FIGURE]:            (  0,   1,   1),
    ClassNames[Class.PLOT]:              (  0,   1, 0.5),
}

DEFAULT_COLOR = (0.2, 0.2, 0.2)
ALPHA = 0.4
FONT_SIZE = 12
TEXT_PADDING_X = 5
TEXT_PADDING_Y = 4
SCALE_FACTOR = 3

def annotate_pdf(pdf_path, json_path, output_path):
    doc = fitz.open(pdf_path)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for page_entry in data:
        page_num = page_entry['page'] - 1
        if page_num < 0 or page_num >= len(doc):
            print(f"Warning: page {page_num + 1} out of bounds")
            continue

        page = doc[page_num]
        width = page.rect.width

        for segment in page_entry['segments']:
            y_start = segment['y_start'] / SCALE_FACTOR
            y_end = segment['y_end'] / SCALE_FACTOR
            label = segment['label']
            rgb = LABEL_COLORS.get(label, DEFAULT_COLOR)

            rect = fitz.Rect(0, y_start, width, y_end)

            shape = page.new_shape()
            shape.draw_rect(rect)
            # shape.finish(fill=rgb, color=rgb, fill_opacity=ALPHA)
            shape.finish(fill=rgb, color=None, fill_opacity=ALPHA)
            shape.commit()

            segment_height = y_end - y_start
            if segment_height >= FONT_SIZE + 2 * TEXT_PADDING_Y:
                text_rect = fitz.Rect(
                    rect.x0 + TEXT_PADDING_X,
                    rect.y0 + TEXT_PADDING_Y,
                    rect.x1 - TEXT_PADDING_X,
                    rect.y1 - TEXT_PADDING_Y
                )
                page.insert_textbox(
                    text_rect,
                    label,
                    fontsize=FONT_SIZE,
                    color=rgb,
                    align=0  # 0 = left, 1 = center, 2 = right, 3 = justify
                )

    doc.save(output_path)
    print(f"Сохранено в: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Annotate a PDF with colored segments from a JSON markup file."
    )
    parser.add_argument("pdf_path", type=str, help="Path to the input PDF file")
    parser.add_argument("json_path", type=str, help="Path to the JSON file with markup")
    parser.add_argument("output_path", type=str, help="Path to save the annotated PDF")

    args = parser.parse_args()

    annotate_pdf(args.pdf_path, args.json_path, args.output_path)
