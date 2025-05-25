import numpy as np
from typing import List, Tuple, Dict, Callable, Optional


def extract_line_features(
    scanline: np.ndarray,
    left_margin: int = 0,
    right_margin: Optional[int] = None,
    white_thresh: int = 250,
    gray_tol: int = 10,
    long_comp_thresh: int = 50
) -> Dict[str, object]:
    """
    Извлекает признаки из строки пикселей.

    Параметры:
    - scanline: одномерный массив пикселей (W или W x C).
    - left_margin: индекс левого отступа (включительно).
    - right_margin: индекс правого отступа (исключительно). Если None, до конца.
    - white_thresh: порог для определения белых пикселей (0-255).
    - gray_tol: допуск для определения серых пикселей (разница каналов).
    - long_comp_thresh: минимальная длина для выделения "длинных" компонент.

    Возвращает словарь с признаками:
      1) 'ratio_nonwhite': отношение не белых пикселей к общей длине.
      2) 'has_nongray': наличие не серых пикселей.
      3) 'n_components': количество связных компонент не белых пикселей.
      4) 'avg_comp_length': средняя длина компонент.
      5) 'std_comp_length': стандартное отклонение длин.
      6) 'min_comp_length': минимальная длина.
      7) 'max_comp_length': максимальная длина.
      8) 'centers_long_comps': центры масс длинных компонент.
    """
    # Определяем границы
    if scanline.ndim == 1:
        data = scanline
    else:
        data = scanline

    end = right_margin if right_margin is not None else data.shape[0]
    segment = data[left_margin:end]
    length = segment.shape[0]
    if length == 0:
        raise ValueError("Segment length is zero. Проверьте margins.")

    # Маска не белых пикселей
    if segment.ndim == 1 or segment.shape[1] == 1:
        # Градает одноканальную
        vals = segment.flatten()
        mask_nonwhite = vals < white_thresh
        mask_nongray = np.zeros_like(mask_nonwhite, dtype=bool)
    else:
        # Цветное изображение (W x C)
        rgb = segment.reshape(-1, segment.shape[-1])
        # Белый: все каналы >= white_thresh
        mask_nonwhite = np.any(rgb < white_thresh, axis=1)
        # Серый: все каналы близки друг к другу
        diffs = rgb.max(axis=1) - rgb.min(axis=1)
        mask_nongray = mask_nonwhite & (diffs > gray_tol)

    # Признак 1: отношение не белых
    ratio_nonwhite = mask_nonwhite.sum() / length
    # Признак 2: наличие не серых
    has_nongray = bool(mask_nongray.any())

    # Находим связные компоненты (True-run lengths)
    comp_lengths: List[int] = []
    comp_starts: List[int] = []
    in_comp = False
    comp_len = 0
    for idx, val in enumerate(mask_nonwhite):
        if val:
            if not in_comp:
                in_comp = True
                comp_len = 1
                comp_starts.append(idx)
            else:
                comp_len += 1
        else:
            if in_comp:
                comp_lengths.append(comp_len)
                in_comp = False
    # Дозапись последней
    if in_comp:
        comp_lengths.append(comp_len)

    n_components = len(comp_lengths)
    if n_components > 0:
        avg_comp_length = float(np.mean(comp_lengths))
        std_comp_length = float(np.std(comp_lengths, ddof=0))
        min_comp_length = int(np.min(comp_lengths))
        max_comp_length = int(np.max(comp_lengths))
    else:
        avg_comp_length = std_comp_length = 0.0
        min_comp_length = max_comp_length = 0

    # Признак 8: центры масс длинных компонен
    centers_long_comps: List[float] = []
    for start, comp_len in zip(comp_starts, comp_lengths):
        if comp_len >= long_comp_thresh:
            center = left_margin + start + comp_len / 2.0
            centers_long_comps.append(center)

    return {
        'ratio_nonwhite': ratio_nonwhite,
        'has_nongray': has_nongray,
        'n_components': n_components,
        'avg_comp_length': avg_comp_length,
        'std_comp_length': std_comp_length,
        'min_comp_length': min_comp_length,
        'max_comp_length': max_comp_length,
        'centers_long_comps': centers_long_comps,
        'comp_lengths': comp_lengths
    }


class DocState:
    UNDEFINED = 0
    BACKGROUND = 1
    TEXT = 2
    TABLE = 3
    IMAGE = 4
    DIAGRAM = 5
    CODE = 6
    LBL_FOUND = 7

StateNames = {
    DocState.UNDEFINED: 'Undefined',
    DocState.BACKGROUND: 'Background',
    DocState.TEXT: 'Text',
    DocState.TABLE: 'Table',
    DocState.IMAGE: 'Image',
    DocState.DIAGRAM: 'Diagram',
    DocState.CODE: 'Code',

    DocState.LBL_FOUND: 'Long Black Line Found',
}


def is_background(features):
    return features['ratio_nonwhite'] == 0

def is_long_black_line(features):
    nc = features['n_components']
    cl = features['comp_lengths'][0]
    return nc == 1 and cl > 200 # TODO: Thresh for cl

def segment_document(
    image: np.ndarray,
    line_feature_func: Callable[[np.ndarray], Dict[str, object]],
) -> List[Tuple[int, int, str]]:
    results: List[Tuple[int, int, str]] = []
    height = image.shape[0]
    prev_scanline = image[0:1]
    prev_features = line_feature_func(prev_scanline)
    and_scanline = prev_scanline
    or_scanline = prev_scanline
    y_start = 0
    state = DocState.BACKGROUND
    for y in range(1, height):
        y_end = y
        scanline = image[y:y+1]
        features = line_feature_func(scanline)

        if is_background(features):
            if state == DocState.BACKGROUND:
                continue
            if state == DocState.UNDEFINED:
                continue
            result = (y_start, y_end, StateNames[state])
            results.append(result)
            y_start = y
            or_scanline = scanline
            continue

        if is_long_black_line(features):
            and_scanline = scanline
            if state == DocState.BACKGROUND:
                state = DocState.LBL_FOUND
            elif state == DocState.UNDEFINED:
                continue # XXX: ???
            elif state == DocState.TABLE:
                continue

        or_scanline = np.logical_or(or_scanline, scanline)
        and_scanline = np.logical_and(and_scanline, scanline)

        prev_scanline = scanline

    return results
