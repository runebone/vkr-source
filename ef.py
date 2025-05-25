import numpy as np
from typing import List, Tuple, Dict, Callable, Optional


def extract_line_features(
    scanline: np.ndarray,
    left_margin: int = 0,
    right_margin: Optional[int] = None,
    white_thresh: int = 250,
    gray_tol: int = 10,
) -> Dict[str, object]:
    """
    Извлекает из одномерного сканлайна следующие признаки:
      1) count_white            — число белых пикселей
      2) count_color            — число цветных (несерых) пикселей
      3) count_gray             — число серых пикселей
      4) comp_lengths           — длины связных компонент не белых пикселей
      5) gap_lengths            — длины пробелов (белых) между такими компонентами
      6) color_comp_lengths     — длины компонент цветных (несерых) пикселей
      7) first_nonwhite_index   — индекс первого не белого пикселя в исходном массиве
    Параметры
    ----------
    scanline : np.ndarray
        Массив пикселей формы (W,) или (W, C).
    left_margin : int
        Левая граница включительно.
    right_margin : Optional[int]
        Правая граница исключая. Если None — до конца.
    white_thresh : int
        Порог для «белого» (0–255).
    gray_tol : int
        Допуск по каналам для «серого».

    Возвращает
    -------
    Dict[str, object]
        Словарь с ключами, перечисленными выше.
    """
    # Выбор сегмента
    end = right_margin if right_margin is not None else scanline.shape[0]
    segment = scanline[left_margin:end]
    if segment.size == 0:
        raise ValueError("Segment length is zero. Проверьте left_margin/right_margin.")

    # Определяем маски
    if segment.ndim == 1 or segment.shape[1] == 1:
        # Одноканальное
        vals = segment.flatten()
        mask_white = vals >= white_thresh
        mask_nonwhite = ~mask_white
        # В одноканальном всё «не белое» считаем «серым»
        mask_color = np.zeros_like(mask_nonwhite, dtype=bool)
        mask_gray = mask_nonwhite.copy()
    else:
        # Многоканальное
        rgb = segment.reshape(-1, segment.shape[-1])
        mask_white = np.all(rgb >= white_thresh, axis=1)
        mask_nonwhite = ~mask_white
        diffs = rgb.max(axis=1) - rgb.min(axis=1)
        mask_color = mask_nonwhite & (diffs > gray_tol)
        mask_gray = mask_nonwhite & ~mask_color

    # Считаем пиксели
    count_white = int(mask_white.sum())
    count_color = int(mask_color.sum())
    count_gray = int(mask_gray.sum())

    # Функция для подсчёта run-length encoding
    def _rle_lengths(mask: np.ndarray) -> List[int]:
        lengths: List[int] = []
        in_run = False
        run_len = 0
        for flag in mask:
            if flag:
                if not in_run:
                    in_run = True
                    run_len = 1
                else:
                    run_len += 1
            else:
                if in_run:
                    lengths.append(run_len)
                    in_run = False
        if in_run:
            lengths.append(run_len)
        return lengths

    comp_lengths = _rle_lengths(mask_nonwhite)
    color_comp_lengths = _rle_lengths(mask_color)

    # Вычисляем длины пробелов между компонентами
    gap_lengths: List[int] = []
    # находим старты и длины nonwhite-компонент
    starts: List[int] = []
    in_comp = False
    curr_len = 0
    for idx, flag in enumerate(mask_nonwhite):
        if flag:
            if not in_comp:
                in_comp = True
                curr_len = 1
                starts.append(idx)
            else:
                curr_len += 1
        else:
            if in_comp:
                in_comp = False
    # если компонент последний не закрыт — закрываем
    # (но для gap_lengths достаточно стартов и comp_lengths)
    for (s, l), (next_s, _) in zip(zip(starts, comp_lengths), zip(starts[1:], comp_lengths[1:])):
        gap = next_s - (s + l)
        if gap > 0:
            gap_lengths.append(gap)

    # Индекс первого не белого пикселя в исходном scanline
    nonwhite_idxs = np.nonzero(mask_nonwhite)[0]
    first_nonwhite_index = (
        int(nonwhite_idxs[0] + left_margin) if nonwhite_idxs.size > 0 else None
    )

    return {
        "count_white": count_white,
        "count_color": count_color,
        "count_gray": count_gray,
        "comp_lengths": comp_lengths,
        "gap_lengths": gap_lengths,
        "color_comp_lengths": color_comp_lengths,
        "first_nonwhite_index": first_nonwhite_index,
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
