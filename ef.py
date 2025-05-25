import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass

class State:
    UNDEFINED         = 0
    BACKGROUND        = 1
    FEW_TEXT          = 2
    MANY_TEXT         = 3
    COLOR             = 4
    MEDIUM_BLACK_LINE = 5
    LONG_BLACK_LINE   = 6

StateNames = {
    State.UNDEFINED:         "Undefined",
    State.BACKGROUND:        "Background",
    State.FEW_TEXT:          "Few Text",
    State.MANY_TEXT:         "Many Text",
    State.COLOR:             "Color",
    State.MEDIUM_BLACK_LINE: "Medium Black Line",
    State.LONG_BLACK_LINE:   "Long Black Line",
}

class Class:
    UNDEFINED  = 0
    BACKGROUND = 1
    TEXT       = 2
    TABLE      = 3
    CODE       = 4
    DIAGRAM    = 5
    FIGURE     = 6
    PLOT       = 7

ClassNames = {
    Class.UNDEFINED:  "Undefined",
    Class.BACKGROUND: "Background",
    Class.TEXT:       "Text",
    Class.TABLE:      "Table",
    Class.CODE:       "Code",
    Class.DIAGRAM:    "Diagram",
    Class.FIGURE:     "Figure",
    Class.PLOT:       "Plot",
}

@dataclass
class LineFeatures:
    count_white: int
    count_color: int
    count_gray: int
    comp_lengths: List[int]
    gap_lengths: List[int]
    gray_comp_lengths: List[int]
    color_comp_lengths: List[int]
    first_nonwhite_index: int | None

@dataclass
class SegmentData:
    start: int
    end: int
    count_long_black_line: int
    count_medium_black_line: int
    count_many_text: int
    count_color: int
    count_few_text: int
    heatmap_black: np.ndarray
    heatmap_color: np.ndarray
    and_black: np.ndarray


def extract_line_features(
    scanline: np.ndarray,
    left_margin: int = 0,
    right_margin: Optional[int] = None,
    white_thresh: int = 250,
    gray_tol: int = 10,
) -> LineFeatures:
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
    gray_comp_lengths = _rle_lengths(mask_gray)
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

    return LineFeatures(
        count_white,
        count_color,
        count_gray,
        comp_lengths,
        gap_lengths,
        gray_comp_lengths,
        color_comp_lengths,
        first_nonwhite_index,
    )

    # return {
    #     "count_white": count_white,
    #     "count_color": count_color,
    #     "count_gray": count_gray,
    #     "comp_lengths": comp_lengths,
    #     "gap_lengths": gap_lengths,
    #     "color_comp_lengths": color_comp_lengths,
    #     "first_nonwhite_index": first_nonwhite_index,
    # }


def handle_undefined(sd: SegmentData):
    pass

def handle_background(sd: SegmentData):
    return ClassNames[Class.BACKGROUND]

def handle_few_text(sd: SegmentData):
    pass

def handle_many_text(sd: SegmentData):
    pass

def handle_color(sd: SegmentData):
    pass

def handle_medium_black_line(sd: SegmentData):
    pass

def handle_long_black_line(sd: SegmentData):
    pass

def classify_segment(state: int, sd: SegmentData):
    handlers = {
        State.UNDEFINED: handle_undefined,
        State.BACKGROUND: handle_background,
        State.FEW_TEXT: handle_few_text,
        State.MANY_TEXT: handle_many_text,
        State.COLOR: handle_color,
        State.MEDIUM_BLACK_LINE: handle_medium_black_line,
        State.LONG_BLACK_LINE: handle_long_black_line,
    }

    handler = handlers.get(state)
    assert handler is not None

    return handler(sd)

def classify_line(feat: LineFeatures):
    cond_background = (
        feat.first_nonwhite_index is None
    )
    if cond_background:
        return State.BACKGROUND

    length = feat.count_white + feat.count_gray + feat.count_color
    has_single_comp = (
        len(feat.comp_lengths) == 1 and
        len(feat.gap_lengths) == 0 and
        len(feat.color_comp_lengths) == 0
    )
    pretty_long_comp = (
        feat.count_gray > length / 2 # XXX: More than half line
    )
    cond_long_black_line = (
        has_single_comp and
        pretty_long_comp
    )
    if cond_long_black_line:
        return State.LONG_BLACK_LINE

    has_medium_sized_comp = (
        any(i > length / 20 for i in feat.gray_comp_lengths) # XXX: magic 20
    )
    cond_medium_black_line = (
        has_medium_sized_comp
    )
    if cond_medium_black_line:
        return State.MEDIUM_BLACK_LINE

    n = 70 # XXX: magic 70
    has_a_fucking_lot_of_comps = (
        len(feat.comp_lengths) > 100 # XXX: magic 100
    )
    has_a_lot_of_comps_and_no_color = (
        len(feat.comp_lengths) > n and
        feat.count_color == 0
    )
    cond_many_text = (
        has_a_fucking_lot_of_comps or
        has_a_lot_of_comps_and_no_color
    )
    if cond_many_text:
        return State.MANY_TEXT

    has_color = (
        feat.count_color > 0
    )
    cond_color = (
        has_color
    )
    if cond_color:
        return State.COLOR

    mean_comp = np.mean(feat.comp_lengths)
    mean_gap = np.mean(feat.gap_lengths)
    std_gap = np.std(feat.gap_lengths)
    z_scores = (np.array(feat.gap_lengths) - mean_gap) / std_gap
    has_huge_gaps = any(abs(z) > 6 for z in z_scores) # XXX: magic 6
    # print(f"mean_comp: {mean_comp}, mean_gap: {mean_gap}, z_scores: {z_scores}")
    has_a_few_comps = (
        len(feat.comp_lengths) <= n
    )
    comps_are_mostly_small = (
        mean_comp < 20 # XXX: magic 20
    )
    gaps_are_mostly_small = (
        mean_gap < 20 # XXX: magic 20
    )
    cond_few_text = (
        has_a_few_comps and
        comps_are_mostly_small and
        gaps_are_mostly_small and
        not has_huge_gaps
    )
    if cond_few_text:
        return State.FEW_TEXT

    return State.UNDEFINED


def classify_line_str(feat: LineFeatures):
    return StateNames[classify_line(feat)]


def update_state(state: int, feat: LineFeatures):
    pass


def segment_document(
    image: np.ndarray,
    line_feature_func: Callable[[np.ndarray], LineFeatures],
) -> List[Tuple[int, int, str]]:
    results: List[Tuple[int, int, str]] = []
    height = image.shape[0]
    prev_scanline = image[0:1]
    prev_features = line_feature_func(prev_scanline)
    and_scanline = prev_scanline
    black_heatmap = prev_scanline
    color_heatmap = prev_scanline
    y_start = 0
    state = State.BACKGROUND
    for y in range(1, height):
        y_end = y
        scanline = image[y:y+1]
        features = line_feature_func(scanline)

        and_scanline = np.logical_and(and_scanline, scanline)

        prev_scanline = scanline

    return results
