import numpy as np
from typing import List, Callable, Optional, Tuple
from dataclasses import dataclass

from states import State, StateNames, Class, ClassNames
from fsm import FSM, assert_not_forbidden_combo

WHITE_THRESH = 200
GRAY_TOL = 10

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
    count_single_long_black_line: int
    count_single_medium_black_line: int
    count_long_black_line: int
    count_medium_black_line: int
    count_total_medium_black_line: int
    count_many_text: int
    count_color: int
    count_few_text: int
    count_undefined: int
    count_white_px: int
    count_color_px: int
    count_gray_px: int
    heatmap_black: np.ndarray
    heatmap_color: np.ndarray

def get_masks(
    segment: np.ndarray,
    white_thresh: int = WHITE_THRESH,
    gray_tol: int = GRAY_TOL,
):
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

    return (mask_white, mask_nonwhite, mask_color, mask_gray)

def extract_line_features(
    scanline: np.ndarray,
    left_margin: int = 0,
    right_margin: Optional[int] = None,
    white_thresh: int = WHITE_THRESH,
    gray_tol: int = GRAY_TOL,
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

    (mask_white,
     mask_nonwhite,
     mask_color,
     mask_gray) = get_masks(segment, white_thresh, gray_tol)

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


def get_min_long_black_line_length(features):
    length = features.count_white + features.count_gray + features.count_color
    return length / 2 # XXX: magic

def get_min_medium_black_line_length(features):
    length = features.count_white + features.count_gray + features.count_color
    return length / 16 # XXX: magic

def classify_line(feat: LineFeatures):
    cond_background = (
        feat.first_nonwhite_index is None
    )
    if cond_background:
        return State.BACKGROUND

    min_long_black_line_length = get_min_long_black_line_length(feat)
    has_single_gray_comp = (
        len(feat.comp_lengths) == 1 and
        len(feat.gap_lengths) == 0 and
        len(feat.color_comp_lengths) == 0
    )
    pretty_long_gray_comp = (
        feat.count_gray > min_long_black_line_length
    )
    cond_long_black_line = (
        has_single_gray_comp and
        pretty_long_gray_comp
    )
    if cond_long_black_line:
        return State.LONG_BLACK_LINE

    min_medium_black_line_length = get_min_medium_black_line_length(feat)
    has_medium_sized_gray_comp = (
        any(i > min_medium_black_line_length
            for i in feat.gray_comp_lengths)
    )
    cond_medium_black_line = (
        has_medium_sized_gray_comp
    )
    if cond_medium_black_line:
        return State.MEDIUM_BLACK_LINE

    n = 80 # XXX: magic
    has_a_fucking_lot_of_comps = (
        len(feat.comp_lengths) > 100 # XXX: magic
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
    has_huge_gaps = any(abs(z) > 6 for z in z_scores) # XXX: magic
    has_a_few_comps = (
        len(feat.comp_lengths) <= n
    )
    comps_are_mostly_small = (
        mean_comp < 20 # XXX: magic
    )
    gaps_are_mostly_small = (
        mean_gap < 20 # XXX: magic
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
    inferred_state = classify_line(feat)
    # assert_not_forbidden_combo(state, inferred_state)
    return FSM[state][inferred_state]

def handle_undefined(sd: SegmentData):
    def text(sd: SegmentData):
        height = sd.end - sd.start
        not_very_high = height < 50 # XXX: Hardcode in pixels, too bad; depends on initial scale
        # had_few_text = sd.count_few_text > 0
        had_a_lot_of_few_text = (sd.count_few_text / height) > 0.4
        # print(sd.count_few_text / height)
        return (
            # (
            #     not_very_high and
            #     had_few_text
            # ) or
            not_very_high or
            had_a_lot_of_few_text
        )

    def code(sd: SegmentData):
        height = sd.end - sd.start

        high_vbls = sd.heatmap_black == height
        padded = np.concatenate(([False], high_vbls, [False]))
        diff = np.diff(padded.astype(int))
        n_vertical_black_lines = len(np.where(diff == 1)[0])

        return n_vertical_black_lines == 2

    def figure(sd: SegmentData):
        height = sd.end - sd.start
        pretty_high = height > 200 # XXX: HARDCODE; dependson init scale
        return pretty_high

    def plot(sd: SegmentData):
        height = sd.end - sd.start

        high_vbls = sd.heatmap_black >= 0.98 * height
        padded = np.concatenate(([False], high_vbls, [False]))
        diff = np.diff(padded.astype(int))
        n_vertical_black_lines = len(np.where(diff == 1)[0])

        return n_vertical_black_lines == 1

    if text(sd):
        return ClassNames[Class.TEXT]

    if code(sd):
        return ClassNames[Class.CODE]

    if figure(sd):
        return ClassNames[Class.FIGURE]

    if plot(sd):
        return ClassNames[Class.PLOT]

    return ClassNames[Class.UNDEFINED]

def handle_background(sd: SegmentData):
    return ClassNames[Class.BACKGROUND]

def handle_few_text(sd: SegmentData):
    return ClassNames[Class.TEXT]

def handle_many_text(sd: SegmentData):
    def table(sd: SegmentData):
        height = sd.end - sd.start

        high_vbls = sd.heatmap_black == height
        padded = np.concatenate(([False], high_vbls, [False]))
        diff = np.diff(padded.astype(int))
        lbl_start_indices = np.where(diff == 1)[0]
        n_vertical_black_lines = len(lbl_start_indices)
        has_more_than_two_vertical_lines = n_vertical_black_lines > 2

        min_space_is_reasonably_small = True
        if has_more_than_two_vertical_lines:
            min_space_is_reasonably_small = min(np.diff(lbl_start_indices)) > 50


        pretty_high = height > 100 # XXX: HARDCODE; dependson init scale

        return (
            pretty_high and
            has_more_than_two_vertical_lines and
            min_space_is_reasonably_small
        )

    def code(sd: SegmentData):
        height = sd.end - sd.start

        high_vbls = sd.heatmap_black == height
        padded = np.concatenate(([False], high_vbls, [False]))
        diff = np.diff(padded.astype(int))
        n_vertical_black_lines = len(np.where(diff == 1)[0])

        has_two_vertical_lines = n_vertical_black_lines == 2
        has_no_mbls = sd.count_single_medium_black_line == 0
        had_many_text = sd.count_many_text > 0
        has_no_color = sd.count_color == 0

        pretty_high = height > 100 # XXX: HARDCODE; dependson init scale

        return (
            pretty_high and
            has_two_vertical_lines and
            has_no_mbls and
            (
                had_many_text or
                has_no_color
            )
        )

    if table(sd):
        return ClassNames[Class.TABLE]

    if code(sd):
        return ClassNames[Class.CODE]

    return ClassNames[Class.TEXT]

def handle_color(sd: SegmentData):
    def plot(sd: SegmentData):
        height = sd.end - sd.start

        high_vbls = sd.heatmap_black >= 0.98 * height
        padded = np.concatenate(([False], high_vbls, [False]))
        diff = np.diff(padded.astype(int))
        n_vertical_black_lines = len(np.where(diff == 1)[0])

        has_single_vertical_axis = n_vertical_black_lines == 1
        has_pretty_small_color_to_white_relation = (sd.count_color_px / sd.count_white_px) < 0.5
        return (
            has_single_vertical_axis and
            has_pretty_small_color_to_white_relation
        )

    def undefined(sd: SegmentData):
        height = sd.end - sd.start
        smol = height < 20
        return smol

    if plot(sd):
        return ClassNames[Class.PLOT]

    if undefined(sd):
        return ClassNames[Class.UNDEFINED]
    
    return ClassNames[Class.FIGURE]

def handle_medium_black_line(sd: SegmentData):
    def text(sd: SegmentData):
        height = sd.end - sd.start
        had_a_lot_of_a_lot_of_text = (sd.count_many_text / height) > 0.4
        pretty_small = height < 60 # XXX: hardcode
        had_a_lot_of_undefined = (sd.count_undefined / height) > 0.7
        had_a_lot_of_few_text = (sd.count_undefined / height) > 0.4
        return (
            had_a_lot_of_a_lot_of_text or
            (
                pretty_small and
                (
                    had_a_lot_of_undefined or
                    had_a_lot_of_few_text
                )
            )
        )

    def figure(sd: SegmentData):
        height = sd.end - sd.start
        has_color = sd.count_color > 0
        has_a_lot_of_mbls = (sd.count_medium_black_line / height) > 0.1
        pretty_high = height > 100 # XXX: HARDCODE; dependson init scale
        return (
            pretty_high and
            (
                has_color or
                has_a_lot_of_mbls
            )
        )

    def plot(sd: SegmentData):
        height = sd.end - sd.start
        # has_a_lot_of_color = (sd.count_color / height) > 0.7
        has_color = sd.count_color > 0
        has_a_few_mbls = (sd.count_medium_black_line / height) < 0.1

        high_vbls = sd.heatmap_black >= 0.98 * height
        padded = np.concatenate(([False], high_vbls, [False]))
        diff = np.diff(padded.astype(int))
        n_vertical_black_lines = len(np.where(diff == 1)[0])

        all_px = sd.count_white_px + sd.count_color_px + sd.count_gray_px
        has_many_white_pixels = sd.count_white_px / all_px > 0.5

        return (
            # has_a_lot_of_color and
            has_color and
            has_a_few_mbls and
            n_vertical_black_lines >= 2 and
            has_many_white_pixels
        )

    def equation(sd: SegmentData):
        height = sd.end - sd.start
        not_very_high = height < 100 # XXX: Hardcode in pixels, too bad; depends on initial scale
        has_a_few_mbls = sd.count_single_medium_black_line < 4
        has_single_mbl = sd.count_single_medium_black_line == 1
        return (
            (
                not_very_high and
                has_a_few_mbls
            ) or
            has_single_mbl
        )

    def diagram(sd: SegmentData):
        return sd.count_single_medium_black_line > 1

    def undefined(sd: SegmentData):
        height = sd.end - sd.start
        smol = height < 20
        return smol

    if plot(sd):
        return ClassNames[Class.PLOT]

    if figure(sd):
        return ClassNames[Class.FIGURE]

    if diagram(sd):
        return ClassNames[Class.DIAGRAM]

    if text(sd):
        return ClassNames[Class.TEXT]

    if equation(sd):
        # return ClassNames[Class.EQUATION]
        return ClassNames[Class.UNDEFINED]

    if undefined(sd):
        return ClassNames[Class.UNDEFINED]

    return ClassNames[Class.DIAGRAM]

def handle_long_black_line(sd: SegmentData):
    def table(sd: SegmentData):
        height = sd.end - sd.start

        high_vbls = sd.heatmap_black == height
        padded = np.concatenate(([False], high_vbls, [False]))
        diff = np.diff(padded.astype(int))
        lbl_start_indices = np.where(diff == 1)[0]
        n_vertical_black_lines = len(lbl_start_indices)
        has_more_than_two_vertical_lines = n_vertical_black_lines > 2

        min_space_is_reasonably_small = True
        if has_more_than_two_vertical_lines:
            min_space_is_reasonably_small = min(np.diff(lbl_start_indices)) > 50

        return (
            has_more_than_two_vertical_lines and
            min_space_is_reasonably_small
        )

    def code(sd: SegmentData):
        height = sd.end - sd.start

        high_vbls = sd.heatmap_black == height
        padded = np.concatenate(([False], high_vbls, [False]))
        diff = np.diff(padded.astype(int))
        n_vertical_black_lines = len(np.where(diff == 1)[0])

        has_two_vertical_lines = n_vertical_black_lines == 2
        has_no_mbls = sd.count_single_medium_black_line == 0
        had_many_text = sd.count_many_text > 0
        has_no_color = sd.count_color == 0
        return (
            has_two_vertical_lines and
            has_no_mbls and
            (
                had_many_text or
                has_no_color
            )
        )

    def diagram(sd: SegmentData):
        has_no_color = sd.count_color == 0
        has_a_few_mbls = sd.count_single_medium_black_line >= 2
        return (
            has_no_color and
            has_a_few_mbls
        )

    def plot(sd: SegmentData):
        height = sd.end - sd.start
        # has_a_lot_of_color = (sd.count_color / height) > 0.7
        has_color = sd.count_color > 0
        has_a_few_lbls = (sd.count_long_black_line / height) < 0.1

        high_vbls = sd.heatmap_black >= 0.98 * height
        padded = np.concatenate(([False], high_vbls, [False]))
        diff = np.diff(padded.astype(int))
        n_vertical_black_lines = len(np.where(diff == 1)[0])

        return (
            # has_a_lot_of_color and
            has_color and
            has_a_few_lbls and
            n_vertical_black_lines >= 2
        )

    def undefined(sd: SegmentData):
        height = sd.end - sd.start
        smol = height < 20
        return smol

    if undefined(sd):
        return ClassNames[Class.UNDEFINED]

    if plot(sd):
        return ClassNames[Class.PLOT]

    if table(sd):
        return ClassNames[Class.TABLE]

    if code(sd):
        return ClassNames[Class.CODE]

    if diagram(sd):
        return ClassNames[Class.DIAGRAM]

    return ClassNames[Class.FIGURE]

def classify_segment(state: int, sd: SegmentData, raw: bool = False):
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

    if raw:
        return StateNames[state]

    return handler(sd)

def update_segment_data(sd: SegmentData, prev_feat, feat: LineFeatures, line: np.ndarray):
    prev_state = classify_line(prev_feat)
    state = classify_line(feat)

    sd.end += 1

    if prev_state != state and state == State.LONG_BLACK_LINE:
        sd.count_single_long_black_line += 1
    elif state == State.LONG_BLACK_LINE:
        sd.count_long_black_line += 1
    elif prev_state != state and state == State.MEDIUM_BLACK_LINE:
        sd.count_single_medium_black_line += 1
    elif state == State.MEDIUM_BLACK_LINE:
        sd.count_medium_black_line += 1
    elif state == State.MANY_TEXT:
        sd.count_many_text += 1
    elif state == State.COLOR:
        sd.count_color += 1
    elif state == State.FEW_TEXT:
        sd.count_few_text += 1
    elif state == State.UNDEFINED:
        sd.count_undefined += 1
    elif state == State.BACKGROUND:
        return

    sd.count_white_px += feat.count_white
    sd.count_color_px += feat.count_color
    sd.count_gray_px += feat.count_gray

    min_medium_black_line_length = get_min_medium_black_line_length(feat)

    # NOTE: Account for several MBLs on single line
    count_total_medium_black_lines = sum(np.array(feat.gray_comp_lengths) >
                                   min_medium_black_line_length)
    sd.count_total_medium_black_line += count_total_medium_black_lines

    (_, _, mask_color, mask_gray) = get_masks(line, WHITE_THRESH, GRAY_TOL)

    sd.heatmap_black += mask_gray.astype(int)
    sd.heatmap_color += mask_color.astype(int)

def segment_document(
    image: np.ndarray,
    line_feature_func: Callable[[np.ndarray], LineFeatures],
    raw: bool = False,
):
    empty_line = np.zeros_like(image[0:1]).reshape(-1, image[0:1].shape[-1]).min(axis=-1).astype(int)
    def empty_segment_data():
        return SegmentData(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            empty_line, empty_line
        )

    def reset_segment_data(sd: SegmentData):
        sd.start = sd.end
        sd.count_single_long_black_line = 0
        sd.count_single_medium_black_line = 0
        sd.count_long_black_line = 0
        sd.count_medium_black_line = 0
        sd.count_total_medium_black_line = 0
        sd.count_many_text = 0
        sd.count_color = 0
        sd.count_few_text = 0
        sd.count_undefined = 0
        sd.count_white_px = 0
        sd.count_color_px = 0
        sd.count_gray_px = 0
        sd.heatmap_black = np.zeros_like(empty_line)
        sd.heatmap_color = np.zeros_like(empty_line)

    # results = np.array([], dtype=object)
    results = []
    height = image.shape[0]
    prev_state = State.BACKGROUND
    prev_feat = line_feature_func(image[0:1])
    sd = empty_segment_data()
    for y in range(1, height):
        line = image[y:y+1]
        feat = line_feature_func(line)
        state = update_state(prev_state, feat)

        bg_started = state == State.BACKGROUND and prev_state != State.BACKGROUND
        bg_finished = state != State.BACKGROUND and prev_state == State.BACKGROUND
        if bg_started or bg_finished:
            class_name = classify_segment(prev_state, sd, raw)
            result = (sd.start, sd.end, class_name)
            # results = np.append(results, result)
            results.append(result)
            reset_segment_data(sd)

        update_segment_data(sd, prev_feat, feat, line)
        prev_state = state
        prev_feat = feat
    class_name = classify_segment(prev_state, sd, raw)
    result = (sd.start, sd.end, class_name)
    # results = np.append(results, result)
    results.append(result)

    # return results.reshape(-1, 3)
    return results

def merge(markup: List[Tuple[int, int, str]]):
    def merge_segments(arr):
        merged = []
        (current_start, current_end, current_class) = arr[0]
        for i in range(1, len(arr)):
            start, end, cls = arr[i]
            if current_class == cls and current_end == start:
                current_end = end
            else:
                merged.append([current_start, current_end, current_class])
                current_start, current_end, current_class = start, end, cls
        merged.append([current_start, current_end, current_class])
        return merged

    tmp_markup = [markup[0]]
    new_markup = [markup[0]]
    prev_start = curr_start = next_start = prev_end = curr_end = next_end = 0
    prev_height = curr_height = next_height = 0
    prev_class = curr_class = next_class = "None"
    
    # Merge small Background segment with nearest larger segment
    for i in range(1, len(markup) - 1):
        (prev_start, prev_end, prev_class) = markup[i-1]
        (curr_start, curr_end, curr_class) = markup[i]
        (next_start, next_end, next_class) = markup[i+1]

        prev_height = prev_end - prev_start
        curr_height = curr_end - curr_start
        next_height = next_end - next_start

        if curr_class == ClassNames[Class.BACKGROUND] and curr_height < 30:
            if prev_height > curr_height:
                curr_class = prev_class
            elif next_height > curr_height:
                curr_class = next_class

        segment = (curr_start, curr_end, curr_class)
        tmp_markup.append(segment)

    segment = (next_start, next_end, next_class)
    tmp_markup.append(segment)

    new_markup = merge_segments(tmp_markup)
    tmp_markup = [new_markup[0]]

    # Remove background between same-class segments
    for i in range(1, len(new_markup) - 1):
        (prev_start, prev_end, prev_class) = new_markup[i-1]
        (curr_start, curr_end, curr_class) = new_markup[i]
        (next_start, next_end, next_class) = new_markup[i+1]

        prev_height = prev_end - prev_start
        curr_height = curr_end - curr_start
        next_height = next_end - next_start

        if (
            prev_class == next_class and
            curr_class == ClassNames[Class.BACKGROUND]
        ):
            curr_class = next_class

        segment = (curr_start, curr_end, curr_class)
        tmp_markup.append(segment)

    segment = (next_start, next_end, next_class)
    tmp_markup.append(segment)

    new_markup = merge_segments(tmp_markup)
    tmp_markup = [new_markup[0]]

    # Make small backgrounds uncertain
    for i in range(len(new_markup)):
        (curr_start, curr_end, curr_class) = new_markup[i]
        curr_height = curr_end - curr_start

        if curr_class == ClassNames[Class.BACKGROUND] and curr_height < 200:
            curr_class = ClassNames[Class.UNDEFINED]

        segment = (curr_start, curr_end, curr_class)
        tmp_markup.append(segment)

    new_markup = merge_segments(tmp_markup)
    tmp_markup = [new_markup[0]]

    # Merge uncertainty with greatest neighbor
    for i in range(1, len(new_markup) - 1):
        (prev_start, prev_end, prev_class) = new_markup[i-1]
        (curr_start, curr_end, curr_class) = new_markup[i]
        (next_start, next_end, next_class) = new_markup[i+1]

        prev_height = prev_end - prev_start
        curr_height = curr_end - curr_start
        next_height = next_end - next_start

        if curr_class == ClassNames[Class.UNDEFINED] and curr_height < 300:
            if prev_class != ClassNames[Class.BACKGROUND] and prev_height > curr_height:
                curr_class = prev_class
            elif next_class != ClassNames[Class.BACKGROUND] and next_height > curr_height:
                curr_class = next_class

        segment = (curr_start, curr_end, curr_class)
        tmp_markup.append(segment)

    if next_class == ClassNames[Class.UNDEFINED] and next_height < 300:
        next_class = curr_class

    segment = (next_start, next_end, next_class)
    tmp_markup.append(segment)

    tmp_markup = tmp_markup[1:]
    second_class = tmp_markup[1][2]
    first_start = tmp_markup[0][0]
    first_end = tmp_markup[0][1]
    tmp_markup[0] = [first_start, first_end, second_class]
    if tmp_markup[-1][2] == ClassNames[Class.BACKGROUND]:
        s = tmp_markup[-1][0]
        e = tmp_markup[-1][1]
        c = tmp_markup[-2][2]
        tmp_markup[-1] = (s, e, c)

    new_markup = merge_segments(tmp_markup)

    print(new_markup)

    return new_markup

def segment_document_raw(
    image: np.ndarray,
    line_feature_func: Callable[[np.ndarray], LineFeatures],
):
    # results = np.array([], dtype=object)
    results = []
    height = image.shape[0]
    for y in range(1, height):
        line = image[y:y+1]
        feat = line_feature_func(line)
        state = classify_line(feat)
        result = (y, y+1, StateNames[state])
        # results = np.append(results, result)
        results.append(result)
    result = (height-1, height,
              StateNames[classify_line(line_feature_func(image[height-1:height]))])
    # results = np.append(results, result)
    results.append(result)

    # return results.reshape(-1, 3)
    return results

def segdoc(image, v):
    if v == 0:
        markup = segment_document_raw(image, lambda sl:
                                extract_line_features(
                                    sl, 0, None, WHITE_THRESH, GRAY_TOL
                                ))
        return markup
        # return np.array(markup).reshape(-1, 3)

    if v == 1:
        markup = segment_document(image, lambda sl:
                                extract_line_features(
                                    sl, 0, None, WHITE_THRESH, GRAY_TOL
                                ), True)
        return markup
        # return np.array(markup).reshape(-1, 3)

    if v == 2:
        markup =  segment_document(image, lambda sl:
                                extract_line_features(
                                    sl, 0, None, WHITE_THRESH, GRAY_TOL
                                ), False)
        return markup
        # return np.array(markup).reshape(-1, 3)

    if v == 3:
        markup =  segment_document(image, lambda sl:
                                extract_line_features(
                                    sl, 0, None, WHITE_THRESH, GRAY_TOL
                                ), False)
        return merge(markup)
        # return np.array(merge(markup).reshape(-1, 3)
