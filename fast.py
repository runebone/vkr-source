import numpy as np
from typing import Callable, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from logic import LineFeatures, SegmentData, State, WHITE_THRESH, update_segment_data, update_state, classify_segment

def find_segments(image: np.ndarray) -> List[Tuple[int, int]]:
    """Возвращает список сегментов (start_y, end_y), где есть небелые строки."""
    white_rows = np.all(image >= WHITE_THRESH, axis=(1, 2))
    padded = np.concatenate(([True], white_rows, [True]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == -1)[0]
    ends = np.where(diff == 1)[0] - 1
    return list(zip(starts, ends))

def find_segments_all(image: np.ndarray) -> List[Tuple[int, int]]:
    """Возвращает список всех сегментов (start_y, end_y), чередуя белые и небелые участки."""
    is_white_row = np.all(image >= WHITE_THRESH, axis=(1, 2)).astype(int)
    diff = np.diff(is_white_row)

    # Индексы смен состояния (True → False или False → True)
    # change_indices = np.where(diff != 0)[0] + 1
    change_indices = np.where(diff != 0)[0]

    # Добавляем границы: начало (0) и конец (image.shape[0])
    boundaries = np.concatenate(([0], change_indices, [image.shape[0] - 1]))

    # Формируем пары (start, end)
    # segments = [(int(boundaries[i]), int(boundaries[i + 1] - 1)) for i in range(len(boundaries) - 1)]
    segments = [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(len(boundaries) - 1)]

    return segments


def classify_segment_range(
    image: np.ndarray,
    segment_range: Tuple[int, int],
    line_feature_func: Callable[[np.ndarray], LineFeatures],
    raw: bool = False,
) -> Tuple[int, int, str]:
    segment_start_y, segment_end_y = segment_range
    empty_line = np.zeros_like(image[0:1]).reshape(-1, image[0:1].shape[-1]).min(axis=-1).astype(int)

    def empty_segment_data():
        return SegmentData(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            empty_line.copy(), empty_line.copy()
        )

    prev_state = State.BACKGROUND
    prev_feat = line_feature_func(image[segment_start_y:segment_start_y + 1])
    sd = empty_segment_data()
    sd.start = int(segment_start_y)

    for y in range(segment_start_y, segment_end_y + 1):
        line = image[y:y + 1]
        feat = line_feature_func(line)
        state = update_state(prev_state, feat)
        update_segment_data(sd, prev_feat, feat, line)
        prev_state = state
        prev_feat = feat

    sd.end = int(segment_end_y)
    class_name = classify_segment(prev_state, sd, raw)
    return (sd.start, sd.end, class_name)


def segment_document(
    image: np.ndarray,
    line_feature_func: Callable[[np.ndarray], 'LineFeatures'],
    raw: bool = False,
    max_workers: int = 8,
) -> List[Tuple[int, int, str]]:
    """Сегментирует изображение документа и классифицирует каждый сегмент в пуле потоков."""
    # segments = find_segments(image)
    segments = find_segments_all(image)
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # with ProcessPoolExecutor(max_workers=max_workers or multiprocessing.cpu_count()) as executor:
        futures = [
            executor.submit(classify_segment_range, image, seg, line_feature_func, raw)
            for seg in segments
        ]
        for future in futures:
            results.append(future.result())

    return results
