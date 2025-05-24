import numpy as np
from scipy.ndimage import label


def extract_line_features(scanline: np.ndarray) -> dict:
    """
    Принимает на вход бинарную сканирующую строку высотой 2 пикселя и шириной W.
    Возвращает словарь с рассчитанными эвристическими признаками.

    Ожидается: scanline.shape == (2, W), значения 0 (белый) и 1 (черный)
    """
    assert scanline.ndim == 2 and scanline.shape[0] == 2, "Input must be 2 x W binary array"
    height, width = scanline.shape
    assert set(np.unique(scanline)).issubset({0, 1}), "Scanline must be binary (0, 1)"
    
    # Объединяем строки по вертикали (если в любом из двух пикселей по высоте есть 1 — считаем пиксель черным)
    line = (scanline.sum(axis=0) > 0).astype(int)

    # 1. Плотность (D)
    density = np.sum(line) / width

    # 2. Компоненты (непрерывные черные участки)
    structure = np.array([1])  # подключаем 1D связность
    labeled_array, num_components = label(line, structure=structure)

    # 3. Ширины компонент
    component_widths = []
    for comp_id in range(1, num_components + 1):
        comp_mask = (labeled_array == comp_id)
        component_widths.append(np.sum(comp_mask))

    if component_widths:
        avg_width = np.mean(component_widths)
        min_width = np.min(component_widths)
        max_width = np.max(component_widths)
        std_width = np.std(component_widths)
    else:
        avg_width = min_width = max_width = std_width = 0.0

    # 4. Центроид (центр масс по X) — нормализованный
    if np.sum(line) > 0:
        centroid = np.sum(np.arange(width) * line) / np.sum(line) / width
    else:
        centroid = 0.0

    # 5. Расстояния между компонентами (gaps)
    if num_components > 1:
        comp_positions = [np.where(labeled_array == i)[0] for i in range(1, num_components + 1)]
        start_positions = [pos[0] for pos in comp_positions]
        gap_widths = np.diff(start_positions)
        std_gap = np.std(gap_widths) if len(gap_widths) > 1 else 0.0
    else:
        std_gap = 0.0

    return {
        "density": density,
        "num_components": num_components,
        "avg_component_width": avg_width,
        "min_component_width": min_width,
        "max_component_width": max_width,
        "std_component_width": std_width,
        "centroid": centroid,
        "std_gap": std_gap
    }
