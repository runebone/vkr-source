import matplotlib.pyplot as plt
from matplotlib import colors as mc
import pandas as pd
import numpy as np

# Данные по количеству процессов
x = [1, 2, 4, 8, 16, 32, 64]

# Названия методов
renamed_columns = {
    "m0": "Построчная",
    "m1": "Первичная",
    "m2": "Уточненная",
    "m3": "Объединенная"
}

alpha = 0.5
colors = [
    mc.to_rgba("tab:blue", alpha),
    mc.to_rgba("tab:orange", alpha),
    mc.to_rgba("tab:green", alpha),
    mc.to_rgba("tab:red", alpha),
]

# Данные для времени разметки
timing_data = [
    [31.8, 41.21, 41.29, 41.31],
    [16.7, 21.72, 21.96, 22.03],
    [10.16, 13.05, 13.11, 13.19],
    [7.03, 8.71, 8.72, 8.96],
    [7.29, 8.83, 8.96, 9.23],
    [7.40, 9.23, 9.24, 9.47],
    [7.82, 9.67, 9.73, 9.79],
]
df_timing = pd.DataFrame(timing_data, columns=["m0", "m1", "m2", "m3"])
df_timing["x"] = x
df_timing.set_index("x", inplace=True)

# Данные для памяти
memory_data = [
    [247.39, 246.58, 246.56, 245.09],
    [374.05, 373.12, 373.26, 373.16],
    [627.94, 625.74, 627.79, 627.11],
    [1085.41, 1096.95, 1098.18, 1122.85],
    [1933.34, 1934.43, 1932.78, 1935.12],
    [3757.83, 3760.95, 3758.53, 3758.38],
    [6437.75, 6444.76, 6438.00, 6440.77],
]
df_memory = pd.DataFrame(memory_data, columns=["m0", "m1", "m2", "m3"])
df_memory["x"] = x
df_memory.set_index("x", inplace=True)

# Параметры шрифта
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

# Штриховки
# hatches = ['/', '\\', '|', '-']
hatches = ['////', '\\\\\\\\', '||||', '----']

# Гистограмма времени разметки
bar_width = 0.2
x_indexes = np.arange(len(x))

plt.figure(figsize=(12, 6))
for i, col in enumerate(df_timing.columns):
    plt.bar(x_indexes + i * bar_width, df_timing[col], width=bar_width,
            label=renamed_columns[col], hatch=hatches[i],
            edgecolor='black', color=colors[i])
plt.xticks(x_indexes + 1.5 * bar_width, labels=[str(i) for i in x])
plt.title("Зависимость времени разметки от количества процессов")
plt.xlabel("Количество процессов")
plt.ylabel("Время разметки, с")
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("tama_bar.pdf")
plt.show()

# Гистограмма использования памяти
plt.figure(figsize=(12, 6))
for i, col in enumerate(df_memory.columns):
    plt.bar(x_indexes + i * bar_width, df_memory[col], width=bar_width,
            label=renamed_columns[col], hatch=hatches[i],
            edgecolor='black', color=colors[i])
plt.xticks(x_indexes + 1.5 * bar_width, labels=[str(i) for i in x])
plt.title("Зависимость использования памяти от количества процессов")
plt.xlabel("Количество процессов")
plt.ylabel("Максимальное использование памяти, МБ")
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("pama_bar.pdf")
plt.show()
