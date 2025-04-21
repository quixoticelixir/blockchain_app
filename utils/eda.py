import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def generate_eda_plots(data):
    # Основная информация
    info = data.info()

    # Статистика
    stats = data.describe()

    # Гистограммы исходных данных
    numeric_cols = data.select_dtypes(include='number').columns.tolist()
    fig1, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
    for i, col in enumerate(numeric_cols[:12]):
        sns.histplot(data[col], kde=True, bins=30, ax=axes[i // 4][i % 4])
        axes[i // 4][i % 4].set_title(f'Distribution of {col}')
    fig1.tight_layout()

    # Log1p преобразование
    log_data = np.log1p(data[numeric_cols])
    fig2, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
    for i, col in enumerate(log_data.columns[:12]):
        sns.histplot(log_data[col], kde=True, bins=30, ax=axes[i // 4][i % 4])
        axes[i // 4][i % 4].set_title(f'Log-transformed {col}')
    fig2.tight_layout()

    return {
        'info': info,
        'stats': stats,
        'original_plots': fig1,
        'log_plots': fig2
    }