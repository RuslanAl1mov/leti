from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tds_core import FlockData


class FlockVisualizer:
    """Строит и сохраняет графики для анализа динамики стаи."""

    def __init__(self, output_dir: str | Path):
        """Инициализирует визуализатор и подготавливает директорию вывода.

        Args:
            output_dir: Путь к папке, куда будут сохраняться изображения.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _flight_type_label(flock: FlockData) -> str:
        """Возвращает русскую подпись типа полета для заголовков графиков."""
        flock_id = flock.flock_id.lower()
        if flock_id.startswith("ff") or flock.group == "1":
            return "локальный полет"
        if flock_id.startswith("hf") or flock.group == "2":
            return "маршрутный полет"
        return "неизвестный тип полета"

    def plot_trajectory(self, flock: FlockData) -> Path:
        """Сохраняет траектории движения всех птиц в стае на плоскости.

        Args:
            flock: Данные одной стаи с координатами птиц во времени.

        Returns:
            Путь к сохраненному PNG-файлу с траекториями.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        for bird_id, df in flock.birds.items():
            ax.plot(df["x"], df["y"], linewidth=1.3, label=bird_id)
            ax.scatter(df["x"].iloc[0], df["y"].iloc[0], s=18)
        ax.set_title(
            f"Траектории: {flock.flock_id} ({self._flight_type_label(flock)})"
        )
        ax.set_xlabel("X (м)")
        ax.set_ylabel("Y (м)")
        ax.axis("equal")
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=8, frameon=False)
        out = self.output_dir / f"{flock.flock_id}_trajectory.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        return out

    @staticmethod
    def _pair_metric_boxplot_data(
        pair_df: pd.DataFrame, value_col: str, scale: float = 1.0
    ) -> tuple[list[np.ndarray], list[str]]:
        birds = sorted(set(pair_df["bird1"]) | set(pair_df["bird2"]))
        data: list[np.ndarray] = []
        labels: list[str] = []
        for bird in birds:
            mask = (pair_df["bird1"] == bird) | (pair_df["bird2"] == bird)
            values = pair_df.loc[mask, value_col].dropna().to_numpy(dtype=float) / scale
            if len(values):
                data.append(values)
                labels.append(bird)
        return data, labels

    def plot_coordinate_overview(self, flock: FlockData, max_birds: int = 6) -> Path:
        """Строит обзор координат `x` и `y` для нескольких птиц во времени.

        Args:
            flock: Данные стаи с временным рядом координат.
            max_birds: Максимальное число птиц, отображаемых на графике.

        Returns:
            Путь к сохраненному PNG-файлу с обзором координат.
        """
        bird_ids = flock.bird_ids[:max_birds]
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for bird_id in bird_ids:
            df = flock.birds[bird_id]
            axes[0].plot(flock.time_seconds, df["x"], label=bird_id, linewidth=1.0)
            axes[1].plot(flock.time_seconds, df["y"], label=bird_id, linewidth=1.0)
        axes[0].set_title(f"Обзор координат: {flock.flock_id}")
        axes[0].set_ylabel("X (м)")
        axes[1].set_ylabel("Y (м)")
        axes[1].set_xlabel("Время (с)")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(ncol=3, fontsize=8, frameon=False)
        out = self.output_dir / f"{flock.flock_id}_coordinates.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        return out

    def plot_tds_heatmap(self, pair_df: pd.DataFrame, flock_id: str, mode: str, window_size: int) -> Optional[Path]:
        """Строит тепловую карту попарных значений TDS для выбранной стаи.

        Args:
            pair_df: Таблица с попарными метриками TDS между птицами.
            flock_id: Идентификатор стаи для фильтрации данных.
            mode: Ось или режим анализа, например `x` или `y`.
            window_size: Размер окна, для которого строится карта.

        Returns:
            Путь к сохраненному файлу, либо `None`, если подходящих данных нет.
        """
        subset = pair_df[
            (pair_df["id"] == flock_id)
            & (pair_df["coord"] == mode)
            & (pair_df["window"] == window_size)
        ]
        if subset.empty:
            return None
        birds = sorted(set(subset["bird1"]) | set(subset["bird2"]))
        matrix = pd.DataFrame(np.nan, index=birds, columns=birds)
        np.fill_diagonal(matrix.values, 1.0)
        for _, row in subset.iterrows():
            matrix.loc[row["bird1"], row["bird2"]] = row["tds"] / 100.0
            matrix.loc[row["bird2"], row["bird1"]] = row["tds"] / 100.0

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix.values, vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(birds)))
        ax.set_yticks(np.arange(len(birds)))
        ax.set_xticklabels(birds)
        ax.set_yticklabels(birds)
        ax.set_title(f"Тепловая карта TDS: {flock_id}, режим={mode}, окно={window_size}")
        for i in range(len(birds)):
            for j in range(len(birds)):
                value = matrix.values[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.85, label="Значение TDS")
        out = self.output_dir / f"{flock_id}_tds_{mode}_w{window_size}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        return out

    def plot_phase_heatmap(
        self, phase_df: pd.DataFrame, flock_id: str, mode: str, window_size: int
    ) -> Optional[Path]:
        """Строит тепловую карту коэффициента фазовой синхронизации."""
        subset = phase_df[
            (phase_df["id"] == flock_id)
            & (phase_df["coord"] == mode)
            & (phase_df["window"] == window_size)
        ]
        if subset.empty:
            return None
        birds = sorted(set(subset["bird1"]) | set(subset["bird2"]))
        matrix = pd.DataFrame(np.nan, index=birds, columns=birds)
        np.fill_diagonal(matrix.values, 1.0)
        for _, row in subset.iterrows():
            matrix.loc[row["bird1"], row["bird2"]] = row["phase_sync"]
            matrix.loc[row["bird2"], row["bird1"]] = row["phase_sync"]

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix.values, vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(birds)))
        ax.set_yticks(np.arange(len(birds)))
        ax.set_xticklabels(birds)
        ax.set_yticklabels(birds)
        ax.set_title(
            f"Тепловая карта коэффициента фазовой синхронизации: {flock_id}, режим={mode}, окно={window_size}"
        )
        for i in range(len(birds)):
            for j in range(len(birds)):
                value = matrix.values[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.85, label="Коэффициент фазовой синхронизации")
        out = self.output_dir / f"{flock_id}_phase_{mode}_w{window_size}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        return out

    def plot_group_comparison(self, summary_df: pd.DataFrame, mode: str) -> Optional[Path]:
        """Сравнивает средний TDS между группами для разных размеров окна.

        Args:
            summary_df: Сводная таблица по стаям и размерам окна.
            mode: Ось или режим анализа, для которого строится сравнение.

        Returns:
            Путь к сохраненному файлу, либо `None`, если данных для режима нет.
        """
        subset = summary_df[summary_df["coord"] == mode].copy()
        if subset.empty:
            return None
        subset = subset.sort_values(["window", "type", "id"])
        windows = sorted(subset["window"].unique())
        groups = list(subset["type"].dropna().unique())

        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        x = np.arange(len(windows))
        for idx, group in enumerate(groups):
            gdf = subset[subset["type"] == group]
            means = [gdf[gdf["window"] == w]["mean_tds"].mean() for w in windows]
            stds = [gdf[gdf["window"] == w]["mean_tds"].std() for w in windows]
            positions = x + (idx - (len(groups) - 1) / 2) * width
            ax.bar(positions, means, width=width, yerr=stds, capsize=4, label=group)

        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in windows])
        ax.set_xlabel("Размер окна (отсчеты)")
        ax.set_ylabel("Средний TDS стаи")
        ax.set_title(f"Сравнение групп по размеру окна ({mode})")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False)
        out = self.output_dir / f"group_comparison_{mode}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        return out

    def plot_lag_trace(self, lag_df: pd.DataFrame, flock_id: str, bird_i: str, bird_j: str, mode: str, window_size: int) -> Optional[Path]:
        """Строит изменение лучшего лага и корреляции по окнам для пары птиц.

        Args:
            lag_df: Таблица с результатами оценки лагов по временным окнам.
            flock_id: Идентификатор стаи.
            bird_i: Идентификатор первой птицы в паре.
            bird_j: Идентификатор второй птицы в паре.
            mode: Ось или режим анализа.
            window_size: Размер окна, для которого выбираются значения.

        Returns:
            Путь к сохраненному файлу, либо `None`, если подходящих данных нет.
        """
        subset = lag_df[
            (lag_df["id"] == flock_id)
            & (lag_df["bird1"] == bird_i)
            & (lag_df["bird2"] == bird_j)
            & (lag_df["coord"] == mode)
            & (lag_df["window"] == window_size)
        ]
        if subset.empty:
            return None
        subset = subset.sort_values("window_index")
        fig, ax1 = plt.subplots(figsize=(10, 4.5))
        ax1.plot(subset["window_index"], subset["best_lag"], marker="o", linewidth=1)
        ax1.set_xlabel("Индекс окна")
        ax1.set_ylabel("Лучший лаг (отсчеты)")
        ax1.set_title(f"Стабильность лага: {flock_id} {bird_i}-{bird_j}, режим={mode}, окно={window_size}")
        ax1.grid(alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(subset["window_index"], subset["best_corr"], linestyle="--", alpha=0.7)
        ax2.set_ylabel("Лучшая корреляция")
        out = self.output_dir / f"{flock_id}_{bird_i}_{bird_j}_{mode}_w{window_size}_lag_trace.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        return out

    def plot_phase_trace(
        self,
        phase_trace_df: pd.DataFrame,
        flock_id: str,
        bird_i: str,
        bird_j: str,
        mode: str,
        window_size: int,
    ) -> Optional[Path]:
        """Строит изменение коэффициента фазовой синхронизации по окнам для пары птиц."""
        subset = phase_trace_df[
            (phase_trace_df["id"] == flock_id)
            & (phase_trace_df["bird1"] == bird_i)
            & (phase_trace_df["bird2"] == bird_j)
            & (phase_trace_df["coord"] == mode)
            & (phase_trace_df["window"] == window_size)
        ]
        if subset.empty:
            return None
        subset = subset.sort_values("window_index")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(subset["window_index"], subset["phase_sync"], marker="o", linewidth=1)
        ax.set_xlabel("Индекс окна")
        ax.set_ylabel("Коэффициент фазовой синхронизации")
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Коэффициент фазовой синхронизации: {flock_id} {bird_i}-{bird_j}, режим={mode}, окно={window_size}"
        )
        ax.grid(alpha=0.3)
        out = self.output_dir / f"{flock_id}_{bird_i}_{bird_j}_{mode}_w{window_size}_phase_trace.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        return out

    def plot_tds_boxplot(
        self, pair_df: pd.DataFrame, flock_id: str, mode: str, window_size: int
    ) -> Optional[Path]:
        """Строит boxplot TDS по птицам: связь каждой птицы со всеми остальными."""
        subset = pair_df[
            (pair_df["id"] == flock_id)
            & (pair_df["coord"] == mode)
            & (pair_df["window"] == window_size)
        ].copy()
        if subset.empty:
            return None

        data, labels = self._pair_metric_boxplot_data(subset, value_col="tds", scale=100.0)
        if not data:
            return None

        fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(labels) + 2), 5))
        ax.boxplot(
            data,
            patch_artist=True,
            boxprops={"facecolor": "#4C78A8", "alpha": 0.7},
            medianprops={"color": "#F58518", "linewidth": 2},
        )
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Значение TDS")
        ax.set_xlabel("Птица")
        ax.set_title(
            f"Боксплот TDS по птицам: {flock_id}, режим={mode}, окно={window_size}"
        )
        ax.grid(axis="y", alpha=0.3)

        out = self.output_dir / f"{flock_id}_tds_{mode}_w{window_size}_boxplot.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        return out

    def plot_phase_boxplot(
        self, phase_df: pd.DataFrame, flock_id: str, mode: str, window_size: int
    ) -> Optional[Path]:
        """Строит boxplot фазовой синхронизации по птицам."""
        subset = phase_df[
            (phase_df["id"] == flock_id)
            & (phase_df["coord"] == mode)
            & (phase_df["window"] == window_size)
        ].copy()
        if subset.empty:
            return None

        data, labels = self._pair_metric_boxplot_data(
            subset, value_col="phase_sync", scale=1.0
        )
        if not data:
            return None

        fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(labels) + 2), 5))
        ax.boxplot(
            data,
            patch_artist=True,
            boxprops={"facecolor": "#54A24B", "alpha": 0.7},
            medianprops={"color": "#E45756", "linewidth": 2},
        )
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Коэффициент фазовой синхронизации")
        ax.set_xlabel("Птица")
        ax.set_title(
            f"Боксплот фазовой синхронизации по птицам: {flock_id}, режим={mode}, окно={window_size}"
        )
        ax.grid(axis="y", alpha=0.3)

        out = self.output_dir / f"{flock_id}_phase_{mode}_w{window_size}_boxplot.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        return out

    def plot_lag_std_boxplot(
        self, lag_df: pd.DataFrame, flock_id: str, mode: str, window_size: int
    ) -> Optional[Path]:
        """Строит boxplot СКО лагов по птицам."""
        subset = lag_df[
            (lag_df["id"] == flock_id)
            & (lag_df["coord"] == mode)
            & (lag_df["window"] == window_size)
        ].copy()
        if subset.empty:
            return None

        lag_std = (
            subset.groupby(["bird1", "bird2"], as_index=False)
            .agg(lag_std=("best_lag", lambda values: float(np.std(values, ddof=0))))
            .sort_values("lag_std")
        )
        if lag_std.empty:
            return None

        data, labels = self._pair_metric_boxplot_data(
            lag_std, value_col="lag_std", scale=1.0
        )
        if not data:
            return None

        fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(labels) + 2), 5))
        ax.boxplot(
            data,
            patch_artist=True,
            boxprops={"facecolor": "#B279A2", "alpha": 0.7},
            medianprops={"color": "#FF9DA6", "linewidth": 2},
        )
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel("СКО лучшего лага")
        ax.set_xlabel("Птица")
        ax.set_title(
            f"Боксплот СКО лагов по птицам: {flock_id}, режим={mode}, окно={window_size}"
        )
        ax.grid(axis="y", alpha=0.3)

        out = self.output_dir / f"{flock_id}_lag_std_{mode}_w{window_size}_boxplot.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        return out

    def plot_lag_std_histogram(
        self,
        lag_df: pd.DataFrame,
        flock_id: str,
        mode: str,
        window_size: int,
        bins: int = 20,
    ) -> Optional[Path]:
        """Строит гистограмму СКО лагов по всем парам птиц для выбранной стаи.

        Args:
            lag_df: Таблица с результатами оценки лагов по временным окнам.
            flock_id: Идентификатор стаи.
            mode: Ось или режим анализа.
            window_size: Размер окна, для которого строится гистограмма.
            bins: Число карманов гистограммы.

        Returns:
            Путь к сохраненному файлу, либо `None`, если подходящих данных нет.
        """
        subset = lag_df[
            (lag_df["id"] == flock_id)
            & (lag_df["coord"] == mode)
            & (lag_df["window"] == window_size)
        ].copy()
        if subset.empty:
            return None

        lag_std = (
            subset.groupby(["bird1", "bird2"], as_index=False)
            .agg(lag_std=("best_lag", lambda values: float(np.std(values, ddof=0))))
            .sort_values("lag_std")
        )
        if lag_std.empty:
            return None

        values = lag_std["lag_std"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.9)
        ax.axvline(np.median(values), color="#F58518", linestyle="--", linewidth=1.5, label=f"Медиана = {np.median(values):.2f}")
        ax.axvline(np.mean(values), color="#54A24B", linestyle="-.", linewidth=1.5, label=f"Среднее = {np.mean(values):.2f}")
        ax.set_xlabel("СКО лучшего лага по окнам")
        ax.set_ylabel("Число пар")
        ax.set_title(
            f"Гистограмма СКО лагов: {flock_id}, режим={mode}, окно={window_size}, карманы={bins}"
        )
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False)

        out = self.output_dir / f"{flock_id}_{mode}_w{window_size}_lag_std_hist_b{bins}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        return out
