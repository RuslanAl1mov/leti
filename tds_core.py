from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SignalMode = Literal["x", "y", "dx", "dy"]


@dataclass(frozen=True)
class TDSParams:
    window_size: int
    step_size: int
    max_lag: int
    lag_tolerance: int = 1
    min_stable_windows: int = 4
    use_absolute_correlation: bool = True


@dataclass
class FlockData:
    flock_id: str
    group: str
    time_seconds: np.ndarray
    birds: Dict[str, pd.DataFrame]

    @property
    def bird_ids(self) -> List[str]:
        return sorted(self.birds.keys())


class PigeonCSVLoader:
    """Loads a flock CSV where columns are stored as X_bird, Y_bird pairs."""

    TIME_COLUMN_PATTERN = re.compile(r"^Time", re.IGNORECASE)
    X_COLUMN_PATTERN = re.compile(r"^X_(?P<bird>[A-Za-z0-9]+)\(m\)$")
    Y_COLUMN_PATTERN = re.compile(r"^Y_(?P<bird>[A-Za-z0-9]+)\(m\)$")

    def load(self, csv_path: str | Path) -> FlockData:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path, sep=";")
        df.columns = [str(c).strip() for c in df.columns]

        time_col = self._find_time_column(df.columns)
        if time_col is None:
            raise ValueError(f"Time column not found in {csv_path.name}")

        df = df.copy()
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])

        numeric_cols = [c for c in df.columns if c != time_col]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Deduplicate repeated timestamps by averaging coordinates.
        df = df.groupby(time_col, as_index=False).mean(numeric_only=True)
        df = df.sort_values(time_col).reset_index(drop=True)

        time_seconds = df[time_col].to_numpy(dtype=float) / 100.0

        birds = self._extract_birds(df)
        flock_id = csv_path.stem
        group = self._infer_group(flock_id)

        return FlockData(
            flock_id=flock_id,
            group=group,
            time_seconds=time_seconds,
            birds=birds,
        )

    def _find_time_column(self, columns: Sequence[str]) -> Optional[str]:
        for col in columns:
            if self.TIME_COLUMN_PATTERN.match(col):
                return col
        return None

    def _extract_birds(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        x_cols: Dict[str, str] = {}
        y_cols: Dict[str, str] = {}
        for col in df.columns:
            x_match = self.X_COLUMN_PATTERN.match(col)
            y_match = self.Y_COLUMN_PATTERN.match(col)
            if x_match:
                x_cols[x_match.group("bird")] = col
            if y_match:
                y_cols[y_match.group("bird")] = col

        birds: Dict[str, pd.DataFrame] = {}
        for bird_id in sorted(set(x_cols) & set(y_cols)):
            bird_df = pd.DataFrame(
                {
                    "x": df[x_cols[bird_id]].astype(float),
                    "y": df[y_cols[bird_id]].astype(float),
                }
            )
            birds[bird_id] = bird_df.interpolate(limit_direction="both")
        if not birds:
            raise ValueError("No bird coordinate pairs found.")
        return birds

    @staticmethod
    def _infer_group(flock_id: str) -> str:
        lower = flock_id.lower()
        if lower.startswith("ff"):
            return "free_flight"
        if lower.startswith("hf"):
            return "homing_flight"
        return "unknown"


class SignalBuilder:
    def __init__(self, smooth_window: int = 5):
        self.smooth_window = max(1, int(smooth_window))

    def build_signal(self, bird_df: pd.DataFrame, mode: SignalMode) -> np.ndarray:
        x = bird_df["x"].to_numpy(dtype=float)
        y = bird_df["y"].to_numpy(dtype=float)

        if mode == "x":
            signal = x
        elif mode == "y":
            signal = y
        elif mode == "dx":
            signal = np.diff(x, prepend=x[0])
        elif mode == "dy":
            signal = np.diff(y, prepend=y[0])
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if self.smooth_window > 1:
            signal = (
                pd.Series(signal)
                .rolling(self.smooth_window, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )
        return signal


class TDSAnalyzer:
    def __init__(self, params: TDSParams):
        self.params = params

    def analyze_flock(self, flock: FlockData, mode: SignalMode) -> Tuple[pd.DataFrame, pd.DataFrame]:
        signals = self._build_signal_table(flock, mode)
        records: List[dict] = []
        lag_records: List[dict] = []

        for bird_i, bird_j in combinations(signals.columns, 2):
            score, lag_series, corr_series = self.compute_tds(signals[bird_i].to_numpy(), signals[bird_j].to_numpy())
            records.append(
                {
                    "flock_id": flock.flock_id,
                    "group": flock.group,
                    "mode": mode,
                    "window_size": self.params.window_size,
                    "step_size": self.params.step_size,
                    "max_lag": self.params.max_lag,
                    "bird_i": bird_i,
                    "bird_j": bird_j,
                    "tds": score,
                    "mean_abs_corr": float(np.nanmean(np.abs(corr_series))) if len(corr_series) else np.nan,
                }
            )
            for window_idx, (lag, corr) in enumerate(zip(lag_series, corr_series)):
                lag_records.append(
                    {
                        "flock_id": flock.flock_id,
                        "group": flock.group,
                        "mode": mode,
                        "window_size": self.params.window_size,
                        "bird_i": bird_i,
                        "bird_j": bird_j,
                        "window_index": window_idx,
                        "best_lag": lag,
                        "best_corr": corr,
                    }
                )

        return pd.DataFrame(records), pd.DataFrame(lag_records)

    def _build_signal_table(self, flock: FlockData, mode: SignalMode) -> pd.DataFrame:
        builder = SignalBuilder()
        signals = {bird_id: builder.build_signal(df, mode) for bird_id, df in flock.birds.items()}
        signal_df = pd.DataFrame(signals)
        signal_df = signal_df.interpolate(limit_direction="both").dropna(axis=1, how="all")
        return signal_df

    def compute_tds(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        p = self.params
        n = min(len(x), len(y))
        x = np.asarray(x[:n], dtype=float)
        y = np.asarray(y[:n], dtype=float)
        if n < p.window_size:
            return np.nan, np.array([]), np.array([])

        lag_values: List[int] = []
        corr_values: List[float] = []
        starts = range(0, n - p.window_size + 1, p.step_size)
        for start in starts:
            end = start + p.window_size
            lag, corr = self._best_lag_in_window(x[start:end], y[start:end])
            lag_values.append(lag)
            corr_values.append(corr)

        lag_arr = np.asarray(lag_values, dtype=float)
        corr_arr = np.asarray(corr_values, dtype=float)
        tds_score = self._lag_stability_score(lag_arr)
        return float(tds_score), lag_arr, corr_arr

    def _best_lag_in_window(self, xw: np.ndarray, yw: np.ndarray) -> Tuple[int, float]:
        xw = np.asarray(xw, dtype=float)
        yw = np.asarray(yw, dtype=float)
        if np.std(xw) < 1e-12 or np.std(yw) < 1e-12:
            return 0, np.nan

        xz = (xw - xw.mean()) / (xw.std() + 1e-12)
        yz = (yw - yw.mean()) / (yw.std() + 1e-12)
        full = np.correlate(xz, yz, mode="full") / len(xz)
        center = len(xz) - 1
        lags = np.arange(-self.params.max_lag, self.params.max_lag + 1)
        valid_idx = center + lags
        valid = (valid_idx >= 0) & (valid_idx < len(full))
        lags = lags[valid]
        corr_vals = full[valid_idx[valid]]
        if len(corr_vals) == 0:
            return 0, np.nan
        scores = np.abs(corr_vals) if self.params.use_absolute_correlation else corr_vals
        best_idx = int(np.nanargmax(scores))
        return int(lags[best_idx]), float(corr_vals[best_idx])

    @staticmethod
    def _align_lag(x: np.ndarray, y: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
        if lag > 0:
            return x[:-lag], y[lag:]
        if lag < 0:
            return x[-lag:], y[:lag]
        return x, y

    @staticmethod
    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    def _lag_stability_score(self, lag_arr: np.ndarray) -> float:
        if len(lag_arr) == 0:
            return np.nan

        stable_mask = np.zeros(len(lag_arr), dtype=bool)
        min_run = self.params.min_stable_windows
        tol = self.params.lag_tolerance

        start = 0
        while start < len(lag_arr):
            end = start + 1
            while end < len(lag_arr) and abs(lag_arr[end] - lag_arr[start]) <= tol:
                end += 1
            if end - start >= min_run:
                stable_mask[start:end] = True
            start = end
        return float(stable_mask.mean())


class BatchPipeline:
    def __init__(self, params_grid: Sequence[TDSParams]):
        self.params_grid = list(params_grid)
        self.loader = PigeonCSVLoader()

    def run(self, csv_files: Sequence[str | Path], modes: Sequence[SignalMode]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, FlockData]]:
        flocks = {Path(path).stem: self.loader.load(path) for path in csv_files}
        pair_frames: List[pd.DataFrame] = []
        lag_frames: List[pd.DataFrame] = []

        for params in self.params_grid:
            analyzer = TDSAnalyzer(params)
            for flock in flocks.values():
                for mode in modes:
                    pair_df, lag_df = analyzer.analyze_flock(flock, mode)
                    pair_frames.append(pair_df)
                    lag_frames.append(lag_df)

        pairwise_results = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
        lag_results = pd.concat(lag_frames, ignore_index=True) if lag_frames else pd.DataFrame()
        return pairwise_results, lag_results, flocks


def summarize_flocks(pairwise_results: pd.DataFrame) -> pd.DataFrame:
    if pairwise_results.empty:
        return pd.DataFrame()
    summary = (
        pairwise_results.groupby(["flock_id", "group", "mode", "window_size", "step_size", "max_lag"], as_index=False)
        .agg(
            mean_tds=("tds", "mean"),
            median_tds=("tds", "median"),
            std_tds=("tds", "std"),
            n_pairs=("tds", "count"),
            mean_abs_corr=("mean_abs_corr", "mean"),
        )
    )
    return summary.sort_values(["group", "mode", "window_size", "flock_id"]).reset_index(drop=True)
