from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import correlation_lags, correlate, hilbert


SignalMode = Literal["x", "y", "dx", "dy"]


@dataclass(frozen=True)
class TDSParams:
    window_size: int
    step_size: int
    max_lag: int
    lag_tolerance: int = 1
    min_stable_windows: int = 4
    stability_window_size: int = 5
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
            return "1"
        if lower.startswith("hf"):
            return "2"
        return "0"


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
                    "id": flock.flock_id,
                    "type": flock.group,
                    "coord": mode,
                    "window": self.params.window_size,
                    "step": self.params.step_size,
                    "lag": self.params.max_lag,
                    "bird1": bird_i,
                    "bird2": bird_j,
                    "tds": int(score * 100),
                    "R": round(float(np.nanmean(np.abs(corr_series))) if len(corr_series) else np.nan, 2),
                }
            )
            for window_idx, (lag, corr) in enumerate(zip(lag_series, corr_series)):
                lag_records.append(
                    {
                        "id": flock.flock_id,
                        "type": flock.group,
                        "coord": mode,
                        "window": self.params.window_size,
                        "bird1": bird_i,
                        "bird2": bird_j,
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

        xz = (xw - xw.mean()) / (xw.std(ddof=1) + 1e-12)
        yz = (yw - yw.mean()) / (yw.std(ddof=1) + 1e-12)
        full = correlate(xz, yz, mode="full")
        lags = correlation_lags(len(xz), len(yz), mode="full")
        valid = np.abs(lags) <= self.params.max_lag
        lags = lags[valid]
        corr_vals = full[valid]
        if len(corr_vals) == 0:
            return 0, np.nan

        scores = np.abs(corr_vals) if self.params.use_absolute_correlation else corr_vals
        candidate_idx = np.flatnonzero(scores == np.nanmax(scores))
        best_idx = int(candidate_idx[np.argmin(np.abs(lags[candidate_idx]))])
        best_lag = int(lags[best_idx])
        xa, ya = self._align_lag(xw, yw, best_lag)
        return best_lag, self._safe_corr(xa, ya)

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
        stable_series = self._lag_stability_series(lag_arr)
        if len(stable_series) == 0:
            return np.nan
        return float(np.mean(stable_series))

    def _lag_stability_series(self, lag_arr: np.ndarray) -> np.ndarray:
        lag_arr = np.asarray(lag_arr, dtype=float)
        if len(lag_arr) == 0:
            return np.array([], dtype=float)
        if np.all(np.isnan(lag_arr)):
            return np.zeros(len(lag_arr), dtype=float)

        window_length = min(max(1, self.params.stability_window_size), len(lag_arr))
        min_stable = min(max(1, self.params.min_stable_windows), window_length)
        tol = float(self.params.lag_tolerance)
        window_number = len(lag_arr) - window_length + 1
        stable = np.zeros(window_number, dtype=float)

        comb_indices = list(combinations(range(window_length), min_stable))
        for start in range(window_number):
            clip = lag_arr[start : start + window_length]
            if np.isnan(clip).all():
                continue
            for idx in comb_indices:
                sample = clip[list(idx)]
                if np.isnan(sample).any():
                    continue
                if float(np.max(sample) - np.min(sample)) <= tol:
                    stable[start] = 1.0
                    break

        delay = window_length // 2
        pad_end = len(lag_arr) - len(stable) - delay
        start_pad = np.full(delay, stable[0], dtype=float)
        end_pad = np.full(max(0, pad_end), stable[-1], dtype=float)
        return np.concatenate((start_pad, stable, end_pad))


class PhaseSyncAnalyzer:
    def __init__(self, params: TDSParams):
        self.params = params

    def analyze_flock(self, flock: FlockData, mode: SignalMode) -> Tuple[pd.DataFrame, pd.DataFrame]:
        signals = self._build_signal_table(flock, mode)
        records: List[dict] = []
        trace_records: List[dict] = []

        for bird_i, bird_j in combinations(signals.columns, 2):
            mean_phase_sync, phase_sync_series = self.compute_phase_sync(
                signals[bird_i].to_numpy(), signals[bird_j].to_numpy()
            )
            records.append(
                {
                    "id": flock.flock_id,
                    "type": flock.group,
                    "coord": mode,
                    "window": self.params.window_size,
                    "step": self.params.step_size,
                    "bird1": bird_i,
                    "bird2": bird_j,
                    "phase_sync": (
                        round(float(mean_phase_sync), 4)
                        if not np.isnan(mean_phase_sync)
                        else np.nan
                    ),
                }
            )
            for window_idx, phase_sync in enumerate(phase_sync_series):
                trace_records.append(
                    {
                        "id": flock.flock_id,
                        "type": flock.group,
                        "coord": mode,
                        "window": self.params.window_size,
                        "bird1": bird_i,
                        "bird2": bird_j,
                        "window_index": window_idx,
                        "phase_sync": (
                            round(float(phase_sync), 4)
                            if not np.isnan(phase_sync)
                            else np.nan
                        ),
                    }
                )

        return pd.DataFrame(records), pd.DataFrame(trace_records)

    def _build_signal_table(self, flock: FlockData, mode: SignalMode) -> pd.DataFrame:
        builder = SignalBuilder()
        signals = {bird_id: builder.build_signal(df, mode) for bird_id, df in flock.birds.items()}
        signal_df = pd.DataFrame(signals)
        signal_df = signal_df.interpolate(limit_direction="both").dropna(axis=1, how="all")
        return signal_df

    def compute_phase_sync(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        p = self.params
        n = min(len(x), len(y))
        x = np.asarray(x[:n], dtype=float)
        y = np.asarray(y[:n], dtype=float)
        if n < p.window_size:
            return np.nan, np.array([])

        phase_sync_values: List[float] = []
        starts = range(0, n - p.window_size + 1, p.step_size)
        for start in starts:
            end = start + p.window_size
            phase_sync_values.append(self._window_phase_sync(x[start:end], y[start:end]))

        phase_sync_arr = np.asarray(phase_sync_values, dtype=float)
        if len(phase_sync_arr) == 0 or np.all(np.isnan(phase_sync_arr)):
            return np.nan, phase_sync_arr
        return float(np.nanmean(phase_sync_arr)), phase_sync_arr

    @staticmethod
    def _window_phase_sync(xw: np.ndarray, yw: np.ndarray) -> float:
        xw = np.asarray(xw, dtype=float)
        yw = np.asarray(yw, dtype=float)
        if len(xw) < 3 or np.std(xw) < 1e-12 or np.std(yw) < 1e-12:
            return np.nan

        phase_x = np.angle(hilbert(xw - np.mean(xw)))
        phase_y = np.angle(hilbert(yw - np.mean(yw)))
        phase_diff = phase_x - phase_y
        return float(np.abs(np.mean(np.exp(1j * phase_diff))))


class BatchPipeline:
    def __init__(self, params_grid: Sequence[TDSParams]):
        self.params_grid = list(params_grid)
        self.loader = PigeonCSVLoader()

    def run(
        self, csv_files: Sequence[str | Path], modes: Sequence[SignalMode]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, FlockData]]:
        flocks = {Path(path).stem: self.loader.load(path) for path in csv_files}
        pair_frames: List[pd.DataFrame] = []
        lag_frames: List[pd.DataFrame] = []
        phase_frames: List[pd.DataFrame] = []
        phase_trace_frames: List[pd.DataFrame] = []

        for params in self.params_grid:
            analyzer = TDSAnalyzer(params)
            phase_analyzer = PhaseSyncAnalyzer(params)
            for flock in flocks.values():
                for mode in modes:
                    pair_df, lag_df = analyzer.analyze_flock(flock, mode)
                    phase_df, phase_trace_df = phase_analyzer.analyze_flock(flock, mode)
                    pair_frames.append(pair_df)
                    lag_frames.append(lag_df)
                    phase_frames.append(phase_df)
                    phase_trace_frames.append(phase_trace_df)

        pairwise_results = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
        lag_results = pd.concat(lag_frames, ignore_index=True) if lag_frames else pd.DataFrame()
        phase_results = pd.concat(phase_frames, ignore_index=True) if phase_frames else pd.DataFrame()
        phase_trace_results = (
            pd.concat(phase_trace_frames, ignore_index=True)
            if phase_trace_frames
            else pd.DataFrame()
        )
        return pairwise_results, lag_results, phase_results, phase_trace_results, flocks


def summarize_flocks(pairwise_results: pd.DataFrame) -> pd.DataFrame:
    if pairwise_results.empty:
        return pd.DataFrame()
    summary = (
        pairwise_results.groupby(["id", "type", "coord", "window", "step", "lag"], as_index=False)
        .agg(
            mean_tds=("tds", "mean"),
            median_tds=("tds", "median"),
            std_tds=("tds", "std"),
            n_pairs=("tds", "count"),
            mean_R=("R", "mean"),
        )
    )
    return summary.sort_values(["type", "coord", "window", "id"]).reset_index(drop=True)


def summarize_phase(phase_results: pd.DataFrame) -> pd.DataFrame:
    if phase_results.empty:
        return pd.DataFrame()
    summary = (
        phase_results.groupby(["id", "type", "coord", "window", "step"], as_index=False)
        .agg(
            mean_phase_sync=("phase_sync", "mean"),
            median_phase_sync=("phase_sync", "median"),
            std_phase_sync=("phase_sync", "std"),
            n_pairs=("phase_sync", "count"),
        )
    )
    return summary.sort_values(["type", "coord", "window", "id"]).reset_index(drop=True)
