from __future__ import annotations

from pathlib import Path
from typing import List
import sys

import pandas as pd

from tds_core import BatchPipeline, TDSParams, summarize_flocks, summarize_phase
from visualization import FlockVisualizer


def discover_csv_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("*.csv"))


def main() -> None:
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else base_dir / "tds_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = discover_csv_files(base_dir)
    if not csv_files:
        raise FileNotFoundError(f"CSV files not found in {base_dir.resolve()}")

    params_grid = [
        TDSParams(
            window_size=60,
            step_size=10,
            max_lag=10,
            lag_tolerance=1,
            min_stable_windows=4,
        ),
        TDSParams(
            window_size=120,
            step_size=20,
            max_lag=15,
            lag_tolerance=1,
            min_stable_windows=4,
        ),
        TDSParams(
            window_size=240,
            step_size=40,
            max_lag=20,
            lag_tolerance=2,
            min_stable_windows=4,
        ),
    ]
    modes = ["x", "y"]

    pipeline = BatchPipeline(params_grid=params_grid)
    pairwise_df, lag_df, phase_df, phase_trace_df, flocks = pipeline.run(
        csv_files=csv_files, modes=modes
    )
    summary_df = summarize_flocks(pairwise_df)
    phase_summary_df = summarize_phase(phase_df)

    pairwise_df.to_csv(output_dir / "pairwise_tds_results.csv", index=False)
    lag_df.to_csv(output_dir / "lag_trace_results.csv", index=False)
    summary_df.to_csv(output_dir / "flock_summary.csv", index=False)
    phase_df.to_csv(output_dir / "pairwise_phase_results.csv", index=False)
    phase_trace_df.to_csv(output_dir / "phase_trace_results.csv", index=False)
    phase_summary_df.to_csv(output_dir / "phase_summary.csv", index=False)

    visualizer = FlockVisualizer(output_dir)
    for flock in flocks.values():
        visualizer.plot_trajectory(flock)
        visualizer.plot_coordinate_overview(flock)

    # heatmaps for the middle window size for a compact default report
    heatmap_window = 60
    for flock_id in flocks:
        for mode in modes:
            visualizer.plot_tds_heatmap(
                pairwise_df, flock_id=flock_id, mode=mode, window_size=heatmap_window
            )
            visualizer.plot_phase_heatmap(
                phase_df, flock_id=flock_id, mode=mode, window_size=heatmap_window
            )

    for mode in modes:
        visualizer.plot_group_comparison(summary_df, mode=mode)

    # representative traces and distributions for each flock and each coordinate mode
    rep = pairwise_df[pairwise_df["window"] == heatmap_window].sort_values(
        "tds", ascending=False
    )
    for flock_id, flock_df in rep.groupby("id"):
        for mode in modes:
            mode_df = flock_df[flock_df["coord"] == mode]
            if mode_df.empty:
                continue

            row = mode_df.iloc[0]
            visualizer.plot_lag_trace(
                lag_df,
                flock_id=flock_id,
                bird_i=row["bird1"],
                bird_j=row["bird2"],
                mode=mode,
                window_size=heatmap_window,
            )
            visualizer.plot_phase_trace(
                phase_trace_df,
                flock_id=flock_id,
                bird_i=row["bird1"],
                bird_j=row["bird2"],
                mode=mode,
                window_size=heatmap_window,
            )
            visualizer.plot_tds_boxplot(
                pairwise_df,
                flock_id=flock_id,
                mode=mode,
                window_size=heatmap_window,
            )
            visualizer.plot_phase_boxplot(
                phase_df,
                flock_id=flock_id,
                mode=mode,
                window_size=heatmap_window,
            )
            visualizer.plot_lag_std_boxplot(
                lag_df,
                flock_id=flock_id,
                mode=mode,
                window_size=heatmap_window,
            )
            visualizer.plot_lag_std_histogram(
                lag_df,
                flock_id=flock_id,
                mode=mode,
                window_size=heatmap_window,
                bins=15,
            )
            visualizer.plot_lag_std_histogram(
                lag_df,
                flock_id=flock_id,
                mode=mode,
                window_size=heatmap_window,
                bins=18,
            )
            visualizer.plot_lag_std_histogram(
                lag_df,
                flock_id=flock_id,
                mode=mode,
                window_size=heatmap_window,
                bins=20,
            )

    print(f"Processed {len(flocks)} flocks")
    print(f"Results saved to: {output_dir.resolve()}")
    print("Top summary rows:")
    print(summary_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
