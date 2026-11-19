"""Collect per-run Kendall Tau, relative-variation quantiles, and FWD."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


DEFAULT_BASE_PATH = (
    Path(__file__).resolve().parents[3]
    / "results/shepplogan/runs_2026-11-11_extrapolation_ablation"
)
METHODS = [
    ("Euclidean", "euclidean"),
    ("LAND", "land"),
    ("LAND RBF", "land_rbf"),
    ("CovCentroids", "cov_centroids"),
]
COLUMNS = [
    "Base Metric",
    "Finsler",
    "Eta",
    "Zeta",
    "Alpha",
    "Kendall Tau Median",
    "Kendall Tau 25%",
    "Kendall Tau 75%",
    "Relative Variation Median",
    "Relative Variation 25%",
    "Relative Variation 75%",
    "FWD",
]


def main(base_path: Path) -> pd.DataFrame:
    rows = []

    for alpha_path in base_path.glob("*/*/*"):
        eta = float(alpha_path.parents[1].name.split("_")[-1])
        zeta = float(alpha_path.parent.name.split("_")[-1])
        alpha = float(alpha_path.name.split("_")[-1])

        for base_metric, key in METHODS:
            for finsler in (False, True):
                suffix = "_randers" if finsler else ""
                metric_path = alpha_path / f"extrapolation_evaluation_{key}{suffix}"
                kendall_path = metric_path / "extrapolation_kendall_tau_test.pth"

                if kendall_path.is_file():
                    kendall_tau = torch.load(
                        kendall_path, map_location="cpu", weights_only=True
                    ).numpy()
                    tau = pd.Series(kendall_tau)
                    tau_median = tau.median()
                    tau_lower = tau.quantile(0.25)
                    tau_upper = tau.quantile(0.75)
                else:
                    tau_median = float("nan")
                    tau_lower = float("nan")
                    tau_upper = float("nan")

                time_path = metric_path / "extrapolated_time_test.pth"
                if time_path.is_file():
                    time_test = torch.load(
                        time_path, map_location="cpu", weights_only=True
                    ).numpy()
                    relative_variation = (
                        100 * (time_test[:, -1] - time_test[:, 0]) / time_test[:, 0]
                    )
                    variation = pd.Series(relative_variation)
                    variation_median = variation.median()
                    variation_lower = variation.quantile(0.25)
                    variation_upper = variation.quantile(0.75)
                else:
                    variation_median = float("nan")
                    variation_lower = float("nan")
                    variation_upper = float("nan")

                fwd_path = metric_path / "metrics_results_t0_25.json"
                try:
                    with fwd_path.open() as file:
                        fwd = json.load(file)["fwd_test"]
                except FileNotFoundError:
                    fwd = float("nan")

                rows.append(
                    {
                        "Base Metric": base_metric,
                        "Finsler": finsler,
                        "Eta": eta,
                        "Zeta": zeta,
                        "Alpha": alpha,
                        "Kendall Tau Median": tau_median,
                        "Kendall Tau 25%": tau_lower,
                        "Kendall Tau 75%": tau_upper,
                        "Relative Variation Median": variation_median,
                        "Relative Variation 25%": variation_lower,
                        "Relative Variation 75%": variation_upper,
                        "FWD": fwd,
                    }
                )

    return pd.DataFrame.from_records(rows, columns=COLUMNS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        type=Path,
        default=DEFAULT_BASE_PATH,
        help="Root directory containing the eta/zeta/alpha result folders",
    )
    args = parser.parse_args()

    summary = main(args.base_path)
    output_path = args.base_path / "extrapolation_ablation_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved {len(summary)} rows to {output_path}")
