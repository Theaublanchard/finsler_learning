"""Shared result loading and formatting helpers for Shepp-Logan analysis notebooks."""

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch


METHODS = [
    ("Euclidean", "euclidean"),
    ("LAND", "land"),
    ("LAND RBF", "land_rbf"),
    ("CovCentroids", "cov_centroids"),
    ("PBCentroids", "pullback_centroids"),
]

BASE_METHODS = METHODS + [
    ("Wasserstein", "wasserstein"),
    ("HyperSpherical", "spherical"),
]


def _load_tensor(path: Path) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=True)
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def load_interpolation_metrics(base_path: Path, key: str) -> pd.DataFrame:
    method_path = base_path / f"geodesic_evaluation_{key}_randers"
    geodesic_distance = _load_tensor(method_path / "geo_dst_list_test.pth")
    kendall_tau = _load_tensor(method_path / "geo_kendall_tau_list_test.pth")

    if kendall_tau.ndim > 1:
        kendall_tau = kendall_tau[0]

    return pd.DataFrame(
        {
            "Geodesic Distance": geodesic_distance,
            "Kendall Tau": kendall_tau,
        }
    )


def load_extrapolation_metrics(base_path: Path, key: str) -> pd.DataFrame:
    method_path = base_path / f"extrapolation_evaluation_{key}_randers"
    kendall_tau = _load_tensor(method_path / "extrapolation_kendall_tau_test.pth")
    time_test = _load_tensor(method_path / "extrapolated_time_test.pth")
    relative_variation = 100 * (time_test[:, -1] - time_test[:, 0]) / time_test[:, 0]

    return pd.DataFrame(
        {
            "Kendall Tau": kendall_tau,
            "Relative Variation": relative_variation,
            "First Time": time_test[:, 0],
        }
    )


def load_quality(base_path: Path, key: str, extrapolation: bool = False) -> dict:
    if extrapolation:
        method_path = base_path / f"extrapolation_evaluation_{key}_randers"
        json_name = "metrics_results_t0_25.json"
    else:
        method_path = base_path / f"geodesic_evaluation_{key}_randers"
        json_name = "metrics_results.json"

    with (method_path / json_name).open() as file:
        metrics = json.load(file)
    return {
        "FWD": metrics.get("fwd_test"),
    }


def load_base_geometry_metrics(
    base_path: Path,
    key: str,
    finsler: bool,
    extrapolation: bool,
) -> pd.DataFrame:
    suffix = "_randers" if finsler else ""
    if extrapolation:
        method_path = base_path / f"extrapolation_evaluation_{key}{suffix}"
        kendall_tau = _load_tensor(method_path / "extrapolation_kendall_tau_test.pth")
        time_test = _load_tensor(method_path / "extrapolated_time_test.pth")
        relative_variation = (
            100 * (time_test[:, -1] - time_test[:, 0]) / time_test[:, 0]
        )
        return pd.DataFrame(
            {
                "Kendall Tau": kendall_tau,
                "Relative Variation": relative_variation,
                "First Time": time_test[:, 0],
            }
        )

    method_path = base_path / f"geodesic_evaluation_{key}{suffix}"
    kendall_tau = _load_tensor(method_path / "geo_kendall_tau_list_test.pth")
    if kendall_tau.ndim > 1:
        kendall_tau = kendall_tau[0]
    geodesic_distance = _load_tensor(method_path / "geo_dst_list_test.pth")
    return pd.DataFrame(
        {
            "Geodesic Distance": geodesic_distance,
            "Kendall Tau": kendall_tau,
        }
    )


def load_base_geometry_quality(
    base_path: Path,
    key: str,
    finsler: bool,
    extrapolation: bool,
) -> dict:
    suffix = "_randers" if finsler else ""
    if extrapolation:
        method_path = base_path / f"extrapolation_evaluation_{key}{suffix}"
        json_name = "metrics_results_t0_25.json"
    else:
        method_path = base_path / f"geodesic_evaluation_{key}{suffix}"
        json_name = "metrics_results.json"
    with (method_path / json_name).open() as file:
        metrics = json.load(file)
    return {"FWD": metrics.get("fwd_test")}


def collect_base_geometry_metrics(
    base_path: Path,
    methods: Iterable[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interpolation_frames = []
    extrapolation_frames = []
    for method_name, key in methods:
        for finsler in (False, True):
            try:
                frame = load_base_geometry_metrics(
                    base_path, key, finsler=finsler, extrapolation=False
                )
                frame["Base Metric"] = method_name
                frame["Finsler"] = finsler
                interpolation_frames.append(frame)
            except FileNotFoundError as error:
                print(f"Interpolation metrics missing for {method_name}, Finsler={finsler}: {error}")

            try:
                frame = load_base_geometry_metrics(
                    base_path, key, finsler=finsler, extrapolation=True
                )
                frame["Base Metric"] = method_name
                frame["Finsler"] = finsler
                extrapolation_frames.append(frame)
            except FileNotFoundError as error:
                print(f"Extrapolation metrics missing for {method_name}, Finsler={finsler}: {error}")

    return (
        pd.concat(interpolation_frames, ignore_index=True),
        pd.concat(extrapolation_frames, ignore_index=True),
    )


def collect_base_geometry_quality(
    base_path: Path,
    methods: Iterable[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interpolation_frames = []
    extrapolation_frames = []
    for method_name, key in methods:
        for finsler in (False, True):
            try:
                frame = pd.DataFrame([
                    load_base_geometry_quality(
                        base_path, key, finsler=finsler, extrapolation=False
                    )
                ])
                frame["Base Metric"] = method_name
                frame["Finsler"] = finsler
                interpolation_frames.append(frame)
            except FileNotFoundError:
                pass

            try:
                frame = pd.DataFrame([
                    load_base_geometry_quality(
                        base_path, key, finsler=finsler, extrapolation=True
                    )
                ])
                frame["Base Metric"] = method_name
                frame["Finsler"] = finsler
                extrapolation_frames.append(frame)
            except FileNotFoundError:
                pass
    return (
        pd.concat(interpolation_frames, ignore_index=True),
        pd.concat(extrapolation_frames, ignore_index=True),
    )


def load_biased_extrapolation_metrics(
    base_path: Path,
    key: str,
    biased: bool,
) -> pd.DataFrame:
    suffix = "_biased_randers" if biased else "_randers"
    method_path = base_path / f"extrapolation_evaluation_{key}{suffix}"
    kendall_tau = _load_tensor(method_path / "extrapolation_kendall_tau_test.pth")
    time_test = _load_tensor(method_path / "extrapolated_time_test.pth")
    relative_variation = (
        100 * (time_test[:, -1] - time_test[:, 0]) / time_test[:, 0]
    )
    return pd.DataFrame(
        {
            "Kendall Tau": kendall_tau,
            "Relative Variation": relative_variation,
            "First Time": time_test[:, 0],
        }
    )


def load_biased_extrapolation_quality(
    base_path: Path,
    key: str,
    biased: bool,
) -> dict:
    suffix = "_biased_randers" if biased else "_randers"
    method_path = base_path / f"extrapolation_evaluation_{key}{suffix}"
    with (method_path / "metrics_results_t0_25.json").open() as file:
        metrics = json.load(file)
    return {"FWD": metrics.get("fwd_test")}


def collect_biased_extrapolation(
    base_path: Path,
    methods: Iterable[tuple[str, str]] = METHODS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    quality_frames = []

    for method_name, key in methods:
        for biased in (False, True):
            try:
                frame = load_biased_extrapolation_metrics(
                    base_path, key, biased=biased
                )
                frame["Base Metric"] = method_name
                frame["Finsler"] = not biased
                frame["Biased"] = biased
                metric_frames.append(frame)
            except FileNotFoundError as error:
                print(
                    f"Biased extrapolation metrics missing for "
                    f"{method_name}, Biased={biased}: {error}"
                )

            try:
                quality = pd.DataFrame([
                    load_biased_extrapolation_quality(
                        base_path, key, biased=biased
                    )
                ])
                quality["Base Metric"] = method_name
                quality["Finsler"] = not biased
                quality["Biased"] = biased
                quality_frames.append(quality)
            except FileNotFoundError:
                pass

    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(quality_frames, ignore_index=True),
    )


def merge_biased_extrapolation_table(
    extrapolation: pd.DataFrame,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    summary = summarize_median_iqr(
        extrapolation,
        ["Base Metric", "Finsler", "Biased"],
        ["Kendall Tau", "Relative Variation"],
    )
    combined = summary.merge(
        quality.rename(columns={"FWD": "FWD Extra"}),
        on=["Base Metric", "Finsler", "Biased"],
        how="outer",
    )
    return combined.rename(
        columns={
            "Kendall Tau Median": "Kendall Tau Median Extra",
            "Kendall Tau IQR": "Kendall Tau IQR Extra",
            "Relative Variation Median": "Relative Variation Median Extra",
            "Relative Variation IQR": "Relative Variation IQR Extra",
        }
    )


def collect_setup_metrics(
    setup_paths: Mapping[str, Path],
    methods: Iterable[tuple[str, str]] = METHODS,
    setup_column: str = "Setting",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interpolation_frames = []
    extrapolation_frames = []

    for setup, base_path in setup_paths.items():
        for method_name, key in methods:
            try:
                frame = load_interpolation_metrics(base_path, key)
                frame["Base Metric"] = method_name
                frame[setup_column] = setup
                interpolation_frames.append(frame)
            except FileNotFoundError as error:
                print(f"Interpolation metrics missing for {setup}: {method_name}: {error}")

            try:
                frame = load_extrapolation_metrics(base_path, key)
                frame["Base Metric"] = method_name
                frame[setup_column] = setup
                extrapolation_frames.append(frame)
            except FileNotFoundError as error:
                print(f"Extrapolation metrics missing for {setup}: {method_name}: {error}")

    interpolation = pd.concat(interpolation_frames, ignore_index=True)
    extrapolation = pd.concat(extrapolation_frames, ignore_index=True)
    return interpolation, extrapolation


def collect_quality(
    setup_paths: Mapping[str, Path],
    methods: Iterable[tuple[str, str]] = METHODS,
    setup_column: str = "Setting",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interpolation_frames = []
    extrapolation_frames = []

    for setup, base_path in setup_paths.items():
        for method_name, key in methods:
            try:
                frame = pd.DataFrame([load_quality(base_path, key)])
                frame["Base Metric"] = method_name
                frame[setup_column] = setup
                interpolation_frames.append(frame)
            except FileNotFoundError as error:
                print(f"Interpolation quality missing for {setup}: {method_name}: {error}")

            try:
                frame = pd.DataFrame([load_quality(base_path, key, extrapolation=True)])
                frame["Base Metric"] = method_name
                frame[setup_column] = setup
                extrapolation_frames.append(frame)
            except FileNotFoundError as error:
                print(f"Extrapolation quality missing for {setup}: {method_name}: {error}")

    interpolation = pd.concat(interpolation_frames, ignore_index=True)
    extrapolation = pd.concat(extrapolation_frames, ignore_index=True)
    return interpolation, extrapolation


def summarize_median_iqr(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: Iterable[str],
) -> pd.DataFrame:
    frame = frame.copy()
    for column in metric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    rows = []
    for keys, group in frame.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        for column in metric_columns:
            values = group[column].dropna()
            row[f"{column} Median"] = values.median()
            row[f"{column} IQR"] = (
                values.quantile(0.25),
                values.quantile(0.75),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def merge_base_geometry_table(
    interpolation: pd.DataFrame,
    extrapolation: pd.DataFrame,
    interpolation_quality: pd.DataFrame,
    extrapolation_quality: pd.DataFrame,
) -> pd.DataFrame:
    inter_summary = summarize_median_iqr(
        interpolation,
        ["Base Metric", "Finsler"],
        ["Kendall Tau"],
    )
    extra_summary = summarize_median_iqr(
        extrapolation,
        ["Base Metric", "Finsler"],
        ["Kendall Tau", "Relative Variation"],
    )

    combined = inter_summary.merge(
        extra_summary,
        on=["Base Metric", "Finsler"],
        how="outer",
    )
    combined = combined.merge(
        interpolation_quality.rename(columns={"FWD": "FWD Inter"}),
        on=["Base Metric", "Finsler"],
        how="outer",
    )
    combined = combined.merge(
        extrapolation_quality.rename(columns={"FWD": "FWD Extra"}),
        on=["Base Metric", "Finsler"],
        how="outer",
    )
    return combined.rename(
        columns={
            "Kendall Tau Median_x": "Kendall Tau Median Inter",
            "Kendall Tau IQR_x": "Kendall Tau IQR Inter",
            "Kendall Tau Median_y": "Kendall Tau Median Extra",
            "Kendall Tau IQR_y": "Kendall Tau IQR Extra",
            "Relative Variation Median": "Relative Variation Median Extra",
            "Relative Variation IQR": "Relative Variation IQR Extra",
        }
    )


def merge_combined_table(
    interpolation: pd.DataFrame,
    extrapolation: pd.DataFrame,
    interpolation_quality: pd.DataFrame,
    extrapolation_quality: pd.DataFrame,
    setup_column: str,
) -> pd.DataFrame:
    inter_summary = summarize_median_iqr(
        interpolation,
        ["Base Metric", setup_column],
        ["Kendall Tau"],
    )
    extra_summary = summarize_median_iqr(
        extrapolation,
        ["Base Metric", setup_column],
        ["Kendall Tau", "Relative Variation"],
    )

    combined = inter_summary.merge(extra_summary, on=["Base Metric", setup_column], how="outer")
    combined = combined.merge(
        interpolation_quality.rename(columns={"FWD": "FWD Inter"}),
        on=["Base Metric", setup_column],
        how="outer",
    )
    combined = combined.merge(
        extrapolation_quality.rename(columns={"FWD": "FWD Extra"}),
        on=["Base Metric", setup_column],
        how="outer",
    )

    combined = combined.rename(
        columns={
            "Kendall Tau Median_x": "Kendall Tau Median Inter",
            "Kendall Tau IQR_x": "Kendall Tau IQR Inter",
            "Kendall Tau Median_y": "Kendall Tau Median Extra",
            "Kendall Tau IQR_y": "Kendall Tau IQR Extra",
            "Relative Variation Median": "Relative Variation Median Extra",
            "Relative Variation IQR": "Relative Variation IQR Extra",
        }
    )
    return combined


def _fmt_num(value, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return r"\cdots"
    return f"{float(value):.{decimals}f}"


def _fmt_median_iqr(median, iqr, decimals: int = 2) -> str:
    if median is None or pd.isna(median):
        return r"\cdots"
    low, high = iqr
    return rf"{float(median):.{decimals}f}\,[{float(low):.{decimals}f}, {float(high):.{decimals}f}]"


def base_latex_table(
    combined: pd.DataFrame,
    caption: str,
    label: str,
    method_order: Iterable[tuple[str, str]] | None = None,
    resize: bool = True,
) -> str:
    if method_order is None:
        method_order = BASE_METHODS
    ordered_names = [name for name, _ in method_order]
    combined = combined.copy()
    combined["Base Metric"] = pd.Categorical(
        combined["Base Metric"],
        categories=ordered_names,
        ordered=True,
    )
    combined = combined.sort_values("Base Metric")

    lines = []
    header = rf"""
\begin{{table}}[ht]
\centering
\caption{{{caption}}}
{"\\label{" + label + "}" if label else ""}
{r"\resizebox{\textwidth}{!}{%" if resize else ""}
\begin{{tabular}}{{lcccccc}}
\toprule
    & & \multicolumn{{2}}{{c}}{{Interpolation}} & \multicolumn{{3}}{{c}}{{Extrapolation}} \\
    \cmidrule(lr){{3-4}} \cmidrule(lr){{5-7}}
    Base metric & Finsler
      & Kendall Tau ($\uparrow$) & FWD ($\downarrow$)
      & Kendall Tau ($\uparrow$) & Relative Variation (\%) & FWD ($\downarrow$) \\
\midrule
"""
    lines.append(header.strip())

    for metric in combined["Base Metric"].drop_duplicates():
        sub = combined[combined["Base Metric"] == metric].sort_values(
            "Finsler", ascending=True
        )
        for index, (_, row) in enumerate(sub.iterrows()):
            metric_name = (
                rf"    \multirow{{{len(sub)}}}{{*}}{{{metric}}}"
                if index == 0
                else "    "
            )
            finsler = r"\checkmark" if bool(row["Finsler"]) else r"\xmark"
            kt_inter = _fmt_median_iqr(
                row["Kendall Tau Median Inter"],
                row["Kendall Tau IQR Inter"],
            )
            kt_extra = _fmt_median_iqr(
                row["Kendall Tau Median Extra"],
                row["Kendall Tau IQR Extra"],
            )
            rv_extra = _fmt_median_iqr(
                row["Relative Variation Median Extra"],
                row["Relative Variation IQR Extra"],
            )
            lines.append(
                f"{metric_name} & {finsler} & "
                f"${kt_inter}$ & ${_fmt_num(row['FWD Inter'])}$ & "
                f"${kt_extra}$ & ${rv_extra}$ & ${_fmt_num(row['FWD Extra'])}$ \\\\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}" + ("}" if resize else ""))
    lines.append(r"\end{table}")
    return "\n".join(lines)


def biased_extrapolation_latex_table(
    combined: pd.DataFrame,
    caption: str,
    label: str,
    method_order: Iterable[tuple[str, str]] | None = None,
    resize: bool = True,
) -> str:
    if method_order is None:
        method_order = METHODS
    ordered_names = [name for name, _ in method_order]
    combined = combined.copy()
    combined["Base Metric"] = pd.Categorical(
        combined["Base Metric"],
        categories=ordered_names,
        ordered=True,
    )
    combined = combined.sort_values(["Base Metric", "Finsler"], ascending=[True, False])

    lines = []
    header = rf"""
\begin{{table}}[ht]
\centering
\caption{{{caption}}}
{"\\label{" + label + "}" if label else ""}
{r"\resizebox{\textwidth}{!}{%" if resize else ""}
\begin{{tabular}}{{llccc}}
\toprule
    & & \multicolumn{{3}}{{c}}{{Extrapolation}} \\
    \cmidrule(lr){{3-5}}
    Base metric & Geometry
      & Kendall Tau ($\uparrow$) & Relative Variation (\%) & FWD ($\downarrow$) \\
\midrule
"""
    lines.append(header.strip())

    for metric in combined["Base Metric"].drop_duplicates():
        sub = combined[combined["Base Metric"] == metric]
        for index, (_, row) in enumerate(sub.iterrows()):
            metric_name = (
                rf"    \multirow{{{len(sub)}}}{{*}}{{{metric}}}"
                if index == 0
                else "    "
            )
            geometry = "Riemann (biased)" if bool(row["Biased"]) else "Finsler"
            kt = _fmt_median_iqr(
                row["Kendall Tau Median Extra"],
                row["Kendall Tau IQR Extra"],
            )
            rv = _fmt_median_iqr(
                row["Relative Variation Median Extra"],
                row["Relative Variation IQR Extra"],
            )
            fwd = _fmt_num(row["FWD Extra"])
            lines.append(
                f"{metric_name} & {geometry} & "
                f"${kt}$ & ${rv}$ & ${fwd}$ \\\\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}" + ("}" if resize else ""))
    lines.append(r"\end{table}")
    return "\n".join(lines)


def dummy_acc_latex_table(
    combined: pd.DataFrame,
    caption: str,
    label: str,
    method_order: Iterable[tuple[str, str]] | None = None,
    resize: bool = True,
) -> str:
    if method_order is None:
        method_order = METHODS
    ordered_names = [name for name, _ in method_order]
    combined = combined.copy()
    combined["Base Metric"] = pd.Categorical(
        combined["Base Metric"],
        categories=ordered_names,
        ordered=True,
    )
    combined = combined.sort_values("Base Metric")

    lines = []
    header = rf"""
\begin{{table}}[ht]
\centering
\caption{{{caption}}}
{"\\label{" + label + "}" if label else ""}
{r"\resizebox{\textwidth}{!}{%" if resize else ""}
\begin{{tabular}}{{lcccccc}}
\toprule
    & & \multicolumn{{2}}{{c}}{{Interpolation}} & \multicolumn{{3}}{{c}}{{Extrapolation}} \\
    \cmidrule(lr){{3-4}} \cmidrule(lr){{5-7}}
    Base metric & Learned $\rho_\theta$
      & Kendall Tau ($\uparrow$) & FWD ($\downarrow$)
      & Kendall Tau ($\uparrow$) & Relative Variation (\%) & FWD ($\downarrow$) \\
\midrule
"""
    lines.append(header.strip())

    for metric in combined["Base Metric"].drop_duplicates():
        sub = combined[combined["Base Metric"] == metric]
        for index, (_, row) in enumerate(sub.iterrows()):
            metric_name = rf"    \multirow{{{len(sub)}}}{{*}}{{{metric}}}" if index == 0 else "    "
            setting = r"\checkmark" if bool(row["Dummy Accuracy"]) else r"\xmark"
            kt_inter = _fmt_median_iqr(row["Kendall Tau Median Inter"], row["Kendall Tau IQR Inter"])
            kt_extra = _fmt_median_iqr(row["Kendall Tau Median Extra"], row["Kendall Tau IQR Extra"])
            rv_extra = _fmt_median_iqr(
                row["Relative Variation Median Extra"],
                row["Relative Variation IQR Extra"],
            )
            fwd_inter = _fmt_num(row["FWD Inter"])
            fwd_extra = _fmt_num(row["FWD Extra"])
            lines.append(
                f"{metric_name} & {setting} & ${kt_inter}$ & ${fwd_inter}$ & "
                f"${kt_extra}$ & ${rv_extra}$ & ${fwd_extra}$ \\\\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}" + ("}" if resize else ""))
    lines.append(r"\end{table}")
    return "\n".join(lines)


def label_noise_latex_table(
    combined: pd.DataFrame,
    caption: str,
    label: str,
    noise_column: str = "Label Noise",
    method_order: Iterable[tuple[str, str]] | None = None,
    resize: bool = True,
) -> str:
    if method_order is None:
        method_order = METHODS
    ordered_names = [name for name, _ in method_order]
    combined = combined.copy()
    combined["Base Metric"] = pd.Categorical(
        combined["Base Metric"],
        categories=ordered_names,
        ordered=True,
    )
    combined = combined.sort_values("Base Metric")

    lines = []
    header = rf"""
\begin{{table}}[ht]
\centering
\caption{{{caption}}}
{"\\label{" + label + "}" if label else ""}
{r"\resizebox{\textwidth}{!}{%" if resize else ""}
\begin{{tabular}}{{lcccccc}}
\toprule
    & & \multicolumn{{2}}{{c}}{{Interpolation}} & \multicolumn{{3}}{{c}}{{Extrapolation}} \\
    \cmidrule(lr){{3-4}} \cmidrule(lr){{5-7}}
    Base metric & Label noise
      & Kendall Tau ($\uparrow$) & FWD ($\downarrow$)
      & Kendall Tau ($\uparrow$) & Relative Variation (\%) & FWD ($\downarrow$) \\
\midrule
"""
    lines.append(header.strip())

    ordered_names = combined["Base Metric"].drop_duplicates().tolist()
    for metric in ordered_names:
        sub = combined[combined["Base Metric"] == metric].sort_values(noise_column)
        for index, (_, row) in enumerate(sub.iterrows()):
            metric_name = (
                rf"    \multirow{{{len(sub)}}}{{*}}{{{metric}}}"
                if index == 0
                else "    "
            )
            kt_inter = _fmt_median_iqr(
                row["Kendall Tau Median Inter"],
                row["Kendall Tau IQR Inter"],
            )
            kt_extra = _fmt_median_iqr(
                row["Kendall Tau Median Extra"],
                row["Kendall Tau IQR Extra"],
            )
            rv_extra = _fmt_median_iqr(
                row["Relative Variation Median Extra"],
                row["Relative Variation IQR Extra"],
            )
            lines.append(
                f"{metric_name} & {_fmt_num(row[noise_column], decimals=1)} & "
                f"${kt_inter}$ & ${_fmt_num(row['FWD Inter'])}$ & "
                f"${kt_extra}$ & ${rv_extra}$ & ${_fmt_num(row['FWD Extra'])}$ \\\\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}" + ("}" if resize else ""))
    lines.append(r"\end{table}")
    return "\n".join(lines)
