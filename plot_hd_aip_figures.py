#!/usr/bin/env python3
"""
Publication-quality matplotlib figures for the HD-AIP manuscript.
Style inspired by figures4papers/scientific-figure-making conventions:
- unified PALETTE, FigureStyle, apply_publication_style, finalize_figure
- PNG and PDF export, 600 dpi
- no seaborn dependency

Usage:
    python plot_hd_aip_figures.py --fasta AIP.fasta --outdir output_figures

Inputs:
    AIP.fasta is optional but recommended. Headers may contain labels such as:
    >seq1|label=0
    >seq2|label=1

Outputs:
    output_figures/*.png
    output_figures/*.pdf
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# figures4papers-style house style
# -----------------------------------------------------------------------------
PALETTE: Dict[str, str] = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#6F6F6F",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "orange": "#D98C28",
    "black": "#222222",
}

DEFAULT_COLORS: List[str] = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["orange"],
    PALETTE["neutral"],
]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 14
    axes_linewidth: float = 2.0
    use_tex: bool = False
    font_family: Tuple[str, ...] = ("DejaVu Sans", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    """Apply a clean publication-oriented matplotlib style."""
    if style is None:
        style = FigureStyle()
    plt.rcParams.update({
        "font.size": style.font_size,
        "font.family": style.font_family,
        "text.usetex": style.use_tex,
        "axes.linewidth": style.axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.major.width": style.axes_linewidth * 0.75,
        "ytick.major.width": style.axes_linewidth * 0.75,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def finalize_figure(fig: matplotlib.figure.Figure,
                    out_path: str | Path,
                    formats: Sequence[str] = ("png", "pdf"),
                    dpi: int = 600) -> List[Path]:
    """Save a figure to multiple formats."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for fmt in formats:
        path = out_path.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        saved.append(path)
    plt.close(fig)
    return saved


def annotate_bars(ax: matplotlib.axes.Axes,
                  bars: Iterable[matplotlib.patches.Rectangle],
                  fmt: str = "{:.3f}",
                  dy: float = 0.008,
                  fontsize: int = 9,
                  rotation: int = 0) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
        )


def add_panel_label(ax: matplotlib.axes.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )


def highlight_xtick(ax: matplotlib.axes.Axes, tick_label: str, color: str = PALETTE["blue_main"]) -> None:
    for tick in ax.get_xticklabels():
        if tick.get_text() == tick_label:
            tick.set_fontweight("bold")
            tick.set_color(color)


# -----------------------------------------------------------------------------
# Data from the HD-AIP manuscript tables
# -----------------------------------------------------------------------------
def table_plm_ablation() -> pd.DataFrame:
    return pd.DataFrame({
        "Feature combination": ["ProtT5 only", "ESM-2 3B only", "Phys + CKSAAP", "PLM-Fusion ML"],
        "ACC": [0.6924, 0.7058, 0.7482, 0.7592],
        "AUC": [0.7388, 0.7566, 0.8137, 0.8186],
        "Original dimension": [2048, 5120, 2029, 9197],
        "Selected dimension": [300, 300, 300, 300],
    })


def table_ctnet_ablation() -> pd.DataFrame:
    return pd.DataFrame({
        "Configuration": ["CNN only\n(w/o Transformer)", "Transformer only\n(w/o ResNet)", "CT-Net"],
        "ACC": [0.7270, 0.7089, 0.7635],
        "AUC": [0.7715, 0.7424, 0.8104],
    })


def table_ensemble() -> pd.DataFrame:
    return pd.DataFrame({
        "Model": ["PLM-Fusion ML", "CT-Net", "Stacking LR", "Dynamic soft\nensemble"],
        "ACC": [0.7592, 0.7532, 0.7697, 0.7749],
        "AUC": [0.8178, 0.8147, 0.8409, 0.8419],
        "MCC": [0.4905, 0.4778, 0.5129, 0.5261],
        "Sn": [0.6377, 0.6305, 0.6502, 0.6800],
        "Sp": [0.8402, 0.8351, 0.8494, 0.8382],
        "Precision": [0.7269, 0.7183, 0.7422, 0.7371],
    })


def table_sota_independent() -> pd.DataFrame:
    return pd.DataFrame({
        "Model": ["iAIPs", "AIP_MDL", "PepNet", "NeXtMD", "HD-AIP"],
        "ACC": [0.6172, 0.6333, 0.6469, 0.6139, 0.6832],
        "AUC": [0.7116, 0.7362, 0.7207, 0.7373, 0.7686],
        "MCC": [0.3567, 0.4079, 0.3477, 0.3634, 0.4434],
        "Sn": [0.8966, 0.9396, 0.8017, 0.9138, 0.8879],
        "Sp": [0.4439, 0.4438, 0.5508, 0.4278, 0.5561],
        "Precision": [0.5000, 0.5117, 0.5254, 0.4977, 0.5538],
    })


# -----------------------------------------------------------------------------
# FASTA parsing and manuscript dataset data
# -----------------------------------------------------------------------------
def parse_fasta_with_labels(fasta_path: str | Path) -> pd.DataFrame:
    """Parse a FASTA file into columns: id, label, sequence, length."""
    records: List[Dict[str, object]] = []
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        return pd.DataFrame(columns=["id", "label", "sequence", "length"])

    header: str | None = None
    seq_parts: List[str] = []

    def flush() -> None:
        if header is None:
            return
        sequence = "".join(seq_parts).strip().upper()
        if not sequence:
            return
        label_match = re.search(r"(?:label|class|y)=([01])", header, flags=re.IGNORECASE)
        label = int(label_match.group(1)) if label_match else None
        rec_id = header.split()[0].lstrip(">")
        records.append({"id": rec_id, "label": label, "sequence": sequence, "length": len(sequence)})

    with fasta_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line
                seq_parts = []
            else:
                seq_parts.append(line)
        flush()

    return pd.DataFrame(records)


def dataset_distribution_from_fasta(fasta_df: pd.DataFrame) -> pd.DataFrame:
    if not fasta_df.empty and "label" in fasta_df and fasta_df["label"].notna().any():
        counts = fasta_df["label"].value_counts().to_dict()
        benchmark_pos = int(counts.get(1, 0))
        benchmark_neg = int(counts.get(0, 0))
    else:
        # Fallback to manuscript Table 1
        benchmark_pos, benchmark_neg = 1678, 2516

    return pd.DataFrame({
        "Dataset": ["Benchmark", "Independent"],
        "Positive AIPs": [benchmark_pos, 116],
        "Negative Non-AIPs": [benchmark_neg, 187],
    })


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def grouped_bar(ax: matplotlib.axes.Axes,
                categories: Sequence[str],
                series: Sequence[Sequence[float]],
                labels: Sequence[str],
                colors: Sequence[str] | None = None,
                ylim: Tuple[float, float] = (0, 1),
                value_fmt: str = "{:.3f}",
                rotate: int = 0) -> None:
    colors = list(colors or DEFAULT_COLORS)
    x = np.arange(len(categories))
    n_series = len(series)
    width = 0.78 / n_series

    for i, values in enumerate(series):
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=labels[i],
            color=colors[i % len(colors)],
            edgecolor=PALETTE["black"],
            linewidth=0.8,
            zorder=3,
        )
        annotate_bars(ax, bars, fmt=value_fmt, dy=(ylim[1] - ylim[0]) * 0.012, fontsize=8, rotation=90 if n_series >= 4 else 0)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=min(len(labels), 6))


def radar_plot(ax: matplotlib.axes.Axes,
               df: pd.DataFrame,
               metrics: Sequence[str],
               label_col: str,
               highlight_label: str = "HD-AIP") -> None:
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)

    for i, row in df.iterrows():
        label = str(row[label_col])
        values = [float(row[m]) for m in metrics]
        values += values[:1]
        is_highlight = label == highlight_label
        color = PALETTE["blue_main"] if is_highlight else DEFAULT_COLORS[(i + 2) % len(DEFAULT_COLORS)]
        lw = 2.8 if is_highlight else 1.3
        alpha = 0.22 if is_highlight else 0.06
        ax.plot(angles, values, color=color, linewidth=lw, label=label)
        ax.fill(angles, values, color=color, alpha=alpha)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)


# -----------------------------------------------------------------------------
# Figure generation
# -----------------------------------------------------------------------------
# def figure_dataset_distribution(fasta_df: pd.DataFrame, outdir: Path) -> None:
#     df = dataset_distribution_from_fasta(fasta_df)
#     df["Total"] = df["Positive AIPs"] + df["Negative Non-AIPs"]
#
#     fig, ax = plt.subplots(figsize=(6.8, 4.8))
#     x = np.arange(len(df))
#     pos = ax.bar(x, df["Positive AIPs"], color=PALETTE["blue_main"], edgecolor=PALETTE["black"], linewidth=0.9, label="Positive AIPs", zorder=3)
#     neg = ax.bar(x, df["Negative Non-AIPs"], bottom=df["Positive AIPs"], color=PALETTE["red_2"], edgecolor=PALETTE["black"], linewidth=0.9, label="Negative Non-AIPs", zorder=3)
#
#     for i, row in df.iterrows():
#         ax.text(i, row["Total"] + max(df["Total"]) * 0.02, f"n={int(row['Total'])}", ha="center", va="bottom", fontsize=11, fontweight="bold")
#         ax.text(i, row["Positive AIPs"] / 2, str(int(row["Positive AIPs"])), ha="center", va="center", fontsize=10, color="white", fontweight="bold")
#         ax.text(i, row["Positive AIPs"] + row["Negative Non-AIPs"] / 2, str(int(row["Negative Non-AIPs"])), ha="center", va="center", fontsize=10, color=PALETTE["black"], fontweight="bold")
#
#     ax.set_xticks(x)
#     ax.set_xticklabels(df["Dataset"])
#     ax.set_ylabel("Number of peptide sequences")
#     ax.set_title("Dataset composition")
#     ax.set_ylim(0, max(df["Total"]) * 1.18)
#     ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
#     ax.legend(loc="upper right")
#     add_panel_label(ax, "A")
#
#     finalize_figure(fig, outdir / "figure_02_dataset_distribution")

#
# def figure_plm_ablation(outdir: Path) -> None:
#     df = table_plm_ablation()
#     fig, ax = plt.subplots(figsize=(8.4, 4.8))
#     grouped_bar(
#         ax,
#         categories=df["Feature combination"],
#         series=[df["ACC"], df["AUC"]],
#         labels=["ACC", "AUC"],
#         colors=[PALETTE["blue_main"], PALETTE["green_3"]],
#         ylim=(0.60, 0.86),
#         rotate=20,
#     )
#     ax.set_ylabel("Cross-validation score")
#     ax.set_title("PLM-Fusion ML stream feature ablation")
#     highlight_xtick(ax, "PLM-Fusion ML")
#     add_panel_label(ax, "A")
#     finalize_figure(fig, outdir / "figure_03_plm_feature_ablation")
#
#
# def figure_ctnet_ablation(outdir: Path) -> None:
#     df = table_ctnet_ablation()
#     fig, ax = plt.subplots(figsize=(7.2, 4.8))
#     grouped_bar(
#         ax,
#         categories=df["Configuration"],
#         series=[df["ACC"], df["AUC"]],
#         labels=["ACC", "AUC"],
#         colors=[PALETTE["blue_secondary"], PALETTE["teal"]],
#         ylim=(0.66, 0.84),
#     )
#     ax.set_ylabel("Cross-validation score")
#     ax.set_title("CT-Net architecture ablation")
#     highlight_xtick(ax, "CT-Net")
#     add_panel_label(ax, "B")
#     finalize_figure(fig, outdir / "figure_04_ctnet_architecture_ablation")
#
#
# def figure_ensemble_strategy(outdir: Path) -> None:
#     df = table_ensemble()
#     metrics = ["ACC", "AUC", "MCC", "Sn", "Sp", "Precision"]
#
#     fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
#     grouped_bar(
#         axes[0],
#         categories=df["Model"],
#         series=[df[m] for m in ["ACC", "AUC", "MCC"]],
#         labels=["ACC", "AUC", "MCC"],
#         colors=[PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"]],
#         ylim=(0.42, 0.88),
#     )
#     axes[0].set_ylabel("Cross-validation score")
#     axes[0].set_title("Core metrics")
#     highlight_xtick(axes[0], "Dynamic soft\nensemble")
#     add_panel_label(axes[0], "C")
#
#     # Radar plot for the final selected strategy and alternatives.
#     ax_radar = fig.add_subplot(122, projection="polar")
#     fig.delaxes(axes[1])
#     radar_plot(ax_radar, df, metrics=metrics, label_col="Model", highlight_label="Dynamic soft\nensemble")
#     ax_radar.set_title("Overall metric balance", y=1.10)
#
#     finalize_figure(fig, outdir / "figure_05_ensemble_strategy_comparison")

#
# def figure_sota_comparison(outdir: Path) -> None:
#     df = table_sota_independent()
#     metrics = ["ACC", "AUC", "MCC", "Sn", "Sp", "Precision"]
#
#     fig = plt.figure(figsize=(13.2, 5.2))
#     gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.30)
#
#     ax_bar = fig.add_subplot(gs[0, 0])
#     ax_radar = fig.add_subplot(gs[0, 1], projection="polar")
#
#     grouped_bar(
#         ax_bar,
#         categories=df["Model"],
#         series=[df[m] for m in ["ACC", "AUC", "MCC"]],
#         labels=["ACC", "AUC", "MCC"],
#         colors=[
#             PALETTE["blue_main"],
#             PALETTE["green_3"],
#             PALETTE["red_strong"],
#         ],
#         ylim=(0.30, 0.82),
#     )
#     ax_bar.set_ylabel("Independent test score")
#     ax_bar.set_title("Core independent-test metrics")
#     highlight_xtick(ax_bar, "HD-AIP")
#     add_panel_label(ax_bar, "A")
#
#     radar_plot(
#         ax_radar,
#         df,
#         metrics=metrics,
#         label_col="Model",
#         highlight_label="HD-AIP",
#     )
#     ax_radar.set_title("SOTA metric profile", y=1.10)
#
#     ax_radar.text(
#         -0.14,
#         1.12,
#         "B",
#         transform=ax_radar.transAxes,
#         fontsize=18,
#         fontweight="bold",
#         va="top",
#         ha="left",
#     )
#
#     finalize_figure(
#         fig,
#         outdir / "figure_06_sota_independent_test_comparison",
#     )


# def figure_length_distribution(fasta_df: pd.DataFrame, outdir: Path) -> None:
#     if fasta_df.empty or "length" not in fasta_df:
#         return
#
#     fig, ax = plt.subplots(figsize=(7.4, 4.8))
#     bins = np.arange(max(0, int(fasta_df["length"].min()) - 1), int(fasta_df["length"].max()) + 2, 1)
#     if fasta_df["label"].notna().any():
#         pos_lengths = fasta_df.loc[fasta_df["label"] == 1, "length"]
#         neg_lengths = fasta_df.loc[fasta_df["label"] == 0, "length"]
#         ax.hist([pos_lengths, neg_lengths], bins=bins, stacked=False, label=["Positive AIPs", "Negative Non-AIPs"],
#                 color=[PALETTE["blue_main"], PALETTE["red_2"]], edgecolor=PALETTE["black"], linewidth=0.35, alpha=0.85)
#     else:
#         ax.hist(fasta_df["length"], bins=bins, color=PALETTE["blue_main"], edgecolor=PALETTE["black"], linewidth=0.35, alpha=0.85)
#
#     median_len = float(fasta_df["length"].median())
#     ax.axvline(median_len, color=PALETTE["black"], linestyle="--", linewidth=1.5, label=f"Median={median_len:.0f}")
#     ax.set_xlabel("Peptide length (aa)")
#     ax.set_ylabel("Number of sequences")
#     ax.set_title("Benchmark peptide length distribution")
#     ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
#     ax.legend(loc="upper right")
#     add_panel_label(ax, "B")
#     finalize_figure(fig, outdir / "figure_supp_sequence_length_distribution")

def figure_external_baseline_comparison(outdir: Path) -> None:
    df = pd.DataFrame({
        "Category": [
            "Traditional ML", "Traditional ML", "Traditional ML",
            "Classic DL", "Classic DL", "Classic DL",
            "HD-AIP components", "HD-AIP components", "HD-AIP components",
        ],
        "Model": [
            "SVM", "Random Forest", "XGBoost",
            "BiLSTM", "TextCNN", "Transformer",
            "CT-Net", "PLM-Fusion ML", "HD-AIP",
        ],
        "ACC": [0.6354, 0.6950, 0.7420, 0.6867, 0.6845, 0.7043, 0.7499, 0.7592, 0.7747],
        "AUC": [0.6666, 0.7508, 0.8061, 0.7027, 0.7314, 0.7535, 0.8141, 0.8178, 0.8394],
        "MCC": [0.2518, 0.3425, 0.4512, 0.3320, 0.3451, 0.3788, 0.4711, 0.4905, 0.5261],
    })

    metrics = ["ACC", "AUC", "MCC"]
    heatmap_data = df.set_index("Model")[metrics]

    # Taller and more publication-friendly vertical layout
    fig = plt.figure(figsize=(8.2, 10.5))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.35, 1.25],
        hspace=0.55,
    )

    ax_heat = fig.add_subplot(gs[0, 0])
    ax_rank = fig.add_subplot(gs[1, 0])

    # -------------------------
    # A. Heatmap
    # -------------------------
    im = ax_heat.imshow(
        heatmap_data.values,
        aspect="auto",
        cmap="YlGnBu",
        vmin=0.25,
        vmax=0.85,
    )

    ax_heat.set_xticks(np.arange(len(metrics)))
    ax_heat.set_xticklabels(metrics)
    ax_heat.set_yticks(np.arange(len(heatmap_data.index)))
    ax_heat.set_yticklabels(heatmap_data.index)

    ax_heat.set_title(
        "External baseline comparison across core metrics",
        pad=14,
    )

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            value = heatmap_data.iloc[i, j]
            text_color = "white" if value > 0.62 else "black"
            ax_heat.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=11,
                color=text_color,
                fontweight="bold" if heatmap_data.index[i] == "HD-AIP" else "normal",
            )

    # Highlight HD-AIP row
    hd_idx = list(heatmap_data.index).index("HD-AIP")
    ax_heat.add_patch(
        plt.Rectangle(
            (-0.5, hd_idx - 0.5),
            len(metrics),
            1,
            fill=False,
            edgecolor=PALETTE["red_strong"],
            linewidth=2.2,
        )
    )

    # Category separators
    for sep in [2.5, 5.5]:
        ax_heat.axhline(sep, color="white", linewidth=2.2)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.025)
    cbar.set_label("Score", rotation=270, labelpad=16)

    add_panel_label(ax_heat, "A")

    # -------------------------
    # B. MCC ranking
    # -------------------------
    rank_df = df.sort_values("MCC", ascending=True)

    colors = []
    for model in rank_df["Model"]:
        if model == "HD-AIP":
            colors.append(PALETTE["red_strong"])
        elif model in ["CT-Net", "PLM-Fusion ML"]:
            colors.append(PALETTE["blue_main"])
        else:
            colors.append(PALETTE["neutral"])

    ax_rank.barh(
        rank_df["Model"],
        rank_df["MCC"],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        height=0.62,
    )

    ax_rank.set_xlabel("MCC")
    ax_rank.set_xlim(0.20, 0.56)
    ax_rank.set_title("MCC-based model ranking", pad=14)

    ax_rank.spines["top"].set_visible(False)
    ax_rank.spines["right"].set_visible(False)

    ax_rank.grid(axis="x", alpha=0.25, linewidth=0.8)

    for y, value in enumerate(rank_df["MCC"]):
        ax_rank.text(
            value + 0.008,
            y,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax_rank.text(
        -0.13,
        1.10,
        "B",
        transform=ax_rank.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
    )

    finalize_figure(fig, outdir / "figure_04_external_baseline_comparison")

def write_summary_tables(fasta_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    dataset_distribution_from_fasta(fasta_df).to_csv(outdir / "dataset_distribution.csv", index=False)
    table_plm_ablation().to_csv(outdir / "plm_feature_ablation.csv", index=False)
    table_ctnet_ablation().to_csv(outdir / "ctnet_architecture_ablation.csv", index=False)
    table_ensemble().to_csv(outdir / "ensemble_strategy_comparison.csv", index=False)
    table_sota_independent().to_csv(outdir / "sota_independent_test_comparison.csv", index=False)
    if not fasta_df.empty:
        desc = fasta_df["length"].describe().to_frame("length_statistics")
        desc.to_csv(outdir / "fasta_length_statistics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HD-AIP manuscript figures.")
    parser.add_argument("--fasta", type=str, default="AIP.fasta", help="Path to AIP.fasta")
    parser.add_argument("--outdir", type=str, default="output_figures", help="Output directory")
    parser.add_argument("--font-size", type=int, default=14, help="Base font size")
    parser.add_argument("--dpi", type=int, default=600, help="Output DPI. finalize_figure uses 600 by default.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    fasta_df = parse_fasta_with_labels(args.fasta)

    apply_publication_style(FigureStyle(font_size=args.font_size, axes_linewidth=2.0))

    # figure_dataset_distribution(fasta_df, outdir)
    # figure_plm_ablation(outdir)
    # figure_ctnet_ablation(outdir)
    # figure_ensemble_strategy(outdir)
    # figure_sota_comparison(outdir)
    # figure_length_distribution(fasta_df, outdir)
    figure_external_baseline_comparison(outdir)
    write_summary_tables(fasta_df, outdir)

    print(f"Saved figures and CSV files to: {outdir.resolve()}")
    if not fasta_df.empty:
        labels = fasta_df["label"].value_counts(dropna=False).to_dict()
        print(f"Parsed {len(fasta_df)} FASTA records. Label counts: {labels}")


if __name__ == "__main__":
    main()
