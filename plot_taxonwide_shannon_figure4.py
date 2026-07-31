"""Create Figure 4 from taxon-wide Shannon positional-concentration results.

The statistic analysed here is positional concentration:

    concentration = 1 - normalized Shannon entropy across positions 0-19

Positive observed-minus-null effects therefore indicate that a motif is more
unevenly localized across mature-miRNA positions than expected under the null.
The globally pooled repertoire is deliberately excluded from taxon panels.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


WITHIN_NULL = "within_sequence"
STRICT_NULL = "positionwise_length_stratified"
NULLS = {
    WITHIN_NULL: {
        "label": "Within-sequence null",
        "marker": "o",
    },
    STRICT_NULL: {
        "label": "Position-wise null",
        "marker": "D",
    },
}

# The order is identical to Figure 3 and follows the displayed topology.
TAXA = [
    ("taxon__10_angiosperms", "Angiosperms"),
    ("taxon__09_nematodes", "Nematodes"),
    ("taxon__08_insects", "Insects"),
    ("taxon__01_fishes_non_tetrapod", "Fishes"),
    ("taxon__02_amphibians", "Amphibians"),
    ("taxon__03_non_avian_reptiles", "Non-avian reptiles"),
    ("taxon__04_aves", "Aves"),
    ("taxon__07_other_mammals", "Other mammals"),
    ("taxon__06_non_human_primates", "Non-human primates"),
    ("taxon__05_human", "Human"),
]
TAXON_LABEL = dict(TAXA)
TAXON_INDEX = {
    dataset: index for index, (dataset, _) in enumerate(TAXA)
}
Y_POSITION = {
    dataset: len(TAXA) - 1 - index
    for index, (dataset, _) in enumerate(TAXA)
}

BASE_ORDER = ("A", "C", "G", "U")
KMERS_2 = tuple(
    first + second for first in BASE_ORDER for second in BASE_ORDER
)

CG_COLOR = "#0072B2"
UA_COLOR = "#D55E00"
OTHER_COLOR = "#8B9299"
CONCENTRATED_COLOR = "#009E73"
UNIFORM_COLOR = "#CC79A7"
K2_COLOR = "#009E73"
K3_COLOR = "#7B61A8"
TEXT_COLOR = "#222222"
MUTED_TEXT = "#61676D"
GRID_COLOR = "#DDE1E5"
LIGHT_NEUTRAL = "#CBD0D5"
ZERO_COLOR = "#50565C"
WHITE = "#FFFFFF"

PANEL_LETTER_SIZE = 9.0
PANEL_TITLE_SIZE = 7.7
AXIS_LABEL_SIZE = 6.5
TICK_SIZE = 5.5
ANNOTATION_SIZE = 5.1


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": 6.5,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": TICK_SIZE,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "axes.edgecolor": ZERO_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.edgecolor": WHITE,
            "savefig.transparent": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def locate_results_root(path: Path) -> Path:
    """Accept the final-results directory or an ancestor containing it."""
    path = Path(path).expanduser().resolve()
    required = {
        "all_datasets_shannon_concentration_results.csv",
        "dataset_inventory.csv",
    }
    candidates = [path]
    candidates.extend(
        candidate.parent
        for candidate in path.rglob(
            "all_datasets_shannon_concentration_results.csv"
        )
    )
    valid: list[Path] = []
    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            if candidate not in valid:
                valid.append(candidate)
    if len(valid) == 1:
        return valid[0]
    if not valid:
        raise FileNotFoundError(
            "Could not locate the combined Shannon result table and "
            f"dataset inventory beneath {path}"
        )
    raise RuntimeError(
        "More than one results directory was found. Pass the intended "
        "directory explicitly:\n" + "\n".join(map(str, valid))
    )


def load_panel_data(
    results_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load, derive and validate the exact Figure 4 plotting tables."""
    results_root = locate_results_root(results_root)
    shannon_path = (
        results_root
        / "all_datasets_shannon_concentration_results.csv"
    )
    inventory_path = results_root / "dataset_inventory.csv"

    shannon = pd.read_csv(shannon_path)
    inventory = pd.read_csv(inventory_path)

    require_columns(
        shannon,
        {
            "dataset",
            "analysis_scope",
            "null_model",
            "k",
            "kmer",
            "observed_normalized_shannon_entropy",
            "observed_positional_concentration",
            "null_mean_positional_concentration",
            "null_lower_95_positional_concentration",
            "null_upper_95_positional_concentration",
            "observed_minus_null_concentration",
            "standardized_effect",
            "BH_q_within_dataset_k_null",
            "direction_at_q_le_alpha",
            "n_permutations",
        },
        shannon_path.name,
    )
    require_columns(
        inventory,
        {
            "dataset",
            "analysis_scope",
            "analysis_group",
            "sequence_occurrences",
        },
        inventory_path.name,
    )

    taxon_ids = set(TAXON_LABEL)
    inventory_taxa = set(
        inventory.loc[
            inventory["analysis_scope"] == "taxon",
            "dataset",
        ]
    )
    if inventory_taxa != taxon_ids:
        raise ValueError(
            "Taxon datasets do not match the ten intended mutually "
            "exclusive groups.\n"
            f"Missing: {sorted(taxon_ids - inventory_taxa)}\n"
            f"Unexpected: {sorted(inventory_taxa - taxon_ids)}"
        )

    taxon = shannon[
        (shannon["analysis_scope"] == "taxon")
        & shannon["dataset"].isin(taxon_ids)
        & shannon["null_model"].isin(NULLS)
        & shannon["k"].isin([2, 3])
    ].copy()
    taxon["taxon_label"] = taxon["dataset"].map(TAXON_LABEL)
    taxon["taxon_leaf_order"] = taxon["dataset"].map(TAXON_INDEX)
    taxon["effect_concentration_percentage_points"] = (
        100.0 * taxon["observed_minus_null_concentration"]
    )
    taxon["significant_q_le_0_05"] = (
        taxon["BH_q_within_dataset_k_null"] <= 0.05
    )

    panel_a = taxon[
        (taxon["null_model"] == STRICT_NULL)
        & (taxon["k"] == 2)
    ].copy()
    panel_a["motif_class"] = np.select(
        [
            panel_a["kmer"] == "CG",
            panel_a["kmer"] == "UA",
        ],
        ["CG", "UA"],
        default="Other 2-mers",
    )
    panel_a["motif_order"] = panel_a["kmer"].map(
        {motif: index for index, motif in enumerate(KMERS_2)}
    )

    panel_b = taxon[
        (taxon["k"] == 2)
        & taxon["kmer"].isin(["CG", "UA"])
    ].copy()

    strict_all = taxon[taxon["null_model"] == STRICT_NULL].copy()
    summary_rows = []
    for (dataset, k), group in strict_all.groupby(
        ["dataset", "k"],
        sort=False,
    ):
        total = len(group)
        significant = group["significant_q_le_0_05"]
        positive = significant & (
            group["observed_minus_null_concentration"] > 0
        )
        negative = significant & (
            group["observed_minus_null_concentration"] < 0
        )
        summary_rows.append(
            {
                "dataset": dataset,
                "taxon_label": TAXON_LABEL[dataset],
                "taxon_leaf_order": TAXON_INDEX[dataset],
                "k": int(k),
                "total_motifs": int(total),
                "significantly_more_concentrated": int(positive.sum()),
                "significantly_more_uniform": int(negative.sum()),
                "no_detectable_deviation": int(
                    total - positive.sum() - negative.sum()
                ),
                "percent_more_concentrated": float(
                    100.0 * positive.mean()
                ),
                "percent_more_uniform": float(
                    100.0 * negative.mean()
                ),
            }
        )
    panel_c = pd.DataFrame(summary_rows)

    panel_d_raw = strict_all[
        (strict_all["k"] == 3)
        & (
            strict_all["kmer"].str.contains("CG", regex=False)
            | strict_all["kmer"].str.contains("UA", regex=False)
        )
    ].copy()
    panel_d_raw["context_class"] = np.where(
        panel_d_raw["kmer"].str.contains("CG", regex=False),
        "Contains CG",
        "Contains UA",
    )
    panel_d_summary = (
        panel_d_raw.groupby(
            ["context_class", "kmer"],
            sort=False,
        )
        .agg(
            cross_taxon_median_effect_pp=(
                "effect_concentration_percentage_points",
                "median",
            ),
            cross_taxon_q1_effect_pp=(
                "effect_concentration_percentage_points",
                lambda values: values.quantile(0.25),
            ),
            cross_taxon_q3_effect_pp=(
                "effect_concentration_percentage_points",
                lambda values: values.quantile(0.75),
            ),
            cross_taxon_minimum_effect_pp=(
                "effect_concentration_percentage_points",
                "min",
            ),
            cross_taxon_maximum_effect_pp=(
                "effect_concentration_percentage_points",
                "max",
            ),
            significant_taxa=(
                "significant_q_le_0_05",
                "sum",
            ),
            positive_effect_taxa=(
                "effect_concentration_percentage_points",
                lambda values: int((values > 0).sum()),
            ),
            negative_effect_taxa=(
                "effect_concentration_percentage_points",
                lambda values: int((values < 0).sum()),
            ),
            taxon_count=("dataset", "nunique"),
        )
        .reset_index()
    )
    panel_d_summary["context_order"] = panel_d_summary[
        "context_class"
    ].map({"Contains CG": 0, "Contains UA": 1})
    panel_d_summary = panel_d_summary.sort_values(
        [
            "context_order",
            "cross_taxon_median_effect_pp",
        ],
        ascending=[True, False],
    ).reset_index(drop=True)

    panel_inventory = inventory[
        inventory["dataset"].isin(taxon_ids)
    ].copy()
    panel_inventory["taxon_label"] = panel_inventory["dataset"].map(
        TAXON_LABEL
    )
    panel_inventory["taxon_leaf_order"] = panel_inventory["dataset"].map(
        TAXON_INDEX
    )

    expected_sizes = {
        "panel_a": 10 * 16,
        "panel_b": 10 * 2 * 2,
        "panel_c": 10 * 2,
        "panel_d_raw": 10 * 16,
        "panel_d_summary": 16,
        "inventory": 10,
    }
    observed_sizes = {
        "panel_a": len(panel_a),
        "panel_b": len(panel_b),
        "panel_c": len(panel_c),
        "panel_d_raw": len(panel_d_raw),
        "panel_d_summary": len(panel_d_summary),
        "inventory": len(panel_inventory),
    }
    if observed_sizes != expected_sizes:
        raise ValueError(
            "Unexpected Figure 4 plotting-table sizes. "
            f"Observed {observed_sizes}; expected {expected_sizes}."
        )

    if set(taxon["n_permutations"].dropna().astype(int)) != {10_000}:
        print(
            "WARNING: the selected Shannon results are not uniformly based "
            "on 10,000 permutations."
        )

    if not np.allclose(
        taxon["observed_positional_concentration"],
        1.0 - taxon["observed_normalized_shannon_entropy"],
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError(
            "Observed positional concentration is inconsistent with "
            "1 - normalized Shannon entropy."
        )

    return (
        panel_a,
        panel_b,
        panel_c,
        panel_d_raw,
        panel_d_summary,
        panel_inventory,
    )


def clean_axis(
    ax: plt.Axes,
    *,
    grid_axis: str | None = None,
) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis is not None:
        ax.grid(
            True,
            axis=grid_axis,
            color=GRID_COLOR,
            linewidth=0.42,
            zorder=0,
        )
    ax.set_axisbelow(True)


def add_panel_header(
    ax: plt.Axes,
    letter: str,
    title: str,
    subtitle: str | None = None,
    *,
    letter_x: float = -0.16,
    title_y: float = 1.11,
    subtitle_y: float = 1.025,
) -> None:
    ax.text(
        letter_x,
        title_y,
        letter,
        transform=ax.transAxes,
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0,
        title_y,
        title,
        transform=ax.transAxes,
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )
    if subtitle:
        ax.text(
            0,
            subtitle_y,
            subtitle,
            transform=ax.transAxes,
            fontsize=ANNOTATION_SIZE,
            color=MUTED_TEXT,
            ha="left",
            va="top",
        )


def taxon_grid(ax: plt.Axes, x_min: float, x_max: float) -> None:
    for dataset, _ in TAXA:
        y = Y_POSITION[dataset]
        ax.plot(
            [x_min, x_max],
            [y, y],
            color=GRID_COLOR,
            linewidth=0.35,
            zorder=0,
        )


def plot_panel_a(ax: plt.Axes, panel_a: pd.DataFrame) -> None:
    """All 16 dinucleotides as a taxon-wise strict-null effect cloud."""
    x_min = -0.75
    x_max = 2.35
    motif_jitter = {
        motif: offset
        for motif, offset in zip(
            KMERS_2,
            np.linspace(-0.17, 0.17, len(KMERS_2)),
        )
    }
    taxon_grid(ax, x_min, x_max)
    ax.axvline(0, color=ZERO_COLOR, linewidth=0.75, zorder=1)

    for dataset, _ in TAXA:
        group = panel_a[panel_a["dataset"] == dataset].copy()
        y = Y_POSITION[dataset]
        effects = group["effect_concentration_percentage_points"]
        ax.plot(
            [effects.min(), effects.max()],
            [y, y],
            color=LIGHT_NEUTRAL,
            linewidth=1.0,
            solid_capstyle="round",
            zorder=1,
        )

        for row in group.itertuples():
            point_y = y + motif_jitter[row.kmer]
            if row.kmer == "CG":
                color = CG_COLOR
                size = 24
                zorder = 4
            elif row.kmer == "UA":
                color = UA_COLOR
                size = 24
                zorder = 4
            else:
                color = OTHER_COLOR
                size = 11
                zorder = 3
            ax.scatter(
                row.effect_concentration_percentage_points,
                point_y,
                s=size,
                marker="o",
                facecolor=(
                    color
                    if row.significant_q_le_0_05
                    else WHITE
                ),
                edgecolor=color,
                linewidth=0.65,
                zorder=zorder,
            )

        maximum = group.loc[
            group["effect_concentration_percentage_points"].idxmax()
        ]
        ax.text(
            min(
                maximum["effect_concentration_percentage_points"] + 0.055,
                x_max - 0.02,
            ),
            y,
            str(maximum["kmer"]),
            fontsize=ANNOTATION_SIZE,
            color=MUTED_TEXT,
            va="center",
            ha="left",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.65, 9.65)
    ax.set_yticks([Y_POSITION[dataset] for dataset, _ in TAXA])
    ax.set_yticklabels([label for _, label in TAXA])
    ax.set_xlabel(
        "Observed − null positional concentration "
        "(percentage points)"
    )
    clean_axis(ax, grid_axis="x")
    add_panel_header(
        ax,
        "A",
        "Dinucleotide concentration landscape across taxa",
        "Position-wise, length-stratified null; all 16 2-mers",
        letter_x=-0.12,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=CG_COLOR,
            markeredgecolor=CG_COLOR,
            markersize=4.4,
            label="CG",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=UA_COLOR,
            markeredgecolor=UA_COLOR,
            markersize=4.4,
            label="UA",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=OTHER_COLOR,
            markeredgecolor=OTHER_COLOR,
            markersize=3.5,
            label="Other 2-mers",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=WHITE,
            markeredgecolor=TEXT_COLOR,
            markersize=3.8,
            label="BH q > 0.05",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.12),
        ncol=4,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.8,
        borderaxespad=0,
    )
    ax.text(
        1.0,
        -0.17,
        "Labels mark the largest positive effect in each taxon.",
        transform=ax.transAxes,
        fontsize=ANNOTATION_SIZE,
        color=MUTED_TEXT,
        ha="right",
        va="top",
    )


def plot_focal_null_axis(
    ax: plt.Axes,
    panel_b: pd.DataFrame,
    *,
    motif: str,
    color: str,
    show_taxon_labels: bool,
) -> None:
    x_min = -0.25
    x_max = 3.35
    subset = panel_b[panel_b["kmer"] == motif]
    taxon_grid(ax, x_min, x_max)
    ax.axvline(0, color=ZERO_COLOR, linewidth=0.75, zorder=1)

    for dataset, _ in TAXA:
        y = Y_POSITION[dataset]
        rows = subset[subset["dataset"] == dataset].set_index(
            "null_model"
        )
        effects = [
            rows.loc[
                null_model,
                "effect_concentration_percentage_points",
            ]
            for null_model in NULLS
        ]
        ax.plot(
            effects,
            [y, y],
            color=LIGHT_NEUTRAL,
            linewidth=0.8,
            zorder=1,
        )
        for null_model, style in NULLS.items():
            row = rows.loc[null_model]
            significant = bool(row["significant_q_le_0_05"])
            ax.scatter(
                row["effect_concentration_percentage_points"],
                y,
                marker=style["marker"],
                s=22,
                facecolor=color if significant else WHITE,
                edgecolor=color,
                linewidth=0.75,
                zorder=3,
            )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.65, 9.65)
    ax.set_yticks([Y_POSITION[dataset] for dataset, _ in TAXA])
    if show_taxon_labels:
        ax.set_yticklabels([label for _, label in TAXA])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Observed − null concentration (pp)")
    ax.set_title(
        motif,
        color=color,
        fontweight="bold",
        pad=3,
    )
    clean_axis(ax, grid_axis="x")


def plot_panel_b(
    axes: list[plt.Axes],
    panel_b: pd.DataFrame,
) -> None:
    left, right = axes
    plot_focal_null_axis(
        left,
        panel_b,
        motif="CG",
        color=CG_COLOR,
        show_taxon_labels=True,
    )
    plot_focal_null_axis(
        right,
        panel_b,
        motif="UA",
        color=UA_COLOR,
        show_taxon_labels=False,
    )
    add_panel_header(
        left,
        "B",
        "Focal dinucleotides under both null models",
        "Connected symbols are calculated from the same observed profile",
        letter_x=-0.24,
        title_y=1.24,
        subtitle_y=1.15,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=OTHER_COLOR,
            markeredgecolor=TEXT_COLOR,
            markersize=4.2,
            label="Within-sequence",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=OTHER_COLOR,
            markeredgecolor=TEXT_COLOR,
            markersize=4.0,
            label="Position-wise",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=TEXT_COLOR,
            markeredgecolor=TEXT_COLOR,
            markersize=3.8,
            label="BH q ≤ 0.05",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=WHITE,
            markeredgecolor=TEXT_COLOR,
            markersize=3.8,
            label="BH q > 0.05",
        ),
    ]
    right.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.25),
        ncol=2,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.7,
        borderaxespad=0,
    )


def plot_panel_c(ax: plt.Axes, panel_c: pd.DataFrame) -> None:
    """Diverging significant share for k=2 and k=3."""
    x_min = -10
    x_max = 100
    taxon_grid(ax, x_min, x_max)
    ax.axvline(0, color=ZERO_COLOR, linewidth=0.75, zorder=1)

    styles = {
        2: {
            "marker": "o",
            "color": K2_COLOR,
            "offset": 0.15,
            "label": "2-mers (n = 16)",
        },
        3: {
            "marker": "s",
            "color": K3_COLOR,
            "offset": -0.15,
            "label": "3-mers (n = 64)",
        },
    }
    for dataset, _ in TAXA:
        base_y = Y_POSITION[dataset]
        rows = panel_c[panel_c["dataset"] == dataset].set_index("k")
        for k, style in styles.items():
            y = base_y + style["offset"]
            positive = float(rows.loc[k, "percent_more_concentrated"])
            negative = -float(rows.loc[k, "percent_more_uniform"])
            ax.plot(
                [0, positive],
                [y, y],
                color=style["color"],
                linewidth=0.8,
                alpha=0.8,
                zorder=1,
            )
            ax.scatter(
                positive,
                y,
                marker=style["marker"],
                s=19,
                facecolor=style["color"],
                edgecolor=style["color"],
                linewidth=0.65,
                zorder=3,
            )
            if negative < 0:
                ax.plot(
                    [negative, 0],
                    [y, y],
                    color=UNIFORM_COLOR,
                    linewidth=0.8,
                    zorder=1,
                )
                ax.scatter(
                    negative,
                    y,
                    marker=style["marker"],
                    s=18,
                    facecolor=UNIFORM_COLOR,
                    edgecolor=UNIFORM_COLOR,
                    linewidth=0.65,
                    zorder=3,
                )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.65, 9.65)
    ax.set_yticks([Y_POSITION[dataset] for dataset, _ in TAXA])
    ax.set_yticklabels([label for _, label in TAXA])
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{abs(value):.0f}")
    )
    ax.set_xlabel("Significant share of motif repertoire (%)")
    clean_axis(ax, grid_axis="x")
    add_panel_header(
        ax,
        "C",
        "Significant share of each motif repertoire",
        "Position-wise null; BH q ≤ 0.05 within each taxon and k",
        letter_x=-0.12,
        title_y=1.18,
        subtitle_y=1.09,
    )
    ax.text(
        x_min + 0.5,
        9.52,
        "more uniform",
        fontsize=ANNOTATION_SIZE,
        color=UNIFORM_COLOR,
        ha="left",
        va="center",
    )
    ax.text(
        x_max - 0.5,
        9.52,
        "more positionally concentrated",
        fontsize=ANNOTATION_SIZE,
        color=CONCENTRATED_COLOR,
        ha="right",
        va="center",
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker=styles[k]["marker"],
            linestyle="none",
            markerfacecolor=styles[k]["color"],
            markeredgecolor=styles[k]["color"],
            markersize=4.2,
            label=styles[k]["label"],
        )
        for k in [2, 3]
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=UNIFORM_COLOR,
            markerfacecolor=UNIFORM_COLOR,
            markeredgecolor=UNIFORM_COLOR,
            linewidth=0.8,
            markersize=3.8,
            label="Significantly more uniform",
        )
    )
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.20),
        ncol=3,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.75,
        borderaxespad=0,
    )


def plot_context_axis(
    ax: plt.Axes,
    panel_d_summary: pd.DataFrame,
    *,
    context_class: str,
    color: str,
) -> None:
    subset = panel_d_summary[
        panel_d_summary["context_class"] == context_class
    ].sort_values(
        "cross_taxon_median_effect_pp",
        ascending=False,
    )
    labels = list(subset["kmer"])
    y_values = np.arange(len(labels) - 1, -1, -1)
    x_min = -0.35
    x_max = 10.8

    for y, row in zip(y_values, subset.itertuples()):
        ax.plot(
            [
                row.cross_taxon_q1_effect_pp,
                row.cross_taxon_q3_effect_pp,
            ],
            [y, y],
            color=color,
            linewidth=1.5,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            row.cross_taxon_median_effect_pp,
            y,
            marker="D",
            s=22,
            facecolor=color,
            edgecolor=color,
            linewidth=0.65,
            zorder=3,
        )
        ax.text(
            10.55,
            y,
            f"{int(row.significant_taxa)}/10",
            fontsize=TICK_SIZE,
            color=MUTED_TEXT,
            ha="center",
            va="center",
        )
        ax.plot(
            [x_min, x_max],
            [y, y],
            color=GRID_COLOR,
            linewidth=0.35,
            zorder=0,
        )

    ax.axvline(0, color=ZERO_COLOR, linewidth=0.75, zorder=1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.65, 7.65)
    ax.set_yticks(y_values)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Median concentration effect across taxa (pp)")
    ax.set_title(
        context_class,
        color=color,
        fontweight="bold",
        pad=3,
    )
    clean_axis(ax, grid_axis="x")
    ax.text(
        10.55,
        7.56,
        "q-sig.\ntaxa",
        fontsize=ANNOTATION_SIZE,
        color=MUTED_TEXT,
        ha="center",
        va="bottom",
    )


def plot_panel_d(
    axes: list[plt.Axes],
    panel_d_summary: pd.DataFrame,
) -> None:
    left, right = axes
    plot_context_axis(
        left,
        panel_d_summary,
        context_class="Contains CG",
        color=CG_COLOR,
    )
    plot_context_axis(
        right,
        panel_d_summary,
        context_class="Contains UA",
        color=UA_COLOR,
    )
    add_panel_header(
        left,
        "D",
        "Positional concentration of focal trinucleotide contexts",
        "Position-wise null; diamonds and bars show cross-taxon median "
        "and interquartile range",
        letter_x=-0.24,
        title_y=1.24,
        subtitle_y=1.15,
    )
    right.text(
        1.0,
        -0.18,
        "Cross-taxon summaries are descriptive, not a phylogenetic "
        "meta-analysis.",
        transform=right.transAxes,
        fontsize=ANNOTATION_SIZE,
        color=MUTED_TEXT,
        ha="right",
        va="top",
    )


def create_figure(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d_summary: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, list[plt.Axes]]]:
    width_inches = 183 / 25.4
    height_inches = 247 / 25.4
    fig = plt.figure(figsize=(width_inches, height_inches))
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.00, 1.02, 0.98, 1.00],
        left=0.13,
        right=0.985,
        bottom=0.050,
        top=0.965,
        hspace=0.60,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    plot_panel_a(ax_a, panel_a)

    grid_b = outer[1, 0].subgridspec(
        1,
        2,
        wspace=0.24,
    )
    ax_b_cg = fig.add_subplot(grid_b[0, 0])
    ax_b_ua = fig.add_subplot(grid_b[0, 1])
    plot_panel_b([ax_b_cg, ax_b_ua], panel_b)

    ax_c = fig.add_subplot(outer[2, 0])
    plot_panel_c(ax_c, panel_c)

    grid_d = outer[3, 0].subgridspec(
        1,
        2,
        wspace=0.30,
    )
    ax_d_cg = fig.add_subplot(grid_d[0, 0])
    ax_d_ua = fig.add_subplot(grid_d[0, 1])
    plot_panel_d([ax_d_cg, ax_d_ua], panel_d_summary)

    return fig, {
        "A": [ax_a],
        "B": [ax_b_cg, ax_b_ua],
        "C": [ax_c],
        "D": [ax_d_cg, ax_d_ua],
    }


def axes_bbox_inches(
    fig: plt.Figure,
    axes: list[plt.Axes],
    *,
    pad_fraction_x: float = 0.18,
    pad_fraction_y: float = 0.25,
) -> Bbox:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [axis.get_tightbbox(renderer) for axis in axes]
    return (
        Bbox.union(boxes)
        .transformed(fig.dpi_scale_trans.inverted())
        .expanded(1 + pad_fraction_x, 1 + pad_fraction_y)
    )


def save_figure_outputs(
    fig: plt.Figure,
    axes_by_panel: dict[str, list[plt.Axes]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "figure4_taxonwide_shannon_concentration"
    svg_metadata = {
        "Title": (
            "Taxon-wide Shannon positional concentration of mature-miRNA "
            "k-mers"
        ),
        "Description": (
            "Permutation effects for ten mutually exclusive taxonomic "
            "groups; positional concentration equals one minus normalized "
            "Shannon entropy over start positions 0-19."
        ),
    }
    pdf_metadata = {
        "Title": svg_metadata["Title"],
        "Subject": svg_metadata["Description"],
    }
    fig.savefig(base.with_suffix(".svg"), metadata=svg_metadata)
    fig.savefig(base.with_suffix(".pdf"), metadata=pdf_metadata)
    fig.savefig(base.with_suffix(".png"), dpi=600)

    panel_names = {
        "A": "figure4_panel_A_2mer_concentration_landscape.svg",
        "B": "figure4_panel_B_CG_UA_two_nulls.svg",
        "C": "figure4_panel_C_significant_repertoire_share.svg",
        "D": "figure4_panel_D_3mer_contexts.svg",
    }
    all_axes = list(fig.axes)
    axis_visibility = {axis: axis.get_visible() for axis in all_axes}
    for panel, axes in axes_by_panel.items():
        selected = set(axes)
        for axis in all_axes:
            axis.set_visible(axis in selected)
        crop = axes_bbox_inches(fig, axes)
        fig.savefig(
            output_dir / panel_names[panel],
            format="svg",
            bbox_inches=crop,
            pad_inches=0.03,
        )
    for axis, visible in axis_visibility.items():
        axis.set_visible(visible)


def write_plotting_tables(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d_raw: pd.DataFrame,
    panel_d_summary: pd.DataFrame,
    panel_inventory: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_a.sort_values(
        ["taxon_leaf_order", "motif_order"]
    ).to_csv(
        output_dir / "figure4_panel_A_plotting_data.csv",
        index=False,
    )
    panel_b.sort_values(
        ["taxon_leaf_order", "kmer", "null_model"]
    ).to_csv(
        output_dir / "figure4_panel_B_plotting_data.csv",
        index=False,
    )
    panel_c.sort_values(
        ["taxon_leaf_order", "k"]
    ).to_csv(
        output_dir / "figure4_panel_C_plotting_data.csv",
        index=False,
    )
    panel_d_raw.sort_values(
        ["context_class", "kmer", "taxon_leaf_order"]
    ).to_csv(
        output_dir / "figure4_panel_D_context_taxon_data.csv",
        index=False,
    )
    panel_d_summary.to_csv(
        output_dir / "figure4_panel_D_context_summary.csv",
        index=False,
    )
    panel_inventory.sort_values("taxon_leaf_order").to_csv(
        output_dir / "figure4_taxon_inventory.csv",
        index=False,
    )


def build_figure(results_root: Path, output_dir: Path) -> Path:
    configure_matplotlib()
    (
        panel_a,
        panel_b,
        panel_c,
        panel_d_raw,
        panel_d_summary,
        panel_inventory,
    ) = load_panel_data(results_root)
    write_plotting_tables(
        panel_a,
        panel_b,
        panel_c,
        panel_d_raw,
        panel_d_summary,
        panel_inventory,
        output_dir,
    )
    fig, axes_by_panel = create_figure(
        panel_a,
        panel_b,
        panel_c,
        panel_d_summary,
    )
    save_figure_outputs(fig, axes_by_panel, output_dir)
    plt.close(fig)
    return output_dir / "figure4_taxonwide_shannon_concentration.svg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Extracted final permutation-results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for editable figures and plotting-data tables",
    )
    args = parser.parse_args()
    result = build_figure(args.results_root, args.output_dir)
    print(f"Created: {result}")


if __name__ == "__main__":
    main()
