"""Create the taxon-wide CG/UA permutation-results Figure 3.

The figure deliberately excludes the globally pooled repertoire from taxon
comparisons. Human is represented as one mutually exclusive terminal group in
the mammalian portion of a schematic cladogram.

Panels
------
A. Overall relative depletion for CG and UA under both null models.
B. Position-resolved CG effects under the position-wise, length-stratified null.
C. Position-resolved UA effects under the same strict null.
D. Taxon-wide contrast of the two null models for UA at start position 0.

All panels are derived directly from the combined final result tables produced
by ``permutation_kmer_statistics.py``.
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
import numpy as np
import pandas as pd


NULLS = {
    "within_sequence": {
        "label": "Within-sequence null",
        "marker": "o",
        "linestyle": "-",
    },
    "positionwise_length_stratified": {
        "label": "Position-wise null",
        "marker": "D",
        "linestyle": "--",
    },
}

# Leaf order follows the displayed cladogram, not a "lower-to-higher" ladder.
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
TAXON_INDEX = {dataset: index for index, (dataset, _) in enumerate(TAXA)}
Y_POSITION = {
    dataset: len(TAXA) - 1 - index
    for index, (dataset, _) in enumerate(TAXA)
}

CG_COLOR = "#0072B2"
UA_COLOR = "#D55E00"
TEXT_COLOR = "#222222"
MUTED_TEXT = "#61676D"
NEUTRAL = "#AEB4BB"
LIGHT_NEUTRAL = "#CBD0D5"
GRID_COLOR = "#DDE1E5"
ZERO_COLOR = "#50565C"
WHITE = "#FFFFFF"
TREE_COLOR = "#555B61"

PANEL_LETTER_SIZE = 9
PANEL_TITLE_SIZE = 7.7
AXIS_LABEL_SIZE = 6.5
TICK_SIZE = 5.6
ANNOTATION_SIZE = 5.2

PROFILE_SCALE_PP_PER_ROW = 9.0


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
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


def locate_results_root(path: Path) -> Path:
    """Accept the final-results root or an ancestor containing it."""
    path = Path(path).expanduser().resolve()
    required = {
        "all_datasets_overall_abundance_permutation_results.csv",
        "all_datasets_positional_permutation_results.csv.gz",
        "dataset_inventory.csv",
    }
    candidates = [path]
    candidates.extend(
        candidate.parent
        for candidate in path.rglob(
            "all_datasets_overall_abundance_permutation_results.csv"
        )
    )
    valid = []
    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            if candidate not in valid:
                valid.append(candidate)
    if len(valid) == 1:
        return valid[0]
    if not valid:
        raise FileNotFoundError(
            "Could not locate the combined final permutation result tables "
            f"beneath {path}"
        )
    raise RuntimeError(
        "More than one final-results directory was found. Pass the intended "
        "directory explicitly:\n" + "\n".join(map(str, valid))
    )


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def load_panel_data(
    results_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, filter, derive and validate the exact four plotting tables."""
    results_root = locate_results_root(results_root)
    overall_path = (
        results_root
        / "all_datasets_overall_abundance_permutation_results.csv"
    )
    positional_path = (
        results_root
        / "all_datasets_positional_permutation_results.csv.gz"
    )
    inventory_path = results_root / "dataset_inventory.csv"

    overall = pd.read_csv(overall_path)
    positional = pd.read_csv(positional_path)
    inventory = pd.read_csv(inventory_path)

    require_columns(
        overall,
        {
            "dataset",
            "analysis_scope",
            "null_model",
            "k",
            "kmer",
            "observed_frequency_percent",
            "null_mean_frequency_percent",
            "observed_minus_null_percentage_points",
            "BH_q_within_dataset_k_null",
            "direction_at_q_le_alpha",
            "n_permutations",
        },
        overall_path.name,
    )
    require_columns(
        positional,
        {
            "dataset",
            "analysis_scope",
            "null_model",
            "k",
            "kmer",
            "start_position_0based",
            "observed_frequency_percent",
            "null_mean_frequency_percent",
            "observed_minus_null_percentage_points",
            "BH_q_within_dataset_k_null",
            "direction_at_q_le_alpha",
            "n_permutations",
        },
        positional_path.name,
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
        inventory.loc[inventory["analysis_scope"] == "taxon", "dataset"]
    )
    if inventory_taxa != taxon_ids:
        raise ValueError(
            "Final-results taxon datasets do not match the intended mutually "
            "exclusive groups.\n"
            f"Missing: {sorted(taxon_ids - inventory_taxa)}\n"
            f"Unexpected: {sorted(inventory_taxa - taxon_ids)}"
        )

    panel_a = overall[
        (overall["analysis_scope"] == "taxon")
        & (overall["k"] == 2)
        & overall["kmer"].isin(["CG", "UA"])
        & overall["dataset"].isin(taxon_ids)
        & overall["null_model"].isin(NULLS)
    ].copy()
    panel_a["taxon_label"] = panel_a["dataset"].map(TAXON_LABEL)
    panel_a["taxon_leaf_order"] = panel_a["dataset"].map(TAXON_INDEX)
    panel_a["significant_q_le_0_05"] = (
        panel_a["BH_q_within_dataset_k_null"] <= 0.05
    )
    panel_a["relative_depletion_percent"] = (
        100
        * (
            panel_a["null_mean_frequency_percent"]
            - panel_a["observed_frequency_percent"]
        )
        / panel_a["null_mean_frequency_percent"]
    )

    panel_bc = positional[
        (positional["analysis_scope"] == "taxon")
        & (positional["k"] == 2)
        & positional["kmer"].isin(["CG", "UA"])
        & positional["dataset"].isin(taxon_ids)
        & (
            positional["null_model"]
            == "positionwise_length_stratified"
        )
        & positional["start_position_0based"].between(0, 19)
    ].copy()
    panel_bc["taxon_label"] = panel_bc["dataset"].map(TAXON_LABEL)
    panel_bc["taxon_leaf_order"] = panel_bc["dataset"].map(TAXON_INDEX)
    panel_bc["significant_q_le_0_05"] = (
        panel_bc["BH_q_within_dataset_k_null"] <= 0.05
    )

    panel_d = positional[
        (positional["analysis_scope"] == "taxon")
        & (positional["k"] == 2)
        & (positional["kmer"] == "UA")
        & positional["dataset"].isin(taxon_ids)
        & positional["null_model"].isin(NULLS)
        & (positional["start_position_0based"] == 0)
    ].copy()
    panel_d["taxon_label"] = panel_d["dataset"].map(TAXON_LABEL)
    panel_d["taxon_leaf_order"] = panel_d["dataset"].map(TAXON_INDEX)
    panel_d["significant_q_le_0_05"] = (
        panel_d["BH_q_within_dataset_k_null"] <= 0.05
    )

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
        "panel_a": 10 * 2 * 2,
        "panel_bc": 10 * 2 * 20,
        "panel_d": 10 * 2,
        "inventory": 10,
    }
    observed_sizes = {
        "panel_a": len(panel_a),
        "panel_bc": len(panel_bc),
        "panel_d": len(panel_d),
        "inventory": len(panel_inventory),
    }
    if observed_sizes != expected_sizes:
        raise ValueError(
            "Unexpected taxon-wide plotting-table sizes. "
            f"Observed {observed_sizes}; expected {expected_sizes}."
        )

    permutation_counts = set(
        pd.concat(
            [
                panel_a["n_permutations"],
                panel_bc["n_permutations"],
                panel_d["n_permutations"],
            ]
        )
        .dropna()
        .astype(int)
    )
    if permutation_counts != {10_000}:
        print(
            "WARNING: expected final 10,000-permutation tables, but found "
            f"{sorted(permutation_counts)}."
        )

    # The focal overall claim should be invariant under both nulls in all taxa.
    unexpected = panel_a[
        ~panel_a["significant_q_le_0_05"]
        | (panel_a["observed_minus_null_percentage_points"] >= 0)
    ]
    if not unexpected.empty:
        raise ValueError(
            "At least one taxon/motif/null block is not significantly depleted "
            "in the overall analysis:\n"
            + unexpected[
                ["dataset", "kmer", "null_model", "direction_at_q_le_alpha"]
            ].to_string(index=False)
        )

    return panel_a, panel_bc, panel_d, panel_inventory


def clean_axis(ax: plt.Axes, *, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(
            True,
            axis=grid_axis,
            color=GRID_COLOR,
            linewidth=0.45,
            zorder=0,
        )
    ax.set_axisbelow(True)


# A nested schematic topology. Leaves are dataset identifiers; branch lengths
# carry no evolutionary-time meaning.
PHYLOGENY = (
    "root",
    [
        "taxon__10_angiosperms",
        (
            "Metazoa",
            [
                (
                    "Ecdysozoa",
                    [
                        "taxon__09_nematodes",
                        "taxon__08_insects",
                    ],
                ),
                (
                    "Chordata",
                    [
                        "taxon__01_fishes_non_tetrapod",
                        (
                            "Tetrapoda",
                            [
                                "taxon__02_amphibians",
                                (
                                    "Amniota",
                                    [
                                        (
                                            "Sauropsida",
                                            [
                                                "taxon__03_non_avian_reptiles",
                                                "taxon__04_aves",
                                            ],
                                        ),
                                        (
                                            "Mammalia",
                                            [
                                                "taxon__07_other_mammals",
                                                (
                                                    "Primates",
                                                    [
                                                        "taxon__06_non_human_primates",
                                                        "taxon__05_human",
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


def tree_center(node) -> float:
    if isinstance(node, str):
        return Y_POSITION[node]
    _, children = node
    child_centers = [tree_center(child) for child in children]
    return float(np.mean(child_centers))


def draw_tree_node(
    ax: plt.Axes,
    node,
    *,
    depth: int,
    depth_step: float = 0.115,
    leaf_x: float = 0.90,
) -> tuple[float, float]:
    if isinstance(node, str):
        return leaf_x, Y_POSITION[node]

    _, children = node
    node_x = 0.06 + depth * depth_step
    child_points = [
        draw_tree_node(
            ax,
            child,
            depth=depth + 1,
            depth_step=depth_step,
            leaf_x=leaf_x,
        )
        for child in children
    ]
    child_y = [point[1] for point in child_points]
    node_y = float(np.mean(child_y))
    ax.plot(
        [node_x, node_x],
        [min(child_y), max(child_y)],
        color=TREE_COLOR,
        linewidth=0.65,
        solid_capstyle="round",
        zorder=2,
    )
    for child_x, y in child_points:
        ax.plot(
            [node_x, child_x],
            [y, y],
            color=TREE_COLOR,
            linewidth=0.65,
            solid_capstyle="round",
            zorder=2,
        )
    return node_x, node_y


def plot_tree_and_taxa(ax: plt.Axes) -> None:
    draw_tree_node(ax, PHYLOGENY, depth=0)
    for dataset, label in TAXA:
        y = Y_POSITION[dataset]
        ax.text(
            0.95,
            y,
            label,
            ha="left",
            va="center",
            fontsize=TICK_SIZE,
            color=TEXT_COLOR,
        )
    ax.set_xlim(0, 1.84)
    ax.set_ylim(-0.65, 9.65)
    ax.axis("off")
    ax.text(
        0.06,
        1.035,
        "Schematic topology",
        transform=ax.transAxes,
        fontsize=ANNOTATION_SIZE,
        color=MUTED_TEXT,
        va="bottom",
    )
    ax.text(
        0.06,
        -0.08,
        "Branch lengths are not scaled.",
        transform=ax.transAxes,
        fontsize=ANNOTATION_SIZE,
        color=MUTED_TEXT,
        va="top",
    )


def spearman_rank_agreement(subset: pd.DataFrame) -> float:
    pivot = subset.pivot(
        index="dataset",
        columns="null_model",
        values="relative_depletion_percent",
    )
    return float(
        pivot["within_sequence"]
        .rank()
        .corr(pivot["positionwise_length_stratified"].rank())
    )


def plot_overall_taxon_effects(
    axes: list[plt.Axes],
    panel_a: pd.DataFrame,
) -> None:
    """Panel A: paired two-null overall relative-depletion effects."""
    motifs = [("CG", CG_COLOR), ("UA", UA_COLOR)]
    maximum = panel_a["relative_depletion_percent"].max()
    x_max = max(70, math.ceil(maximum / 10) * 10)

    for ax, (motif, color) in zip(axes, motifs):
        subset = panel_a[panel_a["kmer"] == motif]
        for dataset, _ in TAXA:
            rows = subset[subset["dataset"] == dataset].set_index(
                "null_model"
            )
            y = Y_POSITION[dataset]
            x_values = [
                float(rows.loc[null_model, "relative_depletion_percent"])
                for null_model in NULLS
            ]
            ax.plot(
                x_values,
                [y, y],
                color=LIGHT_NEUTRAL,
                linewidth=0.7,
                zorder=1,
            )
            for null_model, style in NULLS.items():
                row = rows.loc[null_model]
                significant = bool(row["significant_q_le_0_05"])
                ax.scatter(
                    row["relative_depletion_percent"],
                    y,
                    marker=style["marker"],
                    s=19,
                    facecolor=color if significant else WHITE,
                    edgecolor=color,
                    linewidth=0.7,
                    zorder=3,
                )

        ax.axvline(0, color=ZERO_COLOR, linewidth=0.75, zorder=0)
        ax.set_xlim(-2, x_max)
        ax.set_ylim(-0.65, 9.65)
        ax.set_yticks([])
        ax.set_title(
            f"{motif} relative depletion",
            color=color,
            fontweight="bold",
            pad=3,
        )
        ax.set_xlabel("Relative depletion from null mean (%)")
        clean_axis(ax, grid_axis="x")
        rho = spearman_rank_agreement(subset)
        ax.text(
            0.02,
            -0.16,
            f"Taxon-rank agreement between nulls: ρ = {rho:.2f}",
            transform=ax.transAxes,
            fontsize=ANNOTATION_SIZE,
            color=MUTED_TEXT,
            ha="left",
            va="top",
        )

    axes[0].text(
        -0.94,
        1.12,
        "A",
        transform=axes[0].transAxes,
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        va="top",
    )
    axes[0].text(
        -0.80,
        1.12,
        "Overall taxon-wide depletion under two null models",
        transform=axes[0].transAxes,
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        va="top",
    )


def plot_positional_ridgelines(
    ax: plt.Axes,
    panel_bc: pd.DataFrame,
    *,
    motif: str,
    panel_letter: str,
    show_taxon_labels: bool,
) -> None:
    """Panel B/C: stacked same-scale position-effect profile strips."""
    color = CG_COLOR if motif == "CG" else UA_COLOR
    subset = panel_bc[panel_bc["kmer"] == motif]

    for dataset, label in TAXA:
        y_base = Y_POSITION[dataset]
        rows = subset[subset["dataset"] == dataset].sort_values(
            "start_position_0based"
        )
        positions = rows["start_position_0based"].to_numpy()
        effects = rows[
            "observed_minus_null_percentage_points"
        ].to_numpy()
        y = y_base + effects / PROFILE_SCALE_PP_PER_ROW
        significant = rows["significant_q_le_0_05"].to_numpy()

        ax.plot(
            [-0.2, 19.2],
            [y_base, y_base],
            color=GRID_COLOR,
            linewidth=0.45,
            zorder=0,
        )
        ax.fill_between(
            positions,
            y_base,
            y,
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            positions,
            y,
            color=color,
            linewidth=0.78,
            zorder=2,
        )
        ax.scatter(
            positions[significant],
            y[significant],
            s=8,
            marker="o",
            facecolor=color,
            edgecolor=color,
            linewidth=0.35,
            zorder=3,
        )
        ax.scatter(
            positions[~significant],
            y[~significant],
            s=10,
            marker="o",
            facecolor=WHITE,
            edgecolor=color,
            linewidth=0.65,
            zorder=4,
        )

    if motif == "UA":
        ax.axvspan(-0.42, 0.42, color=UA_COLOR, alpha=0.06, zorder=0)
        ax.text(
            0.25,
            9.52,
            "5′ start",
            fontsize=ANNOTATION_SIZE,
            color=UA_COLOR,
            va="center",
            ha="left",
        )

    # Common scale bar, placed outside the 0–19 positional range.
    scale_x = 20.0
    scale_y = 8.85
    bar_height = 2.0 / PROFILE_SCALE_PP_PER_ROW
    ax.plot(
        [scale_x, scale_x],
        [scale_y, scale_y + bar_height],
        color=ZERO_COLOR,
        linewidth=0.8,
        clip_on=False,
    )
    ax.plot(
        [scale_x - 0.10, scale_x + 0.10],
        [scale_y, scale_y],
        color=ZERO_COLOR,
        linewidth=0.8,
        clip_on=False,
    )
    ax.plot(
        [scale_x - 0.10, scale_x + 0.10],
        [scale_y + bar_height, scale_y + bar_height],
        color=ZERO_COLOR,
        linewidth=0.8,
        clip_on=False,
    )
    ax.text(
        scale_x + 0.18,
        scale_y + bar_height / 2,
        "2 pp",
        fontsize=ANNOTATION_SIZE,
        color=MUTED_TEXT,
        va="center",
        ha="left",
    )

    ax.set_xlim(-0.55, 21.2)
    ax.set_ylim(-0.72, 9.72)
    ax.set_xticks([0, 5, 10, 15, 19])
    ax.set_yticks([Y_POSITION[dataset] for dataset, _ in TAXA])
    if show_taxon_labels:
        ax.set_yticklabels([label for _, label in TAXA])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_xlabel("K-mer start position (0-based)")
    ax.set_title(
        f"{motif} positional effects",
        color=color,
        fontweight="bold",
        pad=3,
    )
    clean_axis(ax)
    ax.text(
        -0.18,
        1.10,
        panel_letter,
        transform=ax.transAxes,
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0,
        1.10,
        "Position-wise, length-stratified null",
        transform=ax.transAxes,
        fontsize=ANNOTATION_SIZE,
        color=MUTED_TEXT,
        va="top",
    )
    ax.text(
        0,
        -0.13,
        "Filled points: BH q ≤ 0.05; open points: BH q > 0.05.",
        transform=ax.transAxes,
        fontsize=ANNOTATION_SIZE,
        color=MUTED_TEXT,
        va="top",
    )


def plot_ua_position_zero(ax: plt.Axes, panel_d: pd.DataFrame) -> None:
    """Panel D: paired null-model effects for the focal 5′ UA position."""
    minimum = panel_d["observed_minus_null_percentage_points"].min()
    maximum = panel_d["observed_minus_null_percentage_points"].max()
    x_min = math.floor(minimum) - 1
    data_x_max = math.ceil(maximum) + 1
    raw_text_x = data_x_max + 1.4
    x_max = raw_text_x + 2.0

    for dataset, _ in TAXA:
        y = Y_POSITION[dataset]
        rows = panel_d[panel_d["dataset"] == dataset].set_index("null_model")
        x_values = [
            float(rows.loc[null_model, "observed_minus_null_percentage_points"])
            for null_model in NULLS
        ]
        ax.plot(
            x_values,
            [y, y],
            color=LIGHT_NEUTRAL,
            linewidth=0.75,
            zorder=1,
        )
        for null_model, style in NULLS.items():
            row = rows.loc[null_model]
            significant = bool(row["significant_q_le_0_05"])
            ax.scatter(
                row["observed_minus_null_percentage_points"],
                y,
                marker=style["marker"],
                s=22,
                facecolor=UA_COLOR if significant else WHITE,
                edgecolor=UA_COLOR,
                linewidth=0.75,
                zorder=3,
            )
        raw_frequency = float(
            rows.loc["within_sequence", "observed_frequency_percent"]
        )
        ax.text(
            raw_text_x,
            y,
            f"{raw_frequency:.1f}%",
            fontsize=TICK_SIZE,
            color=UA_COLOR,
            ha="center",
            va="center",
        )

    for dataset, _ in TAXA:
        y = Y_POSITION[dataset]
        ax.plot(
            [x_min, x_max],
            [y, y],
            color=GRID_COLOR,
            linewidth=0.38,
            zorder=0,
        )
    ax.axvline(0, color=ZERO_COLOR, linewidth=0.8, zorder=1)
    ax.axvline(
        data_x_max + 0.5,
        color=GRID_COLOR,
        linewidth=0.55,
        zorder=0,
    )
    ax.text(
        raw_text_x,
        9.62,
        "Observed UA at position 0",
        fontsize=ANNOTATION_SIZE,
        color=MUTED_TEXT,
        ha="center",
        va="bottom",
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.65, 9.78)
    ax.set_yticks([Y_POSITION[dataset] for dataset, _ in TAXA])
    ax.set_yticklabels([label for _, label in TAXA])
    ax.set_xlabel(
        "Observed − null frequency at position 0 (percentage points)"
    )
    clean_axis(ax, grid_axis="x")
    ax.text(
        -0.075,
        1.10,
        "D",
        transform=ax.transAxes,
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0,
        1.10,
        "UA at the 5′ start: null-model contrast across taxa",
        transform=ax.transAxes,
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        va="top",
    )


def shared_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=TEXT_COLOR,
            markerfacecolor=NEUTRAL,
            markeredgecolor=TEXT_COLOR,
            linewidth=0.8,
            markersize=4,
            label="Within-sequence null",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="--",
            color=TEXT_COLOR,
            markerfacecolor=NEUTRAL,
            markeredgecolor=TEXT_COLOR,
            linewidth=0.8,
            markersize=4,
            label="Position-wise null",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=TEXT_COLOR,
            markeredgecolor=TEXT_COLOR,
            markersize=4,
            label="BH q ≤ 0.05",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=WHITE,
            markeredgecolor=TEXT_COLOR,
            markersize=4,
            label="BH q > 0.05",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.60, 0.996),
        ncol=4,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.15,
        borderaxespad=0,
    )


def create_figure(
    panel_a: pd.DataFrame,
    panel_bc: pd.DataFrame,
    panel_d: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, list[plt.Axes]]]:
    width_inches = 183 / 25.4
    height_inches = 240 / 25.4
    fig = plt.figure(figsize=(width_inches, height_inches))
    outer = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.03, 1.25, 1.02],
        width_ratios=[1, 1],
        left=0.11,
        right=0.985,
        bottom=0.055,
        top=0.955,
        hspace=0.42,
        wspace=0.30,
    )

    grid_a = outer[0, :].subgridspec(
        1,
        3,
        width_ratios=[1.15, 1.0, 1.0],
        wspace=0.20,
    )
    ax_tree = fig.add_subplot(grid_a[0, 0])
    ax_a_cg = fig.add_subplot(grid_a[0, 1])
    ax_a_ua = fig.add_subplot(grid_a[0, 2])
    plot_tree_and_taxa(ax_tree)
    plot_overall_taxon_effects([ax_a_cg, ax_a_ua], panel_a)

    ax_b = fig.add_subplot(outer[1, 0])
    ax_c = fig.add_subplot(outer[1, 1])
    plot_positional_ridgelines(
        ax_b,
        panel_bc,
        motif="CG",
        panel_letter="B",
        show_taxon_labels=True,
    )
    plot_positional_ridgelines(
        ax_c,
        panel_bc,
        motif="UA",
        panel_letter="C",
        show_taxon_labels=False,
    )

    ax_d = fig.add_subplot(outer[2, :])
    plot_ua_position_zero(ax_d, panel_d)
    shared_legend(fig)

    return fig, {
        "A": [ax_tree, ax_a_cg, ax_a_ua],
        "B": [ax_b],
        "C": [ax_c],
        "D": [ax_d],
    }


def axes_bbox_inches(
    fig: plt.Figure,
    axes: list[plt.Axes],
    *,
    pad_fraction_x: float = 0.16,
    pad_fraction_y: float = 0.22,
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
    base = output_dir / "figure3_taxonwide_permutation_results"
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=600)

    panel_names = {
        "A": "figure3_panel_A_overall_taxon_effects.svg",
        "B": "figure3_panel_B_CG_positional_profiles.svg",
        "C": "figure3_panel_C_UA_positional_profiles.svg",
        "D": "figure3_panel_D_UA_position0_contrast.svg",
    }
    all_axes = list(fig.axes)
    axis_visibility = {axis: axis.get_visible() for axis in all_axes}
    ytick_labels = {
        axis: [label.get_text() for label in axis.get_yticklabels()]
        for axis in all_axes
    }
    legend_visibility = {
        legend: legend.get_visible() for legend in fig.legends
    }
    for panel, axes in axes_by_panel.items():
        selected = set(axes)
        for axis in all_axes:
            axis.set_visible(axis in selected)
        for legend in fig.legends:
            legend.set_visible(panel == "A")
        # The assembled figure omits repeated taxon labels from Panel C.
        # Restore them only in its standalone SVG so that every exported panel
        # remains interpretable when edited separately in Inkscape.
        if panel == "C":
            axes[0].set_yticklabels([label for _, label in TAXA])
        crop = axes_bbox_inches(fig, axes)
        fig.savefig(
            output_dir / panel_names[panel],
            format="svg",
            bbox_inches=crop,
            pad_inches=0.03,
        )
        if panel == "C":
            axes[0].set_yticklabels(ytick_labels[axes[0]])
    for axis, visible in axis_visibility.items():
        axis.set_visible(visible)
    for legend, visible in legend_visibility.items():
        legend.set_visible(visible)


def write_plotting_tables(
    panel_a: pd.DataFrame,
    panel_bc: pd.DataFrame,
    panel_d: pd.DataFrame,
    panel_inventory: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_a.to_csv(
        output_dir / "figure3_panel_A_plotting_data.csv",
        index=False,
    )
    panel_bc[panel_bc["kmer"] == "CG"].to_csv(
        output_dir / "figure3_panel_B_plotting_data.csv",
        index=False,
    )
    panel_bc[panel_bc["kmer"] == "UA"].to_csv(
        output_dir / "figure3_panel_C_plotting_data.csv",
        index=False,
    )
    panel_d.to_csv(
        output_dir / "figure3_panel_D_plotting_data.csv",
        index=False,
    )
    panel_inventory.sort_values("taxon_leaf_order").to_csv(
        output_dir / "figure3_taxon_inventory.csv",
        index=False,
    )


def build_figure(results_root: Path, output_dir: Path) -> Path:
    configure_matplotlib()
    panel_a, panel_bc, panel_d, panel_inventory = load_panel_data(
        results_root
    )
    write_plotting_tables(
        panel_a,
        panel_bc,
        panel_d,
        panel_inventory,
        output_dir,
    )
    fig, axes_by_panel = create_figure(panel_a, panel_bc, panel_d)
    save_figure_outputs(fig, axes_by_panel, output_dir)
    plt.close(fig)
    return output_dir / "figure3_taxonwide_permutation_results.svg"


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
