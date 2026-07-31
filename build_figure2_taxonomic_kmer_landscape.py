#!/usr/bin/env python3
"""
Build publication Figure 2 from the taxon-wise mature-miRNA k-mer outputs.

The figure deliberately uses only observed frequencies:
  A. taxonomy-aligned dinucleotide bubble atlas
  B. CG positional profiles
  C. UA positional profiles
  D. organism-level PCA of complete positional dinucleotide profiles

The SVG is written directly so that text remains editable in Inkscape.
All axis/tick text is Arial at >=5 pt.

Dependencies: Python 3, numpy, pandas.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 192.0
POSITIONS = list(range(20))
KMERS = [a + b for a in "ACGU" for b in "ACGU"]
PCA_MIN_SEQUENCES = 100

GROUP_ORDER = [
    "Angiosperms",
    "Nematodes",
    "Insects",
    "Fishes (non-tetrapod vertebrates)",
    "Amphibians",
    "Non-avian reptiles",
    "Aves",
    "Other mammals",
    "Non-human primates",
    "Human",
]

DISPLAY_NAME = {
    "Angiosperms": "Angiosperms",
    "Nematodes": "Nematodes",
    "Insects": "Insects",
    "Fishes (non-tetrapod vertebrates)": "Non-tetrapod fishes",
    "Amphibians": "Amphibians",
    "Non-avian reptiles": "Non-avian reptiles",
    "Aves": "Aves",
    "Other mammals": "Other mammals",
    "Non-human primates": "Non-human primates",
    "Human": "Human",
}

ABBREVIATION = {
    "Angiosperms": "Angio.",
    "Nematodes": "Nemat.",
    "Insects": "Insects",
    "Fishes (non-tetrapod vertebrates)": "Fishes",
    "Amphibians": "Amph.",
    "Non-avian reptiles": "Reptiles",
    "Aves": "Aves",
    "Other mammals": "Mammals",
    "Non-human primates": "Primates",
    "Human": "Human",
}

GROUP_COLOR = {
    "Angiosperms": "#3A7D44",
    "Nematodes": "#C58A1B",
    "Insects": "#D55E00",
    "Fishes (non-tetrapod vertebrates)": "#56A9D6",
    "Amphibians": "#009E73",
    "Non-avian reptiles": "#0072B2",
    "Aves": "#8E5EA2",
    "Other mammals": "#6F7F8F",
    "Non-human primates": "#3B528B",
    "Human": "#111111",
}

GROUP_MARKER = {
    "Angiosperms": "circle",
    "Nematodes": "diamond",
    "Insects": "triangle",
    "Fishes (non-tetrapod vertebrates)": "downtriangle",
    "Amphibians": "square",
    "Non-avian reptiles": "pentagon",
    "Aves": "triangle",
    "Other mammals": "circle",
    "Non-human primates": "diamond",
    "Human": "star",
}

CG_COLOR = "#126E75"
UA_COLOR = "#D55E00"
TEXT_COLOR = "#202124"
MUTED_COLOR = "#676B70"
GRID_COLOR = "#D8DADD"
BRANCH_COLOR = "#555A60"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fnum(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def attrs(**kwargs: object) -> str:
    out = []
    for key, value in kwargs.items():
        if value is None:
            continue
        key = key.replace("_", "-")
        out.append(f'{key}="{esc(value)}"')
    return " ".join(out)


def text(
    x: float,
    y: float,
    value: object,
    *,
    size: float = 5.2,
    weight: str = "normal",
    anchor: str = "start",
    fill: str = TEXT_COLOR,
    rotate: float | None = None,
    italic: bool = False,
    extra: str = "",
) -> str:
    transform = None if rotate is None else f"rotate({rotate} {fnum(x)} {fnum(y)})"
    # The SVG viewBox is expressed in millimetres. Convert typographic points
    # to millimetre-valued user units so 5 pt renders as 1.764 mm physically.
    size_user_units = size * 25.4 / 72.0
    attribute_text = attrs(
        x=fnum(x),
        y=fnum(y),
        fill=fill,
        font_family="Arial, Helvetica, sans-serif",
        font_size=fnum(size_user_units),
        font_weight=weight,
        font_style="italic" if italic else None,
        text_anchor=anchor,
        transform=transform,
        data_font_size_pt=fnum(size),
    )
    return f"<text {attribute_text} {extra}>{esc(value)}</text>"


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = BRANCH_COLOR,
    width: float = 0.22,
    opacity: float = 1.0,
    dash: str | None = None,
) -> str:
    attribute_text = attrs(
        x1=fnum(x1),
        y1=fnum(y1),
        x2=fnum(x2),
        y2=fnum(y2),
        stroke=stroke,
        stroke_width=fnum(width),
        stroke_opacity=fnum(opacity),
        stroke_dasharray=dash,
        vector_effect="non-scaling-stroke",
    )
    return f"<line {attribute_text} />"


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = "none",
    stroke: str = "none",
    width: float = 0.2,
    opacity: float = 1.0,
    rx: float | None = None,
    extra: str = "",
) -> str:
    attribute_text = attrs(
        x=fnum(x),
        y=fnum(y),
        width=fnum(w),
        height=fnum(h),
        fill=fill,
        stroke=stroke,
        stroke_width=fnum(width),
        opacity=fnum(opacity),
        rx=fnum(rx) if rx is not None else None,
        vector_effect="non-scaling-stroke",
    )
    return f"<rect {attribute_text} {extra}/>"


def circle(
    cx: float,
    cy: float,
    r: float,
    *,
    fill: str,
    stroke: str = "none",
    width: float = 0.2,
    opacity: float = 1.0,
    extra: str = "",
) -> str:
    attribute_text = attrs(
        cx=fnum(cx),
        cy=fnum(cy),
        r=fnum(r),
        fill=fill,
        stroke=stroke,
        stroke_width=fnum(width),
        opacity=fnum(opacity),
        vector_effect="non-scaling-stroke",
    )
    if extra.lstrip().startswith("<"):
        return f"<circle {attribute_text}>{extra}</circle>"
    return f"<circle {attribute_text} {extra}/>"


def polyline(
    points: list[tuple[float, float]],
    *,
    fill: str = "none",
    stroke: str = TEXT_COLOR,
    width: float = 0.25,
    opacity: float = 1.0,
    close: bool = False,
    extra: str = "",
) -> str:
    pts = " ".join(f"{fnum(x)},{fnum(y)}" for x, y in points)
    tag = "polygon" if close else "polyline"
    attribute_text = attrs(
        points=pts,
        fill=fill,
        stroke=stroke,
        stroke_width=fnum(width),
        opacity=fnum(opacity),
        stroke_linejoin="round",
        stroke_linecap="round",
        vector_effect="non-scaling-stroke",
    )
    return f"<{tag} {attribute_text} {extra}/>"


def marker(
    x: float,
    y: float,
    shape: str,
    size: float,
    color: str,
    *,
    opacity: float = 1.0,
    stroke: str = "#FFFFFF",
    width: float = 0.18,
) -> str:
    if shape == "circle":
        return circle(x, y, size, fill=color, stroke=stroke, width=width, opacity=opacity)
    if shape == "square":
        return rect(
            x - size,
            y - size,
            2 * size,
            2 * size,
            fill=color,
            stroke=stroke,
            width=width,
            opacity=opacity,
        )

    if shape == "diamond":
        angles = [-90, 0, 90, 180]
    elif shape == "triangle":
        angles = [-90, 30, 150]
    elif shape == "downtriangle":
        angles = [90, 210, 330]
    elif shape == "pentagon":
        angles = [-90, -18, 54, 126, 198]
    elif shape == "star":
        angles = []
        for i in range(10):
            angles.append(-90 + i * 36)
        points = []
        for i, angle in enumerate(angles):
            radius = size if i % 2 == 0 else size * 0.43
            rad = math.radians(angle)
            points.append((x + radius * math.cos(rad), y + radius * math.sin(rad)))
        return polyline(
            points,
            fill=color,
            stroke=stroke,
            width=width,
            opacity=opacity,
            close=True,
        )
    else:
        raise ValueError(f"Unknown marker shape: {shape}")

    points = []
    for angle in angles:
        rad = math.radians(angle)
        points.append((x + size * math.cos(rad), y + size * math.sin(rad)))
    return polyline(
        points,
        fill=color,
        stroke=stroke,
        width=width,
        opacity=opacity,
        close=True,
    )


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    vals = tuple(max(0, min(255, int(round(x)))) for x in rgb)
    return "#" + "".join(f"{x:02X}" for x in vals)


def mix_color(low: str, high: str, fraction: float) -> str:
    fraction = max(0.0, min(1.0, fraction))
    a = hex_to_rgb(low)
    b = hex_to_rgb(high)
    return rgb_to_hex(tuple(x + fraction * (y - x) for x, y in zip(a, b)))


def positional_concentration(values: np.ndarray) -> float:
    """1 - normalized Shannon entropy across positions; 0 means perfectly even."""
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total <= 0:
        return 0.0
    q = values / total
    q = q[q > 0]
    return float(1.0 + np.sum(q * np.log(q)) / np.log(len(values)))


def macroclade(group: str) -> str:
    if group == "Angiosperms":
        return "Angiosperms"
    if group in {"Nematodes", "Insects"}:
        return "Invertebrate animals"
    return "Vertebrates"


def load_and_prepare(input_dir: Path):
    aggregate_path = input_dir / "taxonwise_2mer_aggregate_tidy.csv"
    organism_path = input_dir / "organism_level_2mer_positional_tidy.csv.gz"
    membership_path = input_dir / "selected_taxon_membership_observed_counts.csv"

    for path in (aggregate_path, organism_path, membership_path):
        if not path.exists():
            raise FileNotFoundError(path)

    aggregate = pd.read_csv(aggregate_path)
    aggregate = aggregate[
        (aggregate["start_position_0based"].isin(POSITIONS))
        & (aggregate["kmer"].isin(KMERS))
    ].copy()

    observed = pd.read_csv(membership_path)
    group_counts = (
        observed.groupby("analysis_group", as_index=True)
        .agg(
            organisms=("organism_prefix", "nunique"),
            unique_sequences=("observed_unique_sequences", "sum"),
        )
        .reindex(GROUP_ORDER)
    )

    atlas_rows = []
    for group in GROUP_ORDER:
        subset = aggregate[aggregate["analysis_group"].eq(group)]
        for kmer in KMERS:
            profile = (
                subset[subset["kmer"].eq(kmer)]
                .set_index("start_position_0based")
                .reindex(POSITIONS)["pooled_frequency_percent"]
                .to_numpy(dtype=float)
            )
            if np.isnan(profile).any():
                raise ValueError(f"Incomplete profile for {group}, {kmer}")
            atlas_rows.append(
                {
                    "analysis_group": group,
                    "kmer": kmer,
                    "mean_observed_frequency_percent": float(profile.mean()),
                    "positional_concentration": positional_concentration(profile),
                }
            )
    atlas = pd.DataFrame(atlas_rows)

    profiles = aggregate[aggregate["kmer"].isin(["CG", "UA"])][
        [
            "analysis_group",
            "start_position_0based",
            "kmer",
            "pooled_count",
            "pooled_eligible_unique_sequences",
            "pooled_frequency_percent",
            "organism_balanced_mean_frequency_percent",
            "contributing_organisms",
        ]
    ].copy()
    profiles["analysis_group"] = pd.Categorical(
        profiles["analysis_group"], categories=GROUP_ORDER, ordered=True
    )
    profiles = profiles.sort_values(
        ["kmer", "analysis_group", "start_position_0based"]
    ).reset_index(drop=True)
    profiles["analysis_group"] = profiles["analysis_group"].astype(str)

    organism = pd.read_csv(organism_path)
    baseline = (
        organism[organism["start_position_0based"].eq(0)]
        .groupby(
            ["organism_prefix", "mirbase_organism_name", "analysis_group"],
            as_index=False,
        )["eligible_unique_sequences"]
        .max()
    )
    keep = baseline.loc[
        baseline["eligible_unique_sequences"].ge(PCA_MIN_SEQUENCES),
        "organism_prefix",
    ]
    pca_long = organism[
        organism["organism_prefix"].isin(keep)
        & organism["start_position_0based"].isin(POSITIONS)
        & organism["kmer"].isin(KMERS)
    ].copy()
    wide = pca_long.pivot(
        index="organism_prefix",
        columns=["start_position_0based", "kmer"],
        values="frequency_percent",
    )
    expected_columns = pd.MultiIndex.from_product(
        [POSITIONS, KMERS], names=["start_position_0based", "kmer"]
    )
    wide = wide.reindex(columns=expected_columns)
    if wide.isna().any().any():
        missing = int(wide.isna().sum().sum())
        raise ValueError(f"PCA feature matrix contains {missing} missing values")

    feature_means = wide.mean(axis=0).to_numpy(dtype=float)
    matrix = wide.to_numpy(dtype=float) - feature_means
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    scores = u[:, :2] * singular[:2]
    variance = singular**2 / (len(matrix) - 1)
    explained = variance / variance.sum()

    pca = (
        pd.DataFrame(
            {
                "organism_prefix": wide.index,
                "PC1": scores[:, 0],
                "PC2": scores[:, 1],
            }
        )
        .merge(baseline, on="organism_prefix", how="left")
        .reset_index(drop=True)
    )

    # Stabilize otherwise arbitrary PCA signs for reproducible visual orientation:
    # plants to the right on PC1; invertebrate animals upward on PC2.
    angio_pc1 = pca.loc[pca["analysis_group"].eq("Angiosperms"), "PC1"].mean()
    other_pc1 = pca.loc[~pca["analysis_group"].eq("Angiosperms"), "PC1"].mean()
    if angio_pc1 < other_pc1:
        pca["PC1"] *= -1
        vt[0, :] *= -1

    invert_groups = {"Nematodes", "Insects"}
    invert_pc2 = pca.loc[pca["analysis_group"].isin(invert_groups), "PC2"].mean()
    other_pc2 = pca.loc[~pca["analysis_group"].isin(invert_groups), "PC2"].mean()
    if invert_pc2 < other_pc2:
        pca["PC2"] *= -1
        vt[1, :] *= -1

    pca["macroclade"] = pca["analysis_group"].map(macroclade)

    loadings = pd.DataFrame(
        {
            "start_position_0based": [c[0] for c in expected_columns],
            "kmer": [c[1] for c in expected_columns],
            "feature_mean_frequency_percent": feature_means,
            "PC1_loading": vt[0, :],
            "PC2_loading": vt[1, :],
        }
    )

    return aggregate, group_counts, atlas, profiles, pca, loadings, explained


def add_panel_a(
    parts: list[str],
    atlas: pd.DataFrame,
    group_counts: pd.DataFrame,
) -> None:
    parts.append('<g id="panel-A_taxonomic_motif_atlas">')
    parts.append(text(2.0, 7.0, "A", size=9.0, weight="bold"))
    parts.append(
        text(
            9.0,
            7.0,
            "Taxonomy-aligned observed dinucleotide landscape",
            size=7.0,
            weight="bold",
        )
    )
    parts.append(
        text(
            9.0,
            11.0,
            "Bubble area: mean observed frequency; colour: positional concentration",
            size=5.0,
            fill=MUTED_COLOR,
        )
    )

    row_y = {group: 21.0 + i * 8.1 for i, group in enumerate(GROUP_ORDER)}
    leaf_x = 29.5
    label_x = 72.0
    atlas_x0 = 78.0
    atlas_dx = 6.35

    # Taxonomy-based rectangular cladogram (topology only; no branch lengths).
    y_angio = row_y["Angiosperms"]
    y_nem = row_y["Nematodes"]
    y_ins = row_y["Insects"]
    y_fish = row_y["Fishes (non-tetrapod vertebrates)"]
    y_amph = row_y["Amphibians"]
    y_rep = row_y["Non-avian reptiles"]
    y_aves = row_y["Aves"]
    y_mamm = row_y["Other mammals"]
    y_prim = row_y["Non-human primates"]
    y_human = row_y["Human"]

    y_ecdy = (y_nem + y_ins) / 2
    y_primates = (y_prim + y_human) / 2
    y_mammalia = (y_mamm + y_primates) / 2
    y_sauropsid = (y_rep + y_aves) / 2
    y_amniota = (y_sauropsid + y_mammalia) / 2
    y_tetrapod = (y_amph + y_amniota) / 2
    y_vertebrate = (y_fish + y_tetrapod) / 2
    y_metazoa = (y_ecdy + y_vertebrate) / 2

    def split(parent_x, child_x, y1, y2):
        parts.append(line(parent_x, y1, parent_x, y2, width=0.24))
        parts.append(line(parent_x, y1, child_x, y1, width=0.24))
        parts.append(line(parent_x, y2, child_x, y2, width=0.24))

    split(4.5, 8.0, y_angio, y_metazoa)
    parts.append(line(8.0, y_angio, leaf_x, y_angio, width=0.24))
    split(8.0, 12.5, y_ecdy, y_vertebrate)
    split(12.5, leaf_x, y_nem, y_ins)
    split(12.5, 16.5, y_fish, y_tetrapod)
    parts.append(line(16.5, y_fish, leaf_x, y_fish, width=0.24))
    split(16.5, 20.5, y_amph, y_amniota)
    parts.append(line(20.5, y_amph, leaf_x, y_amph, width=0.24))
    split(20.5, 24.0, y_sauropsid, y_mammalia)
    split(24.0, leaf_x, y_rep, y_aves)
    split(24.0, 27.0, y_mamm, y_primates)
    parts.append(line(27.0, y_mamm, leaf_x, y_mamm, width=0.24))
    split(27.0, leaf_x, y_prim, y_human)

    # Tip symbols, names and sample counts.
    for group in GROUP_ORDER:
        y = row_y[group]
        parts.append(
            marker(
                leaf_x,
                y,
                GROUP_MARKER[group],
                0.72,
                GROUP_COLOR[group],
                stroke="#FFFFFF",
                width=0.12,
            )
        )
        parts.append(
            text(
                label_x,
                y - 0.55,
                DISPLAY_NAME[group],
                size=5.25,
                anchor="end",
                weight="bold" if group in {"Human", "Angiosperms"} else "normal",
            )
        )
        organisms = int(group_counts.loc[group, "organisms"])
        sequences = int(group_counts.loc[group, "unique_sequences"])
        parts.append(
            text(
                label_x,
                y + 2.05,
                f"n={organisms}; {sequences:,} unique sequences",
                size=5.0,
                anchor="end",
                fill=MUTED_COLOR,
            )
        )

    # Dimer labels and subtle emphasis around CG and UA.
    for j, kmer in enumerate(KMERS):
        x = atlas_x0 + j * atlas_dx
        color = TEXT_COLOR
        weight = "normal"
        if kmer == "CG":
            color, weight = CG_COLOR, "bold"
        elif kmer == "UA":
            color, weight = UA_COLOR, "bold"
        parts.append(text(x, 15.2, kmer, size=5.2, weight=weight, anchor="middle", fill=color))

    for kmer, color in (("CG", CG_COLOR), ("UA", UA_COLOR)):
        j = KMERS.index(kmer)
        x = atlas_x0 + j * atlas_dx
        parts.append(
            rect(
                x - 3.0,
                17.0,
                6.0,
                77.0,
                fill="none",
                stroke=color,
                width=0.28,
                opacity=0.8,
                rx=0.8,
            )
        )

    # Atlas bubbles. Positional concentration is displayed on a fixed 0-0.04 scale.
    max_frequency_for_size = 10.5
    max_concentration_for_color = 0.04
    for i, group in enumerate(GROUP_ORDER):
        y = row_y[group]
        for j, kmer in enumerate(KMERS):
            row = atlas[
                atlas["analysis_group"].eq(group) & atlas["kmer"].eq(kmer)
            ].iloc[0]
            frequency = float(row["mean_observed_frequency_percent"])
            concentration = float(row["positional_concentration"])
            radius = 0.52 + 2.25 * math.sqrt(
                min(frequency, max_frequency_for_size) / max_frequency_for_size
            )
            color = mix_color(
                "#E8F1EE",
                "#075E67",
                min(concentration / max_concentration_for_color, 1.0),
            )
            x = atlas_x0 + j * atlas_dx
            parts.append(
                circle(
                    x,
                    y,
                    radius,
                    fill=color,
                    stroke="#FFFFFF",
                    width=0.15,
                    extra=(
                        f'<title>{esc(group)}; {esc(kmer)}; mean frequency '
                        f'{frequency:.2f}%; positional concentration '
                        f'{100*concentration:.2f}%</title>'
                    ),
                )
            )

    # Bubble size legend.
    legend_y = 104.2
    parts.append(text(78.0, legend_y, "Mean frequency", size=5.0, fill=MUTED_COLOR))
    lx = 102.0
    for frequency in (3.0, 6.0, 9.0):
        radius = 0.52 + 2.25 * math.sqrt(frequency / max_frequency_for_size)
        parts.append(circle(lx, legend_y - 0.6, radius, fill="#B8D3CE", stroke="#FFFFFF", width=0.15))
        parts.append(text(lx + 3.2, legend_y, f"{frequency:.0f}%", size=5.0, fill=MUTED_COLOR))
        lx += 15.5

    # Positional-concentration color legend.
    parts.append(text(148.0, legend_y, "Position concentration", size=5.0, fill=MUTED_COLOR))
    parts.append(
        rect(
            173.1,
            legend_y - 2.7,
            7.0,
            2.6,
            fill="url(#pbi-gradient)",
            stroke="#BFC3C7",
            width=0.12,
        )
    )
    parts.append(text(172.7, legend_y + 2.2, "0", size=5.0, anchor="end", fill=MUTED_COLOR))
    parts.append(text(180.4, legend_y + 2.2, "4%", size=5.0, anchor="end", fill=MUTED_COLOR))
    parts.append(
        text(
            4.5,
            111.0,
            "Taxonomy-based topology; branch lengths are not estimated.",
            size=5.0,
            fill=MUTED_COLOR,
            italic=True,
        )
    )
    parts.append("</g>")


def add_profile_panel(
    parts: list[str],
    profiles: pd.DataFrame,
    *,
    panel_letter: str,
    kmer: str,
    x0: float,
    x1: float,
    show_labels: bool,
) -> None:
    panel_id = f"panel-{panel_letter}_{kmer}_profiles"
    color = CG_COLOR if kmer == "CG" else UA_COLOR
    parts.append(f'<g id="{panel_id}">')
    parts.append(text(x0, 120.0, panel_letter, size=9.0, weight="bold"))
    parts.append(
        text(
            x0 + 7.0,
            120.0,
            f"{kmer} positional profiles",
            size=6.6,
            weight="bold",
            fill=color,
        )
    )
    parts.append(
        text(
            x0 + 7.0,
            124.0,
            "Pooled observed frequency; common 0–18% scale",
            size=5.0,
            fill=MUTED_COLOR,
        )
    )

    plot_left = x0 + (18.0 if show_labels else 5.0)
    plot_right = x1 - 1.5
    row_top = 128.7
    row_step = 5.25
    ridge_height = 4.0
    ymax = 18.0

    for i, group in enumerate(GROUP_ORDER):
        baseline = row_top + i * row_step + ridge_height
        top = baseline - ridge_height
        parts.append(line(plot_left, baseline, plot_right, baseline, stroke=GRID_COLOR, width=0.14))
        if show_labels:
            parts.append(
                text(
                    plot_left - 1.1,
                    baseline - 0.6,
                    ABBREVIATION[group],
                    size=5.0,
                    anchor="end",
                    fill=TEXT_COLOR,
                )
            )

        subset = profiles[
            profiles["analysis_group"].eq(group) & profiles["kmer"].eq(kmer)
        ].sort_values("start_position_0based")
        values = subset["pooled_frequency_percent"].to_numpy(dtype=float)
        xs = np.linspace(plot_left, plot_right, len(POSITIONS))
        ys = baseline - np.clip(values / ymax, 0, 1) * ridge_height
        area_points = [(plot_left, baseline)] + list(zip(xs, ys)) + [(plot_right, baseline)]
        parts.append(
            polyline(
                area_points,
                fill=color,
                stroke="none",
                opacity=0.14,
                close=True,
            )
        )
        parts.append(
            polyline(
                list(zip(xs, ys)),
                fill="none",
                stroke=color,
                width=0.32,
                opacity=0.95,
            )
        )
        # Position 0 is marked explicitly because UA has a characteristic 5' peak.
        parts.append(circle(xs[0], ys[0], 0.43, fill=color, stroke="#FFFFFF", width=0.12))

    bottom = row_top + (len(GROUP_ORDER) - 1) * row_step + ridge_height
    for pos in (0, 5, 10, 15, 19):
        x = plot_left + (plot_right - plot_left) * pos / 19.0
        parts.append(line(x, bottom, x, bottom + 1.0, stroke=BRANCH_COLOR, width=0.16))
        parts.append(text(x, bottom + 3.6, pos, size=5.0, anchor="middle", fill=MUTED_COLOR))
    parts.append(
        text(
            (plot_left + plot_right) / 2,
            bottom + 7.0,
            "K-mer start position (0-based)",
            size=5.4,
            anchor="middle",
        )
    )
    # A compact frequency reference bracket.
    parts.append(line(plot_left - 0.8, row_top, plot_left - 0.8, row_top + ridge_height, stroke=BRANCH_COLOR, width=0.16))
    parts.append(text(plot_left - 1.4, row_top + 1.0, "18", size=5.0, anchor="end", fill=MUTED_COLOR))
    parts.append(text(plot_left - 1.4, row_top + ridge_height, "0", size=5.0, anchor="end", fill=MUTED_COLOR))
    parts.append("</g>")


def covariance_ellipse(
    points_xy: np.ndarray,
    xscale,
    yscale,
    *,
    color: str,
    opacity: float,
) -> str:
    display = np.column_stack(
        [
            np.array([xscale(x) for x in points_xy[:, 0]]),
            np.array([yscale(y) for y in points_xy[:, 1]]),
        ]
    )
    center = display.mean(axis=0)
    cov = np.cov(display, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # Approximately 80% normal-data ellipse: sqrt(chi-square_2(0.80)) = 1.794.
    factor = 1.794
    rx, ry = factor * np.sqrt(np.maximum(eigenvalues, 0))
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    attribute_text = attrs(
        cx=fnum(center[0]),
        cy=fnum(center[1]),
        rx=fnum(rx),
        ry=fnum(ry),
        fill=color,
        fill_opacity=fnum(opacity),
        stroke=color,
        stroke_width="0.25",
        stroke_opacity="0.7",
        transform=(
            f"rotate({fnum(angle)} {fnum(center[0])} {fnum(center[1])})"
        ),
        vector_effect="non-scaling-stroke",
    )
    return f"<ellipse {attribute_text} />"


def add_panel_d(
    parts: list[str],
    pca: pd.DataFrame,
    explained: np.ndarray,
    *,
    x0: float,
    x1: float,
) -> None:
    parts.append('<g id="panel-D_organism_level_PCA">')
    parts.append(text(x0, 120.0, "D", size=9.0, weight="bold"))
    parts.append(
        text(
            x0 + 7.0,
            120.0,
            "Organism-level positional landscape",
            size=6.6,
            weight="bold",
        )
    )
    parts.append(
        text(
            x0 + 7.0,
            124.0,
            f"PCA: {len(pca)} organisms with ≥{PCA_MIN_SEQUENCES} unique sequences",
            size=5.0,
            fill=MUTED_COLOR,
        )
    )

    plot_left, plot_right = x0 + 5.0, x1 - 1.0
    plot_top, plot_bottom = 129.0, 176.2
    xmin, xmax = pca["PC1"].min(), pca["PC1"].max()
    ymin, ymax = pca["PC2"].min(), pca["PC2"].max()
    xpad = 0.08 * (xmax - xmin)
    ypad = 0.10 * (ymax - ymin)
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad

    def xs(value):
        return plot_left + (float(value) - xmin) / (xmax - xmin) * (plot_right - plot_left)

    def ys(value):
        return plot_bottom - (float(value) - ymin) / (ymax - ymin) * (plot_bottom - plot_top)

    # Neutral zero axes.
    if xmin <= 0 <= xmax:
        parts.append(line(xs(0), plot_top, xs(0), plot_bottom, stroke=GRID_COLOR, width=0.16))
    if ymin <= 0 <= ymax:
        parts.append(line(plot_left, ys(0), plot_right, ys(0), stroke=GRID_COLOR, width=0.16))

    macro_style = {
        "Angiosperms": ("#3A7D44", "circle"),
        "Invertebrate animals": ("#D98200", "triangle"),
        "Vertebrates": ("#2C7FB8", "circle"),
    }

    for macro, (color, _) in macro_style.items():
        subset = pca[pca["macroclade"].eq(macro)]
        if len(subset) >= 3:
            points_xy = subset[["PC1", "PC2"]].to_numpy(dtype=float)
            parts.append(
                covariance_ellipse(
                    points_xy,
                    xs,
                    ys,
                    color=color,
                    opacity=0.08,
                )
            )

    # Individual organisms are colored by macroclade so the PCA emphasizes
    # broad evolutionary structure rather than ten competing legend entries.
    for row in pca.itertuples(index=False):
        group = row.analysis_group
        if group == "Human":
            color, shape, size, opacity = "#111111", "star", 1.35, 1.0
        else:
            color, shape = macro_style[row.macroclade]
            size, opacity = 0.75, 0.72
        parts.append(
            marker(
                xs(row.PC1),
                ys(row.PC2),
                shape,
                size,
                color,
                opacity=opacity,
                stroke="#FFFFFF",
                width=0.14,
            )
        )

    # Macroclade centroid labels.
    label_offsets = {
        "Angiosperms": (2.0, -2.0),
        "Invertebrate animals": (1.5, -2.0),
        "Vertebrates": (1.5, 3.0),
    }
    for macro, (color, _) in macro_style.items():
        subset = pca[pca["macroclade"].eq(macro)]
        cx = xs(subset["PC1"].mean())
        cy = ys(subset["PC2"].mean())
        dx, dy = label_offsets[macro]
        parts.append(line(cx, cy, cx + dx - 0.3, cy + dy, stroke=color, width=0.18, opacity=0.8))
        parts.append(
            text(
                cx + dx,
                cy + dy + 0.6,
                macro,
                size=5.0,
                weight="bold",
                fill=color,
            )
        )

    human = pca[pca["analysis_group"].eq("Human")]
    if len(human) == 1:
        hx, hy = xs(human.iloc[0]["PC1"]), ys(human.iloc[0]["PC2"])
        parts.append(line(hx, hy, hx + 2.0, hy + 2.2, stroke="#111111", width=0.18))
        parts.append(text(hx + 2.3, hy + 2.8, "Human", size=5.0, weight="bold"))

    # Plot frame and axes.
    parts.append(rect(plot_left, plot_top, plot_right - plot_left, plot_bottom - plot_top, fill="none", stroke="#8C9196", width=0.18))
    parts.append(
        text(
            (plot_left + plot_right) / 2,
            181.0,
            f"PC1 ({100*explained[0]:.1f}%)",
            size=5.4,
            anchor="middle",
        )
    )
    parts.append(
        text(
            x0 + 1.2,
            (plot_top + plot_bottom) / 2,
            f"PC2 ({100*explained[1]:.1f}%)",
            size=5.4,
            anchor="middle",
            rotate=-90,
        )
    )

    # Compact legend using macroclades, plus the separately highlighted human.
    legend_y = 188.0
    legend_entries = [
        ("Angiosperms", "#3A7D44", "circle"),
        ("Invertebrate animals", "#D98200", "triangle"),
        ("Vertebrates", "#2C7FB8", "circle"),
        ("Human", "#111111", "star"),
    ]
    lx = x0 + 5.0
    for label, color, shape in legend_entries:
        parts.append(marker(lx, legend_y - 0.5, shape, 0.65, color, stroke="#FFFFFF", width=0.1))
        parts.append(text(lx + 1.7, legend_y, label, size=5.0, fill=MUTED_COLOR))
        lx += 15.0 if label in {"Human", "Vertebrates"} else 23.0

    parts.append("</g>")


def build_svg(
    atlas: pd.DataFrame,
    group_counts: pd.DataFrame,
    profiles: pd.DataFrame,
    pca: pd.DataFrame,
    explained: np.ndarray,
) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'width="{FIGURE_WIDTH_MM}mm" height="{FIGURE_HEIGHT_MM}mm" '
        f'viewBox="0 0 {FIGURE_WIDTH_MM} {FIGURE_HEIGHT_MM}">',
        "<title>Figure 2. Taxonomy-aligned mature-miRNA dinucleotide landscape</title>",
        (
            "<desc>Observed dinucleotide frequencies across ten taxonomic groups, "
            "CG and UA positional profiles, and organism-level PCA.</desc>"
        ),
        "<defs>",
        '<linearGradient id="pbi-gradient" x1="0%" x2="100%" y1="0%" y2="0%">',
        '<stop offset="0%" stop-color="#E8F1EE"/>',
        '<stop offset="100%" stop-color="#075E67"/>',
        "</linearGradient>",
        "</defs>",
        rect(0, 0, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, fill="#FFFFFF"),
    ]
    add_panel_a(parts, atlas, group_counts)
    add_profile_panel(
        parts,
        profiles,
        panel_letter="B",
        kmer="CG",
        x0=2.0,
        x1=53.0,
        show_labels=True,
    )
    add_profile_panel(
        parts,
        profiles,
        panel_letter="C",
        kmer="UA",
        x0=54.0,
        x1=105.0,
        show_labels=False,
    )
    add_panel_d(parts, pca, explained, x0=106.0, x1=182.0)
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.home() / "Downloads" / "taxonwise_kmer_outputs",
        help="Directory created by taxonwise_kmer_positional_bias.ipynb",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for Figure 2 SVG and supporting tables",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        aggregate,
        group_counts,
        atlas,
        profiles,
        pca,
        loadings,
        explained,
    ) = load_and_prepare(args.input_dir)

    svg = build_svg(atlas, group_counts, profiles, pca, explained)
    svg_path = args.output_dir / "Figure2_taxonomic_kmer_landscape.svg"
    svg_path.write_text(svg, encoding="utf-8")

    atlas.merge(
        group_counts.reset_index(), on="analysis_group", how="left"
    ).to_csv(args.output_dir / "Figure2A_motif_atlas_data.csv", index=False)
    profiles.to_csv(args.output_dir / "Figure2BC_CG_UA_profiles.csv", index=False)
    pca.to_csv(args.output_dir / "Figure2D_PCA_scores.csv", index=False)
    loadings.to_csv(args.output_dir / "Figure2D_PCA_loadings.csv", index=False)

    provenance = {
        "figure": "Figure2_taxonomic_kmer_landscape.svg",
        "input_directory": str(args.input_dir.resolve()),
        "aggregation_in_panels_A_B_C": (
            "Pooled observed counts after within-organism exact-sequence deduplication"
        ),
        "positions": POSITIONS,
        "panel_A": {
            "bubble_area": "Mean pooled observed frequency across positions 0-19",
            "bubble_colour": (
                "Positional concentration = 1 - H(profile)/ln(20); "
                "displayed on a fixed 0-0.04 scale"
            ),
            "cladogram": "Taxonomy-based topology only; branch lengths not estimated",
        },
        "panels_B_C": {
            "line": "Pooled observed positional frequency",
            "common_y_scale_percent": [0, 18],
        },
        "panel_D": {
            "minimum_unique_sequences_per_organism": PCA_MIN_SEQUENCES,
            "organisms_retained": int(len(pca)),
            "features": "16 dimers x 20 positions = 320 observed frequency features",
            "preprocessing": "Feature-wise mean centering; no variance scaling",
            "method": "PCA by singular value decomposition",
            "explained_variance_percent_PC1": float(100 * explained[0]),
            "explained_variance_percent_PC2": float(100 * explained[1]),
        },
        "typography": {
            "font": "Arial",
            "minimum_axis_and_tick_font_pt": 5.0,
            "svg_text_preserved": True,
        },
    }
    (args.output_dir / "Figure2_provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

    print(svg_path)
    print(f"PCA organisms retained: {len(pca)}")
    print(
        f"Explained variance: PC1={100*explained[0]:.2f}%, "
        f"PC2={100*explained[1]:.2f}%"
    )


if __name__ == "__main__":
    main()
