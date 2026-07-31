#!/usr/bin/env python3
"""
Permutation-centred positional k-mer statistics for mature miRNAs.

Primary questions
-----------------
1. Is each 2-mer or 3-mer globally scarce/abundant?
2. At which positions is it depleted/enriched?
3. Is its positional profile unusually concentrated or unusually uniform?

Two exact nucleotide-shuffle null models
----------------------------------------
1. within_sequence:
   Independently Fisher-Yates shuffle every mature miRNA. This preserves each
   sequence's length and exact mononucleotide composition.

2. positionwise_length_stratified:
   Independently permute every nucleotide column among sequences of the same
   length. This preserves length-specific mononucleotide counts at every
   position while destroying adjacent-base associations.

Overlapping k-mers are always recounted after nucleotide shuffling.

The script supports resumable NumPy checkpoint arrays. It uses only numpy and
pandas and is suitable for Google Colab CPU runtimes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_FASTA_SHA256 = (
    "3c521fc9bea3c7993e71cf188b11caf73e2d40ac83454e32876455317cc6d342"
)
EXPECTED_FASTA_RECORDS = 48_885
ALPHABET = "ACGU"
BASE_TO_INT = {base: i for i, base in enumerate(ALPHABET)}
POSITIONS = tuple(range(20))
KMERS = {
    2: tuple(a + b for a in ALPHABET for b in ALPHABET),
    3: tuple(a + b + c for a in ALPHABET for b in ALPHABET for c in ALPHABET),
}
NULL_MODELS = ("within_sequence", "positionwise_length_stratified")


def parse_fasta(path: Path) -> Iterable[tuple[str, str]]:
    header = None
    pieces: list[str] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(pieces).upper()
                header = line[1:].strip()
                pieces = []
            else:
                if header is None:
                    raise ValueError(
                        f"Sequence before first FASTA header at line {line_number}"
                    )
                pieces.append(line)
    if header is not None:
        yield header, "".join(pieces).upper()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_") or "dataset"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    p = values[valid]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[valid] = restored
    return result


def empirical_two_sided(null_values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Conservative two-sided empirical p-values with +1 correction."""
    null_values = np.asarray(null_values)
    observed = np.asarray(observed)
    n_perm = null_values.shape[0]
    upper = (np.sum(null_values >= observed, axis=0) + 1.0) / (n_perm + 1.0)
    lower = (np.sum(null_values <= observed, axis=0) + 1.0) / (n_perm + 1.0)
    return np.minimum(1.0, 2.0 * np.minimum(upper, lower))


def direction_labels(
    differences: np.ndarray,
    q_values: np.ndarray,
    *,
    positive: str,
    negative: str,
    alpha: float,
) -> np.ndarray:
    labels = np.full(np.asarray(differences).shape, "no detectable deviation", dtype=object)
    significant = np.isfinite(q_values) & (q_values <= alpha)
    labels[significant & (differences > 0)] = positive
    labels[significant & (differences < 0)] = negative
    return labels


def dataset_fingerprint(sequences: list[str]) -> str:
    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(sequence.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_sequence_datasets(
    fasta_path: Path,
    manifest_path: Path,
    *,
    strict_fasta: bool,
) -> tuple[dict[str, list[str]], pd.DataFrame, dict]:
    fasta_sha = sha256_file(fasta_path)
    if strict_fasta and fasta_sha != EXPECTED_FASTA_SHA256:
        raise ValueError(
            "mature.fa SHA-256 mismatch.\n"
            f"Observed: {fasta_sha}\nExpected: {EXPECTED_FASTA_SHA256}"
        )

    manifest = pd.read_csv(manifest_path)
    required = {
        "group_order",
        "group_code",
        "analysis_group",
        "organism_prefix",
        "mirbase_organism_name",
        "raw_mature_records",
        "unique_sequences_within_organism",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Taxon manifest missing columns: {missing}")
    if manifest["organism_prefix"].duplicated().any():
        raise ValueError("Taxon manifest contains duplicate organism prefixes")

    selected_prefixes = set(manifest["organism_prefix"])
    all_sequences: list[str] = []
    within_organism: dict[str, set[str]] = defaultdict(set)
    raw_selected = Counter()
    total_records = 0
    invalid: list[tuple[str, str]] = []

    for header, sequence in parse_fasta(fasta_path):
        total_records += 1
        identifier = header.split()[0]
        bad = sorted(set(sequence).difference(ALPHABET))
        if bad:
            invalid.append((identifier, "".join(bad)))
            continue
        all_sequences.append(sequence)
        prefix = identifier.split("-", 1)[0]
        if prefix in selected_prefixes:
            raw_selected[prefix] += 1
            within_organism[prefix].add(sequence)

    if strict_fasta and total_records != EXPECTED_FASTA_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_FASTA_RECORDS:,} records; observed {total_records:,}"
        )
    if invalid:
        raise ValueError(f"Invalid A/C/G/U sequences; examples: {invalid[:5]}")

    absent = selected_prefixes.difference(within_organism)
    if absent:
        raise ValueError(f"Manifest prefixes absent from FASTA: {sorted(absent)}")

    observed_raw = manifest["organism_prefix"].map(raw_selected).astype(int)
    observed_unique = manifest["organism_prefix"].map(
        lambda prefix: len(within_organism[prefix])
    )
    if strict_fasta:
        if not np.array_equal(
            observed_raw.to_numpy(), manifest["raw_mature_records"].to_numpy()
        ):
            raise ValueError("Manifest raw counts do not match mature.fa")
        if not np.array_equal(
            observed_unique.to_numpy(),
            manifest["unique_sequences_within_organism"].to_numpy(),
        ):
            raise ValueError("Manifest unique counts do not match mature.fa")

    datasets: dict[str, list[str]] = {
        "universal_global_unique": sorted(set(all_sequences))
    }
    for group, rows in manifest.groupby("analysis_group", sort=False):
        sequences: list[str] = []
        for prefix in rows["organism_prefix"]:
            sequences.extend(sorted(within_organism[prefix]))
        code = rows["group_code"].iloc[0]
        datasets[f"taxon__{code}"] = sequences

    inventory_rows = []
    group_lookup = (
        manifest[
            ["group_code", "group_order", "analysis_group"]
        ]
        .drop_duplicates()
        .set_index("group_code")
        .to_dict("index")
    )
    for dataset_name, sequences in datasets.items():
        lengths = Counter(map(len, sequences))
        if dataset_name == "universal_global_unique":
            group_code = "universal"
            group_order = 0
            analysis_group = "All mature miRNAs (global unique repertoire)"
        else:
            group_code = dataset_name.removeprefix("taxon__")
            group_order = int(group_lookup[group_code]["group_order"])
            analysis_group = group_lookup[group_code]["analysis_group"]
        inventory_rows.append(
            {
                "dataset": dataset_name,
                "analysis_scope": (
                    "universal"
                    if dataset_name == "universal_global_unique"
                    else "taxon"
                ),
                "group_code": group_code,
                "group_order": group_order,
                "analysis_group": analysis_group,
                "sequence_occurrences": len(sequences),
                "unique_sequence_strings": len(set(sequences)),
                "minimum_length": min(lengths),
                "maximum_length": max(lengths),
                "median_length": float(np.median([len(s) for s in sequences])),
                "length_histogram": json.dumps(dict(sorted(lengths.items()))),
                "sequence_fingerprint_sha256": dataset_fingerprint(sequences),
            }
        )
    inventory = pd.DataFrame(inventory_rows)

    audit = {
        "fasta_path": str(fasta_path),
        "fasta_sha256": fasta_sha,
        "fasta_records": total_records,
        "global_unique_sequences": len(datasets["universal_global_unique"]),
        "selected_organism_prefixes": len(manifest),
        "selected_within_organism_unique_sequence_occurrences": int(
            observed_unique.sum()
        ),
    }
    return datasets, inventory, audit


def encode_by_length(sequences: list[str]) -> dict[int, np.ndarray]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for sequence in sequences:
        grouped[len(sequence)].append(sequence)

    encoded: dict[int, np.ndarray] = {}
    lookup = np.full(256, 255, dtype=np.uint8)
    for base, code in BASE_TO_INT.items():
        lookup[ord(base)] = code

    for length, members in sorted(grouped.items()):
        raw = np.frombuffer("".join(members).encode("ascii"), dtype=np.uint8)
        matrix = lookup[raw].reshape(len(members), length)
        if np.any(matrix > 3):
            raise ValueError(f"Encoding failure for length {length}")
        encoded[length] = matrix
    return encoded


def count_kmers(
    encoded_by_length: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts2 = np.zeros((len(POSITIONS), 16), dtype=np.int32)
    counts3 = np.zeros((len(POSITIONS), 64), dtype=np.int32)
    denominators2 = np.zeros(len(POSITIONS), dtype=np.int64)
    denominators3 = np.zeros(len(POSITIONS), dtype=np.int64)

    for length, matrix in encoded_by_length.items():
        n_sequences = len(matrix)
        for position in POSITIONS:
            if length >= position + 2:
                codes2 = (
                    matrix[:, position].astype(np.int16) * 4
                    + matrix[:, position + 1]
                )
                counts2[position] += np.bincount(codes2, minlength=16).astype(
                    np.int32
                )
                denominators2[position] += n_sequences
            if length >= position + 3:
                codes3 = (
                    matrix[:, position].astype(np.int16) * 16
                    + matrix[:, position + 1].astype(np.int16) * 4
                    + matrix[:, position + 2]
                )
                counts3[position] += np.bincount(codes3, minlength=64).astype(
                    np.int32
                )
                denominators3[position] += n_sequences
    return counts2, counts3, denominators2, denominators3


def shuffled_encoded(
    encoded_by_length: dict[int, np.ndarray],
    null_model: str,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    shuffled: dict[int, np.ndarray] = {}
    if null_model == "within_sequence":
        for length, original in encoded_by_length.items():
            matrix = original.copy()
            n_sequences = len(matrix)
            rows = np.arange(n_sequences)
            for right in range(length - 1, 0, -1):
                left = rng.integers(0, right + 1, size=n_sequences)
                temporary = matrix[rows, right].copy()
                matrix[rows, right] = matrix[rows, left]
                matrix[rows, left] = temporary
            shuffled[length] = matrix
        return shuffled

    if null_model == "positionwise_length_stratified":
        for length, original in encoded_by_length.items():
            matrix = np.empty_like(original)
            n_sequences = len(original)
            if n_sequences <= 1:
                matrix[:] = original
            else:
                for position in range(length):
                    matrix[:, position] = original[
                        rng.permutation(n_sequences), position
                    ]
            shuffled[length] = matrix
        return shuffled

    raise ValueError(f"Unknown null model: {null_model}")


def replicate_seed(
    base_seed: int,
    dataset_name: str,
    null_model: str,
    replicate: int,
) -> np.random.SeedSequence:
    dataset_code = zlib.crc32(dataset_name.encode("utf-8"))
    null_code = zlib.crc32(null_model.encode("utf-8"))
    return np.random.SeedSequence(
        [
            int(base_seed) & 0xFFFFFFFF,
            dataset_code,
            null_code,
            int(replicate) & 0xFFFFFFFF,
        ]
    )


def checkpoint_paths(checkpoint_dir: Path, dataset_name: str, null_model: str):
    stem = f"{safe_name(dataset_name)}__{safe_name(null_model)}"
    return {
        "counts2": checkpoint_dir / f"{stem}__counts2.npy",
        "counts3": checkpoint_dir / f"{stem}__counts3.npy",
        "state": checkpoint_dir / f"{stem}__state.json",
    }


def open_or_create_checkpoints(
    paths: dict[str, Path],
    *,
    n_perm: int,
    dataset_fingerprint_value: str,
    base_seed: int,
    dataset_name: str,
    null_model: str,
) -> tuple[np.memmap, np.memmap, int]:
    expected = {
        "version": 1,
        "dataset": dataset_name,
        "null_model": null_model,
        "n_permutations": int(n_perm),
        "dataset_fingerprint_sha256": dataset_fingerprint_value,
        "base_seed": int(base_seed),
        "shape_counts2": [n_perm, len(POSITIONS), 16],
        "shape_counts3": [n_perm, len(POSITIONS), 64],
    }

    resume = False
    completed = 0
    if all(path.exists() for path in paths.values()):
        try:
            state = json.loads(paths["state"].read_text(encoding="utf-8"))
            comparable = {key: state.get(key) for key in expected}
            if comparable == expected:
                candidate_counts2 = np.load(paths["counts2"], mmap_mode="r+")
                candidate_counts3 = np.load(paths["counts3"], mmap_mode="r+")
                shapes_valid = (
                    list(candidate_counts2.shape) == expected["shape_counts2"]
                    and list(candidate_counts3.shape)
                    == expected["shape_counts3"]
                )
                dtypes_valid = (
                    candidate_counts2.dtype == np.int32
                    and candidate_counts3.dtype == np.int32
                )
                candidate_completed = int(state.get("completed", 0))
                completed_valid = 0 <= candidate_completed <= n_perm
                if shapes_valid and dtypes_valid and completed_valid:
                    resume = True
                    completed = candidate_completed
                    return (
                        candidate_counts2,
                        candidate_counts3,
                        completed,
                    )
        except (OSError, ValueError, EOFError, json.JSONDecodeError) as error:
            print(
                "Checkpoint is incomplete or incompatible; recreating it: "
                f"{paths['state'].name} ({type(error).__name__})"
            )

    # Remove only this generated checkpoint block before recreating it. Completed
    # statistical summaries live separately and are never removed here.
    for path in paths.values():
        if path.exists():
            path.unlink()

    paths["state"].parent.mkdir(parents=True, exist_ok=True)
    counts2 = np.lib.format.open_memmap(
        paths["counts2"],
        mode="w+",
        dtype=np.int32,
        shape=(n_perm, len(POSITIONS), 16),
    )
    counts3 = np.lib.format.open_memmap(
        paths["counts3"],
        mode="w+",
        dtype=np.int32,
        shape=(n_perm, len(POSITIONS), 64),
    )
    state = dict(expected)
    state["completed"] = 0
    paths["state"].write_text(json.dumps(state, indent=2), encoding="utf-8")
    return counts2, counts3, 0


def update_checkpoint_state(
    state_path: Path,
    *,
    completed: int,
) -> None:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["completed"] = int(completed)
    state["updated_unix_time"] = time.time()
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def generate_null_counts(
    encoded_by_length: dict[int, np.ndarray],
    *,
    dataset_name: str,
    dataset_fingerprint_value: str,
    null_model: str,
    n_perm: int,
    base_seed: int,
    checkpoint_dir: Path,
    checkpoint_every: int,
) -> tuple[np.memmap, np.memmap, dict[str, Path]]:
    paths = checkpoint_paths(checkpoint_dir, dataset_name, null_model)
    counts2, counts3, completed = open_or_create_checkpoints(
        paths,
        n_perm=n_perm,
        dataset_fingerprint_value=dataset_fingerprint_value,
        base_seed=base_seed,
        dataset_name=dataset_name,
        null_model=null_model,
    )
    if completed:
        print(
            f"Resuming {dataset_name} / {null_model} at "
            f"{completed:,}/{n_perm:,}"
        )

    start_time = time.time()
    for replicate in range(completed, n_perm):
        rng = np.random.default_rng(
            replicate_seed(base_seed, dataset_name, null_model, replicate)
        )
        permuted = shuffled_encoded(encoded_by_length, null_model, rng)
        replicate_counts2, replicate_counts3, _, _ = count_kmers(permuted)
        counts2[replicate] = replicate_counts2
        counts3[replicate] = replicate_counts3

        finished = replicate + 1
        if finished % checkpoint_every == 0 or finished == n_perm:
            counts2.flush()
            counts3.flush()
            update_checkpoint_state(paths["state"], completed=finished)
            elapsed = time.time() - start_time
            rate = (finished - completed) / elapsed if elapsed > 0 else float("nan")
            print(
                f"  {dataset_name} / {null_model}: "
                f"{finished:,}/{n_perm:,} ({rate:.2f} permutations/s)"
            )
    return counts2, counts3, paths


def null_summary_columns(
    null_values: np.ndarray,
    observed: np.ndarray,
) -> dict[str, np.ndarray]:
    null_mean = np.mean(null_values, axis=0)
    null_sd = np.std(null_values, axis=0, ddof=1)
    lower, upper = np.quantile(null_values, [0.025, 0.975], axis=0)
    difference = observed - null_mean
    standardized = np.divide(
        difference,
        null_sd,
        out=np.full_like(difference, np.nan, dtype=float),
        where=null_sd > 0,
    )
    p_value = empirical_two_sided(null_values, observed)
    return {
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_lower_95": lower,
        "null_upper_95": upper,
        "difference": difference,
        "standardized_effect": standardized,
        "empirical_p": p_value,
    }


def positional_summary(
    observed_counts: np.ndarray,
    null_counts: np.ndarray,
    denominators: np.ndarray,
    *,
    dataset_name: str,
    null_model: str,
    k: int,
    n_perm: int,
    alpha: float,
) -> pd.DataFrame:
    observed_frequency = np.divide(
        observed_counts * 100.0,
        denominators[:, None],
        where=denominators[:, None] > 0,
    )
    null_frequency = np.divide(
        null_counts.astype(np.float64) * 100.0,
        denominators[None, :, None],
        where=denominators[None, :, None] > 0,
    )
    statistics = null_summary_columns(null_frequency, observed_frequency)
    p_values = statistics["empirical_p"].reshape(-1)
    q_values = bh_adjust(p_values).reshape(observed_frequency.shape)
    directions = direction_labels(
        statistics["difference"],
        q_values,
        positive="enriched",
        negative="depleted",
        alpha=alpha,
    )

    rows = []
    for position_index, position in enumerate(POSITIONS):
        for motif_index, motif in enumerate(KMERS[k]):
            rows.append(
                {
                    "dataset": dataset_name,
                    "null_model": null_model,
                    "k": k,
                    "start_position_0based": position,
                    "kmer": motif,
                    "eligible_sequences": int(denominators[position_index]),
                    "observed_count": int(
                        observed_counts[position_index, motif_index]
                    ),
                    "observed_frequency_percent": float(
                        observed_frequency[position_index, motif_index]
                    ),
                    "null_mean_frequency_percent": float(
                        statistics["null_mean"][position_index, motif_index]
                    ),
                    "null_sd_frequency_percent": float(
                        statistics["null_sd"][position_index, motif_index]
                    ),
                    "null_lower_95_frequency_percent": float(
                        statistics["null_lower_95"][position_index, motif_index]
                    ),
                    "null_upper_95_frequency_percent": float(
                        statistics["null_upper_95"][position_index, motif_index]
                    ),
                    "observed_minus_null_percentage_points": float(
                        statistics["difference"][position_index, motif_index]
                    ),
                    "standardized_effect": float(
                        statistics["standardized_effect"][
                            position_index, motif_index
                        ]
                    ),
                    "empirical_p_two_sided": float(
                        statistics["empirical_p"][position_index, motif_index]
                    ),
                    "BH_q_within_dataset_k_null": float(
                        q_values[position_index, motif_index]
                    ),
                    "direction_at_q_le_alpha": directions[
                        position_index, motif_index
                    ],
                    "alpha": alpha,
                    "n_permutations": n_perm,
                }
            )
    return pd.DataFrame(rows)


def overall_summary(
    observed_counts: np.ndarray,
    null_counts: np.ndarray,
    denominators: np.ndarray,
    *,
    dataset_name: str,
    null_model: str,
    k: int,
    n_perm: int,
    alpha: float,
) -> pd.DataFrame:
    denominator_total = int(denominators.sum())
    observed_total = observed_counts.sum(axis=0)
    null_total = null_counts.sum(axis=1)
    observed_frequency = observed_total * 100.0 / denominator_total
    null_frequency = null_total.astype(np.float64) * 100.0 / denominator_total
    statistics = null_summary_columns(null_frequency, observed_frequency)
    q_values = bh_adjust(statistics["empirical_p"])
    directions = direction_labels(
        statistics["difference"],
        q_values,
        positive="enriched",
        negative="depleted",
        alpha=alpha,
    )

    rows = []
    for motif_index, motif in enumerate(KMERS[k]):
        rows.append(
            {
                "dataset": dataset_name,
                "null_model": null_model,
                "k": k,
                "kmer": motif,
                "eligible_kmer_windows_positions_0_19": denominator_total,
                "observed_count_positions_0_19": int(observed_total[motif_index]),
                "observed_frequency_percent": float(
                    observed_frequency[motif_index]
                ),
                "null_mean_frequency_percent": float(
                    statistics["null_mean"][motif_index]
                ),
                "null_sd_frequency_percent": float(
                    statistics["null_sd"][motif_index]
                ),
                "null_lower_95_frequency_percent": float(
                    statistics["null_lower_95"][motif_index]
                ),
                "null_upper_95_frequency_percent": float(
                    statistics["null_upper_95"][motif_index]
                ),
                "observed_minus_null_percentage_points": float(
                    statistics["difference"][motif_index]
                ),
                "standardized_effect": float(
                    statistics["standardized_effect"][motif_index]
                ),
                "empirical_p_two_sided": float(
                    statistics["empirical_p"][motif_index]
                ),
                "BH_q_within_dataset_k_null": float(q_values[motif_index]),
                "direction_at_q_le_alpha": directions[motif_index],
                "alpha": alpha,
                "n_permutations": n_perm,
            }
        )
    return pd.DataFrame(rows)


def concentration_from_frequency(frequency: np.ndarray) -> np.ndarray:
    """Return 1 - normalized Shannon entropy over the position axis."""
    frequency = np.asarray(frequency, dtype=float)
    if frequency.ndim == 2:
        # positions x motifs -> motifs x positions
        profiles = frequency.T
    elif frequency.ndim == 3:
        # permutations x positions x motifs -> permutations x motifs x positions
        profiles = np.transpose(frequency, (0, 2, 1))
    else:
        raise ValueError("Frequency array must have 2 or 3 dimensions")
    totals = profiles.sum(axis=-1, keepdims=True)
    q = np.divide(
        profiles,
        totals,
        out=np.zeros_like(profiles, dtype=float),
        where=totals > 0,
    )
    log_q = np.zeros_like(q)
    positive = q > 0
    log_q[positive] = np.log(q[positive])
    entropy = -np.sum(q * log_q, axis=-1)
    normalized = entropy / math.log(len(POSITIONS))
    concentration = 1.0 - normalized
    concentration = np.where(totals.squeeze(-1) > 0, concentration, np.nan)
    return concentration


def entropy_summary(
    observed_counts: np.ndarray,
    null_counts: np.ndarray,
    denominators: np.ndarray,
    *,
    dataset_name: str,
    null_model: str,
    k: int,
    n_perm: int,
    alpha: float,
) -> pd.DataFrame:
    observed_frequency = observed_counts * 100.0 / denominators[:, None]
    null_frequency = (
        null_counts.astype(np.float64)
        * 100.0
        / denominators[None, :, None]
    )
    observed_concentration = concentration_from_frequency(observed_frequency)
    null_concentration = concentration_from_frequency(null_frequency)
    statistics = null_summary_columns(
        null_concentration, observed_concentration
    )
    q_values = bh_adjust(statistics["empirical_p"])
    directions = direction_labels(
        statistics["difference"],
        q_values,
        positive="more positionally concentrated",
        negative="more positionally uniform",
        alpha=alpha,
    )

    rows = []
    for motif_index, motif in enumerate(KMERS[k]):
        observed_c = observed_concentration[motif_index]
        rows.append(
            {
                "dataset": dataset_name,
                "null_model": null_model,
                "k": k,
                "kmer": motif,
                "observed_normalized_shannon_entropy": float(1.0 - observed_c),
                "observed_positional_concentration": float(observed_c),
                "null_mean_positional_concentration": float(
                    statistics["null_mean"][motif_index]
                ),
                "null_sd_positional_concentration": float(
                    statistics["null_sd"][motif_index]
                ),
                "null_lower_95_positional_concentration": float(
                    statistics["null_lower_95"][motif_index]
                ),
                "null_upper_95_positional_concentration": float(
                    statistics["null_upper_95"][motif_index]
                ),
                "observed_minus_null_concentration": float(
                    statistics["difference"][motif_index]
                ),
                "standardized_effect": float(
                    statistics["standardized_effect"][motif_index]
                ),
                "empirical_p_two_sided": float(
                    statistics["empirical_p"][motif_index]
                ),
                "BH_q_within_dataset_k_null": float(q_values[motif_index]),
                "direction_at_q_le_alpha": directions[motif_index],
                "alpha": alpha,
                "n_permutations": n_perm,
            }
        )
    return pd.DataFrame(rows)


def summarize_null(
    observed_counts2: np.ndarray,
    observed_counts3: np.ndarray,
    denominators2: np.ndarray,
    denominators3: np.ndarray,
    null_counts2: np.ndarray,
    null_counts3: np.ndarray,
    *,
    dataset_name: str,
    null_model: str,
    n_perm: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positional_frames = []
    overall_frames = []
    entropy_frames = []
    for k, observed, null, denominators in (
        (2, observed_counts2, null_counts2, denominators2),
        (3, observed_counts3, null_counts3, denominators3),
    ):
        positional_frames.append(
            positional_summary(
                observed,
                null,
                denominators,
                dataset_name=dataset_name,
                null_model=null_model,
                k=k,
                n_perm=n_perm,
                alpha=alpha,
            )
        )
        overall_frames.append(
            overall_summary(
                observed,
                null,
                denominators,
                dataset_name=dataset_name,
                null_model=null_model,
                k=k,
                n_perm=n_perm,
                alpha=alpha,
            )
        )
        entropy_frames.append(
            entropy_summary(
                observed,
                null,
                denominators,
                dataset_name=dataset_name,
                null_model=null_model,
                k=k,
                n_perm=n_perm,
                alpha=alpha,
            )
        )
    return (
        pd.concat(positional_frames, ignore_index=True),
        pd.concat(overall_frames, ignore_index=True),
        pd.concat(entropy_frames, ignore_index=True),
    )


def remove_checkpoint_arrays(paths: dict[str, Path]) -> None:
    for key in ("counts2", "counts3", "state"):
        path = paths[key]
        if path.exists():
            path.unlink()


def write_combined_outputs(
    output_dir: Path,
    positional_frames: list[pd.DataFrame],
    overall_frames: list[pd.DataFrame],
    entropy_frames: list[pd.DataFrame],
) -> None:
    positional = pd.concat(positional_frames, ignore_index=True)
    overall = pd.concat(overall_frames, ignore_index=True)
    entropy = pd.concat(entropy_frames, ignore_index=True)
    positional.to_csv(
        output_dir / "all_datasets_positional_permutation_results.csv.gz",
        index=False,
        compression="gzip",
    )
    overall.to_csv(
        output_dir / "all_datasets_overall_abundance_permutation_results.csv",
        index=False,
    )
    entropy.to_csv(
        output_dir / "all_datasets_shannon_concentration_results.csv",
        index=False,
    )

    taxon_position = positional[
        positional["dataset"].str.startswith("taxon__")
    ]
    if len(taxon_position):
        consistency = (
            taxon_position.groupby(
                ["null_model", "k", "start_position_0based", "kmer"],
                as_index=False,
            )
            .agg(
                taxa_evaluated=("dataset", "nunique"),
                taxa_with_negative_effect=(
                    "observed_minus_null_percentage_points",
                    lambda values: int((values < 0).sum()),
                ),
                taxa_with_positive_effect=(
                    "observed_minus_null_percentage_points",
                    lambda values: int((values > 0).sum()),
                ),
                taxa_significantly_depleted=(
                    "direction_at_q_le_alpha",
                    lambda values: int((values == "depleted").sum()),
                ),
                taxa_significantly_enriched=(
                    "direction_at_q_le_alpha",
                    lambda values: int((values == "enriched").sum()),
                ),
                median_effect_percentage_points=(
                    "observed_minus_null_percentage_points",
                    "median",
                ),
                minimum_effect_percentage_points=(
                    "observed_minus_null_percentage_points",
                    "min",
                ),
                maximum_effect_percentage_points=(
                    "observed_minus_null_percentage_points",
                    "max",
                ),
            )
        )
        consistency.to_csv(
            output_dir / "taxon_direction_consistency_summary.csv",
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Directory for temporary memory-mapped permutation arrays. "
            "Use a local /content path in Colab, not a mounted Drive path."
        ),
    )
    parser.add_argument("--n-perm-universal", type=int, default=1_000)
    parser.add_argument("--n-perm-taxa", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20_260_726)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--run-taxa",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--keep-null-arrays",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--strict-fasta",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.n_perm_universal < 2 or args.n_perm_taxa < 2:
        raise ValueError("At least two permutations are required")
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must lie between 0 and 1")

    output_dir = args.output_dir
    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else output_dir / "checkpoints"
    )
    per_dataset_dir = output_dir / "per_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    per_dataset_dir.mkdir(parents=True, exist_ok=True)

    datasets, inventory, audit = load_sequence_datasets(
        args.fasta,
        args.manifest,
        strict_fasta=args.strict_fasta,
    )
    inventory.to_csv(output_dir / "dataset_inventory.csv", index=False)
    metadata_by_dataset = inventory.set_index("dataset")[
        ["analysis_scope", "group_code", "group_order", "analysis_group"]
    ].to_dict("index")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    selected_dataset_names = ["universal_global_unique"]
    if args.run_taxa:
        selected_dataset_names.extend(
            name for name in datasets if name.startswith("taxon__")
        )

    positional_frames: list[pd.DataFrame] = []
    overall_frames: list[pd.DataFrame] = []
    entropy_frames: list[pd.DataFrame] = []

    run_start = time.time()
    for dataset_number, dataset_name in enumerate(selected_dataset_names, start=1):
        sequences = datasets[dataset_name]
        fingerprint = dataset_fingerprint(sequences)
        n_perm = (
            args.n_perm_universal
            if dataset_name == "universal_global_unique"
            else args.n_perm_taxa
        )
        print(
            f"\nDataset {dataset_number}/{len(selected_dataset_names)}: "
            f"{dataset_name} ({len(sequences):,} sequence occurrences; "
            f"{n_perm:,} permutations per null)"
        )
        encoded = encode_by_length(sequences)
        observed2, observed3, denominator2, denominator3 = count_kmers(encoded)

        observed_rows = []
        for k, counts, denominators in (
            (2, observed2, denominator2),
            (3, observed3, denominator3),
        ):
            for position_index, position in enumerate(POSITIONS):
                for motif_index, motif in enumerate(KMERS[k]):
                    observed_rows.append(
                        {
                            "dataset": dataset_name,
                            "k": k,
                            "start_position_0based": position,
                            "kmer": motif,
                            "count": int(counts[position_index, motif_index]),
                            "eligible_sequences": int(
                                denominators[position_index]
                            ),
                            "frequency_percent": float(
                                100.0
                                * counts[position_index, motif_index]
                                / denominators[position_index]
                            ),
                        }
                    )
        pd.DataFrame(observed_rows).to_csv(
            per_dataset_dir
            / f"{safe_name(dataset_name)}__observed_positional_counts.csv",
            index=False,
        )

        for null_model in NULL_MODELS:
            prefix = f"{safe_name(dataset_name)}__{safe_name(null_model)}"
            positional_output_path = (
                per_dataset_dir / f"{prefix}__positional.csv.gz"
            )
            overall_output_path = (
                per_dataset_dir / f"{prefix}__overall_abundance.csv"
            )
            entropy_output_path = (
                per_dataset_dir / f"{prefix}__shannon_concentration.csv"
            )
            completion_path = per_dataset_dir / f"{prefix}__complete.json"
            expected_completion = {
                "version": 1,
                "dataset": dataset_name,
                "null_model": null_model,
                "dataset_fingerprint_sha256": fingerprint,
                "n_permutations": n_perm,
                "base_seed": args.seed,
                "alpha": args.alpha,
            }
            completed_summary_available = False
            if (
                completion_path.exists()
                and positional_output_path.exists()
                and overall_output_path.exists()
                and entropy_output_path.exists()
            ):
                completion = json.loads(
                    completion_path.read_text(encoding="utf-8")
                )
                completed_summary_available = (
                    {
                        key: completion.get(key)
                        for key in expected_completion
                    }
                    == expected_completion
                )

            if completed_summary_available:
                print(
                    f"Using completed summary: {dataset_name} / {null_model}"
                )
                positional = pd.read_csv(positional_output_path)
                overall = pd.read_csv(overall_output_path)
                entropy = pd.read_csv(entropy_output_path)
                positional_frames.append(positional)
                overall_frames.append(overall)
                entropy_frames.append(entropy)
                write_combined_outputs(
                    output_dir,
                    positional_frames,
                    overall_frames,
                    entropy_frames,
                )
                continue

            null2, null3, paths = generate_null_counts(
                encoded,
                dataset_name=dataset_name,
                dataset_fingerprint_value=fingerprint,
                null_model=null_model,
                n_perm=n_perm,
                base_seed=args.seed,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=args.checkpoint_every,
            )
            positional, overall, entropy = summarize_null(
                observed2,
                observed3,
                denominator2,
                denominator3,
                null2,
                null3,
                dataset_name=dataset_name,
                null_model=null_model,
                n_perm=n_perm,
                alpha=args.alpha,
            )
            metadata = metadata_by_dataset[dataset_name]
            for frame in (positional, overall, entropy):
                frame.insert(1, "analysis_scope", metadata["analysis_scope"])
                frame.insert(2, "group_code", metadata["group_code"])
                frame.insert(3, "group_order", metadata["group_order"])
                frame.insert(4, "analysis_group", metadata["analysis_group"])
            positional_frames.append(positional)
            overall_frames.append(overall)
            entropy_frames.append(entropy)

            positional.to_csv(
                positional_output_path,
                index=False,
                compression="gzip",
            )
            overall.to_csv(
                overall_output_path,
                index=False,
            )
            entropy.to_csv(
                entropy_output_path,
                index=False,
            )
            completion_record = dict(expected_completion)
            completion_record["completed_unix_time"] = time.time()
            completion_path.write_text(
                json.dumps(completion_record, indent=2),
                encoding="utf-8",
            )

            write_combined_outputs(
                output_dir,
                positional_frames,
                overall_frames,
                entropy_frames,
            )
            if not args.keep_null_arrays:
                del null2, null3
                remove_checkpoint_arrays(paths)

    provenance = {
        **audit,
        "positions": list(POSITIONS),
        "null_models": list(NULL_MODELS),
        "n_permutations_universal": args.n_perm_universal,
        "n_permutations_taxa": args.n_perm_taxa if args.run_taxa else 0,
        "base_seed": args.seed,
        "checkpoint_directory": str(checkpoint_dir),
        "alpha": args.alpha,
        "multiple_testing": (
            "Benjamini-Hochberg separately within each dataset, k and null model"
        ),
        "empirical_p": (
            "Two-sided 2*min(tails), with +1 numerator and denominator correction"
        ),
        "universal_deduplication": "Global exact-sequence deduplication",
        "taxon_deduplication": (
            "Exact deduplication within organism; conserved occurrences across "
            "organisms retained"
        ),
        "elapsed_seconds": time.time() - run_start,
    }
    (output_dir / "analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    print(f"\nCompleted. Results: {output_dir}")


if __name__ == "__main__":
    main()
