"""Export molecule-specific phonon displacements to a JMOL-ready XYZ file.

This utility focuses on a subset of atoms (e.g. one molecule) embedded in a
larger supercell.  You must supply the atom indices that belong to that
sub-molecule as they appear in the supercell; the script extracts their
positions and eigenvector components from Phonopy's ``band.hdf5`` data and
writes an XYZ file containing both equilibrium coordinates and displacement
vectors.

Usage example
-------------
```
python Harmonic_phonons/Visualize_phonons/create_jmol_from_indexed_naphthalene.py \
    --structure supercell.extxyz --structure-format extxyz \
    --mass-structure geometry.in --mass-format aims \
    --band-file band.hdf5 --mode-index 39 --skip-branches 3 \
    --structure-indices 10,6,26,2,22,19,35,15,31,47,43,63,39,59,54,70,50,66 \
    --eigenvector-indices 10,6,26,2,22,19,35,15,31,11,7,27,3,23,18,34,14,30 \
    --normalize-displacements --output-dir naph_modes
```
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence

import h5py
import numpy as np
from ase import Atoms
from ase.io import read

LOGGER = logging.getLogger(__name__)

# NOTE: Update these lists for other systems. They encode the atom indices of the
# target molecule within the supercell and the corresponding eigenvector mapping.
DEFAULT_STRUCTURE_INDICES = [
    10,
    6,
    26,
    2,
    22,
    19,
    35,
    15,
    31,
    47,
    43,
    63,
    39,
    59,
    54,
    70,
    50,
    66,
]
DEFAULT_EIGENVECTOR_INDICES = [
    10,
    6,
    26,
    2,
    22,
    19,
    35,
    15,
    31,
    11,
    7,
    27,
    3,
    23,
    18,
    34,
    14,
    30,
]


def parse_indices(value: str) -> List[int]:
    """Parse a comma/space separated list of integers or @file reference."""
    value = value.strip()
    if not value:
        raise ValueError("Index list cannot be empty.")

    if value.startswith("@"):
        path = Path(value[1:]).expanduser().resolve()
        data = path.read_text(encoding="utf-8")
        tokens = data.replace(",", " ").split()
    else:
        tokens = value.replace(",", " ").split()

    try:
        return [int(tok) for tok in tokens]
    except ValueError as exc:
        raise ValueError(f"Could not parse integer indices from '{value}'.") from exc


def parse_args() -> argparse.Namespace:
    """Configure and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create JMOL-style XYZ files for phonon modes restricted to a set of "
            "supercell atom indices."
        )
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=Path("supercell.extxyz"),
        help="Supercell structure file readable by ASE.",
    )
    parser.add_argument(
        "--structure-format",
        default="extxyz",
        help="ASE format hint for the supercell structure. Use 'auto' to guess.",
    )
    parser.add_argument(
        "--mass-structure",
        type=Path,
        default=None,
        help="Optional structure file used solely to obtain atomic masses.",
    )
    parser.add_argument(
        "--mass-format",
        default="aims",
        help="ASE format hint for the mass structure (ignored if not provided).",
    )
    parser.add_argument(
        "--band-file",
        type=Path,
        default=Path("band.hdf5"),
        help="Phonopy band structure HDF5 file.",
    )
    parser.add_argument(
        "--mode-index",
        type=int,
        default=0,
        help="Mode index counted after skipped branches (0-based).",
    )
    parser.add_argument(
        "--skip-branches",
        type=int,
        default=3,
        help="Number of leading branches to skip (e.g. acoustic modes).",
    )
    parser.add_argument(
        "--qpoint-index",
        type=int,
        default=0,
        help="Index of the q-point along the phonon path (0-based).",
    )
    parser.add_argument(
        "--structure-indices",
        default=None,
        help=(
            "Indices of atoms forming the target molecule within the supercell. "
            "Provide as comma/space separated list or '@file' reference. "
            "Defaults to the hard-coded indices for naphthalene."
        ),
    )
    parser.add_argument(
        "--eigenvector-indices",
        default=None,
        help=(
            "Indices selecting rows from the eigenvector matrix. Defaults to the "
            "same list as --structure-indices."
        ),
    )
    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=1.0,
        help="Multiply displacement vectors by this value before writing.",
    )
    parser.add_argument(
        "--normalize-displacements",
        action="store_true",
        help="Scale displacements so the maximum absolute component equals 1.",
    )
    parser.add_argument(
        "--mass-weight",
        action="store_true",
        help=(
            "Apply Phonopy-style 1/sqrt(m_i) mass-weighting using masses from the "
            "--mass-structure or the supercell if not provided."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output XYZ file. Defaults to 'molecule_mode_<idx>'.",
    )
    parser.add_argument(
        "--output-prefix",
        default="molecule_mode",
        help="Prefix used when constructing default output directory and filename.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity of console logging.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    """Initialise logging with a consistent message format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_structure(structure_path: Path, fmt: str) -> Atoms:
    """Read an ASE structure with optional format hint."""
    path = structure_path.expanduser().resolve()
    fmt_hint = None if fmt.lower() == "auto" else fmt
    LOGGER.info("Reading structure from %s", path)
    return read(str(path), format=fmt_hint)


def build_mass_matrix(structure: Atoms) -> np.ndarray:
    """Construct the diagonal mass-weight matrix required by Phonopy."""
    masses = structure.get_masses()
    factors = np.repeat(1.0 / np.sqrt(masses), 3)
    matrix = np.diag(factors)
    LOGGER.debug("Mass matrix shape: %s", matrix.shape)
    return matrix


def extract_mode_data(
    band_file: Path,
    qpoint_index: int,
    branch_index: int,
    mass_matrix: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    """Return the frequency and eigen-displacements for the specified branch."""
    band_path = band_file.expanduser().resolve()
    LOGGER.info("Reading phonon data from %s", band_path)
    with h5py.File(band_path, "r") as handle:
        frequencies = handle["frequency"]
        eigenvectors = handle["eigenvector"]

        frequency = float(frequencies[qpoint_index][0][branch_index])
        raw_eigenvectors = eigenvectors[qpoint_index][0].T
        LOGGER.debug(
            "Eigenvector matrix shape (transposed view): %s",
            raw_eigenvectors.shape,
        )

        if mass_matrix is not None:
            eigenvectors_weighted = raw_eigenvectors @ mass_matrix
        else:
            eigenvectors_weighted = raw_eigenvectors

        displacements = eigenvectors_weighted[branch_index].reshape((-1, 3))

    return frequency, displacements


def select_atoms(structure: Atoms, indices: Sequence[int]) -> Atoms:
    """Slice an ASE Atoms object using integer indices."""
    try:
        subset = structure[indices]
    except Exception as exc:
        raise IndexError(
            "Failed to slice structure with provided indices. Verify they refer "
            "to valid atoms in the supercell."
        ) from exc
    return subset


def apply_index_map(
    displacements: np.ndarray,
    index_map: Sequence[int],
) -> np.ndarray:
    """Reorder displacement rows according to the provided index map."""
    try:
        subset = displacements[index_map]
    except Exception as exc:
        raise IndexError(
            "Eigenvector index mapping must contain valid indices into the full "
            "displacement array."
        ) from exc
    return subset


def compute_scaling(displacements: np.ndarray, scaling: float, normalize: bool) -> float:
    """Compute the final scaling factor applied to displacements."""
    factor = scaling
    if normalize:
        max_component = np.max(np.abs(displacements))
        if max_component == 0.0:
            LOGGER.warning("All displacement components are zero; skipping normalisation.")
        else:
            factor *= 1.0 / max_component
    return factor


def write_xyz(
    structure: Atoms,
    displacements: np.ndarray,
    frequency_cm1: float,
    mode_number: int,
    scaling_factor: float,
    output_path: Path,
) -> None:
    """Write structure and displacement data to an XYZ file."""
    symbols = structure.get_chemical_symbols()
    positions = structure.get_positions()

    LOGGER.info("Writing XYZ output to %s", output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(structure)}\n")
        handle.write(f"# {frequency_cm1:.6f} cm-1, branch # {mode_number}\n")

        for symbol, pos, disp in zip(symbols, positions, displacements):
            dx, dy, dz = np.real(disp) * scaling_factor
            handle.write(
                f"{symbol} "
                f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                f"{dx:.6f} {dy:.6f} {dz:.6f}\n"
            )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    structure_indices = (
        parse_indices(args.structure_indices)
        if args.structure_indices
        else DEFAULT_STRUCTURE_INDICES
    )
    eigenvector_indices = (
        parse_indices(args.eigenvector_indices)
        if args.eigenvector_indices
        else DEFAULT_EIGENVECTOR_INDICES
    )

    supercell = load_structure(args.structure, args.structure_format)
    molecule = select_atoms(supercell, structure_indices)

    mass_reference = (
        load_structure(args.mass_structure, args.mass_format)
        if args.mass_structure is not None
        else supercell
    )
    mass_matrix = build_mass_matrix(mass_reference) if args.mass_weight else None

    branch_index = args.skip_branches + args.mode_index
    frequency_native, displacements_full = extract_mode_data(
        band_file=args.band_file,
        qpoint_index=args.qpoint_index,
        branch_index=branch_index,
        mass_matrix=mass_matrix,
    )

    displacements_subset = apply_index_map(displacements_full, eigenvector_indices)
    scaling = compute_scaling(displacements_subset, args.scaling_factor, args.normalize_displacements)

    frequency_cm1 = frequency_native * 33.35641
    LOGGER.info(
        "Mode frequency: %.6f (native units) -> %.6f cm^-1",
        frequency_native,
        frequency_cm1,
    )

    if len(displacements_subset) != len(molecule):
        raise ValueError(
            "Number of displacement vectors does not match number of selected atoms. "
            "Ensure the structure and eigenvector index lists describe the same atoms."
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path(f"{args.output_prefix}_{args.mode_index:03d}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.output_prefix}_{args.mode_index:03d}.xyz"
    write_xyz(
        structure=molecule,
        displacements=displacements_subset,
        frequency_cm1=frequency_cm1,
        mode_number=branch_index,
        scaling_factor=scaling,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
