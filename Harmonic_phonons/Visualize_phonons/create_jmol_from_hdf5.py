"""Export phonon mode displacements from a Phonopy HDF5 file to XYZ.

The script reads an input geometry and the Phonopy ``band.hdf5`` output,
extracts a single vibrational mode, and writes the equilibrium positions plus
eigenvector displacements in a JMOL-friendly XYZ file.  It supports optional
mass-weighting, configurable scaling.

Usage example
-------------
```
python Harmonic_phonons/Visualize_phonons/create_jmol_from_hdf5.py \
    --structure geometry.in --structure-format aims \
    --band-file band.hdf5 --mode-index 1 --skip-branches 3 \
    --output-dir visualizations
```
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from ase import Atoms
from ase.io import read

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Configure and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Create a JMOL-style XYZ file for a selected phonon mode."
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=Path("geometry.in"),
        help="Input structure file readable by ASE.",
    )
    parser.add_argument(
        "--structure-format",
        default="aims",
        help="ASE format hint for the structure file. Use 'auto' to guess.",
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
        help="Mode index counted after any skipped branches (0-based).",
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
        "--scaling-factor",
        type=float,
        default=1.0,
        help="Multiply displacement vectors by this value before writing.",
    )
    parser.add_argument(
        "--mass-weight",
        action="store_true",
        help="Apply 1/sqrt(m_i) mass-weighting to eigenvectors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output XYZ file. Defaults to mode-specific folder.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity of console logging.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    """Initialise logging with a succinct format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_structure(structure_path: Path, fmt: str) -> Atoms:
    """Read an ASE structure, expanding user paths and handling format hints."""
    LOGGER.info("Reading structure from %s", structure_path)
    formatted_path = structure_path.expanduser().resolve()
    format_hint = None if fmt.lower() == "auto" else fmt
    structure = read(str(formatted_path), format=format_hint)
    LOGGER.debug("Loaded structure with %d atoms.", len(structure))
    return structure


def load_mode_data(
    band_path: Path,
    mode_index: int,
    qpoint_index: int,
    skip_branches: int,
    mass_matrix: np.ndarray | None,
) -> Tuple[float, np.ndarray]:
    """Extract the frequency and displacement array for the chosen mode."""
    LOGGER.info(
        "Extracting mode index %d (skip=%d) at q-point %d from %s",
        mode_index,
        skip_branches,
        qpoint_index,
        band_path,
    )
    with h5py.File(band_path.expanduser().resolve(), "r") as handle:
        frequencies = handle["frequency"]
        eigenvectors = handle["eigenvector"]

        LOGGER.debug("Frequency dataset shape: %s", frequencies.shape)
        LOGGER.debug("Eigenvector dataset shape: %s", eigenvectors.shape)

        branch_index = skip_branches + mode_index
        frequency = float(frequencies[qpoint_index][0][branch_index])

        raw_eigenvectors = eigenvectors[qpoint_index][0].T
        LOGGER.debug("Raw eigenvector matrix shape: %s", raw_eigenvectors.shape)

        if mass_matrix is not None:
            LOGGER.debug("Applying mass-weighting to eigenvectors.")
            mode_eigenvectors = raw_eigenvectors @ mass_matrix
        else:
            mode_eigenvectors = raw_eigenvectors

        displacements = mode_eigenvectors[branch_index].reshape((-1, 3))

    return frequency, displacements


def build_mass_matrix(structure: Atoms) -> np.ndarray:
    """Construct the diagonal mass-weight matrix used by Phonopy."""
    masses = structure.get_masses()
    repeat = np.repeat(1.0 / np.sqrt(masses), 3)
    mass_matrix = np.diag(repeat)
    LOGGER.debug("Mass matrix shape: %s", mass_matrix.shape)
    return mass_matrix


def write_xyz(
    structure: Atoms,
    displacements: np.ndarray,
    frequency_thz: float,
    mode_index: int,
    scaling_factor: float,
    output_path: Path,
) -> None:
    """Write an XYZ file including displacements suited for JMOL."""
    symbols = structure.get_chemical_symbols()
    positions = structure.get_positions()

    LOGGER.info("Writing XYZ for mode %d to %s", mode_index, output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(structure)}\n")
        handle.write(f"# {frequency_thz:.6f} cm-1, branch # {mode_index}\n")

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

    structure = load_structure(args.structure, args.structure_format)
    mass_matrix = build_mass_matrix(structure) if args.mass_weight else None

    frequency_native, displacements = load_mode_data(
        band_path=args.band_file,
        mode_index=args.mode_index,
        qpoint_index=args.qpoint_index,
        skip_branches=args.skip_branches,
        mass_matrix=mass_matrix,
    )

    frequency_cm1 = frequency_native * 33.35641
    LOGGER.info(
        "Mode frequency: %.6f (native units) / %.6f cm^-1",
        frequency_native,
        frequency_cm1,
    )

    mode_folder = (
        args.output_dir
        if args.output_dir is not None
        else Path(f"mode_index_{args.mode_index:03d}")
    )
    output_dir = mode_folder.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"Mode_{args.mode_index:03d}.xyz"
    write_xyz(
        structure=structure,
        displacements=displacements,
        frequency_thz=frequency_cm1,
        mode_index=args.mode_index,
        scaling_factor=args.scaling_factor,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
