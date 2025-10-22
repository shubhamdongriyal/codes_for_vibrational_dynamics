"""Phonon workflow driven by a committee of MACE models.

The script loads an optimized structure, evaluates forces using several
MACE models, passes the averaged forces to Phonopy, and produces phonon
band structures plus DOS plots.  It has been refactored to expose the most
useful options through a command-line interface and to provide structured
logging instead of opaque print statements.

Usage example
-------------
```
python mace_phonopy_committee_mean-naphthalene.py \
    --structure minimization_committee_mean-naphthalene/relaxed.extxyz \
    --model-root . \
    --output-dir phonons-naphthalene
```
Adjust arguments as needed or run with ``--help`` to see all options.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read, write
from mace.calculators import MACECalculator
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.atoms import PhonopyAtoms

LOGGER = logging.getLogger(__name__)


# Default K-point route tailored for the naphthalene structure.
DEFAULT_BAND_PATH = [
    (0, 0, 0),
    (0, 0.5, 0),
    (0, 0.5, 0.5),
    (0, 0, 0.5),
    (0, 0, 0),
    (-0.5, 0, 0.5),
    (-0.5, 0.5, 0.5),
    (0, 0.5, 0),
    (-0.5, 0.5, 0),
    (-0.5, 0, 0),
    (0, 0, 0),
]


def parse_args() -> argparse.Namespace:
    """Return parsed command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate phonon properties via a MACE committee ensemble."
    )
    parser.add_argument(
        "--structure",
        default="minimization_committee_mean-naphthalene/relaxed.extxyz",
        help="Input structure file (read using ASE).",
    )
    parser.add_argument(
        "--structure-format",
        default="extxyz",
        help="Optional ASE format hint (set to 'auto' to let ASE guess).",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("."),
        help="Directory that contains the individual model sub-directories.",
    )
    parser.add_argument(
        "--model-prefix",
        default="nn",
        help="Directory name prefix used to locate individual models.",
    )
    parser.add_argument(
        "--model-filename",
        default="naphthalene_mace_tight_swa.model",
        help="Model file searched for within each model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phonons-naphthalene"),
        help="Destination directory for phonopy outputs.",
    )
    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        default=(2, 2, 2),
        help="Diagonal supercell definition passed to Phonopy.",
    )
    parser.add_argument(
        "--displacement",
        type=float,
        default=0.01,
        help="Displacement distance (Å) for finite-difference forces.",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=1e-4,
        help="Symmetry precision used by Phonopy.",
    )
    parser.add_argument(
        "--band-points",
        type=int,
        default=60,
        help="Number of interpolation points along each band path segment.",
    )
    parser.add_argument(
        "--mesh",
        type=int,
        nargs=3,
        default=(20, 20, 20),
        help="Monkhorst-Pack mesh used for total DOS.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Computation device handed to MACE (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity of the console logger.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip Matplotlib plotting and only write band data files.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    """Initialise the module-level logger with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def configure_matplotlib_defaults() -> None:
    """Apply a consistent Matplotlib style for the generated plots."""
    plt.rcParams.update(
        {
            "figure.figsize": (8, 6),
            "figure.dpi": 300,
            "font.family": "Arial",
            "font.size": 20,
            "axes.labelsize": 20,
            "axes.titlesize": 25,
            "axes.grid": True,
            "lines.linewidth": 3,
            "lines.markersize": 8,
            "legend.fontsize": 16,
        }
    )


def find_model_paths(
    model_root: Path,
    model_prefix: str,
    model_filename: str,
) -> List[Path]:
    """Return absolute paths to all MACE model files under ``model_root``."""
    candidates = []
    for directory in model_root.iterdir():
        if not directory.is_dir() or not directory.name.startswith(model_prefix):
            continue
        model_path = directory / model_filename
        if model_path.is_file():
            candidates.append(model_path.resolve())
            LOGGER.debug("Discovered model: %s", model_path.resolve())
    return sorted(candidates)


def phonopy_atoms_from_ase(structure: Atoms) -> PhonopyAtoms:
    """Convert an ASE Atoms instance to a PhonopyAtoms container."""
    return PhonopyAtoms(
        symbols=structure.get_chemical_symbols(),
        positions=structure.get_positions(),
        cell=structure.get_cell(),
    )


def initialise_calculators(
    model_paths: Iterable[Path], device: str
) -> List[MACECalculator]:
    """Build MACE calculators, one per model path."""
    calculators = []
    for model_path in model_paths:
        LOGGER.info("Loading MACE model: %s", model_path)
        calculators.append(
            MACECalculator(
                model_paths=str(model_path),
                device=device,
                default_dtype="float64",
            )
        )
    return calculators


def evaluate_committee_forces(
    supercells: Sequence[PhonopyAtoms],
    calculators: Sequence[MACECalculator],
    model_paths: Sequence[Path],
    output_dir: Path,
) -> List[np.ndarray]:
    """Evaluate forces with each model and return the averaged forces."""
    forces: List[np.ndarray] = []
    calc_output_dir = output_dir / "mace_calculations"
    calc_output_dir.mkdir(parents=True, exist_ok=True)

    for index, scell in enumerate(supercells):
        LOGGER.info(
            "Processing displaced structure %d/%d",
            index + 1,
            len(supercells),
        )
        scell_ase = Atoms(
            symbols=scell.get_chemical_symbols(),
            positions=scell.get_positions(),
            cell=scell.get_cell(),
            pbc=True,
        )
        structure_dir = calc_output_dir / f"structure_{index:03d}"
        structure_dir.mkdir(parents=True, exist_ok=True)

        model_forces = []
        model_energies = []

        for model_id, (calculator, model_path) in enumerate(zip(calculators, model_paths)):
            scell_ase.set_calculator(calculator)
            LOGGER.debug("  Running model %d (%s)", model_id, model_path.name)
            energy = scell_ase.get_potential_energy()
            force = scell_ase.get_forces()
            model_forces.append(force)
            model_energies.append(energy)

            write(
                str(structure_dir / f"prediction_{index:03d}_model_{model_id:02d}.extxyz"),
                scell_ase,
                format="extxyz",
            )

        mean_force = sum(model_forces) / len(model_forces)
        mean_energy = sum(model_energies) / len(model_energies)

        scell_ase.set_calculator(None)
        scell_ase.arrays["force"] = mean_force
        scell_ase.info["energy"] = mean_energy

        write(
            str(structure_dir / f"prediction_{index:03d}.extxyz"),
            scell_ase,
            format="extxyz",
        )
        forces.append(mean_force)

    LOGGER.info("Finished evaluating forces for %d displaced structures.", len(forces))
    return forces


def run_phonopy_workflow(
    structure_path: Path,
    structure_format: str,
    model_paths: Sequence[Path],
    output_dir: Path,
    supercell_diag: Sequence[int],
    displacement: float,
    symprec: float,
    band_points: int,
    mesh: Sequence[int],
    device: str,
    plot: bool,
) -> None:
    """Execute the phonon workflow and persist all outputs."""
    if not model_paths:
        raise RuntimeError("No MACE models were discovered. Check your inputs.")

    LOGGER.info("Reading structure from %s", structure_path)
    fmt = None if structure_format.lower() == "auto" else structure_format
    structure = read(str(structure_path), format=fmt)
    LOGGER.debug("Loaded structure with %d atoms.", len(structure))

    phonopy_atoms = phonopy_atoms_from_ase(structure)
    supercell_matrix = np.diag(supercell_diag)
    phonon = Phonopy(phonopy_atoms, supercell_matrix, symprec=symprec)
    phonon.generate_displacements(distance=displacement)
    LOGGER.info(
        "Generated %d displaced supercells with displacement %.4f Å.",
        len(phonon.supercells_with_displacements),
        displacement,
    )

    calculators = initialise_calculators(model_paths, device)
    forces = evaluate_committee_forces(
        phonon.supercells_with_displacements,
        calculators,
        model_paths,
        output_dir,
    )

    LOGGER.info("Attaching averaged forces to phonopy model.")
    phonon.set_forces(forces)
    phonon.produce_force_constants()

    yaml_path = output_dir / "phonopy.yaml"
    yaml_fc_path = output_dir / "phonopy_force_constants.yaml"
    phonon.save(str(yaml_path))
    phonon.save(filename=str(yaml_fc_path), settings={"force_constants": True})
    LOGGER.info("Wrote phonopy configuration to %s", yaml_path)

    qpoints, connections = get_band_qpoints_and_path_connections(
        [DEFAULT_BAND_PATH],
        npoints=band_points,
    )
    phonon.run_band_structure(qpoints, path_connections=connections, with_eigenvectors=True)
    phonon.write_yaml_band_structure(filename=str(output_dir / "band.yaml"))
    phonon.write_hdf5_band_structure(filename=str(output_dir / "band.hdf5"))
    LOGGER.info("Band structure data saved to YAML and HDF5 formats.")

    phonon.run_mesh(mesh)
    phonon.run_total_dos()

    if plot:
        LOGGER.info("Plotting band structure and DOS.")
        fig = phonon.plot_band_structure_and_dos()
        fig.savefig(str(output_dir / "band_structure_and_dos.pdf"))
        plt.close(fig)

    conversion = phonon.get_unit_conversion_factor()
    gamma_freqs = phonon.get_frequencies([0.0, 0.0, 0.0])
    LOGGER.info("Conversion factor to THz: %.6f", conversion)
    LOGGER.info("Gamma-point frequencies (native units): %s", gamma_freqs)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if not args.no_plot:
        configure_matplotlib_defaults()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = find_model_paths(
        args.model_root.expanduser().resolve(),
        args.model_prefix,
        args.model_filename,
    )
    LOGGER.info("Found %d model(s) for committee evaluation.", len(model_paths))

    run_phonopy_workflow(
        structure_path=Path(args.structure).expanduser().resolve(),
        structure_format=args.structure_format,
        model_paths=model_paths,
        output_dir=output_dir,
        supercell_diag=args.supercell,
        displacement=args.displacement,
        symprec=args.symprec,
        band_points=args.band_points,
        mesh=args.mesh,
        device=args.device,
        plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
