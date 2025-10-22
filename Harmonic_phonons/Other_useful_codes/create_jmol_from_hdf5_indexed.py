import os
import numpy as np
import networkx as nx
import h5py
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from ase import Atoms
from ase.io import read, write

# Identify molecules based on bonding connectivity
def identify_molecules(atoms):
    """
    Identify molecules based on bonding connectivity using a graph.
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    masses = atoms.get_masses()

    # Create a graph to represent the connections
    G = nx.Graph()

    # Define bond thresholds for C-C and C-H
    bond_thresholds = {
        'C-C': 1.6,  # Approximate bond length for C-C in angstroms
        'C-H': 1.2,  # Slightly larger than C-H bond length
    }

    # Add edges based on bond thresholds
    for i in range(len(atoms)):
        if i==2867:
            x=5
        for j in range(i + 1, len(atoms)):
            distance = np.linalg.norm(positions[i] - positions[j])
            # Check for C-C bonds
            if symbols[i] == 'C' and symbols[j] == 'C' and distance < bond_thresholds['C-C']:
                G.add_edge(i, j)
            # Check for C-H bonds
            elif (symbols[i] == 'C' and symbols[j] == 'H' and distance < bond_thresholds['C-H']) or \
                 (symbols[i] == 'H' and symbols[j] == 'C' and distance < bond_thresholds['C-H']):
                G.add_edge(i, j)

    # Identify connected components (molecules)
    connected_components = list(nx.connected_components(G))
    molecule_indices = [list(component) for component in connected_components]

    molecule_coms = []
    for molecule in molecule_indices:
        atomic_positions = np.array([positions[i] for i in molecule])
        atomic_masses = np.array([masses[i] for i in molecule])

        # Compute weighted center of mass
        com = np.sum(atomic_positions.T * atomic_masses, axis=1) / np.sum(atomic_masses)
        molecule_coms.append(com)

    return molecule_indices, molecule_coms

# def find_nearest_molecules(mol_com, target_idx, num_neighbors=5):
#     """
#     Find the nearest neighbor molecules to a given molecule index.

#     Parameters:
#     - mol_com (list of np.array): List of center of mass (COM) coordinates for each molecule.
#     - target_idx (int): Index of the molecule whose neighbors we want to find.
#     - num_neighbors (int): Number of nearest neighbors to return.

#     Returns:
#     - List of nearest neighbor molecule indices.
#     """

#     target_com = mol_com[target_idx]  # Get COM of the target molecule
#     distances = []

#     for i, com in enumerate(mol_com):
#         if i != target_idx:  # Skip self
#             dist = np.linalg.norm(target_com - com)  # Euclidean distance
#             distances.append((i, dist))  # Store index and distance

#     # Sort by distance and get closest neighbors
#     distances.sort(key=lambda x: x[1])
#     nearest_neighbors = [idx for idx, _ in distances[:num_neighbors]]

#     return nearest_neighbors



# Read the relaxed structure
structure = read('../relaxed.extxyz')
positions = structure.get_positions()
masses = np.diag(np.repeat(1/np.sqrt(structure.get_masses()), 3))

atom_types = structure.get_chemical_symbols()
mol_idx, mol_com = identify_molecules(structure)
print(f'Number of molecules: {len(mol_idx)}')

# nearest_neighbors = find_nearest_molecules(mol_com, target_idx=-1, num_neighbors=101)
# print(f'Nearest neighbors of guests: {nearest_neighbors}')

mol_idx_to_keep = [159, 160, 163, 164, 143, 147, 144, 148]


pentacene_idx = np.array([mol_idx[232]])
print(f'Pentacene indices: {pentacene_idx}')

guest_nn_idx = np.array([idx for neighbor in mol_idx_to_keep for idx in mol_idx[neighbor]])
print(f'Guest nearest neighbor indices: {guest_nn_idx}')


# # Reading the hdf5 file
mode_index = 2

with h5py.File("../band.hdf5", "r") as f:
    mode_freq = f['frequency'][0][0][mode_index+3]
    print(mode_freq)
    print(f['eigenvector'].shape)
    mode_eigenvectors = f['eigenvector'][0][0].T @ masses
    # mode_eigenvectors = f['eigenvector'][0][0].T
    print(mode_eigenvectors.shape)
    displacements = mode_eigenvectors[mode_index+3].reshape((-1, 3))
    print(f"Frequency: {mode_freq * 33.35641} cm-1")

# print(displacements[2847])


frequency_factor = 1/np.max(displacements)

scaling_factor_pentacene = 1*frequency_factor
scaling_factor_nn_atoms = 3*frequency_factor

# Extracting the pentacene mode
pentacene_positions = positions[pentacene_idx].reshape(-1, 3)
print(pentacene_positions.shape)
pentacene_displacements = displacements[pentacene_idx].reshape(-1, 3)
pentacene_atom_types = np.array(atom_types)[pentacene_idx].reshape(-1,)
print(pentacene_atom_types.shape)

print(pentacene_displacements.shape)


# # Extracting the guest nearest neighbor mode
nn_positions = positions[guest_nn_idx].reshape(-1, 3)
nn_displacements = displacements[guest_nn_idx].reshape(-1, 3)
nn_atom_types = np.array(atom_types)[guest_nn_idx].reshape(-1,)

# # Total number of atoms
pentacene_total_atoms = len(pentacene_positions)
nn_total_atoms = len(nn_positions)



if not os.path.exists(f'mode_index_2'):
    os.mkdir(f'mode_index_2')
os.chdir(f'mode_index_2')

with open(f"Host-guest_mode_2.xyz", "w") as f:
    f.write(f"{nn_total_atoms + pentacene_total_atoms}\n")
    f.write(f"# {mode_freq * 33.35641:.6f} cm-1, branch # 1\n")
    for i, (pos, disp, atom_type) in enumerate(zip(pentacene_positions, pentacene_displacements, pentacene_atom_types)):
        f.write(f"{atom_type} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {np.real(disp[0])*scaling_factor_pentacene:.6f} {np.real(disp[1])*scaling_factor_pentacene:.6f} {np.real(disp[2])*scaling_factor_pentacene:.6f}\n")
    for i, (pos, disp, atom_type) in enumerate(zip(nn_positions, nn_displacements, nn_atom_types)):
        f.write(f"{atom_type} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {np.real(disp[0])*scaling_factor_nn_atoms:.6f} {np.real(disp[1])*scaling_factor_nn_atoms:.6f} {np.real(disp[2])*scaling_factor_nn_atoms:.6f}\n")