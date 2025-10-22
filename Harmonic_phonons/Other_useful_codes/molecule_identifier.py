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


structure = read('../relaxed.extxyz')
mol_idx, mol_com = identify_molecules(structure)

mol_com = np.array(mol_com).reshape(-1, 3)
print(mol_com.shape)


# first_four = [232, 147, 160, 143, 164]
# second_side_two = [163, 144,  148, 159]
# lateral_further_side = [131, 176]
# next_further_side = [26, 72]
# lateral_other_side = [132, 175]

# extra = [45, 53]
# lateral_first_two = [25, 73]
# lateral_other_side_extra = [179, 128]



# new indexes

first_two = [232]
second_two = []
third_two = []
fourth_two = []

#53, 60

extra_one = [159, 160]
#159
extra_two = [163, 164]

#155, 156

fifth_two = []
sixth_two = []
# 45, 38

#53 and 38 are side molecules


extra_three = [ 143]
extra_four = [147, 144, 148]
#143, 140,  [147, 144, 148, 30]
# seventh_two = [1]

#139, 148

#147, 140

eigth_two = []
#172, 176, 180

# lateral
ninth_three = [ ]
seventh_three = [ ]
thriteenth_three = []
# 131, 26, 128, 127, 22, 132

tenth_three = []
#76, 179, 72, 68, 175
eleven_three = []

#  171

mol_idx_to_keep = first_two + second_two + third_two + fourth_two + fifth_two + sixth_two + seventh_three +eigth_two + ninth_three + tenth_three + eleven_three + thriteenth_three + extra_one + extra_two + extra_three + extra_four

print(mol_idx_to_keep)

atomic_idx = []
for idx in mol_idx_to_keep:
    # print(mol_idx[idx])
    for j in mol_idx[idx]:
        atomic_idx.append(j)


write('molecule_index.extxyz', structure[atomic_idx], 'extxyz')
# indices = np.arange(len(mol_idx))
# print(mol_com)
# plt.figure(figsize=(20, 10))
# plt.scatter(mol_com[:, 0], mol_com[:, 1])
# for idx, com in zip(indices, mol_com):
#     plt.text(com[0], com[1], str(idx))
# plt.savefig('molecule_index.png')




# tuple_list = []
# for idx, sublist in enumerate(mol_idx):
#     for value in sublist:
#         tuple_list.append((value, idx))
# tuple_list.sort(key=lambda x: x[0])
# flattened_second_indices = [second for _, second in tuple_list]

# write('molecule_index.extxyz', structure, 'extxyz')