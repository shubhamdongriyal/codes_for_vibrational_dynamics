import h5py
import numpy as np
from ase.io import read
import os

mode_index = 2722

structure = read('relaxed.extxyz', format='extxyz')
positions = structure.get_positions()

with h5py.File("band.hdf5", "r") as f:
    mode_freq = f['frequency'][0][0][mode_index+3]
    mode_eigenvectors = f['eigenvector'][0][0][mode_index+3]
    displacements = mode_eigenvectors.reshape((-1, 3))
    print(f"Frequency: {mode_freq * 33.35641} cm-1")

# os.mkdir(f'mode_index_1')
os.chdir(f'mode_index_37')
with open(f"Host-guest_mode_37.xyz", "w") as f:
   f.write(f"{len(structure)}\n")
   f.write(f"# {mode_freq * 33.35641:.6f} cm-1, branch # 37\n")
   for i, (pos, disp) in enumerate(zip(positions, displacements)):
       f.write(f"{structure.get_chemical_symbols()[i]} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {np.real(disp[0]):.6f} {np.real(disp[1]):.6f} {np.real(disp[2]):.6f}\n")
