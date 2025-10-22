import h5py
import numpy as np
from ase.io import read
import os

mode_index = 1272

structure = read('../relaxed.extxyz', format='extxyz')
positions = structure.get_positions()
positions_guest = positions[-36:]
atom_types = structure.get_chemical_symbols()[-36:]
masses = np.diag(np.repeat(1/np.sqrt(structure.get_masses()), 3))

with h5py.File("../band.hdf5", "r") as f:
    mode_freq = f['frequency'][0][0][mode_index+3]
    print(mode_freq)
    print(f['eigenvector'].shape)
    mode_eigenvectors = f['eigenvector'][0][0].T @ masses
    displacements = mode_eigenvectors[mode_index+3].reshape((-1, 3))
    print(f"Frequency: {mode_freq * 33.35641} cm-1")
    displacements_guest = displacements[-36:]

frequency_factor = 1/np.max(displacements)
scaling_factor = 1*frequency_factor

if not os.path.exists(f'mode_index_9'):
    os.mkdir(f'mode_index_9')
os.chdir(f'mode_index_9')
with open(f"Guest_mode_9.xyz", "w") as f:
   f.write(f"{36}\n")
   f.write(f"# {mode_freq * 33.35641:.6f} cm-1, branch # 12\n")
   for i, (at, pos, disp) in enumerate(zip(atom_types, positions_guest, displacements_guest)):
       f.write(f"{at} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {np.real(disp[0]) * scaling_factor:.6f} {np.real(disp[1])* scaling_factor:.6f} {np.real(disp[2])*scaling_factor:.6f}\n")