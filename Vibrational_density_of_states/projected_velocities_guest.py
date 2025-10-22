import os, sys
import numpy as np
import h5py
from ase.io import read, write


file = sys.argv[1]
filename = file.split("/")[-1]
modeIndex = int(sys.argv[2])
os.chdir(os.curdir)

print('Reading eigen vectors')
with h5py.File("/ada/ptmp/mpsd/shubsharma/HDNNP/mace/naphthalene_com_tight/host_guest/phonons_error_propagation/phonons_com_mean/band.hdf5", "r") as f:
    frequencies = f['frequency'][:,0,:]
    eigenvectors = f['eigenvector'][:,0,:,:]

eigenvectors = eigenvectors[0,:,:]
eigenvectors = eigenvectors.T
frequencies = frequencies[0,3:] * 33.356 # Convert from THz to cm^-1
num_modes = eigenvectors.shape[0]  # Number of modes (after removing acoustic modes)
num_atoms = eigenvectors.shape[1] // 3  # Divide by 3 for Cartesian components
eigenvectors_reshaped = eigenvectors[:, :num_atoms * 3].reshape(num_modes, num_atoms, 3)
print(f'Shape of Eigen vectors: {eigenvectors_reshaped.shape}')

print('Reading velocities')
velTraj = read(file, ':', format='extxyz')

print('Calculating projected velocities')
masses = np.diag(np.concatenate((np.zeros((2844)),(np.sqrt(velTraj[0].get_masses())))))
projections_sum = []
for i in range(1, len(velTraj)):
    velTraj[i]=np.vstack((np.zeros((2844, 3)), velTraj[i].get_positions()))
    projections = np.real(eigenvectors_reshaped[modeIndex] * (masses @ velTraj[i]))
    projection_sum = np.sum(projections, axis=(0, 1))
    projections_sum.append(projection_sum)
    # print(f'Projections sum: {projections_sum}')

np.savetxt(f'guest_projected_velocities_{modeIndex}_{filename}.txt', projections_sum)
    # print(f'Norm of projection: {np.linalg.norm(projections, axis = 1)}')
    # print(f'Projections shape: {projections.shape}')

    # velTraj[i].set_positions(projections.reshape(-1, 3))
    
    # write(f'projected_{modeIndex}_{filename}', velTraj[i], append=True, format='extxyz')

# from matplotlib import pyplot as plt
# plt.plot(np.linalg.norm(projections, axis = 1))
# plt.savefig(f'projected_{modeIndex}_{filename}.png')