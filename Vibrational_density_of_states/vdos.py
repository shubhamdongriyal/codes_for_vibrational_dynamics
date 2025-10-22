import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, fft as sp_fft
from py4vasp import Calculation
import scipy.io as sio
from scipy import signal
from scipy import stats
from ase.io import read
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# Boltzmann Constant in [eV/K]
kb = 8.617332478E-5
# electron volt in [Joule]
ev = 1.60217733E-19
# Avogadro's Constant
Navogadro = 6.0221412927E23


def getVACF(velocity,Nion,method='Fourier'):
    """ Velocity Autocorrelation Function """
    # assume velocities.shape = (nstep,natoms,3)

    if method == 'Direct':
        VAF2 = np.zeros((len(velocity))*2 - 1)
        for i in range(Nion):
            for j in range(3):
                vel_dummy=velocity[:,i,j]
                VAF2 += signal.fftconvolve(vel_dummy,
                                         vel_dummy[::-1],
                                         mode='full')        # two-sided VAF    
        VAF2 /=  np.sum(velocity**2)
        VAF = VAF2[len(velocity)-1:]
        
    elif method == 'Fourier':
        ft = np.fft.rfft(velocity, axis=0) # Fourier transform
        ac = ft*np.conjugate(ft) # Power spectrum
        mean_ac = np.real(np.mean((np.mean(ac, axis=1)),axis=1))
        VAF = np.fft.irfft(mean_ac)[:int(len(velocity)/2)+1]/np.average(velocity**2)/len(velocity)
    
    return VAF


def getVDOS(velocity,potim,Nions,method = 'Fourier',window=True,npad= 5,unit='THz',tonext=True,max_freq_plot=1.2e14):
    velocity -= np.mean(velocity, axis=0)
    dt=potim
    # assume velocities.shape = (nstep,natoms,3)
    axis = 0
    
     # Velocities are not padded or windowed, but VACF does
    VACF=getVACF(velocity, Nions,method=method)
    if window:
       VACF_win=VACF*np.hanning(2*VACF.shape[axis])[VACF.shape[axis]:]
    else:
        VACF_win=VACF
        
    if npad is not None:
       if method == 'Fourier':
           npad = 2*(npad+1)-1
       nadd = (VACF_win.shape[axis]-1)*npad-1
       VACF_win = np.append(VACF_win, np.zeros(nadd))
    
            
    full_VDOS = sp_fft.fftn(VACF_win,axes=axis)
    freqq =sp_fft.fftfreq(VACF_win.shape[axis],dt)
    split_idx = abs(freqq-max_freq_plot).argmin()
    freqPos = freqq[:split_idx]
    VDOS=np.real(full_VDOS[:split_idx])
                
    if unit.lower() == 'cm-1':
        freqPos *= 33.35640951981521
    if unit.lower() == 'mev':
        freqPos *= 4.13567
    
    return freqPos,VDOS,VACF_win

def read_trajectories(nruns, mdlen):
    """
    Read nve trajectories from multiple runs
    Parameters:
    nruns (int): Number of runs
    mdlen (int): time in ps
    """
    
    time_in_ps = 1000 * mdlen  # Time in fs
    projModes1 = []
    for i in range(1, nruns+1):
        try:
            data = read(f'/ada/ptmp/mpsd/shubsharma/HDNNP/mace/naphthalene_com_tight/vdos/naphthalene/nve20ps_convergence_test/vel-commean_300_traj/nap80K-nvt.velocities_0_{i}.xyz', index=':'+str(time_in_ps))
            projModes1.append(data)
        except FileNotFoundError:
            continue  # Skip the file if it's not found
    return projModes1

mdlen = 20 # change this
num_traj = 300 # change this
time = np.arange(0, mdlen*1000 + 1, 1)
projModes1 = read_trajectories(num_traj, mdlen)


freqs= []
VDOSs= []
VACFs=[]
timees =[]

time_window = 20e-12 # change this
potim = 1e-15
Nions = 36
masses = np.diag(np.sqrt(projModes1[0][0].get_masses()))
trajlength=int(time_window/potim)
for j in range(len(projModes1)):
    vel = np.asanyarray([projModes1[j][i].get_positions() for i in range(len(projModes1[j]))])
    vel = masses @ vel
    #vaf, vaf2 = getVACF(vel, Nions)
    freq,VDOS,VACF=getVDOS(vel,potim,Nions,method='Fourier',window=True,npad=3)
    freqs.append(freq)
    #VAFs.append(vaf)
    VDOSs.append(VDOS)
    VACFs.append(VACF)

VDOS_mean = np.mean(VDOSs, axis=0)
VDOS_std = np.std(VDOSs, axis=0, ddof=1)

VAF_mean = np.mean(VACFs, axis=0)
VAF_std = np.std(VACFs, axis=0, ddof=1)

 
os.mkdir(f'naph_{num_traj}_traj_{mdlen}_ps')
os.chdir(f'naph_{num_traj}_traj_{mdlen}_ps')

freqs[0] = freqs[0] * (33.356 / (10**12)) # Convert Hz to cm^-1

np.savetxt(f'vdoses.txt',np.concatenate([freqs[0].reshape(-1, 1), np.array(VDOSs).T], axis=1), delimiter='\t', header='Frequency (cm-1)\tVDOS')
np.savetxt(f'vdos_naph_{num_traj}_traj_{mdlen}_ps.txt', np.concatenate((freqs[0].reshape(-1,1), VDOS_mean.reshape(-1,1), VDOS_std.reshape(-1,1)), axis=1), delimiter='\t', header='Frequency (cm-1)\tVDOS_mean\tVDOS_std')
np.savetxt(f'vacf_naph_{num_traj}_traj_{mdlen}_ps.txt', np.concatenate((np.arange(VAF_mean.shape[0]).reshape(-1,1), VAF_mean.reshape(-1,1), VAF_std.reshape(-1,1)), axis=1), delimiter='\t', header='Time (fs)\tVAF_mean\tVAF_std')





