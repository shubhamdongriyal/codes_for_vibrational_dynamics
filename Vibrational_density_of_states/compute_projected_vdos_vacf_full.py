import os
import sys
from flask import g
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



def getvelocity(positions, lattice_vec, Niter, potim):
    dpos=np.diff(positions, axis=0)
    # apply periodic boundary condition
    dpos[dpos > 0.5] -= 1.0
    dpos[dpos <-0.5] += 1.0
    # Velocity in Angstrom per femtosecond
    for i in range(Niter-1):
        dpos[i,:,:] = np.linalg.multi_dot([dpos[i,:,:], lattice_vec]) / potim
    return dpos

def welch(M, sym=1):
    """Welch window. Function skeleton shamelessly stolen from
    scipy.signal.bartlett() and others. PWTOOL"""
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1,dtype=float)
    odd = M % 2
    if not sym and not odd:
        M = M+1
    n = np.arange(0,M)
    w = 1.0-((n-0.5*(M-1))/(0.5*(M-1)))**2.0
    if not sym and not odd:
        w = w[:-1]
    return w

def pad_zeros(arr, axis=0, where='end', nadd=None, upto=None, tonext=None,
              tonext_min=None):
    """Pad an nd-array with zeros. Default is to append an array of zeros of
    the same shape as `arr` to arr's end along `axis`.

    Parameters
    ----------
    arr :  nd array
    axis : the axis along which to pad
    where : string {'end', 'start'}, pad at the end ("append to array") or
        start ("prepend to array") of `axis`
    nadd : number of items to padd (i.e. nadd=3 means padd w/ 3 zeros in case
        of an 1d array)
    upto : pad until arr.shape[axis] == upto
    tonext : bool, pad up to the next power of two (pad so that the padded
        array has a length of power of two)
    tonext_min : int, when using `tonext`, pad the array to the next possible
        power of two for which the resulting array length along `axis` is at
        least `tonext_min`; the default is tonext_min = arr.shape[axis]

    Use only one of nadd, upto, tonext.

    Returns
    -------
    padded array

    Examples
    --------
    >>> # 1d
    >>> pad_zeros(a)
    array([1, 2, 3, 0, 0, 0])
    >>> pad_zeros(a, nadd=3)
    array([1, 2, 3, 0, 0, 0])
    >>> pad_zeros(a, upto=6)
    array([1, 2, 3, 0, 0, 0])
    >>> pad_zeros(a, nadd=1)
    array([1, 2, 3, 0])
    >>> pad_zeros(a, nadd=1, where='start')
    array([0, 1, 2, 3])
    >>> # 2d
    >>> a=arange(9).reshape(3,3)
    >>> pad_zeros(a, nadd=1, axis=0)
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8],
           [0, 0, 0]])
    >>> pad_zeros(a, nadd=1, axis=1)
    array([[0, 1, 2, 0],
           [3, 4, 5, 0],
           [6, 7, 8, 0]])
    >>> # up to next power of two
    >>> 2**arange(10)
    array([  1,   2,   4,   8,  16,  32,  64, 128, 256, 512])
    >>> pydos.pad_zeros(arange(9), tonext=True).shape
    (16,)
    """
    if tonext == False:
        tonext = None
    lst = [nadd, upto, tonext]
    assert lst.count(None) in [2,3], "`nadd`, `upto` and `tonext` must be " +\
           "all None or only one of them not None"
    if nadd is None:
        if upto is None:
            if (tonext is None) or (not tonext):
                # default
                nadd = arr.shape[axis]
            else:
                tonext_min = arr.shape[axis] if (tonext_min is None) \
                             else tonext_min
                # beware of int overflows starting w/ 2**arange(64), but we
                # will never have such long arrays anyway
                two_powers = 2**np.arange(30)
                assert tonext_min <= two_powers[-1], ("tonext_min exceeds "
                    "max power of 2")
                power = two_powers[np.searchsorted(two_powers,
                                                  tonext_min)]
                nadd = power - arr.shape[axis]
        else:
            nadd = upto - arr.shape[axis]
    if nadd == 0:
        return arr
    add_shape = list(arr.shape)
    add_shape[axis] = nadd
    add_shape = tuple(add_shape)
    if where == 'end':
        return np.concatenate((arr, np.zeros(add_shape, dtype=arr.dtype)), axis=axis)
    elif where == 'start':
        return np.concatenate((np.zeros(add_shape, dtype=arr.dtype), arr), axis=axis)
    else:
        raise Exception("illegal `where` arg: %s" %where)

def slicetake(a, sl, axis=None, copy=False):
    """The equivalent of numpy.take(a, ..., axis=<axis>), but accepts slice
    objects instead of an index array. Also by default, it returns a *view* and
    no copy.

    Parameters
    ----------
    a : numpy ndarray
    sl : slice object, list or tuple of slice objects
        axis=<int>
            one slice object for *that* axis
        axis=None
            `sl` is a list or tuple of slice objects, one for each axis.
            It must index the whole array, i.e. len(sl) == len(a.shape).
    axis : {None, int}
    copy : bool, return a copy instead of a view

    Returns
    -------
    A view into `a` or copy of a slice of `a`.

    Examples
    --------
    >>> from numpy import s_
    >>> a = np.random.rand(20,20,20)
    >>> b1 = a[:,:,10:]
    >>> # single slice for axis 2
    >>> b2 = slicetake(a, s_[10:], axis=2)
    >>> # tuple of slice objects
    >>> b3 = slicetake(a, s_[:,:,10:])
    >>> (b2 == b1).all()
    True
    >>> (b3 == b1).all()
    True
    >>> # simple extraction too, sl = integer
    >>> (a[...,5] == slicetake(a, 5, axis=-1))
    True
    """
    # The long story
    # --------------
    #
    # 1) Why do we need that:
    #
    # # no problem
    # a[5:10:2]
    #
    # # the same, more general
    # sl = slice(5,10,2)
    # a[sl]
    #
    # But we want to:
    #  - Define (type in) a slice object only once.
    #  - Take the slice of different arrays along different axes.
    # Since numpy.take() and a.take() don't handle slice objects, one would
    # have to use direct slicing and pay attention to the shape of the array:
    #
    #     a[sl], b[:,:,sl,:], etc ...
    #
    # We want to use an 'axis' keyword instead. np.r_() generates index arrays
    # from slice objects (e.g r_[1:5] == r_[s_[1:5] ==r_[slice(1,5,None)]).
    # Since we need index arrays for numpy.take(), maybe we can use that? Like
    # so:
    #
    #     a.take(r_[sl], axis=0)
    #     b.take(r_[sl], axis=2)
    #
    # Here we have what we want: slice object + axis kwarg.
    # But r_[slice(...)] does not work for all slice types. E.g. not for
    #
    #     r_[s_[::5]] == r_[slice(None, None, 5)] == array([], dtype=int32)
    #     r_[::5]                                 == array([], dtype=int32)
    #     r_[s_[1:]]  == r_[slice(1, None, None)] == array([0])
    #     r_[1:]
    #         ValueError: dimensions too large.
    #
    # The returned index arrays are wrong (or we even get an exception).
    # The reason is given below.
    # Bottom line: We need this function.
    #
    # The reason for r_[slice(...)] gererating sometimes wrong index arrays is
    # that s_ translates a fancy index (1:, ::5, 1:10:2, ...) to a slice
    # object. This *always* works. But since take() accepts only index arrays,
    # we use r_[s_[<fancy_index>]], where r_ translates the slice object
    # prodced by s_ to an index array. THAT works only if start and stop of the
    # slice are known. r_ has no way of knowing the dimensions of the array to
    # be sliced and so it can't transform a slice object into a correct index
    # array in case of slice(<number>, None, None) or slice(None, None,
    # <number>).
    #
    # 2) Slice vs. copy
    #
    # numpy.take(a, array([0,1,2,3])) or a[array([0,1,2,3])] return a copy of
    # `a` b/c that's "fancy indexing". But a[slice(0,4,None)], which is the
    # same as indexing (slicing) a[:4], return *views*.

    if axis is None:
        slices = sl
    else:
        # Note that these are equivalent:
        #   a[:]
        #   a[s_[:]]
        #   a[slice(None)]
        #   a[slice(None, None, None)]
        #   a[slice(0, None, None)]
        slices = [slice(None)] * a.ndim
        slices[axis] = sl
    # a[...] can take a tuple or list of slice objects
    # a[x:y:z, i:j:k] is the same as
    # a[(slice(x,y,z), slice(i,j,k))] == a[[slice(x,y,z), slice(i,j,k)]]
    slices = tuple(slices)
    if copy:
        return a[slices].copy()
    else:
        return a[slices]
        
def num_sum(arr, axis=None, keepdims=False, **kwds):
    """This numpy.sum() with some features implemented which can be found in
    numpy v1.7 and later: `axis` can be a tuple to select arbitrary axes to sum
    over.

    We also have a `keepdims` keyword, which however works completely different
    from numpy. Docstrings shamelessly stolen from numpy and adapted here
    and there.

    Parameters
    ----------
    arr : nd array
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed. The default (`axis` =
        `None`) is to perform a sum over all the dimensions of the input array.
        `axis` may be negative, in which case it counts from the last to the
        first axis.
        If this is a tuple of ints, a sum is performed on multiple
        axes, instead of a single axis or all the axes as before.
    keepdims : bool, optional
        If this is set to True, the axes from `axis` are left in the result
        and the reduction (sum) is performed for all remaining axes. Therefore,
        it reverses the `axis` to be summed over.
    **kwds : passed to np.sum().

    Examples
    --------
    >>> a=rand(2,3,4)
    >>> num.sum(a)
    12.073636268676152
    >>> a.sum()
    12.073636268676152
    >>> num.sum(a, axis=1).shape
    (2, 4)
    >>> num.sum(a, axis=(1,)).shape
    (2, 4)
    >>> # same as axis=1, i.e. it inverts the axis over which we sum
    >>> num.sum(a, axis=(0,2), keepdims=True).shape
    (2, 4)
    >>> # numpy's keepdims has another meaning: it leave the summed axis (0,2)
    >>> # as dimension of size 1 to allow broadcasting
    >>> numpy.sum(a, axis=(0,2), keepdims=True).shape
    (1, 3, 1)
    >>> num.sum(a, axis=(1,)) - num.sum(a, axis=1)
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.]])
    >>> num.sum(a, axis=(0,2)).shape
    (3,)
    >>> num.sum(a, axis=(0,2)) - a.sum(axis=0).sum(axis=1)
    array([ 0.,  0.,  0.])
    """
    # Recursion rocks!
    def _sum(arr, tosum):
        if len(tosum) > 0:
            # Choose axis to sum over, remove from list w/ remaining axes.
            axis = tosum.pop(0)
            _arr = arr.sum(axis=axis)
            # arr has one dim less now. Rename remaining axes accordingly.
            _tosum = [xx - 1 if xx > axis else xx for xx in tosum]
            return _sum(_arr, _tosum)
        else:
            return arr

    axis_is_int = isinstance(axis, int)
    if axis is None:
        if keepdims:
            raise Exception("axis=None + keepdims=True makes no sense")
        else:
            return np.sum(arr, axis=axis, **kwds)
    elif axis_is_int and not keepdims:
        return np.sum(arr, axis=axis, **kwds)
    else:
        if axis_is_int:
            tosum = [axis]
        elif isinstance(axis, tuple) or isinstance(axis, list):
            tosum = list(axis)
        else:
            raise Exception("illegal type for axis: %s" % str(type(axis)))
        if keepdims:
            alldims = range(arr.ndim)
            tosum = [xx for xx in alldims if xx not in tosum]
        return _sum(arr, tosum)

def getVACF(velocity,Nion):
    """ Velocity Autocorrelation Function """
    # assume velocities.shape = (nstep,natoms,3)

    VAF2 = np.zeros((len(velocity))*2 - 1)
    for i in range(Nion):
        for j in range(3):
            vel_dummy=velocity[:,i,j]
            VAF2 += signal.fftconvolve(vel_dummy,
                                         vel_dummy[::-1],
                                         mode='full')        # two-sided VAF    
    VAF2 /=  np.sum(velocity**2)
    VAF = VAF2[len(velocity)-2:]
    return VAF,VAF2


def getVDOS(velocity,potim,Nions,method,window=True,npad= 5,unit='THz',tonext=True,max_freq_plot=1.2e14):
    dt=potim
    # assume velocities.shape = (nstep,natoms,3)
    axis = 0
    if  window:
        sl = [None]*velocity.ndim
        sl[axis] = slice(None)  # ':'
        vel2 = velocity*(welch(velocity.shape[axis])[tuple(sl)])
    else:
        vel2=velocity

    if npad is not None:
        nadd = (vel2.shape[axis]-1)*npad
        if tonext:
            vel2 = pad_zeros(vel2, tonext=True,
                             tonext_min=vel2.shape[axis] + nadd,
                             axis=axis)
        else:
            vel2 = pad_zeros(vel2, tonext=False, nadd=nadd, axis=axis)
    
    if method == 'VACF': # Velocities are not padded or windowed, but VACF does
        VACF,VACF2=getVACF(velocity, Nions)
        if window:
            VACF2_win=VACF2*welch(2*velocity.shape[axis]-1)
        else:
            VACF2_win=VACF2
        
        if npad is not None:
            nadd = (VACF2_win.shape[axis]-1)*npad-1
            if tonext:
                VACF2_win = pad_zeros(VACF2_win, tonext=True,
                             tonext_min=VACF2_win.shape[axis] + nadd,
                             axis=axis)
            else:
                VACF2_win = pad_zeros(VACF2_win, tonext=False, nadd=nadd, axis=axis)
    
            
        full_VDOS = sp_fft.fftn(VACF2_win,axes=axis)
        freqq =sp_fft.fftfreq(VACF2_win.shape[axis],dt)
        split_idx = abs(freqq-max_freq_plot).argmin()
        freqPos = freqq[:split_idx]
        VDOS=full_VDOS[:split_idx]
    elif method == 'direct': # velocities windowed and padded
        full_VDOS = abs(sp_fft.fftn(vel2,axes=axis))**2
        freqq = sp_fft.fftfreq(vel2.shape[0],dt)
        split_idx = abs(freqq-max_freq_plot).argmin()
        freqPos = freqq[:split_idx]
        fft_vel = slicetake(full_VDOS, slice(0, split_idx), axis=axis, copy=False)
        VDOS = num_sum(fft_vel, axis=axis, keepdims=True)
    else: 
        print ('Invalid method')
    
        
    if unit.lower() == 'cm-1':
        freqPos *= 33.35640951981521
    if unit.lower() == 'mev':
        freqPos *= 4.13567
    
    return freqPos,VDOS

def read_trajectories(nruns, mdlen):
    """
    Read nve trajectories from multiple runs
    Parameters:
    nruns (int): Number of runs
    mdlen (int): time in ps
    """
    
    time_in_ps = 1000 * mdlen  # Time in fs
    projModes1 = []
    # for i in range(1, nruns+1):
    #     try:
    #         print(i + 4)
    #         data = read(f'/ada/ptmp/mpsd/shubsharma/HDNNP/mace/naphthalene_com_tight/host_guest/vdos/projected_velocities/full/projected_1276_nap100K-nve.velocities_0_0{i+4}.xyz', index=':'+str(time_in_ps)) #change
    #         projModes1.append(data)
    #     except FileNotFoundError:
    #         continue  # Skip the file if it's not found
    # return projModes1

    for i in range(1, nruns + 1):
        try:
            # Add leading zero for single-digit numbers
            if i + 4 < 10:
                filename = f'full_projected_velocities_{modeIndex}_nap100K-nve.velocities_0_0{i+4}.xyz.txt'
            else:
                filename = f'full_projected_velocities_{modeIndex}_nap100K-nve.velocities_0_{i+4}.xyz.txt'

            data = np.loadtxt(filename)
            projModes1.append(data)
            print(projModes1)
        except FileNotFoundError:
            continue
        return projModes1

    
modeIndex = int(sys.argv[1])
mdlen = 15 # change this
num_traj = 27 # change this
time = np.arange(0, mdlen*1000 + 1, 1)
projModes1 = read_trajectories(num_traj, mdlen)


freqs= []
VAFs= []
VDOSs= []
timees =[]
VAFs = []

time_window = 15e-12 # change this
potim = 1e-15
Nions = 1 # change this
trajlength=int(time_window/potim)


from scipy.optimize import curve_fit
def lorentzian(x, x0, gamma, A):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2)) 

def lifetime_to_linewidth(lifetime_s):
    """Convert lifetime in seconds to linewidth in cm-1."""
    c = 2.9979e10  # Speed of light in cm/s
    linewidth = 1 / (2 * np.pi * lifetime_s)
    return linewidth

def linewidth_to_lifetime(linewidth_cm1):
    """Convert linewidth in cm-1 to lifetime in seconds."""
    c = 2.9979e10  # Speed of light in cm/s
    lifetime = 1 / (2 * np.pi * linewidth_cm1)
    return lifetime

r_squareds = []
x0_fits = []
gamma_fits = []
a_fits = []
y_fits = []

lifetime_s = 5e-12 # 5 ps  

for j in range(len(projModes1)):  
    vel = np.asanyarray([projModes1[j][i] for i in range(len(projModes1[j]))])
    vel = vel.reshape(-1, 1)
    vel = np.hstack((vel, np.zeros((vel.shape[0], 2)))) 
    vel = vel.reshape(vel.shape[0], Nions, 3)
    vaf, vaf2 = getVACF(vel, Nions)
    freq,VDOS=getVDOS(vel,potim,Nions,method='direct',window=True,npad=5)

    popt, pcov = curve_fit(lorentzian, freq, VDOS, p0=[freq[np.argmax(VDOS)], lifetime_to_linewidth(lifetime_s), np.max(VDOS)])
    x0_fit, gamma_fit, a_fit = popt
    y_fit = lorentzian(freq, *popt)
    residuals = VDOS - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((VDOS - np.mean(VDOS))**2)
    r_squared = 1 - (ss_res / ss_tot)
    r_squareds.append(r_squared)
    x0_fits.append(x0_fit)
    gamma_fits.append(gamma_fit)
    a_fits.append(a_fit)
    y_fits.append(y_fit)

    freqs.append(freq)
    VAFs.append(vaf)
    VDOSs.append(VDOS)
 

VDOS_mean = np.mean(VDOSs, axis=0)
VDOS_std = np.std(VDOSs, axis=0)

VAF_mean = np.mean(VAFs, axis=0)
VAF_std = np.std(VAFs, axis=0)

# r_squared_mean = np.mean(r_squareds)

gamma_mean = np.mean(gamma_fits)
gamma_std = np.std(gamma_fits)

lifetime_mean = np.mean([linewidth_to_lifetime(gamma) for gamma in gamma_fits])
lifetime_std = np.std([linewidth_to_lifetime(gamma) for gamma in gamma_fits])
print(f'\n')


np.savetxt(f'y_fits_{num_traj}_traj_{mdlen}_ps.txt', np.hstack((freqs[0].reshape(-1,1), np.array(y_fits).T)), delimiter='\t', header='Frequency (Hz)\tVDOS_fits')
np.savetxt(f'r_squared_{num_traj}_traj_{mdlen}_ps.txt', r_squareds, delimiter='\t', header='R_squared')
np.savetxt(f'lifetimes_info_{num_traj}_traj_{mdlen}_ps.txt', np.hstack((lifetime_mean, lifetime_std, gamma_mean, gamma_std)), delimiter='\t', header='Lifetime_mean (s)\tLifetime_std (s)\tGamma_mean (Hz)\tGamma_std (Hz)')
np.savetxt(f'vdoses_{num_traj}_traj_{mdlen}_ps.txt', np.hstack((freqs[0].reshape(-1, 1), np.array(VDOSs).T)), delimiter='\t', header='VDOSs')
np.savetxt(f'projected_vdos_{num_traj}_traj_{mdlen}_ps.txt', np.concatenate((freqs[0].reshape(-1,1), VDOS_mean.reshape(-1,1), VDOS_std.reshape(-1,1)), axis=1), delimiter='\t', header='Frequency (Hz)\tVDOS_mean\tVDOS_std')
np.savetxt(f'projected_vacf_{num_traj}_traj_{mdlen}_ps.txt', np.concatenate((time.reshape(-1,1), VAF_mean.reshape(-1,1), VAF_std.reshape(-1,1)), axis=1), delimiter='\t', header='Time (fs)\tVAF_mean\tVAF_std')

# sio.savemat('Data_VDOS_1x1x1_window'+str(int(time_window/1e-12))+'ps.mat', {'VDOSs':VDOSs, 'VAFs':VAFs, 'freqs':freqs, 'timees':timees, 'VDOS_mean':VDOS_mean, 'VDOS_std':VDOS_std,'VAF_mean':VAF_mean, 'VAF_std':VAF_std})

# plt.figure()
# plt.errorbar(freqs[0], VDOS_mean, yerr=VDOS_std, fmt='o-', label='VDOS')
# plt.xlabel('Frequency (THz)')
# plt.ylabel('VDOS')
# plt.legend()
# plt.savefig('VDOS.png', dpi=150)
# plt.close()

# time = np.arange(0, 20001)
# print(len(time), len(VAF_mean))

# plt.figure()
# plt.errorbar(time, VAF_mean, yerr=VAF_std, fmt='o-', label='VAF')
# plt.xlabel('Time (fs)')
# plt.ylabel('VAF')
# plt.legend()
# plt.savefig('VAF.png', dpi=150)
# plt.close()


# fig_vdos = go.Figure()

# fig_vdos.add_trace(go.Scatter(
#     x=freqs[0], y=VDOS_mean,
#     error_y=dict(type='data', array=VDOS_std),
#     mode='lines+markers',
#     name='VDOS'
# ))

# fig_vdos.update_layout(
#     title="VDOS Plot",
#     xaxis_title="Frequency (THz)",
#     yaxis_title="VDOS"
# )

# # Save as HTML
# fig_vdos.write_html("VDOS_plot.html")


# # VAF Plot with Plotly
# fig_vaf = go.Figure()

# fig_vaf.add_trace(go.Scatter(
#     x=time, y=VAF_mean,
#     error_y=dict(type='data', array=VAF_std),
#     mode='lines+markers',
#     name='VAF'
# ))

# fig_vaf.update_layout(
#     title="VAF Plot",
#     xaxis_title="Time (fs)",
#     yaxis_title="VAF"
# )

# # Save as HTML
# fig_vaf.write_html("VAF_plot.html")
