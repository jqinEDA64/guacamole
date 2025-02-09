####################################
# guac_math.py
#
# This file contains all the math
# functions needed for contact
# resistance modeling
####################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import os


# Custom definition of "large number"
FLOAT_MAX = 1e30


####################################
# ERROR HANDLING
####################################


def err_out(msg) :
    print("ERROR: " + msg)
    quit()


####################################
# KERNELS AND KERNEL OPS
####################################


# Get maximum grid spacing required to
# have a well-resolved kernel.
#
# Inputs:
# - G : G is the one-dimensional kernel
#       parameter (either Lorentzian width
#       or Gaussian standard deviation).
#       It can be passed either as a scalar
#       or as a NumPy array. All values must
#       be positive.
#
# Outputs:
# - dE: Maximum grid spacing required to
#       resolve all the relevant kernels.
def getMinResolution(G) :
    # TODO jqin: reactivate this function once
    #            resampling is sorted out
    return 1e5
    if   isinstance(G, (int, float)):
        if G == 0 :  # No smoothing; just don't apply any kernel
            return FLOAT_MAX
        return G/4.0
    elif isinstance(G, np.ndarray):
        return np.min(G)/4.0
    else:
        err_out("Input data must be a scalar or a NumPy array.")
        return 0
        
        
# Returns the current energy resolution, given
# the energy values. The energy values are
# assumed to be evenly spaced.
#
# Input:
# - E_vals: Current energy values [eV].
#
# Output: 
# - dE    : Resolution of 1D energy grid [eV].
def getEnergyResolution(E_vals):

    dE_vals = np.diff(E_vals)

    # Check that E_vals is strictly increasing
    if not np.all(dE_vals > 0) :
        err_out("E_vals is not strictly increasing")
        
    # Check that E_vals is evenly spaced (to 1% error)
    if np.ptp(dE_vals) > 0.01*dE_vals[0] :
        err_out("E_vals is not evenly spaced")
        
    # Return spacing
    dE = np.median(dE_vals)
    return dE


# Returns the half-size of the 
# Lorentzian kernel of width Gamma,
# up to 99% of the integrated
# content. (Could adjust the CDF to
# desired accuracy.)

# If the length of the 1D kernel is
# 2N+1, this function returns N+1.
#
# Inputs:
# - dE    : grid discretization (float)
# - Gamma : Lorentzian width    (float)
def __getLorentzianWidth(Gamma, dE):
    if dE > getMinResolution(Gamma) :
        err_out("Insufficient resolution for Lorentzian kernel")
    return (int)(33*Gamma/dE)


# Returns the half-size of the
# Gaussian kernel of standard deviation Sigma,
# up to 3*Sigma (99.7%) of the integrated content.
#
# If the length of the 1D kernel is 
# 2N+1, this function returns N+1.
#
# Inputs:
# - dE    : grid discretization (float)
# - Sigma : Gaussian st. dev. (float)
def __getGaussianWidth(Sigma, dE):
    if dE > getMinResolution(Sigma) :
        err_out("Insufficient resolution for Gaussian kernel")
    return (int)(3*Sigma/dE)+1


# Returns the Lorentzian kernel of 
# width Gamma. 
#
# The length of the 1D kernel is
# guaranteed to be odd, such that
# the kernel has a well-defined 
# center position. 
#
# Inputs:
# - dE    : grid discretization
# - Gamma : Lorentzian width
def __getLorentzianKernel(Gamma, dE):
    N = __getLorentzianWidth(Gamma, dE)
    
    # Create the kernel. The middle element
    # has index = N and value = 0.
    # The total length is 2N + 1.
    out = np.arange(-N, N+1)
    
    scale = dE/Gamma
    out = scipy.stats.cauchy.pdf(out*scale)
    return out / np.sum(out)  # Normalize
    
    
# Returns the Gaussian kernel of 
# standard deviation = Sigma.
#
# The length of the 1D kernel is
# guaranteed to be odd, such that
# the kernel has a well-defined 
# center position. 
#
# Inputs:
# - dE    : grid discretization
# - Sigma : Gaussian st. dev.
def __getGaussianKernel(Sigma, dE):
    N = __getGaussianWidth(Sigma, dE)
    
    # Create the kernel. The middle element
    # has index = N and value = 0.
    # The total length is 2N + 1.
    out = np.arange(-N, N+1)
    
    scale = dE/Sigma
    out = scipy.stats.norm.pdf(out*scale)
    return out / np.sum(out)  # Normalize
    

# Main function of this section.
# Returns appropriate kernel size, given
# the kernel parameter and grid spacing
def getKernelWidth(Param, dE, kerneltype = "Lorentzian"):
    if   kerneltype == "Lorentzian" :
        return __getLorentzianWidth(Param, dE)
    elif kerneltype == "Gaussian" :
        return __getGaussianWidth  (Param, dE)
    else :
        err_out("Kernel type " + kerneltype + " not recognized")
    
    
# Main function of this section.
# Returns appropriate kernel (of 
# automatically-inferred size), given
# the kernel parameter and grid spacing
def getKernel(Param, dE, kerneltype = "Lorentzian"):

    # Return identity if Param = 0
    # NOTE: requires kernels of ANY type to reduce
    #       to a delta-function when Param = 0
    if   Param == 0 :
        return np.array([1])

    if   kerneltype == "Lorentzian" :
        return __getLorentzianKernel(Param, dE)
    elif kerneltype == "Gaussian" :
        return __getGaussianKernel  (Param, dE)
    else :
        err_out("Kernel type " + kerneltype + " not recognized")
        return 0


# Resampling function to increase the 
# energy resolution, if needed. Since the
# function is assumed non-periodic, cannot
# use Fourier methods. Instead, use 
# real-space interpolation methods. In this 
# case, it is not necessary or helpful for
# the resampling factor "f" to be an integer
# or inverse of an integer. 
#
# Inputs: 
# - arr_in: Input array to be resampled.
# - f     : Resampling factor (usually > 1)
# 
# Output:
# - arr_out: Result of resampling.
def doResample(arr_in, f):

    # TODO jqin: Perhaps, apply appropriate amount of 
    #            Gaussian pre-smoothing to avoid aliasing
    #            errors and resample correctly

    # Create cubic spline from input data
    N_in   = arr_in.shape[0]
    x_in   = np.arange(N_in)
    spline = scipy.interpolate.CubicSpline(x_in, arr_in)
    
    # Define resampling points
    N_out = (int)(N_in*f)
    x_out = np.linspace(0, N_in-1, N_out)
    
    # Compute the resampled values by spline interp
    arr_out = [spline(x) for x in x_out]
    return np.asarray(arr_out)
    
    # Try FFT resampling
    # Looks horrible for CNT since lots of aliasing
    #return scipy.signal.resample(arr_in, N_out)
    

####################################
# CONVOLUTIONS AND RELATED OPS
####################################


# Accumulates a small array (from kernel)
# into a larger array (for entire convolution)
# with index checking.
#
# Inputs:
# - arr_1 : Small array (length N1)
# - arr_2 : Large array (length N2 > N1)
# - i     : Index of arr_2 to accumulate arr_1 into arr_2.
def __accumIntoArr(arr_1, arr_2, i):
    N1 = arr_1.shape[0]
    N2 = arr_2.shape[0]
    if N2 < N1 :
        err_out("Array accumulation size mismatch")
    if (i < 0) or (i+N1 > N2) :
        err_out("Array accumulation out-of-bounds")
    
    # Perform accumulation op (with NumPy slicing)
    arr_2[i:i+N1] += arr_1

    
# Compute the nonuniform convolution of
# a single-parameter kernel with an input
# array. 
#
# The size of the output array will be 
# larger than the size of the input array.
# This difference in size depends on the
# kernel sizes. 
#
# Inputs:
# - arr_in   : Input array of values to be 
#              convoluted with nonuniform kernel.
# - E_in     : Energies of the input array (same as
#              x-coordinates, essentially). It is assumed
#              that dE = E_in[i+1]-E_in[i], such that the
#              entries of E_in are evenly spaced.
# - arr_param: Array of kernel parameters. Must
#              have same size as arr_in. Can also be a constant
#              in which case the same value is used everywhere.
# - kern_type: Either "Lorentzian" or "Gaussian" (currently)
#
# Outputs:
# - arr_out  : Output array (convoluted version
#              of arr_in). Is larger than
#              arr_in due to nonuniform, nonperiodic
#              convolution.
# - E_out    : Output array of energies (equivalent of E_in).
#              Is larger than E_in due to nonuniform convolution.
def getNonUnifConv(arr_in, E_in, arr_param, kern_type = "Lorentzian"):
    
    N_in = arr_in.shape[0]
    if N_in != E_in.shape[0] :
        err_out("arr_in and E_in must have same shape")
        
    # Implicit energy spacing 
    dE = E_in[1] - E_in[0]
    
    # Detect the correct parameter type and create
    # its array if needed
    param = None
    if   isinstance(arr_param, float):
        param = arr_param*np.ones(N_in)
    elif isinstance(arr_param, np.ndarray):
        if N_in != arr_param.shape[0] :
            err_out("arr_in and arr_param must have same shape")
        param = arr_param
    
    # Compute the appropriate shape of arr_out
    min_index = 0  # min_index < 0
    max_index = 0  # max_index > N_in
    for i in range(N_in) :
        N = getKernelWidth(param[i], dE, kern_type)
        min_index = min(min_index, i-N)
        max_index = max(max_index, i+N)
        
    # Create arr_out
    N_out = max_index-min_index+1
    arr_out = np.zeros(N_out)
    
    # Accumulate non-uniform convolution values into output
    for i in range(N_in) :
        __accumIntoArr(arr_in[i]*getKernel(param[i], dE, kern_type), \
                       arr_out,                                      \
                       i - min_index - getKernelWidth(param[i], dE, kern_type))
    
    # Create E_out
    E0    = E_in[0] + min_index*dE  # NOTE: min_index < 0
    E_out = np.asarray([E0 + i*dE for i in range(N_out)])
    
    return arr_out, E_out


# Charge neutrality level computation.
# Computes the energy level at which the output has the
# same amount of charge as the input.
#
# Inputs:
# - D_in   : Density of states of original system.
# - E_in   : Energy levels of states of original system.
# - CNL_in : Input charge neutrality level. For semiconductors
#            CNL_in is not unique (it may be anywhere in the bandgap).
#            CNL_in does not have to be one of the entries of E_in;
#            we use interpolation to compute the total charge.
# - D_out  : Density of states of modified system.
# - E_out  : Energy levels of states of modified system.
#
# Output:
# - CNL_out: Output charge neutrality level. This better be unique!
#            If not unique, (i.e., MIGS density is zero), then 
#            presumably interpolation will fail.
def getCNL(D_in, E_in, CNL_in, D_out, E_out):
    
    # Sanity check that total number of states is preserved
    Q_tot_in  = np.sum(D_in )
    Q_tot_out = np.sum(D_out)
    Q_tot_err = (Q_tot_out-Q_tot_in)/Q_tot_in*100
    if abs(Q_tot_err) > 0.5 :  # Accept rel. err. less than 0.5%
        err_out("Inputs to CNL computation do not satisfy Sum Rule")
    
    # Compute the functions Q_in(E_in) and Q_out(E_out),
    # the total charges at zero temperature as a function
    # of the Fermi level E_{in, out}.
    Q_in  = scipy.integrate.cumulative_trapezoid(D_in , x = E_in , initial = 0)
    Q_out = scipy.integrate.cumulative_trapezoid(D_out, x = E_out, initial = 0)
    
    # Compute Q(CNL_in), the total amount of electronic charge
    # in the original ("in") material.
    Q = scipy.interpolate.CubicSpline(E_in, Q_in)(CNL_in)
    
    # Compute CNL_out, the charge neutrality level of the output
    # density of states. Again use spline interpolation but
    # "invert" it simply by using E(Q) instead of Q(E).
    CNL_out = scipy.interpolate.CubicSpline(Q_out, E_out)(Q)
    
    return CNL_out


####################################
# FILE I/O
####################################

# Saves the figure from matplotlib and
# displays it with Python viewer.
#
# Example of usage:
#      fig, ax = plt.subplots()
#      ax.plot(x, y)
#      ax.set_xlabel("x-axis")
#      ax.legend()
#      saveAndShow(fig, "img.png")
#
# Requires some packages. I think
# requires Matplotlib, Tkinter, and PyQt6
# as described here:
# # https://stackoverflow.com/questions/77507580/userwarning-figurecanvasagg-is-non-interactive-and-thus-cannot-be-shown-plt-sh
#
def saveAndShow(fig, fig_path):
    png_path = fig_path + ".png"
    fig.savefig(png_path, dpi=500)
    plt.show()
    

# Load a density of states from
# file and return the energy and DoS
# values as NumPy arrays.
#
# The DoS gets sorted and the user can
# also specify a new energy to be defined
# as the "0" energy reference.
#
# Inputs:
# - file_path: Path to CSV file for DoS.
# - [E_zero] : Energy (in the original scale)
#              which is set to zero (in the new scale).
#
# Outputs:
# - E_vals   : Energy values [eV]
# - D_vals   : Density of states [unknown units; must be checked manually]
def loadDoSFromFile(file_path, E_zero = None):

    # Read from disk
    df = pd.read_csv(file_path)
    
    # Check for correct column headers
    if not "E"   in df.columns :
        err_out("E not found in DoS CSV file")
    if not "DOS" in df.columns :
        err_out("DOS not found in DoS CSV file")
        
    # Sort based on E values
    df = df.sort_values(by = "E") 
    
    E_vals = df["E"]
    D_vals = df["DOS"]

    # Adjust the energy reference if needed
    if E_zero is not None :
        E_vals -= E_zero

    # Get median energy spacing
    dE = getEnergyResolution(E_vals)
        
    # Force E_vals to be EXACTLY evenly spaced
    # Center the extrapolation around E = 0
    dE_abs = np.abs(np.diff(E_vals)).tolist()
    i0 = dE_abs.index(min(dE_abs))
    E0 = E_vals[i0]
    E_vals = np.asarray([E0 + dE*(i-i0) for i in range(E_vals.shape[0])])
    
    return E_vals, D_vals

    
####################################
# CUSTOM DATASETS (STORED IN THIS
# PROJECT'S CODEBASE)
####################################


# Loads the CNT (11,0) DoS from file (stored in guacamole codebase).
# Applies a small amount of Gaussian smoothing because
# otherwise the van Hove singularities in the 1D DoS
# will lead to heavy aliasing and thus accuracy issues
# in CNL and related computations.
#
# Outputs: 
# - E_vals : Energy values [eV] 
# - D_vals : Density of states [eV^(-1) nm^(-2)]
def loadCNT_11_0_DoSFromFile():
    
    # Read CNT density of states from disk
    E_vals, D_vals = loadDoSFromFile("raw_data/DoS_CNT_11_0_clean.dat")
    
    # Reduce aliasing error by a small amount of smoothing
    D_vals, E_vals = getNonUnifConv(D_vals, E_vals, 0.01, "Gaussian")
    
    # Estimate CNT diameter (see raw_data/README)
    d_CNT = 0.8 # [nm]
    
    # Convert D_vals from [eV^(-1) m^(-1)] to [eV^(-1) nm^(-2)]
    D_vals = (1e-9 / (np.pi*d_CNT)) * D_vals
    
    return E_vals, D_vals
    




