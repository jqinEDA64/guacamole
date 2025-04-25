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

# Override kernel resolution checking
# Needed for pre-smoothing of CNT 
# density of states, for example,
# since the pre-smoothing is applied 
# even before upsampling in the energy
# domain.
#
# Must be reset to "False" after overriding.
OVERRIDE_ENERGY_RESOLUTION_CHECK = True


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

    if  OVERRIDE_ENERGY_RESOLUTION_CHECK == True :
        return FLOAT_MAX

    if   isinstance(G, (int, float)):
        if G == 0 :  # No smoothing; just don't apply any kernel
            return FLOAT_MAX
        return G/4.0
    elif isinstance(G, np.ndarray):
        return np.min(G[G > 0])/4.0  # Min value of positive G
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
# - Gamma : Lorentzian width    (float) [eV]
# - E_vals: Input energy values [eV]
# - dE    : grid discretization (float) [eV]
def __getLorentzianWidth(Gamma, E_vals, dE):
    return int((np.max(E_vals)-np.min(E_vals))/dE)


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

# Returns the half-size of the
# logistic kernel of width kT.
#
# If the length of the 1D kernel is 
# 2N+1, this function returns N+1.
#
# Inputs:
# - dE    : grid discretization (float)
# - kT    : Distribution width  (float)
def __getLogisticWidth(kT, dE):
    if dE > getMinResolution(kT) :
        err_out("Insufficient resolution for Gaussian kernel")
    return (int)(10*kT/dE)+1

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
# - E_vals: Energy values
# - Gamma : Lorentzian width
def __getLorentzianKernel(Gamma, E_vals, dE):
    N = __getLorentzianWidth(Gamma, E_vals, dE)
    
    # Create the kernel. The middle element
    # has index = N and value = 0.
    # The total length is 2N + 1.
    out = np.arange(-N, N+1)
    
    scale = dE/Gamma
    out = scipy.stats.cauchy.pdf(out*scale)

    # TODO jqin: Replace this PDF-based computation
    #            with more accurate CDF computation

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
    
# Returns the Logistic kernel of 
# width = kT
#
# The length of the 1D kernel is
# guaranteed to be odd, such that
# the kernel has a well-defined 
# center position. 
#
# Inputs:
# - dE    : grid discretization
# - kT    : Logistic width (thermal voltage)
def __getLogisticKernel(kT, dE):
    N = __getLogisticWidth(kT, dE)
    
    # Create the kernel. The middle element
    # has index = N and value = 0.
    # The total length is 2N + 1.
    out = np.arange(-N, N+1)
    
    FD_der_vals = scipy.stats.logistic.pdf(out, loc = 0, scale = kT/dE)
    FD_der_vals = FD_der_vals / np.sum(FD_der_vals) # normalize
    return FD_der_vals

# Main function of this section.
# Returns appropriate kernel size, given
# the kernel parameter and grid spacing
def getKernelWidth(Param, E_vals, dE, kerneltype = "Lorentzian"):
    if   kerneltype == "Lorentzian" :
        return __getLorentzianWidth(Param, E_vals, dE)
    elif kerneltype == "Gaussian" :
        return __getGaussianWidth  (Param, dE)
    elif kerneltype == "Logistic" :
        return __getLogisticWidth  (Param, dE)
    else :
        err_out("Kernel type " + kerneltype + " not recognized")
    
    
# Main function of this section.
# Returns appropriate kernel (of 
# automatically-inferred size), given
# the kernel parameter and grid spacing
def getKernel(Param, E_vals, dE, kerneltype = "Lorentzian"):

    # Return identity if Param = 0
    # NOTE: requires kernels of ANY type to reduce
    #       to a delta-function when Param = 0
    if   Param == 0 :
        return np.array([1])

    if   kerneltype == "Lorentzian" :
        return __getLorentzianKernel(Param, E_vals, dE)
    elif kerneltype == "Gaussian" :
        return __getGaussianKernel  (Param, dE)
    elif kerneltype == "Logistic" :
        return __getLogisticKernel  (Param, dE)
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
# - E_in  : Values of the energy at arr_in indices.
# - f     : Resampling factor (usually > 1),
#           or array of energy values at which
#           to perform the resampling
# - interp_type : Either "cubic" or "linear" interpolation
# 
# Output:
# - arr_out: Result of resampling.
def doResample(arr_in, E_in, f, interp_type = "cubic"):

    # Define resampling points
    E_in_min = np.min(E_in)
    E_in_max = np.max(E_in)
    dE = getEnergyResolution(E_in)
    
    E_out = None
    if   isinstance(f, (int, float)):
        E_out = np.arange(E_in_min, E_in_max, dE/f)
    elif isinstance(f, np.ndarray):
        if ( np.min(f) < E_in_min ) or ( np.max(f) > E_in_max ) :
            err_out("Resampling energy array is out-of-range")
        E_out = f
    
    # Compute the resampled values by real-space interpolation.
    # The default choice is "cubic", although "linear" interpolation
    # can avoid overshoot errors of discontinuous functions (if we
    # don't want to go through the trouble of prefiltering them
    # before the resampling).
    arr_out = None
    if   interp_type == "cubic" :
        spline = scipy.interpolate.CubicSpline(E_in, arr_in)
        arr_out = np.asarray([spline(E) for E in E_out])
    elif interp_type == "linear" :
        arr_out = np.interp(E_out, E_in, arr_in)

    return arr_out
    

####################################
# CONVOLUTIONS AND RELATED OPS
####################################

    
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
    dE = getEnergyResolution(E_in)
    
    # Detect the correct parameter type and create
    # its array if needed
    param = None
    if   isinstance(arr_param, float):
        param = arr_param*np.ones(N_in)
    elif isinstance(arr_param, np.ndarray):
        if N_in != arr_param.shape[0] :
            err_out("arr_in and arr_param must have same shape")
        param = arr_param
    else :
        raise Exception("arr_param must be either float or NumPy array")
    
    # Compute the appropriate shape of arr_out
    min_index = 0  # min_index < 0
    max_index = 0  # max_index > N_in
    for i in range(N_in) :
        N = getKernelWidth(param[i], E_in, dE, kern_type)
        min_index = min(min_index, i-N)
        max_index = max(max_index, i+N)
        
    # Create arr_out
    N_out = max_index-min_index+1
    arr_out = np.zeros(N_out)
    
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
    
    # Accumulate non-uniform convolution values into output
    for i in range(N_in) :
        __accumIntoArr(arr_in[i]*getKernel(param[i], E_in, dE, kern_type), \
                       arr_out,                                      \
                       i - min_index - getKernelWidth(param[i], E_in, dE, kern_type))
    
    # Create E_out
    E0    = E_in[0] + min_index*dE  # NOTE: min_index < 0
    E_out = np.asarray([E0 + i*dE for i in range(N_out)])
    
    return arr_out, E_out


# Given a function of energy X(E), uses
# density of states D(E) to map X to new
# energies based on nonuniform convolution.
#
# Essentially, weighted averaged of the new
# X(E) after E -> E' mapping due to spectral
# broadening.
#
# Inputs:
# - X_in     : Values of the input function
# - D_in     : Density of states
# - E_in     : Input energies
# - arr_param: Parameters for the nonuniform convolution (may be
#              either NumPy array or float)
# - kern_type: Convolution type (either "Lorentzian" or "Gaussian")
#
# Outputs:
# - X_out    : Mapped values of X(E)
# - E_out    : Corresponding output energies
def getNonUnifMap(X_in, D_in, E_in, arr_param, kern_type = "Lorentzian"):
    
    XD_out, E_out = getNonUnifConv(np.multiply(X_in, D_in), E_in, arr_param, kern_type)
    D_out , E_out = getNonUnifConv(D_in                   , E_in, arr_param, kern_type)
    
    eps = 1e-15  # Regularization
    X_out = np.divide(XD_out, D_out + eps)
    return X_out, E_out


# Given two functions f1(E) and f2(E),
# returns f(E) = min(f1(E), f2(E)).
# 
# Only returns f(E) in the region of E
# where both f1(E) and f2(E) are known.
#
# Inputs:
# - f1_vals: f1(E) values
# - f2_vals: f2(E) values
# - E1_vals: Energy values for f1.
# - E2_vals: Energy values for f2. Does not
#            necessarily have to match E1_vals.
#
# Output:
# - f_vals: min(f1, f2) for all E in E_vals.
# - E_vals: Values of energy [eV]
def getMinFunction(f1_vals, E1_vals, f2_vals, E2_vals):
    
    # Compute E_vals
    dE     = min(getEnergyResolution(E1_vals), getEnergyResolution(E2_vals))
    E_min  = max(np.min(E1_vals), np.min(E2_vals))
    E_max  = min(np.max(E1_vals), np.max(E2_vals))
    E_vals = np.arange(E_min, E_max, dE)
    
    # Compute both values and return minimum
    f1_interp_vals = doResample(f1_vals, E1_vals, E_vals, interp_type = "linear")
    f2_interp_vals = doResample(f2_vals, E2_vals, E_vals, interp_type = "linear")
    fm_interp_vals = np.minimum(f1_interp_vals, f2_interp_vals)
    
    return fm_interp_vals, E_vals


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

# This function loads DOS from file but does not do any sorting or energy shifting.
#
# Inputs:
# - file_path: Path to CSV file for DoS.
#
# Outputs:
# - E_vals   : Energy values [eV]
# - D_vals   : Density of states [unknown units; must be checked manually]
def loadDoSFromFile_nosort(file_path) :

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

    return E_vals, D_vals
    

####################################
# CUSTOM DATASETS (STORED IN THIS
# PROJECT'S CODEBASE)
####################################


# Loads the CNT DoS from file (stored in guacamole codebase).
# Applies a small amount of Gaussian smoothing because
# otherwise the van Hove singularities in the 1D DoS
# will lead to heavy aliasing and thus accuracy issues
# in CNL and related computations.
#
# Inputs:
# - a1     : CNT parameter 1 (int)
# - a2     : CNT parameter 2 (int)
#
# Outputs: 
# - E_vals : Energy values [eV] 
# - D_vals : Density of states [eV^(-1) nm^(-2)]
def loadCNTDoSFromFile(a1, a2):
    
    # Read CNT density of states from disk
    filename = "raw_data/DoS_CNT_" + str(a1) + "_" + str(a2) + "_clean.dat"
    E_vals, D_vals = loadDoSFromFile(filename)
    
    # Reduce aliasing error by a small amount of smoothing
    global OVERRIDE_ENERGY_RESOLUTION_CHECK
    OVERRIDE_ENERGY_RESOLUTION_CHECK = True   # Temporarily override resolution checking
    D_vals, E_vals = getNonUnifConv(D_vals, E_vals, 0.01, "Gaussian")
    OVERRIDE_ENERGY_RESOLUTION_CHECK = False  # Restore default setting
    
    # Estimate CNT diameter (see raw_data/README)
    d_CNT = 1.0 # [nm]
    
    # Convert D_vals from [eV^(-1) m^(-1)] to [eV^(-1) nm^(-2)]
    D_vals = (1e-9 / (np.pi*d_CNT)) * D_vals
    
    return E_vals, D_vals


def loadCNT_11_0_DoSFromFile():
    
    return loadCNTDoSFromFile(11, 0)
    

