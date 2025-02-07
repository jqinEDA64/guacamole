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


####################################
# KERNELS AND KERNEL OPS
####################################


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
    return (int)(33*Gamma/dE)

# Returns the half-size of the
# Gaussian kernel of standard deviation Sigma,
# up to 3*Sigma of the integrated content.
#
# If the length of the 1D kernel is 
# 2N+1, this function returns N+1.
#
# Inputs:
# - dE    : grid discretization (float)
# - Sigma : Gaussian st. dev. (float)
def __getGaussianWidth(Sigma, dE):
    return (int)(3*Sigma/dE)

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
# st. dev. Sigma. 
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
        print("ERROR: kernel type " + kerneltype + " not recognized")
        quit()
        return 0
    
    
# Main function of this section.
# Returns appropriate kernel (of 
# automatically-inferred size), given
# the kernel parameter and grid spacing
def getKernel(Param, dE, kerneltype = "Lorentzian"):
    if   kerneltype == "Lorentzian" :
        return __getLorentzianKernel(Param, dE)
    elif kerneltype == "Gaussian" :
        return __getGaussianKernel  (Param, dE)
    else :
        print("ERROR: kernel type " + kerneltype + " not recognized")
        quit()
        return 0


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
        print("ERROR: Array accumulation size mismatch")
        quit()
    if (i < 0) or (i+N1 > N2) :
        print("ERROR: Array accumulation out-of-bounds")
        quit()
    
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
#              that dE = E_in[i+1]-E_in[i].
# - arr_param: Array of kernel parameters. Must
#              have same size as arr_in.
# - kern_type: Either "Lorentzian" or "Gaussian" (currently)
#
# Output:
# - arr_out  : Output array (convoluted version
#              of arr_in). Is larger than
#              arr_in due to nonuniform, nonperiodic
#              convolution.
# - E_out    : Output array of energies (equivalent of E_in).
#              Is larger than E_in due to nonuniform convolution.
def getNonUnifConv(arr_in, E_in, arr_param, kern_type = "Lorentzian"):
    
    N_in = arr_in.shape[0]
    if N_in != arr_param.shape[0] :
        print("ERROR: arr_in and arr_param must have same shape")
    if N_in != E_in.shape[0] :
        print("ERROR: arr_in and E_in must have same shape")
        
    # Implicit energy spacing
    dE = E_in[1] - E_in[0]    
    
    # Compute the appropriate shape of arr_out
    min_index = 0  # min_index < 0
    max_index = 0  # max_index > N_in
    for i in range(N_in) :
        N = getKernelWidth(arr_param[i], dE, kern_type)
        min_index = min(min_index, i-N)
        max_index = max(max_index, i+N)
        
    # Create arr_out
    N_out = max_index-min_index+1
    arr_out = np.zeros(N_out)
    
    # Accumulate non-uniform convolution values into output
    for i in range(N_in) :
        __accumIntoArr(arr_in[i]*getKernel(arr_param[i], dE, kern_type), \
                       arr_out,                                          \
                       i - min_index - getKernelWidth(arr_param[i], dE, kern_type))
    
    # Create E_out
    E0    = E_in[0] + min_index*dE  # NOTE: min_index < 0
    E_out = np.asarray([E0 + i*dE for i in range(N_out)])
    
    return arr_out, E_out


####################################
# PLOTTING FUNCTIONALITY
####################################

# Saves the figure from matplotlib and
# displays it with native Linux program.
#
# Example of usage:
#      fig, ax = plt.subplots()
#      ax.plot(x, y)
#      ax.set_xlabel("x-axis")
#      ax.legend()
#
# ... etc.
def saveAndShow(fig, fig_path):
    png_path = fig_path + ".png"
    fig.savefig(png_path, dpi=500)
    os.system("gio open " + png_path + " &")
    #os.system("imgcat " + png_path + " &")
