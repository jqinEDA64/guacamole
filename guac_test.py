#################################
# guac_test.py
#
# Testbench for guac_* stuff
#
#################################

from guac_math    import *
from guac_classes import *
import numpy as np
import matplotlib.pyplot as plt

def test_Lorentzian_Kernel(Gamma, dE):
    k = getKernel(Gamma, dE, "Lorentzian")
    print("Sum of Lorentzian kernel = " + str(np.sum(k)))
    fig, ax = plt.subplots()
    ax.plot(k)
    saveAndShow(fig, "test_out/Lorentzian")
    
    
def test_Gaussian_Kernel(Gamma, dE):
    k = getKernel(Gamma, dE, "Gaussian")
    print("Sum of Gaussian kernel = " + str(np.sum(k)))
    fig, ax = plt.subplots()
    ax.plot(k)
    saveAndShow(fig, "test_out/Gaussian")
    
    
def test_NonUnif_Conv_0(Gamma, dE):
    
    # Create array with "bandgap" in the middle
    # The middle of the bandgap is defined as E = 0
    E_band = 10   # [eV]
    E_vals = np.arange(-E_band, E_band, dE)
    
    # Create "density of states" with bandgap
    E_gap = 2.5
    D_vals = np.asarray([1 if abs(E) > E_gap else 0 for E in E_vals])
    
    # Create uniform array of Gamma values
    G_vals = Gamma*np.ones(E_vals.shape[0])
    
    # Compute non-uniform convolution
    D_out, E_out = getNonUnifConv(D_vals, E_vals, G_vals, "Lorentzian")
    
    # Ensure that input and output satisfy the "Sum Rule"
    # preserving the total number of states
    S_in = np.sum(D_vals)
    S_out= np.sum(D_out )
    print("Sum Rule Check:")
    print("  Sum of inputs = " + str(S_in))
    print("  Sum of outputs= " + str(S_out))
    print("  Relative error= " + str((S_out-S_in)/S_in*100) + " [%]")
    
    # Plot input and output on the same graph
    fig, ax = plt.subplots()
    ax.plot(E_vals, D_vals, label = "Original")
    ax.plot(E_out , D_out , label = "Convoluted")
    ax.legend()
    ax.set_title("Test of nonuniform convolution")
    saveAndShow(fig, "test_out/NonUnif_Conv_0")

#################################
# TEST BENCH
#################################

#test_Lorentzian_Kernel(0.5, 0.1)
#test_Gaussian_Kernel(5, 0.1)

test_NonUnif_Conv_0(2, 0.1)


