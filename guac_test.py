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
    
    
def test_NonUnif_Conv(Gamma, dE):
    
    # Create array with "bandgap" in the middle
    # The middle of the bandgap is defined as E = 0
    E_band = 5   # [eV]
    E_vals = np.arange(-E_band, E_band, dE)
    
    # Create "density of states" with bandgap
    D_vals = np.asarray([1 if abs(E) > 1.5 else 0 for E in E_vals])
    
    # Create uniform array of Gamma values
    G_vals = Gamma*np.ones(E_vals.shape[0])
    
    # Compute non-uniform convolution
    D_out, E_out = getNonUnifConv(D_vals, E_vals, G_vals, "Gaussian")
    
    # Plot input and output on the same graph
    fig, ax = plt.subplots()
    ax.plot(E_vals, D_vals, label = "Original")
    ax.plot(E_out , D_out , label = "Convoluted")
    ax.legend()
    ax.title("Test of nonuniform convolution")
    saveAndShow(fig, "test_out/NonUnif_Conv")

#################################
# TEST BENCH
#################################

#test_Lorentzian_Kernel(5, 0.1)
#test_Gaussian_Kernel(5, 0.1)

test_NonUnif_Conv(2, 0.1)


