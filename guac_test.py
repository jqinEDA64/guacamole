#################################
# guac_test.py
#
# Testbench for guac_* stuff
#################################

from guac_math    import *
from guac_classes import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


def test_NonUnif_Conv_CNT(Gamma):

    # Load CNT density of states from disk
    E_vals, D_vals = loadCNT_11_0_DoSFromFile()
    
    # Create uniform array of Gamma values
    G_vals = Gamma*np.ones(E_vals.shape[0])
    
    # Compute non-uniform convolution
    #D_out, E_out = getNonUnifConv(D_vals, E_vals, G_vals, "Lorentzian")
    D_out, E_out = getNonUnifConv(D_vals, E_vals, G_vals, "Gaussian")
    
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
    ax.plot(E_vals, D_vals, label = "Original DoS", color = "blue")
    ax.plot(E_out , D_out , label = "Convoluted DoS", color = "red")
    ax.legend()
    ax.set_title("Test of nonuniform convolution")
    saveAndShow(fig, "test_out/NonUnif_Conv_0")


def test_CNL_CNT(Gamma_V, Gamma_C):
    
    # Load CNT density of states from disk
    E_vals, D_vals = loadCNT_11_0_DoSFromFile()
    
    # Get the DoS of only valence and only conduction bands
    # NOTE: Assumes that E = 0 lies inside the bandgap!!!
    D_V_vals = np.where(E_vals <= 0, D_vals, 0)
    D_C_vals = np.where(E_vals >  0, D_vals, 0)
    
    # Create Gamma values which may be different in the
    # valence and conduction bands
    G_vals = np.asarray([Gamma_V if E < 0 else Gamma_C for E in E_vals])
    
    # FIRST CASE: constant Gamma
    #G_vals = 0.03*np.ones(E_vals.shape[0])
    
    # SECOND CASE: semimetal broadening
    #E_Dirac = 0.5
    #G_vals = 0.03*np.asarray([min(1, np.abs(E-E_Dirac)) for E in E_vals])
    
    # Compute non-uniform convolution
    D_out  , E_out = getNonUnifConv(D_vals  , E_vals, G_vals, "Lorentzian")
    D_C_out, E_out = getNonUnifConv(D_C_vals, E_vals, G_vals, "Lorentzian")
    D_V_out, E_out = getNonUnifConv(D_V_vals, E_vals, G_vals, "Lorentzian")
    #D_out, E_out = getNonUnifConv(D_vals, E_vals, G_vals, "Gaussian")
    
    # Ensure that input and output satisfy the "Sum Rule"
    # preserving the total number of states
    S_in = np.sum(D_vals)
    S_out= np.sum(D_out )
    
    # Compute the new CNL
    CNL = getCNL(D_vals, E_vals, 0, D_out, E_out)
    
    # Plot input and output on the same graph
    fig, ax = plt.subplots(2, sharex = True)  # Ensure the plots share a common x-axis
    fig.suptitle("MIGS and CNL of CNT")
    
    ax[0].plot(E_vals, D_vals, label = "Original DoS",   color = "black")
    ax[0].plot(E_out , D_out , label = "Convoluted DoS", color = "purple")
    ax[0].fill_between(E_out, 0, D_V_out, alpha = 0.5, color = "blue", label = "Valence band states")
    ax[0].fill_between(E_out, 0, D_C_out, alpha = 0.5, color = "red" , label = "Conduction band states")
    ax[0].set_title("Density of states")
    ax[0].set_ylabel("DoS [eV$^{-1}$ m$^{-1}$]")
    CNL_label = "CNL = " + str(round(float(CNL), 2)) + " [eV]"
    ax[0].axvline(x = CNL, label = CNL_label, linestyle = "dashed", color = "black")
    ax[0].legend(bbox_to_anchor=(1.05, 1.0), loc = "upper left")
    
    # Also plot Gamma on this graph
    ax[1].plot(E_vals, G_vals, color = "red")
    ax[1].set_title("Metal-semiconductor coupling")
    ax[1].set_xlabel("Energy [eV]")
    ax[1].set_ylabel("$\\Gamma(E)$ [eV]")
    plt.tight_layout()
    saveAndShow(fig, "test_out/CNL_CNT")


def test_resample():
    E_in , D_in  = loadDoSFromFile("raw_data/DoS_CNT_11_0_clean.dat")
    E_out = doResample(E_in, 4)
    D_out = doResample(D_in, 4)
    
    plt.plot(E_in , D_in , color = "blue")
    plt.plot(E_out, D_out, color = "red" )
    plt.title("Test of real-space resampling")
    plt.show()

#################################
# TEST BENCH
#################################

#test_Lorentzian_Kernel(3, 1)
#test_Gaussian_Kernel(5, 1)

#test_NonUnif_Conv_0(2, 0.1)
#test_NonUnif_Conv_CNT(0.03)

test_CNL_CNT(0.04, 0.04)
#test_load_DoS()

#test_resample()

