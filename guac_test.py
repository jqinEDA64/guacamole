#################################
# guac_test.py
#
# Testbench for guac_* stuff
#################################

from guac_math    import *
from guac_physics import *
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
    
    # Create Gamma values which may be different in the
    # valence and conduction bands
    G_vals = np.asarray([Gamma_V if E < 0 else Gamma_C for E in E_vals])
    
    # Compute upsampling factor required to resolve the
    # Lorentzian kernels (upsampling factor does not have
    # to be an integer); perform upsampling
    f = getEnergyResolution(E_vals) / getMinResolution(G_vals) + 0.1
    D_vals = doResample(D_vals, E_vals, f)
    G_vals = doResample(G_vals, E_vals, f, interp_type = "linear")
    E_vals = doResample(E_vals, E_vals, f)
    
    # Get the DoS of only valence and only conduction bands
    # NOTE: Assumes that E = 0 lies inside the bandgap!!!
    D_V_vals = np.where(E_vals <= 0, D_vals, 0)
    D_C_vals = np.where(E_vals >  0, D_vals, 0)
    
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
    
    ############################################
    # Analytical section
    ############################################
    
    # Estimate MIGS density and CNL from analytical formula
    E_V = -0.45  # [eV]
    E_C =  0.45  # [eV]
    E_MIGS, D_MIGS, CNL_est = getEstMIGSDoS(E_vals, D_vals, G_vals, E_V, E_C)
    
    ############################################
    # End analytical section
    ############################################
    
    # Plot input and output on the same graph
    fig, ax = plt.subplots(2, sharex = True)  # Ensure the plots share a common x-axis
    fig.suptitle("MIGS and CNL of CNT")
    
    ax[0].plot(E_vals, D_vals, label = "Original DoS",   color = "black")
    ax[0].plot(E_out , D_out , label = "Convoluted DoS", color = "purple")
    ax[0].plot(E_MIGS, D_MIGS, label = "Analytical MIGS", color = "green")
    ax[0].fill_between(E_out, 0, D_V_out, alpha = 0.5, color = "blue", label = "Valence band states")
    ax[0].fill_between(E_out, 0, D_C_out, alpha = 0.5, color = "red" , label = "Conduction band states")
    ax[0].set_title("Density of states")
    ax[0].set_ylabel("DoS [eV$^{-1}$ nm$^{-2}$]")
    CNL_label = "CNL = " + str(round(float(CNL), 2)) + " [eV]"
    ax[0].axvline(x = CNL, label = CNL_label, linestyle = "dashed", color = "black")
    CNL_analytic_label = "Analytical CNL = " + str(round(float(CNL_est), 2)) + " [eV]"
    ax[0].axvline(x = CNL_est, label = CNL_analytic_label, linestyle = "dashed", color = "green")
    ax[0].legend(bbox_to_anchor=(1.05, 1.0), loc = "upper left")
    
    # Also plot Gamma on this graph
    ax[1].plot(E_vals, G_vals, color = "red")
    ax[1].set_title("Metal-semiconductor coupling")
    ax[1].set_xlabel("Energy [eV]")
    ax[1].set_ylabel("$\\Gamma(E)$ [eV]")
    plt.tight_layout()
    saveAndShow(fig, "test_out/CNL_CNT")


# This resampling test showed that it is important to
# apply a small amount of low-pass filtering to CNT DoS
# to avoid aliasing due to the discontinuity near van Hove singularities.
#
# The real-space resampling seems more robust than FFT
# and has the bonus of being non-periodic (so valence and conduction band
# states never mix).
def test_resample():

    # Load CNT density of states from disk
    E_in, D_in = loadCNT_11_0_DoSFromFile()
    E_out = doResample(E_in, 4)
    D_out = doResample(D_in, 4)
    
    plt.plot(E_in , D_in , color = "blue", label = "Original")
    plt.plot(E_out, D_out, color = "red" , label = "Resampled (4X)")
    plt.legend()
    plt.title("Test of real-space resampling")
    plt.show()


def test_thermal_expectation():
    
    E_vals = np.linspace(-10, 10, 500)
    D_vals = np.ones(500)
    
    kT = 0.025
    E_F = 0
    D_avg = getThermalExpectation(D_vals, E_vals, kT, E_F)
    print("D_avg = " + str(D_avg))


def test_ballistic_speed():
    
    # Load CNT density of states from disk
    E_in, D_in = loadCNT_11_0_DoSFromFile()
    D_in = doResample(D_in, E_in, 4)
    E_in = doResample(E_in, E_in, 4)
    E_V = -0.45
    E_C = +0.45
    me  = 0.067
    mh  = 0.067
    
    # Compute ballistic transport velocity
    # of carriers in bare CNT
    V_in = getBallisticSpeed(E_in, E_V, E_C, mh, me, dim = 1)
    
    # Compute ballistic transport velocity
    # of carriers in "X", CNT underneath metal
    G = 0.01
    V_out, E_out = getNonUnifMap (V_in, E_in, E_in, G, kern_type = "Lorentzian")
    
    # Compute convolved density of states
    D_out, E_out = getNonUnifConv(D_in, E_in,       G, kern_type = "Lorentzian")
    
    #print((str)(getBallisticSpeed(1, E_V, E_C, mh, me, dim = 1)))
    
    # Plotting
    '''
    plt.plot(E_in , V_in , color = "red" , label = "Original")
    plt.plot(E_out, V_out, color = "blue", label = "Convolved")
    plt.legend()
    plt.xlabel("Energy [eV]")
    plt.ylabel("Transport velocity [m/s]")
    plt.show()
    '''
    
    # TODO jqin: continue working on this and get the well-known
    #            RQ = 6.5 kOhm for single CNT. May need to compute RQ of bare
    #            CNT rather than CNT-under-metal.
    
    # Compute quantum resistance [Ohm . um]
    E_F = 0.4
    kT  = 0.025
    RQ  = getRQ(V_out, D_out, E_out, E_F, kT)
    
    # Convert quantum resistance from [Ohm*um] to [Ohm / CNT]
    d_CNT_um = 0.8e-3           # CNT diameter [um]
    RQ_1  = RQ / (np.pi*d_CNT_um) # Larger CNT -> lower RQ
    print("RQ = " + str(RQ_1) + " [Ohm] for 1 CNT")
    
    # Compute specific contact conductivity
    gC = getGC(G, D_out, E_out, E_F, kT)
    
    # Compute Rc for various contact lengths Lc
    Rsh = 0  # Ballistic transport in CNT
    Lc_vals = np.logspace(0, 2, 50)  # from 10^0 to 10^2
    
    Rc_vals = getR_cf(RQ, gC, Rsh, Lc_vals*1e-3)
    Rc_vals = Rc_vals / (np.pi*d_CNT_um)  # Normalize to 1 CNT
    
    plt.plot(Lc_vals, Rc_vals*1e-3)
    plt.xlabel("Contact length [nm]")
    plt.ylabel("Contact resistance [k$\\Omega$ per CNT]")
    plt.xscale("log")
    plt.show()


# Compare my analytical/numerical results with the
# NEGF study here: 10.1109/LED.2022.3185991 
def test_SK_paper_MIGS():
    
    # Define strength of metal-semiconductor couplings
    G_1 = 3.5e-3
    G_2 = 10e-3
    G_3 = 0.3
    
    # TODO jqin: test again with full DOS (not restricted to +- 3 eV).
    #            is there something funny with logarithmic divergence?
    
    # Load CNT density of states from disk
    E_in, D_in = loadCNTDoSFromFile(11, 0)
    #D_in = D_in[np.abs(E_in) < 3]
    #E_in = E_in[np.abs(E_in) < 3]    
    
    #print("Bandgap pts = " + str(0.8/getEnergyResolution(E_in)))
    
    D_in = doResample(D_in, E_in, 10)
    E_in = doResample(E_in, E_in, 10)
    
    print("Bandgap pts = " + str(0.8/getEnergyResolution(E_in)))
    print("33*G_3/dE = " + str(33*G_3/getEnergyResolution(E_in)))
    #return
    
    ###############################
    # PLOT MIGS DENSITY
    ###############################

    D_in     = 1e14*D_in  # Convert to [eV^(-1) cm^(-2)] instead of [eV^(-1) nm^(-2)]
    D_1, E_1 = getNonUnifConv(D_in, E_in, G_1, "Lorentzian")
    D_2, E_2 = getNonUnifConv(D_in, E_in, G_2, "Lorentzian")
    D_3, E_3 = getNonUnifConv(D_in, E_in, G_3, "Lorentzian")
    
    plt.plot(D_in, E_in, label = "CNT"                , color = "black")
    plt.plot(D_1 , E_1 , label = "$\\Delta = 3.5$ meV", color = "red")
    plt.plot(D_2 , E_2 , label = "$\\Delta = 10$ meV" , color = "blue")
    plt.plot(D_3 , E_3 , label = "$\\Delta = 0.3$ eV" , color = "green")
    plt.xlabel("Density of states [eV$^{-1}$ cm$^{-2}$]")
    plt.ylim(-1, 1)
    plt.xscale("log")
    plt.legend()
    plt.ylabel("Energy [eV]")
    plt.show()
    
    # Compute the Fermi-level pinning factors
    d   = 0.2 # [nm]
    eps = 8
    D_1 = 1e-14*D_1
    D_2 = 1e-14*D_2
    D_3 = 1e-14*D_3
    S_1 = getEstPinFactor(D_1, E_1, d, eps, 0)
    S_2 = getEstPinFactor(D_2, E_2, d, eps, 0)
    S_3 = getEstPinFactor(D_3, E_3, d, eps, 0)
    print("S_1 = " + str(S_1))
    print("S_2 = " + str(S_2))
    print("S_3 = " + str(S_3))
    
    # Schottky barrier physics
    E_FM = -1
    D_in = 1e-14*D_in
    CNL_1   = getCNL(D_in, E_in, 0, D_1, E_1)
    E_Sch_1 = getSchottkyEnergy(D_1, E_1, CNL_1, E_FM, d, eps)
    CNL_2   = getCNL(D_in, E_in, 0, D_2, E_2)
    E_Sch_2 = getSchottkyEnergy(D_2, E_2, CNL_2, E_FM, d, eps)
    CNL_3   = getCNL(D_in, E_in, 0, D_3, E_3)
    E_Sch_3 = getSchottkyEnergy(D_3, E_3, CNL_3, E_FM, d, eps)
    print("E_Sch_1 = " + str(E_Sch_1))
    print("E_Sch_2 = " + str(E_Sch_2))
    print("E_Sch_3 = " + str(E_Sch_3))
    # NOTE: E_V = -0.45 eV.
    
    return
    
    
# Compare my analytical/numerical results with the
# NEGF study here: 10.1109/LED.2022.3185991 
def test_SK_paper_RCLC():
    
    # Define strength of metal-semiconductor couplings
    G_1 = 3.5e-3
    G_2 = 10e-3
    G_3 = 0.3
    
    # Load CNT density of states from disk
    E_in, D_in = loadCNTDoSFromFile(11, 0)
    D_in  = doResample(D_in, E_in, 10)
    E_in  = doResample(E_in, E_in, 10)
    GQ_in = np.asarray([1/6.45 if abs(E) > 0.46 else 0 for E in E_in])
    
    # Compute DOS and conductance/CNT per energy after spectral broadening
    GQ_1, E_1 = getNonUnifConv(GQ_in, E_in, G_1, kern_type = "Lorentzian")
    GQ_2, E_1 = getNonUnifConv(GQ_in, E_in, G_2, kern_type = "Lorentzian")
    GQ_3, E_1 = getNonUnifConv(GQ_in, E_in, G_3, kern_type = "Lorentzian")
    
    D_1, E_1 = getNonUnifConv(D_in, E_in, G_1, "Lorentzian")
    D_2, E_2 = getNonUnifConv(D_in, E_in, G_2, "Lorentzian")
    D_3, E_3 = getNonUnifConv(D_in, E_in, G_3, "Lorentzian")
    
    # Define external parameters
    E_F = -0.55  # [eV]
    kT  = 0.025  # [eV]
    
    # Compute quantum resistance for each situation
    RQ_1 = 1/getThermalExpectation(GQ_1, E_1, kT, E_F, D_vals = None)
    RQ_2 = 1/getThermalExpectation(GQ_2, E_2, kT, E_F, D_vals = None)
    RQ_3 = 1/getThermalExpectation(GQ_3, E_3, kT, E_F, D_vals = None)
    
    # Inner function to compute contact resistance
    # as a function of contact length
    def getRCLC(RQ_in, G_in, D_in, E_in):
    
        # TODO jqin: Map G_in?
    
        # Compute specific contact conductivity
        gC = getGC(G_in, D_in, E_in, E_F, kT)
        
        # Compute Rc for various contact lengths Lc
        Rsh = 0  # Ballistic transport in CNT
        Lc_vals = np.logspace(0, 2.5, 100)
        Rc_vals = getR_cf(RQ_in, gC, Rsh, Lc_vals*1e-3)
        
        return Lc_vals, Rc_vals
       
   
    # Compute contact resistances as a function of contact length 
    Lc_1, Rc_1 = getRCLC(RQ_1, G_1, D_1, E_1)
    Lc_2, Rc_2 = getRCLC(RQ_2, G_2, D_2, E_2)
    Lc_3, Rc_3 = getRCLC(RQ_3, G_3, D_3, E_3)
    
    # Usually plot 2*RC (cf. Franklin paper, Solomon paper).
    # Sheng-Kai plots the resistance of device = 2*RC
    # since 2 contacts
    Rc_1 = 2*Rc_1
    Rc_2 = 2*Rc_2
    Rc_3 = 2*Rc_3
    
    plt.plot(Lc_1, Rc_1, label = "$\\Delta = 3.5$ meV", color = "red")
    plt.plot(Lc_2, Rc_2, label = "$\\Delta = 10$ meV" , color = "blue")
    plt.plot(Lc_3, Rc_3, label = "$\\Delta = 0.3$ eV" , color = "green")
    plt.xlabel("Contact length [nm]")
    plt.ylabel("Contact resistance [k$\\Omega$ per CNT]")
    plt.legend()
    plt.xscale("log")
    plt.show()
    
    return
    

#################################
# TEST BENCH
#################################


#test_Lorentzian_Kernel(3, 1)
#test_Gaussian_Kernel(5, 1)

#test_NonUnif_Conv_0(2, 0.1)
#test_NonUnif_Conv_CNT(0.03)

#test_CNL_CNT(0.02, 0.04)
#test_load_DoS()

#test_thermal_expectation()

#test_resample()

#test_ballistic_speed()

#test_SK_paper_MIGS()
test_SK_paper_RCLC()


