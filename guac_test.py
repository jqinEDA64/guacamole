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
    D_vals = doResample(D_vals, E_vals, f, "linear")
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
    #ax[0].plot(E_MIGS, D_MIGS, label = "Analytical MIGS", color = "green")
    ax[0].fill_between(E_out, 0, D_V_out, alpha = 0.5, color = "blue", label = "Valence band states")
    ax[0].fill_between(E_out, 0, D_C_out, alpha = 0.5, color = "red" , label = "Conduction band states")
    ax[0].set_title("Density of states")
    ax[0].set_ylabel("DoS [eV$^{-1}$ nm$^{-2}$]")
    CNL_label = "CNL = " + str(round(float(CNL), 2)) + " [eV]"
    ax[0].axvline(x = CNL, label = CNL_label, linestyle = "dashed", color = "black")
    #CNL_analytic_label = "Analytical CNL = " + str(round(float(CNL_est), 2)) + " [eV]"
    #ax[0].axvline(x = CNL_est, label = CNL_analytic_label, linestyle = "dashed", color = "green")
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
    E_in, D_in = loadCNTDoSFromFile(13, 0)
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
    
    # Load S-K MIGS result from file
    def loadMIGSFromFile(filepath):
        df = pd.read_csv(filepath, sep = "\s+", header=None, names = ["D", "E"])
        D  = df["D"] * 1e-2
        E  = df["E"] - 0.465
        return D, E
    
    D_11, E_11 = loadMIGSFromFile("./raw_data/sk_MIGS_red")
    D_22, E_22 = loadMIGSFromFile("./raw_data/sk_MIGS_blue")
    D_33, E_33 = loadMIGSFromFile("./raw_data/sk_MIGS_green")
    
    '''
    plt.rcParams['font.size'  ] = 13
    #plt.plot(D_in, E_in, label = "CNT"                , color = "black")
    plt.plot(D_1 , E_1 , label = "$\\Gamma = 3.5$ meV", color = "red")
    plt.plot(D_2 , E_2 , label = "$\\Gamma = 10$ meV" , color = "blue")
    plt.plot(D_3 , E_3 , label = "$\\Gamma = 0.3$ eV" , color = "green")
    plt.plot(D_11, E_11, label = "$\\Gamma = 3.5$ meV", color = "red", linestyle = "dashed")
    plt.plot(D_22, E_22, label = "$\\Gamma = 10$ meV" , color = "blue", linestyle = "dashed")
    plt.plot(D_33, E_33, label = "$\\Gamma = 0.3$ eV" , color = "green", linestyle = "dashed")
    plt.xlabel("Density of states [eV$^{-1}$ cm$^{-2}$]")
    plt.ylim(-1, 1)
    plt.xscale("log")
    plt.legend()
    plt.ylabel("Energy [eV]")
    plt.title("Metal-induced gap states of $(13,0)$ CNT\nat various values of $\\Gamma(E)$")
    plt.show()
    '''
    
    
    plt.rcParams['figure.dpi'] = 500
    #plt.tick_params(axis='both', labelsize=2)
    plt.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
    #plt.plot(D_in, E_in, label = "CNT"                , color = "black")
    plt.plot(E_1 , D_1,  color = "red")
    plt.plot(E_2 , D_2,  color = "blue")
    plt.plot(E_3 , D_3,  color = "green")
    plt.plot(E_11, D_11, color = "red", linestyle = "dashed")
    plt.plot(E_22, D_22, color = "blue", linestyle = "dashed")
    plt.plot(E_33, D_33, color = "green", linestyle = "dashed")
    #plt.xlabel("Density of states [eV$^{-1}$ cm$^{-2}$]")
    plt.xlim(-0.56, 0.56)
    plt.ylim(0.9e13, 1.1e17)
    plt.yscale("log")
    #plt.legend()
    #plt.xlabel("Energy [eV]")
    #plt.title("Metal-induced gap states of $(13,0)$ CNT\nat various values of $\\Gamma(E)$")
    plt.savefig("test_out/MIGS.png", dpi=1000)
    plt.show()
    
    return
    
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
    E_in, D_in = loadCNTDoSFromFile(13, 0)  # [eV^(-1) nm^(-2)]
    D_in  = doResample(D_in, E_in, 10)
    E_in  = doResample(E_in, E_in, 10)
    M_in  = np.asarray([4 if np.abs(E) > 0.43 else 0 for E in E_in])
    
    # External parameters
    E_F = -0.48  # [eV] (Schottky energy level assuming in valence band [S-K paper])
    kT  = 0.025  # [eV]
    
    # Toggle between S-K paper (no Schottky physics)
    # and more realistic Schottky contact
    doSchottkyPhysics = False
    d = 0.2 # [nm]
    eps = 8 
    def getSchottkyEnergyLevel(D_in, E_in, D_vals, E_vals):
        if not doSchottkyPhysics :
            return E_F
        
        # Compute CNL
        CNL   = getCNL(D_in, E_in, 0, D_vals, E_vals)
        
        # Compute Schottky energy
        E_FM  = -1  # [eV] (Fermi energy of the metal)
        E_Sch = getSchottkyEnergy(D_vals, E_vals, CNL, E_FM, d, eps)
        return E_Sch
    
    # Inner function to do most of the work
    # Returns the 
    # (1) Quantum resistance of the contact
    # (2) Specific contact conductivity of the contact
    # based on the Schottky barrier computation
    # and conservation of modes across the interface.
    def getRQandGC(G) :
        
        # Compute the density of states of X (including MIGS)
        D_out, E_out = getNonUnifConv(D_in, E_in, G, "Lorentzian")
        
        # Compute the number of modes of X
        M_out, E_out = getNonUnifConv(M_in, E_in, G, "Lorentzian")
        
        # Compute the Schottky barrier level
        E_F = getSchottkyEnergyLevel(D_in, E_in, D_out, E_out)
        
        # Compute the number of modes at interface
        # and the density of states of electrons which can
        # cross the interface
        M_int, D_int, E_int = getInterfaceModesAndDOS(M_out, D_out, E_out, \
                                                      M_in , D_in , E_in )
        
        '''                                              
        plt.plot(E_in, M_in, label = "Modes of S")
        plt.plot(E_out, M_out, label = "Modes of X")
        plt.plot(E_int, M_int, label = "Modes of interface")
        plt.legend()
        plt.xlabel("Energy [eV]")
        plt.ylabel("Modes")
        plt.show()
        
        plt.plot(E_in , D_in , label = "DOS of S")
        plt.plot(E_out, D_out, label = "DOS of X")
        plt.plot(E_int, D_int, label = "DOS of interface")
        plt.legend()
        plt.xlabel("Energy [eV]")
        plt.ylabel("Modes")
        plt.show()
        '''
        
        # Compute the quantum resistance
        RQ = getRQfromModes(M_int, E_int, E_F, kT)  # [Ohm per CNT]
        R0 = RQ*(np.pi*1.0*1e-3)                    # [Ohm um]
        
        # Compute the specific contact conductivity
        #gC = getGC(G, D_int, E_int, E_F, kT)        # [Ohm^(-1) um^(-2)]
        gC = getGC(G, D_int, E_int, E_F, kT)
        
        print("G = " + str(G) + " [eV]")
        print("  E_F = " + str(round (E_F, 3    )) + " [eV]")
        print("  RQ  = " + str(format(RQ , '.2e')) + " [Ohm per CNT]")
        print("  gC  = " + str(format(gC , '.2e')) + " [Ohm^(-1) um^(-2)]")
        print("  Lt  = " + str(format(1e3/getBeta(gC, R0), '.2e')) + " [nm]")
        
        # Return these values
        return R0, RQ, gC
    
    R01, RQ1, gC1 = getRQandGC(G_1)
    R02, RQ2, gC2 = getRQandGC(G_2)
    R03, RQ3, gC3 = getRQandGC(G_3)
    
    Lc   = np.logspace(0, 2.5, 100)
    Rc_1 = 2*RQ1/R01*1e-3*getR_cf(R01, gC1, 0, Lc*1e-3)
    Rc_2 = 2*RQ2/R02*1e-3*getR_cf(R02, gC2, 0, Lc*1e-3)
    Rc_3 = 2*RQ3/R03*1e-3*getR_cf(R03, gC3, 0, Lc*1e-3)
    
    # Usually plot 2*RC (cf. Franklin paper, Solomon paper).
    # Sheng-Kai plots the resistance of device = 2*RC
    # since 2 contacts
    # Rc_1 = 2*Rc_1
    # Rc_2 = 2*Rc_2
    # Rc_3 = 2*Rc_3
    
    # Load S-K MIGS result from file
    def loadRCFromFile(filepath):
        df = pd.read_csv(filepath, sep = "\s+")
        Lc = df["Lc"]
        Rc = df["Rc"]
        return Lc, Rc
    
    Lc_11, Rc_11 = loadRCFromFile("raw_data/sk_RC_red")
    Lc_22, Rc_22 = loadRCFromFile("raw_data/sk_RC_blue")
    Lc_33, Rc_33 = loadRCFromFile("raw_data/sk_RC_green")
    
    plt.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
    plt.plot(Lc_11, Rc_11, color = "red", linestyle = "dashed")
    plt.plot(Lc_22, Rc_22, color = "blue", linestyle = "dashed")
    plt.plot(Lc_33, Rc_33, color = "green", linestyle = "dashed")
    plt.plot(Lc, Rc_1, label = "$\\Delta = 3.5$ meV", color = "red")
    plt.plot(Lc, Rc_2, label = "$\\Delta = 10$ meV" , color = "blue")
    plt.plot(Lc, Rc_3, label = "$\\Delta = 0.3$ eV" , color = "green")
    #plt.xlabel("Contact length [nm]")
    #plt.ylabel("Contact resistance [k$\\Omega$ per CNT]")
    #plt.legend()
    plt.xscale("log")
    plt.xlim(1, 100)
    #plt.ylim(0, 30)
    plt.savefig("test_out/Rc.png", dpi=1000)
    plt.show()
    
    return
    

# Test the quantum resistance computation (which depends on
# Schottky barrier). The quantum resistance should go to 6.45 kOhm
# when the Fermi level is inside the transport band.
def test_RQ():

    E_in, D_in = loadCNTDoSFromFile(11, 0)
    D_in  = doResample(D_in, E_in, 10, "linear")
    E_in  = doResample(E_in, E_in, 10, "linear")
    M_in  = np.asarray([2 if np.abs(E) > 0.45 else 0 for E in E_in])
    
    kT = 0.025
    E_F_vals = np.linspace(-0.7, 0.7, 200)
    RQ_vals  = np.asarray([getRQfromModes(M_in, E_in, E_F, kT) for E_F in E_F_vals])
    
    plt.plot(E_F_vals, 1e-3*RQ_vals)
    plt.title("Quantum resistance as a function of Fermi energy")
    plt.axvline(x =  0.45, linestyle = "dashed", color = "black")
    plt.axvline(x = -0.45, linestyle = "dashed", color = "black")
    plt.axhline(y =  6.45, label = "$R_Q=6.45$ k$\\Omega$", linestyle = "dashed", color = "red")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Fermi energy $E_F$ [eV]")
    plt.ylabel("Quantum resistance $R_Q$ [k$\\Omega$ per CNT]")
    plt.show()
    
def test_MoS2() :
    df = pd.read_csv("./raw_data/DoS_MoS2_monolayer.dat", sep=',', skipinitialspace=True)
    E_in = df["E"]-0.8
    D_in = df["DOS"]*6  # 6 = approx. scaling to MoS2 DOS in ev^(-1) nm^(-2)
    D_in[48] = 0
    D_in[49] = 0
    
    E_vals = np.arange(np.min(E_in), np.max(E_in), 0.005)
    D_vals = np.interp(E_vals, E_in, D_in)
    D_vals = doResample(D_vals, E_vals, 5, "linear")
    E_vals = doResample(E_vals, E_vals, 5, "linear")
    D_V_vals = np.where(E_vals <= 0, D_vals, 0)
    D_C_vals = np.where(E_vals >  0, D_vals, 0)
    print("dE = " + str(getEnergyResolution(E_vals)))
    
    E_C    = 0.8
    E_V    = -0.8
    
    G1     = 0.01
    def getPrefactor(E):
        if E < E_V :
            return 2
        else :
            return min(abs(E-1.2), 1)
            
    G1_vals= np.array([G1*getPrefactor(E) for E in E_vals])
    
    D_out, E_out   = getNonUnifConv(D_vals, E_vals, G1_vals, "Lorentzian")
    D_C_out, E_out = getNonUnifConv(D_C_vals, E_vals, G1_vals, "Lorentzian")
    D_V_out, E_out = getNonUnifConv(D_V_vals, E_vals, G1_vals, "Lorentzian")
    
    CNL = getCNL(D_vals, E_vals, 0, D_out, E_out)
    #E_F = getSchottkyEnergyLevel(D_vals, E_vals, D_out, E_out)
    print("CNL = " + str(CNL))
    
    E_FM = E_C + 0.3
    d = 0.2
    eps = 5.5
    E_F = getSchottkyEnergy(D_vals, E_vals, CNL, E_FM, d, eps)
    print("E_F = " + str(E_F))
    

    
   
    # Plot input and output on the same graph
    #fig, ax = plt.subplots(1)  # Ensure the plots share a common x-axis
    #fig.suptitle("MIGS and CNL of Bi-on-MoS$_2$")
    
    
    #plt.rcParams['figure.dpi'] = 1000
    plt.plot(E_vals, D_vals, label = "Original DoS",   color = "black")
    plt.plot(E_out , D_out , label = "Convoluted DoS", color = "purple")
    #ax[0].plot(E_MIGS, D_MIGS, label = "Analytical MIGS", color = "green")
    plt.fill_between(E_out, 0, D_V_out, alpha = 0.5, color = "blue", label = "Valence band states")
    plt.fill_between(E_out, 0, D_C_out, alpha = 0.5, color = "red" , label = "Conduction band states")
    plt.title("Density of states")
    plt.ylabel("DoS [eV$^{-1}$ nm$^{-2}$]")
    plt.xlabel("Energy [eV]")
    #EF_label = "C = " + str(round(float(CNL), 2)) + " [eV]"
    plt.axvline(x = E_F, linestyle = "dashed", color = "black")
    #CNL_analytic_label = "Analytical CNL = " + str(round(float(CNL_est), 2)) + " [eV]"
    #ax[0].axvline(x = CNL_est, label = CNL_analytic_label, linestyle = "dashed", color = "green")
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc = "upper left")
    plt.show()
    
    # Also plot Gamma on this graph
    #ax[1].plot(E_vals, G_vals, color = "red")
    #ax[1].set_title("Metal-semiconductor coupling")
    #ax[1].set_xlabel("Energy [eV]")
    #ax[1].set_ylabel("$\\Gamma(E)$ [eV]")
    #plt.tight_layout()
    #saveAndShow(fig, "test_out/CNL_MoS2")
    
   # plt.plot(E_vals, G1_vals)
   # plt.show()
    
   # plt.plot(E_vals, D_vals)
   # plt.plot(E_out, D_out)
   # plt.show()

#################################
# TEST BENCH
#################################


#test_Lorentzian_Kernel(3, 1)
#test_Gaussian_Kernel(5, 1)

#test_NonUnif_Conv_0(2, 0.1)
#test_NonUnif_Conv_CNT(0.03)

#test_CNL_CNT(0.03, 0.04)
#test_load_DoS()

#test_thermal_expectation()

#test_resample()

#test_ballistic_speed()

#test_SK_paper_MIGS()
#test_SK_paper_RCLC()

#test_RQ()
#test_MoS2()