####################################
# guac_classes_test.py
#
# This file tests the classes and
# functionality in guac_classes.
####################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import os
import copy
from guac_classes  import *

# Verifies the correctness of DoS generation in CNT class. 
def test_DoS(n, m, t, G) :
    test_cnt = CNT_Contact(n, m, t)
    test_cnt.setInteraction(G)
    d        = test_cnt.diameter
    plt.plot(test_cnt.E0, test_cnt.D0*1e14, label = "$\\Gamma = $0", color = "black")
    plt.plot(test_cnt.E1, test_cnt.D1*1e14, label = "$\\Gamma = $" + str(G), color = "green")
    plt.xlim(-test_cnt.Eg, test_cnt.Eg)
    plt.ylim(1e12,1e15)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Energy [eV]")
    plt.ylabel("Density of states [eV$^{-1}$ cm$^{-2}$]")
    plt.savefig("test_out/test_DoS", dpi=500)
    print("Diameter of (" + str(n) + ", " + str(m) + ") CNT = " + str(d) + " [nm]")
    print("Ec = " + str(test_cnt.Ec) + ", Ev = " + str(test_cnt.Ev) + ", Eg = " + str(test_cnt.Eg))
    #print("Max value of DoS = " + str(np.max(test_cnt.D0)))

# Test the correctness of computed CNT density of states. There are two tests here:
# (1) Confirm that the E = 0 DOS of (10, 10) and (12, 12) metallic CNT is about 2 eV^(-1) nm^(-1) [see Deji's book, section 5.3]
# (2) Confirm that the area density of electrons in multiple nanotubes is about 1 electron per 2.62 square-Angstrom.
#     Note that the area density of carbon atoms is 1 C atom per 2.62 square-Angstrom, and that each C atom contributes
#     one electron to the pz orbital (which gives this DOS). See https://poplab.stanford.edu/pdfs/PopVarshneyRoy-GrapheneThermal-MRSbull12.pdf
def test_DOS_correctness() :

    # TEST 1: E = 0 density of state for armchair metallic CNTs
    E1, D1 = getCNT_DOS(10, 10, 3.0)
    E2, D2 = getCNT_DOS(12, 12, 3.0)
    print("Density of states at midgap = " + str(D1[int(len(D1)/2)]))
    print("Density of states at midgap = " + str(D2[int(len(D2)/2)]))

    # TEST 2: Area density of electrons in arbitrarily selected CNTs
    cnt_1 = CNT_Contact(10, 0, 3.0)
    cnt_2 = CNT_Contact(14, 7, 3.0)
    cnt_3 = CNT_Contact(8, 4, 3.0)

    def getD_tot(E, D) :
        return getEnergyResolution(E) * np.sum(D)/2  # Integrate only up to the Fermi level

    def getA_per_electron(E, D) :
        return 1e2/getD_tot(E, D)

    #print("Dtot [nm^(-2)] = " + str(getD_tot(cnt_1.E0, cnt_1.D0)) + ", " +  str(getD_tot(cnt_2.E0, cnt_2.D0)) + ", " + str(getD_tot(cnt_3.E0, cnt_3.D0)))
    print("1/Dtot [A^2] = " + str(getA_per_electron(cnt_1.E0, cnt_1.D0)) + ", " +  str(getA_per_electron(cnt_2.E0, cnt_2.D0)) + ", " + str(getA_per_electron(cnt_3.E0, cnt_3.D0)))

# Verifies the correctness of CNT EF, CNL, RQ, etc. computation.
def test_CNT_RQ() :

    # Create metal-CNT contact
    test_cnt = CNT_Contact(13, 0, 3.18)

    # Set important parameters
    test_cnt.WM = 5.5   # Workfunction of the metal [eV]
    test_cnt.XS = 4.0   # Electron affinity of the semiconductor [eV]

    # Set the interaction strength, which triggers the 
    # computation of many other things
    test_cnt.setInteraction(0.01)

    E0, D0 = loadCNTDoSFromFile(13, 0)
    plt.plot(E0[np.abs(E0)< 0.6], D0[np.abs(E0) < 0.6], color = "black")

    # Plot and print results
    plt.plot(test_cnt.E0[np.abs(test_cnt.E0) < 0.6], test_cnt.D0[np.abs(test_cnt.E0) < 0.6])
    plt.plot(test_cnt.E1[np.abs(test_cnt.E1) < 0.6], test_cnt.D1[np.abs(test_cnt.E1) < 0.6])
    plt.axvline(x = test_cnt.EF, label = "$E_F$", linestyle = "dashed", color = "black")
    plt.yscale("log")
    plt.ylim(1e-1, 1e4)
    plt.savefig("test_out/test_DoS", dpi = 500)
    print("--------------------")
    print("Metal-on-CNT contact")
    print("--------------------")
    print("CNL = " + str(test_cnt.CNL) + " [eV]")
    print("EF  = " + str(test_cnt.EF ) + " [eV]")
    print("Ec = " + str(test_cnt.Ec) + ", Ev = " + str(test_cnt.Ev) + ", Eg = " + str(test_cnt.Eg))

# Plots Rc vs Lc and prints Lt.
def test_CNT_Lc(n, m, t, WM, G) :

    # Create metal-CNT contact
    test_cnt = CNT_Contact(n, m, t)

    # Set important parameters
    test_cnt.WM =  WM  # Workfunction of the metal [eV]

    # Set the interaction strength, which triggers the 
    # computation of many other things
    test_cnt.setInteraction(G)

    print("-------------------------")
    print("CNT transfer length test: ")
    print("-------------------------")
    print("  WM = " + str(WM) + " [eV]")
    print("  G  = " + str(G) + " [eV]")
    print("  Lt = " + str(round(test_cnt.getLt(),2)) + " [nm]")
    print("  Barrier height = " + str(round(test_cnt.getSchottkyBarrier_p(),2)) + " [eV]")
    print("  RQ = " + str(round(test_cnt.RQ/np.pi/test_cnt.diameter/1e3,2)) + " [kOhm per CNT]")
    print("  RC(Lc = inf) = " + str(round(test_cnt.getRC_perCNT(1e5)/1e3, 2)) + " [kOhm per CNT]")

    # Plot and print results
    Lc_vals = np.logspace(0, 2, 50)
    Rc_vals = 1e-3*np.array([test_cnt.getRC_perCNT(Lc) for Lc in Lc_vals])

    plt.plot(Lc_vals, Rc_vals)
    plt.xlabel("Contact length [nm]")
    plt.ylabel("Contact resistance [k$\\Omega$ per CNT]")
    plt.title("Contact resistance as a function of length")
    plt.xscale("log")
    plt.savefig("test_out/Rc_Lc", dpi = 500)

# Plots the dependence of contact resistance on CNT bandgap.
def test_Rc_Eg() :

    # Hopping parameter
    t = 3.0

    # Create many CNTs with various chiralities
    cnts = []
    for n in range(1, 20) :
        for m in range(0, n) :
            if ((n-m) % 3) == 0 :  # Metallic CNT (don't include)
                continue
            if n+m > 24 :          # Unusually large CNT (don't include)
                continue
            cnts.append(CNT_Contact(n, m, t))
            print("Added (n,m) = (" + str(n) + "," + str(m) + ") CNT")

    # Create only CNTs with diameter around 1.6 nm
    #cnts = []
    #cnts.append(CNT_Contact(13, 0, t))
    #cnts.append(CNT_Contact(8,  7, t))

    #plt.plot(cnts[0].E0, cnts[0].D0, label = "(13, 0)")
    #plt.plot(cnts[1].E0, cnts[1].D0, label = "(8, 7)")
    #plt.xlabel("Energy [eV]")
    #plt.ylabel("Density of states [$eV^{-1} nm^{-2}$]")
    #plt.savefig("test_out/DoS_2", dpi = 500)
    #plt.clf()

    # Store the diameters
    cnt_diameters = [cnt.diameter for cnt in cnts]

    # Store the bandgaps
    cnt_bandgaps  = [cnt.Eg       for cnt in cnts]

    # Set the electrostatics
    for cnt in cnts :
        cnt.WM = 5.0   # Workfunction of the metal [eV]
        cnt.setInteraction(0.01)
        print("(n,m) = (" + str(cnt.n) + "," + str(cnt.m) + "): CNL = " + str(cnt.CNL) + ", EF  = " + str(cnt.EF) + ", Ev = " + str(cnt.Ev))

    # Store the Schottky barriers
    barriers      = [cnt.EF - cnt.Ev for cnt in cnts]

    # Plot Schottky barrier as a function of CNT diameter
    plt.plot(cnt_diameters, barriers, 'o')
    #for i in range(len(cnts)):  # Add labels
    #    text = "(" + str(cnts[i].n) + ", " + str(cnts[i].m) + ")"
    #    plt.annotate(text, (cnt_diameters[i], barriers[i]))
    plt.xlabel("CNT diameter [nm]")
    plt.ylabel("$\\Phi_{Bp}$ [eV]")
    plt.title("$p$-type Schottky barrier height vs. CNT diameter:\n$\\Gamma(E) = 10$ meV, $\\chi_S = W_{Graphene} - E_g/2$, $W_M = 5$ eV")
    plt.savefig("test_out/Barrier_diameter", dpi = 500)
    plt.clf()

    # Plot Schottky barrier as a function of CNT bandgap
    plt.plot(cnt_bandgaps, barriers, 'o')
    #for i in range(len(cnts)):  # Add labels
    #    text = "(" + str(cnts[i].n) + ", " + str(cnts[i].m) + ")"
    #    plt.annotate(text, (cnt_bandgaps[i], barriers[i]))
    plt.xlabel("CNT bandgap [eV]")
    plt.ylabel("$\\Phi_{Bp}$ [eV]")
    plt.title("$p$-type Schottky barrier height vs. CNT bandgap:\n$\\Gamma(E) = 10$ meV, $\\chi_S = W_{Graphene} - E_g/2$, $W_M = 5$ eV")
    plt.savefig("test_out/Barrier_Eg", dpi = 500)
    plt.clf()


# Test the thermionic vs. extra thermionic vs. quantum tunneling conductances.
def test_CNT_Extra() :

    # Create metal-CNT contact
    test_cnt_0 = CNT_Contact(13, 0, 3.0)

    # Set important parameters
    test_cnt_0.WM = 8   # Workfunction of the metal [eV]

    # Set the interaction strength, which triggers the 
    # computation of many other things
    test_cnt_0.setInteraction(0.01)

    # Create another CNT with much shorter depletion length
    #Ld = 3  # Depletion length [nm]
    #test_cnt_1    = copy.deepcopy(test_cnt_0)
    #test_cnt_1.Ld = Ld

    print("Lt = " + str(test_cnt_0.getLt()))
    print("Barrier height = " + str(test_cnt_0.getSchottkyBarrier_p()))

    # Plot and print results
    #Lc_vals = np.logspace(0, 2, 50)
    #Rc_0_vals = 1e-3*np.array([test_cnt_0.getRC_perCNT(Lc) for Lc in Lc_vals])
    #Rc_1_vals = 1e-3*np.array([test_cnt_1.getRC_perCNT(Lc) for Lc in Lc_vals])

   # plt.plot(Lc_vals, Rc_0_vals, label = "$L_d = \\infty$")
    #plt.plot(Lc_vals, Rc_1_vals, label = "$L_d$ = " + str(Ld))
    #plt.xlabel("Contact length [nm]")
    #plt.ylabel("Contact resistance [k$\\Omega$ per CNT]")
    #plt.title("Contact resistance as a function of length")
    #plt.xscale("log")
    #plt.legend()
    #plt.savefig("test_out/Rc_Lc", dpi = 500)

# Verifies the correctness of DoS generation in MoS2 class. 
def test_MoS2_DoS() :
    pct_vals = np.linspace(0, 3, 10)
    valence_area_vals = []
    for pct in pct_vals :
        test_mos2 = MoS2_Contact(pct)
        valence_area_vals.append(np.sum(test_mos2.D0[test_mos2.E0 < 0]))
        plt.plot(test_mos2.E0, test_mos2.D0, label = str(round(pct,2)) + "%")
    
    test_mos2 = MoS2_Contact(0)
    plt.axvline(x = test_mos2.CNL_0, color = "black", linestyle = "dashed", label = "CNL")
    plt.xlabel("Energy [eV]")
    plt.ylabel("Density of states [eV$^{-1}$nm$^{-2}$]")
    plt.legend()
    plt.savefig("test_out/Dos_MoS2", dpi=500)
    plt.clf()

    plt.plot(pct_vals, valence_area_vals)
    plt.savefig("test_out/Dos_MoS2_valence_area", dpi=500)
    
# Computes the total areas of Mo and S valence bands and their defect contributions.
# This will show how much Mo and S contribute to the defect, and how much of the 
# defect is contributed by the valence band vs. conduction band.
def test_MoS2_defect_statistics() :
    
    def getArea(filepath) :
        E_vals, D_vals = loadDoSFromFile_nosort(filepath)
        return np.trapz(D_vals, E_vals)
    
    folder = "./raw_data/MoS2/"
    Area_M_defect            = getArea(folder + "MoS2 M defect DOS.csv")
    Area_M_valence_pristine  = getArea(folder + "MoS2 M valence DOS (no defect).csv")
    Area_M_valence_defective = getArea(folder + "MoS2 M valence DOS (with defect).csv")
    Area_S_defect            = getArea(folder + "MoS2 S defect DOS.csv")
    Area_S_valence_pristine  = getArea(folder + "MoS2 S valence DOS (no defect).csv")
    Area_S_valence_defective = getArea(folder + "MoS2 S valence DOS (with defect).csv")

    print("Area of M defect = " + str(round(Area_M_defect, 2)))
    print("Area of M valence band (pristine)  = " + str(round(Area_M_valence_pristine, 2)))
    print("Area of M valence band (defective) = " + str(round(Area_M_valence_defective, 2)))
    print("Area of S defect = " + str(round(Area_S_defect, 2)))
    print("Area of S valence band (pristine)  = " + str(round(Area_S_valence_pristine, 2)))
    print("Area of S valence band (defective) = " + str(round(Area_S_valence_defective, 2)))
    print("")
    print("Contribution of M valence states to M defect = " + str(round(Area_M_defect/(Area_M_valence_pristine-Area_M_valence_defective)*1e2, 3)) + " [%]")
    print("Contribution of S valence states to S defect = " + str(round(Area_S_defect/(Area_S_valence_pristine-Area_S_valence_defective)*1e2, 3)) + " [%]")

# Tests the FLP in MoS2 with various sulfur vacancy concentrations.
def test_MoS2_FLP_1() :
    G_vals      = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]  # Metal-semiconductor interaction strength [eV]
    WM_vals     = np.arange(4.25, 6, 0.25)  # Metal  workfunctions [eV]
    defect_vals = np.arange(0, 3, 0.1)      # Defect probabilities [%]

    # Inner function to compute the slope
    # of the best-fit line through some data
    def compLstSqSlope(x, y) :
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond = None)[0]
        return m

    for G in G_vals :
        slopes = []
        for d in defect_vals :
            SB_vals = []
            for WM in WM_vals :
                test_mos2 = MoS2_Contact(d)
                test_mos2.WM = WM
                test_mos2.setInteraction(G)
                SB_vals.append(test_mos2.getSchottkyBarrier_n())
            slopes.append(compLstSqSlope(WM_vals, SB_vals))
        plt.plot(defect_vals, slopes, label = "$\\Gamma = $" + str(G) + " [eV]")

    plt.legend()
    plt.xlabel("Sulfur vacancy defect conc. [%]")
    plt.ylabel("Fermi-level pinning factor, $S$")
    plt.title("Metal-on-MoS$_2$ contact:\nStrength of Fermi-level pinning")
    plt.savefig("test_out/MoS2_Pinning_factors", dpi = 500)

# Tests the FLP in MoS2 and plots WM and EF. 
def test_MoS2_FLP_2() :
    G = 0.1                                # Metal-semiconductor interaction strength [eV]
    WM_vals     = [4.2, 4.45, 5.2, 5.6]
    defect_pct  = 1.5                      # Probability of Sulfur vacancies

    EF_vals = []
    for WM in WM_vals :
        test_mos2 = MoS2_Contact(defect_pct)
        test_mos2.WM = WM
        test_mos2.setInteraction(G)
        EF_vals.append(test_mos2.EF)

    test_mos2 = MoS2_Contact(defect_pct)
    test_mos2.setInteraction(G)
    plt.plot(test_mos2.D0, test_mos2.E0, color = "black", label = "Isolated MoS$_2$")
    plt.plot(test_mos2.D1, test_mos2.E1, color = "purple", label = "MoS$_2$-under-metal")
    plt.axhline(y = test_mos2.Ec, color = "black", linestyle = "dashed")
    plt.axhline(y = test_mos2.Ev, color = "black", linestyle = "dashed")
    plt.legend()

    # Plot WM and EF points
    colors = ["red", "blue", "green", "orange"]
    E_vac = test_mos2.Ec + test_mos2.XS
    for i in range(len(WM_vals)) :
        D_vals = [300, 500]
        E_vals = [EF_vals[i], E_vac-WM_vals[i]]
        plt.plot(D_vals, E_vals, linestyle = "dashed", marker = "o", color = colors[i])

    plt.xlabel("Density of states [eV$^{-1}$ nm$^{-2}$]")
    plt.ylabel("Energy [eV]")
    plt.ylim(-2.0, 2.0)
    plt.xlim(0, 600)
    plt.title("MoS$_2$ density of states and Fermi-level pinning:\n$[V_S]$ = " + str(defect_pct) + "%, $\\Gamma(E)$ = " + str(G) + " [eV]")
    plt.savefig("test_out/MoS2_DoS_01", dpi = 500)

    for i in range(len(WM_vals)) :
        print("EF_M = " + str(E_vac-WM_vals[i]) + " [eV], EF_MS = " + str(EF_vals[i]) + " [eV]")

# Computes the n- and p-type Schottky barriers for a semimetal 
# with varying semimetal workfunction.
def test_MoS2_Semimetal() :
    
    G = 0.05                              # Metal-semiconductor interaction strength [eV]
    WM_vals     = np.linspace(2.5, 7, 50)
    defect_pct  = 0.2                      # Probability of Sulfur vacancies

    def getGVals(G, WM, E_vac, dE, E_vals) :
        G_vals = np.array([G*min(1, np.abs((E-(E_vac-WM))/dE)) for E in E_vals])
        return G_vals

    Barrier_n_vals = []
    Barrier_p_vals = []
    for WM in WM_vals :
        test_mos2 = MoS2_Contact(defect_pct)
        test_mos2.WM = WM
        E_vac = test_mos2.Ec + test_mos2.XS
        G_vals = getGVals(G, WM, E_vac, 1, test_mos2.E0)
        test_mos2.setInteraction(G_vals)
        Barrier_n_vals.append(test_mos2.getSchottkyBarrier_n())
        Barrier_p_vals.append(test_mos2.getSchottkyBarrier_p())

    plt.plot(WM_vals, Barrier_n_vals, label = "$\\Phi_{Bn}$")
    plt.plot(WM_vals, Barrier_p_vals, label = "$\\Phi_{Bp}$")
    plt.xlabel("Semimetal workfunction [eV]")
    plt.ylabel("Schottky barrier height [eV]")
    plt.legend()
    plt.savefig("test_out/MoS2_semimetal_barrierheights", dpi = 500)
    plt.clf()

    test_mos2 = MoS2_Contact(defect_pct)
    plt.plot(E_vac-WM_vals, Barrier_n_vals, label = "$\\Phi_{Bn}$")
    plt.plot(E_vac-WM_vals, Barrier_p_vals, label = "$\\Phi_{Bn}$")
    plt.axvline(x=test_mos2.Ec, color = "black", linestyle = "dashed")
    plt.axvline(x=test_mos2.Ev, color = "black", linestyle = "dashed")
    plt.xlabel("Energy [eV]")
    plt.ylabel("Schottky barrier height [eV]")
    plt.savefig("test_out/MoS2_semimetal_barrierheights_2", dpi = 500)


# Test of CNT doping.
def test_CNT_Doping(n, m, t, WM, G, nd) :

    # Create metal-CNT contact
    test_cnt = CNT_Contact(n, m, t)

    # Set important parameters
    test_cnt.WM =  WM  # Workfunction of the metal [eV]

    # Set the interaction strength, which triggers the 
    # computation of many other things
    test_cnt.setInteraction(G)

    # Set the CNT doping strength
    # Positive for p-doping, negative for n-doping
    test_cnt.setExtensionDoping_CNT(nd)

    print("----------------")
    print("CNT doping test:")
    print("----------------")
    print("  WM = " + str(WM) + " [eV]")
    print("  G  = " + str(G) + " [eV]")
    print("  Lt = " + str(round(test_cnt.getLt(),2)) + " [nm]")
    print("  Barrier height = " + str(round(test_cnt.getSchottkyBarrier_p(),2)) + " [eV]")
    print("  RQ = " + str(round(test_cnt.RQ/np.pi/test_cnt.diameter/1e3,2)) + " [kOhm per CNT]")
    print("  RC(Lc = inf) = " + str(round(test_cnt.getRC_perCNT(1e5)/1e3, 2)) + " [kOhm per CNT]")
    print("  ")
    print("  Doping = " + str(round(nd, 2)) + " [electrons * nm^(-1)]")
    print("  Depletion length = " + str(round(test_cnt.Ld, 2)) + " [nm]")
    print("  Fermi level in extension = " + str(round(test_cnt.EFd, 2)) + " [eV]")

    print(test_cnt.getConductances(10))

    return 

    # Plot and print results
    Lc_vals = np.logspace(0, 2, 50)
    Rc_vals = 1e-3*np.array([test_cnt.getRC_perCNT(Lc) for Lc in Lc_vals])

    plt.plot(Lc_vals, Rc_vals)
    plt.xlabel("Contact length [nm]")
    plt.ylabel("Contact resistance [k$\\Omega$ per CNT]")
    plt.title("Contact resistance as a function of length")
    plt.xscale("log")
    plt.savefig("test_out/Rc_Lc", dpi = 500)

#test_DoS(13, 0, 3.2, 0.3)
#test_CNT_RQ()
#test_CNT_Lc(16, 0, 3.22, 5.5, 0.01)
#test_Rc_Eg()
#test_CNT_Extra()
#test_DOS_correctness()
#est_MoS2_defect_statistics()
#test_MoS2_DoS()

#test_MoS2_FLP_1()
#test_MoS2_FLP_2()

#test_MoS2_Semimetal()

test_CNT_Doping(8, 0, 3, 4, 0.1, -0.5)