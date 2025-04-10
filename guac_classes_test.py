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
from guac_classes  import *

# Verifies the correctness of DoS generation in CNT class. 
def test_DoS() :
    test_cnt = CNT_Contact(13, 0, 2.74)
    plt.plot(test_cnt.E0, test_cnt.D0)
    plt.savefig("test_out/test_DoS", dpi=500)
    print("Diameter of (13, 0) CNT = " + str(test_cnt.d) + " [nm]")
    print("Ec = " + str(test_cnt.Ec) + ", Ev = " + str(test_cnt.Ev) + ", Eg = " + str(test_cnt.Eg))
    print("Max value of DoS = " + str(np.max(test_cnt.D0)))

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
def test_CNT_Lc() :

    # Create metal-CNT contact
    test_cnt = CNT_Contact(13, 0, 3.0)

    # Set important parameters
    test_cnt.WM = 5.0   # Workfunction of the metal [eV]
    test_cnt.XS = 4.0   # Electron affinity of the semiconductor [eV]

    # Set the interaction strength, which triggers the 
    # computation of many other things
    test_cnt.setInteraction(0.01)

    print("Lt = " + str(test_cnt.getLt()))

    # Plot and print results
    Lc_vals = np.linspace(10, 100, 50)
    Rc_vals = [test_cnt.getRC(Lc) for Lc in Lc_vals]

    plt.plot(Lc_vals, Rc_vals)
    plt.xlabel("Contact length [nm]")
    plt.ylabel("Contact resistance [Ohm per CNT]")
    plt.title("Contact resistance as a function of length")
    plt.savefig("test_out/Rc_Lc", dpi = 500)

#test_DoS()
test_CNT_RQ()