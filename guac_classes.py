####################################
# guac_class.py
#
# This file computes the contact
# resistance for various kinds of 
# low-d materials (CNT, MoS2) from
# the given assumptions.
####################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import os
from guac_math    import *
from guac_physics import *
from gen_CNT_DoS  import *


# This is the base class for any low-d contact.
#
# NOTE: the parameters d, eps, WM, XS must be set manually by the user.
class Contact :

    # These parameters must be set manually by the user
    d = 0.2    # Metal-semiconductor distance [nm]
    eps = 8    # Relative dielectric constant of semiconductor [no units]
    WM = 0     # Workfunction of the metal [eV]
    XS = 0     # Electron affinity of the semiconductor [eV]
    Rsh= 0     # Sheet resistance of the semiconductor [Ohm per sq]

    # These parameters are computed.
    E0 = None  # Energy values for density of states (no metal) [eV]
    D0 = None  # Density of states (no metal) [eV^(-1) nm^(-2)]
    G0 = None  # Broadening energy corresponding to the E0 values [eV]
    E1 = None  # Energy values for density of states (with metal) [eV]
    D1 = None  # Density of states (with metal) [eV^(-1) nm^(-2)]

    # These parameters are computed.
    Eg = 0     # Bandgap of the semiconductor [eV]
    Ec = 0     # Conduction band minimum of the semiconductor [eV]
    Ev = 0     # Valence band maximum of the semiconductor [eV]
    CNL= 0     # Charge neutrality level of the semiconductor-under-metal [eV]
    EF = 0     # Fermi level of the metal-semiconductor system [eV]
    gC = 0     # Specific conductivity of metal-semiconductor interface [Ohm^(-1) nm^(-2)]
    RQ = 0     # Quantum resistance of contact (accounts for Schottky barrier) [Ohm nm]

    # These parameters are environmental and usually shouldn't be changed.
    kT = 0.025 # Thermal voltage [eV]

    def __init__(self) :
        dE = 0.01          # [eV]
        self._compDoS(dE)  # Initialize the semiconductor DoS.
                           # May want to increase energy resolution later.
        
        # Compute Eg, Ec, Ev
        E_bandgap_vals = self.E0[self.D0 == 0]
        self.Ec = np.max(E_bandgap_vals)
        self.Ev = np.min(E_bandgap_vals)
        self.Eg = self.Ec-self.Ev
    
    #########################################
    # Externally accessible physics functions
    # These functions are used to set the
    # contact properties
    #########################################

    # Set metal-CNT interaction strength
    #
    # Inputs:
    # - G: Metal-semiconductor interaction strength (either scalar or NumPy array)
    # - E: Energy values (None if G is scalar; NumPy array if G is array)
    #
    # Outputs: None, but computes and stores the MIGS density along with other
    # contact properties. 
    def setInteraction(self, G, E = None):

        if E is not None :  # G and E are NumPy arrays
            raise Exception("Not implemented yet")
        else :              # G is a scalar and E is None
            self.D1, self.E1 = getNonUnifConv(self.D0, self.E0, G, "Lorentzian")
            self.G0          = G

        # Compute charge neutrality level (CNL)
        self.CNL = getCNL(self.D0, self.E0, 0.5*(self.Ev+self.Ec), self.D1, self.E1)

        # Compute Fermi level of system (equivalent to Schottky barrier position)
        E_FM     = self.Ec - (self.WM - self.XS)  # Fermi energy of isolated metal
        self.EF  = getSchottkyEnergy(self.D1, self.E1, self.CNL, E_FM, self.d, self.eps)

        ind = int(len(self.D1)/2)
        print("Density of states in middle = " + str(self.D1[ind]))

        # Compute specific contact conductivity
        self.gC = getGC(G, self.D1, self.E1, self.EF, self.kT)

        # Compute quantum resistance, accounting for Schottky barrier
        self.RQ = self._compRQ()

        return

    # Sets the depletion length of the extension region.
    # This depends on doping.
    def setExtensionDepletionLength(self, Ld) :
        self.Ld = Ld

    #########################################
    # Externally accessible physics functions
    # These functions are used to compute the
    # contact properties
    #########################################

    # Get the contact resistance for contact of length Lc
    #
    # Inputs:
    # - Lc : Contact length [nm]
    #
    # Outputs:
    # - Rc : Contact resistance [Ohm . nm]
    def getRC(self, Lc) :
        rc_contributions = [self._getRC_thermionic(Lc), self._getRC_localthermionic(Lc), self._getRC_tunneling(Lc)]
        return np.sum(np.reciprocal(rc_contributions))
    
    # Get the transfer length for contact
    #
    # Outputs:
    # - Lt : Contact resistance transfer length [nm]
    def getLt(self) :
        alpha = getAlpha(self.Rsh, self.RQ)
        beta  = getBeta (self.gC , self.RQ)
        return getTransferLength(alpha, beta)

    #############################
    # Protected methods
    #############################

    # Computes the density of states of the semiconductor.
    # dE is the energy spacing of the computed DoS.
    #
    # Input:
    # - dE: Energy spacing of the desired DoS [eV]
    #
    # Output:
    # - self.E0 : Energy values of the DoS [eV]
    # - self.D0 : Density of states of the DoS [eV^(-1) nm^(-2)].
    def _compDoS(self, dE) :
        raise Exception("_compDoS not implemented for generic base class")
    
    # Computes the electrostatic properties (CNL, EF, etc.) of
    # the semiconductor.
    def _compRQ(self) :
        raise Exception("_compRQ not implemented for generic base class")

    # Computes the thermionic contact resistance
    # of the semiconductor.
    #
    # Inputs:
    # - Lc : Contact length [nm]
    def _getRC_thermionic(self, Lc) :
        return getR_cf(self.RQ, self.gC, self.Rsh, Lc)
    
    # Computes the 'extra' thermionic contact resistance
    # of the semiconductor due to local Schottky barrier
    # lowering from doping.
    def _getRC_localthermionic(self, Lc) :
        # TODO
        return FLOAT_MAX
    
    # Computes the contact resistance of the
    # semiconductor due to quantum tunneling.
    def _getRC_tunneling(self, Lc) :
        # TODO
        return FLOAT_MAX


# This class handles all contact physics related to CNT.
# Prefer for a single CNT object to refer to a particular 
# CNT under a particular metal, etc.
#
# If testing multiple CNTs in different contacts, better to
# create multiple distinct objects.
class CNT_Contact(Contact) :
    
    #####################
    # CNT fixed variables
    #####################

    a = 0.142 # Carbon-carbon distance in graphene [nm]

    # CNT constructor.
    # - n = First  chiral number
    # - m = Second chiral number
    # - t = Hopping energy [eV]
    def __init__(self, n, m, t) :
        self.n = n
        self.m = m
        self.t = t

        # Compute CNT diameter [nm]
        # See https://www.photon.t.u-tokyo.ac.jp/~maruyama/kataura/chirality.html
        self.diameter = self.a*np.sqrt(3*(n*n+n*m+m*m))/np.pi

        # Compute DoS
        super().__init__()

        # Smooth DoS (regulate singularities somewhat)
        self.D0, self.E0 = getNonUnifConv(self.D0, self.E0, 0.01, "Gaussian")

    #########################################
    # Externally accessible physics functions
    #########################################

    def getRC(self, Lc) :
        # Convert from [Ohm.nm] to [Ohm per CNT]
        Rc = super().getRc(Lc)
        return self.d*Rc

    #########################################
    # Internal computation functions
    #########################################

    def _compDoS(self, dE) :
        self.E0, self.D0 = getCNT_DOS(self.n, self.m, self.t)
        f = getEnergyResolution(self.E0) / dE + 0.1
        self.D0 = doResample(self.D0, self.E0, f, "linear")
        self.E0 = doResample(self.E0, self.E0, f, "linear")
        self.D0 = (1e-9 / (np.pi*self.diameter)) * self.D0
        return self.E0, self.D0
    
    # Sets the quantum resistance self.RQ in [Ohm.nm]
    def _compRQ(self) :
        
        # Number of transport modes per energy of S
        M_in  = np.asarray([4 if D > 0 else 0 for D in self.D0])

        # Compute the number of modes of X
        M_out, E_out = getNonUnifConv(M_in, self.E0, self.G0, "Lorentzian")

        # Compute the number of modes at interface
        # and the density of states of electrons which can
        # cross the interface
        M_int, D_int, E_int = getInterfaceModesAndDOS(M_out, self.D1, self.E1, \
                                                      M_in , self.D0, self.E0)
        
        self.RQ = getRQfromModes(M_int, E_int, self.EF, self.kT)  # [Ohm per CNT]
        self.RQ = (1.0/self.diameter/np.pi) * self.RQ             # [Ohm . nm]
    

