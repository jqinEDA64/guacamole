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
    d  = 0.2   # Metal-semiconductor distance [nm]
    eps= 8     # Relative dielectric constant of semiconductor [no units]
    WM = 0     # Workfunction of the metal [eV]
    XS = 0     # Electron affinity of the semiconductor [eV]
    Rsh= 0     # Sheet resistance of the semiconductor [Ohm per sq]
    Lsc= 1     # Electrostatic scale length of the contact [nm]
    Ld = 1e5   # Depletion length of the semiconductor outside the contact [nm].
               # This is the 'effective' depletion length as defined by
               # Carlo Gilardi in https://doi.org/10.1109/TED.2022.3190464, not
               # the full depletion length. Arbitrarily initialized to a huge
               # number to suppress quantum tunneling; should change this 
               # according to the technology.

    # These parameters are computed.
    E0 = None  # Energy values for density of states (no metal) [eV]
    D0 = None  # Density of states (no metal) [eV^(-1) nm^(-2)]
    G0 = None  # Broadening energy corresponding to the E0 values [eV]
    E1 = None  # Energy values for density of states (with metal) [eV]
    D1 = None  # Density of states (with metal) [eV^(-1) nm^(-2)]
    E2 = None  # Energy values for density of states (at semiconductor to semiconductor-under-metal interface) [eV]
    D2 = None  # Density of states (at interface) [eV^(-1) nm^(-2)]

    # These parameters are computed.
    Eg = 0     # Bandgap of the semiconductor [eV]
    Ec = 0     # Conduction band minimum of the semiconductor [eV]
    Ev = 0     # Valence band maximum of the semiconductor [eV]
    CNL= 0     # Charge neutrality level of the semiconductor-under-metal [eV]
    CNL_0 = 0  # Charge neutrality level of the isolated semiconductor [eV]
    EF = 0     # Fermi level of the metal-semiconductor system [eV]
    gC = 0     # Specific conductivity of metal-semiconductor interface [Ohm^(-1) nm^(-2)]
    RQ = 0     # Quantum resistance of contact (accounts for Schottky barrier) [Ohm nm]
    mEff_e = 0 # Electron effective mass [me]
    mEff_h = 0 # Hole effective mass [me]

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

        # Compute CNL of isolated semiconductor
        self._setCNL0()
    
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

        #if E is not None :  # G and E are NumPy arrays
        #    raise Exception("Not implemented yet")
        #else :              # G is a scalar and E is None
        #    self.D1, self.E1 = getNonUnifConv(self.D0, self.E0, G, "Lorentzian")
        #    self.G0          = G

        self.D1, self.E1 = getNonUnifConv(self.D0, self.E0, G, "Lorentzian")
        self.G0          = G

        # Compute charge neutrality level (CNL)
        self.CNL = getCNL(self.D0, self.E0, self.CNL_0, self.D1, self.E1)

        # Compute Fermi level of system (equivalent to Schottky barrier position)
        E_FM     = self.Ec - (self.WM - self.XS)  # Fermi energy of isolated metal
        self.EF  = getSchottkyEnergy(self.D1, self.E1, self.CNL, E_FM, self.d, self.eps)

        # Compute quantum resistance, accounting for Schottky barrier
        # and specific contact conductivity. This sets self.RQ.
        self._compRQ()

        # Compute specific contact conductivity. This sets self.gC.
        self.gC = 1e-6*getGC(self.G0, self.D2, self.E2, self.EF, self.kT)

    # Sets the depletion length of the extension region.
    # This depends on doping.
    def setExtensionDepletionLength(self, Ld) :
        self.Ld = Ld

    #########################################
    # Externally accessible physics functions
    # These functions are used to compute the
    # contact properties
    #########################################

    # Get the p-type Schottky barrier height [eV].
    def getSchottkyBarrier_p(self) :
        return self.EF - self.Ev

    # Get the n-type Schottky barrier height [eV].
    def getSchottkyBarrier_n(self) :
        return self.Ec - self.EF

    # Get the contact resistance for contact of length Lc
    #
    # Inputs:
    # - Lc : Contact length [nm]
    #
    # Outputs:
    # - Rc : Contact resistance [Ohm . nm]
    def getRC(self, Lc) :
        rc_contributions = [self._getRC_thermionic(Lc), self._getRC_localthermionic(Lc), self._getRC_tunneling(Lc)]
        out = np.reciprocal(np.sum(np.reciprocal(rc_contributions)))
        return out
        
    # Get the transfer length for contact
    #
    # Outputs:
    # - Lt : Contact resistance transfer length [nm]
    def getLt(self) :
        alpha = getAlpha(self.Rsh    , 1e-3*self.RQ)
        beta  = getBeta (1e6*self.gC , 1e-3*self.RQ)
        print("gC = " + str(self.gC))
        print("RQ = " + str(self.RQ))
        return 1e3*getTransferLength(alpha, beta)

    #############################
    # Protected methods
    #############################

    # Sets the charge neutrality level of the isolated semiconductor.
    # This may be anywhere in the bandgap for a semiconductor with no defects.
    # Here, we arbitrarily set it to midgap.
    def _setCNL0(self) :
        self.CNL_0 = 0.5*(self.Ev+self.Ec)

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
    #
     # Outputs:
    # - Rth: Thermionic contact resistance [Ohm . nm]
    def _getRC_thermionic(self, Lc) :
        out = 1e3*getR_cf(1e-3*self.RQ, 1e6*self.gC, self.Rsh, 1e-3*Lc)
        return out
    
    # Computes the 'extra' thermionic contact resistance
    # of the semiconductor due to local Schottky barrier
    # lowering from doping. This is overriden in the CNT class
    # because the transport modes in CNT are "per CNT" rather than "per nm".
    #
    # Inputs:
    # - Lc : Contact length [nm]
    #
    # Outputs:
    # - Rth_extra : Local "extra" contribution to thermionic contact resistance [Ohm nm]
    def _getRC_localthermionic(self, Lc) :
        isN = np.abs(self.EF - self.Ec) < np.abs(self.EF - self.Ev)
        rth = getR_th_extra(self.G0, self._getModes(self.E0), self.E0, \
                            self.EF, self.Ec, self.Ev, \
                            self.Lsc, self.Ld, self.mEff_e, self.kT, isN)
        return rth
    
    # Computes the contact resistance of the
    # semiconductor due to quantum tunneling.
    def _getRC_tunneling(self, Lc) :
        # TODO
        return FLOAT_MAX
    
    # Computes the number of transport modes as a function of energy.
    #
    # Inputs:
    # - E_vals : NumPy array of energies [eV] 
    #
    # Outputs:
    # - M_vals : Modes per energy [eV^(-1)], corresponding to E_vals
    def _getModes(self, E_vals) :
        raise Exception("_getModes not implemented for generic base class")



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

        # Compute DoS, Ec, Ev
        super().__init__()

        # Electron affinity of CNT approximated by graphene workfunction [Greg]
        W_Graphene = 4.5  # Workfunction of Graphene [eV]
        self.XS    = W_Graphene - self.Eg/2

        # Compute effective mass
        self.mEff_e = getCNT_effectivemass(n, m, t)
        self.mEff_h = self.mEff_e

        # Smooth DoS (regulate singularities somewhat)
        self.D0, self.E0 = getNonUnifConv(self.D0, self.E0, 0.01, "Gaussian")

    #########################################
    # Externally accessible physics functions
    #########################################

    # Converts getRC() (returns value in [Ohm nm])
    # to [Ohm per CNT].
    #
    # Input: 
    # - Lc : Contact length [nm]
    #
    # Output:
    # - Rc : Contact resistance [Ohm per CNT]
    def getRC_perCNT(self, Lc) :
        Rc = super().getRC(Lc)

        # Convert from [Ohm.nm] to [Ohm per CNT]
        return Rc/(np.pi*self.diameter)

    #########################################
    # Internal computation functions
    #########################################

    def _compDoS(self, dE) :
        self.E0, self.D0 = getCNT_DOS(self.n, self.m, self.t)  # [ev^(-1) nm^(-1)]
        f = getEnergyResolution(self.E0) / dE + 0.1
        self.D0 = doResample(self.D0, self.E0, f, "linear")
        self.E0 = doResample(self.E0, self.E0, f, "linear")
        self.D0 = (1e0 / (np.pi*self.diameter)) * self.D0      # [ev^(-1) nm^(-2)]
        return self.E0, self.D0
    
    # Sets the quantum resistance self.RQ in [Ohm.nm]
    def _compRQ(self) :
        
        # Number of transport modes per energy of S
        M_in  = self._getModes(self.E0)

        # Compute the number of modes of X
        M_out, E_out = getNonUnifConv(M_in, self.E0, self.G0, "Lorentzian")

        # Compute the number of modes at interface
        # and the density of states of electrons which can
        # cross the interface
        M_int, self.D2, self.E2 = getInterfaceModesAndDOS(M_out, self.D1, self.E1, \
                                                          M_in , self.D0, self.E0)

        self.RQ = getRQfromModes(M_int, self.E2, self.EF, self.kT)  # [Ohm per CNT]
        self.RQ = (self.diameter*np.pi) * self.RQ                   # [Ohm . nm]

    def _getRC_localthermionic(self, Lc) :
        rth_extra = super()._getRC_localthermionic(Lc)

        # Convert from [Ohm per CNT] to [Ohm nm]
        return np.pi*self.diameter*rth_extra

    # Computes the number of transport modes as a function of energy. 
    def _getModes(self, E_vals) :

        # Number of transport modes per energy of S
        M0  = np.asarray([4 if D > 0 else 0 for D in self.D0])

        # Resample to the input energies
        return doResample(M0, self.E0, E_vals, "linear")


# This class handles all contact physics related to 2D TMD.
class TMD_Contact(Contact) :
    
    #####################
    # TMD fixed variables
    #####################

    # TMD constructor.
    def __init__(self) :

        # Compute DoS, Ec, Ev
        super().__init__()

    #########################################
    # Externally accessible physics functions
    #########################################


    #########################################
    # Internal computation functions
    #########################################

    def _compDoS(self, dE) :
        self.E0, self.D0 = getCNT_DOS(self.n, self.m, self.t)  # [ev^(-1) nm^(-1)]
        f = getEnergyResolution(self.E0) / dE + 0.1
        self.D0 = doResample(self.D0, self.E0, f, "linear")
        self.E0 = doResample(self.E0, self.E0, f, "linear")
        self.D0 = (1e0 / (np.pi*self.diameter)) * self.D0      # [ev^(-1) nm^(-2)]
        return self.E0, self.D0
    
    # Sets the quantum resistance self.RQ in [Ohm.nm]
    def _compRQ(self) :

        # Number of transport modes per energy of S
        M_in  = self._getModes(self.E0)

        # Compute the number of modes of X
        M_out, E_out = getNonUnifConv(M_in, self.E0, self.G0, "Lorentzian")

        # Compute the number of modes at interface
        # and the density of states of electrons which can
        # cross the interface
        M_int, self.D2, self.E2 = getInterfaceModesAndDOS(M_out, self.D1, self.E1, \
                                                          M_in , self.D0, self.E0)
        return FLOAT_MAX

    def _getRC_localthermionic(self, Lc) :
        
        # TODO
        return FLOAT_MAX

    # Computes the number of transport modes as a function of energy. 
    def _getModes(self, E_vals) :

        # TODO
        return np.ones(E_vals.shape)
    

# This class describes metal contacts to MoS2.
#
# We support MoS2 with nonzero concentration of Sulfur vacancies. See https://dx.doi.org/10.1021/acs.jpcc.0c04203?ref=pdf 
class MoS2_Contact(TMD_Contact) :

    # MoS2 constructor.
    #
    # Here, defect_pct is the Sulfur vacancy defect density [%]
    def __init__(self, defect_pct = 0.0) :
        
        if defect_pct > 3 or defect_pct < 0:
            raise Exception("Cannot handle Sulfur vacancy defect density outside of [0.0 %, 3.0 %]")

        self.defect_pct = defect_pct

        # Compute DoS, Ec, Ev
        super().__init__()
        self.Ec = 0.575              # [eV] (in the conventions of our DOS)
        self.Ev = -1.108             # [eV] (in the conventions of our DOS)
        self.Eg = self.Ec - self.Ev  # This gives 1.67 eV, smaller than the true value of 1.8 eV

        # Electron affinity of MoS2
        self.XS = 4.2    # [eV]

        # Relative permittivity of MoS2 (out-of-plane)
        self.eps = 3

        # Compute effective mass (sources?)
        self.mEff_e = 0.45  # [me]
        self.mEff_h = 0.54  # [me]

    #########################################
    # Internal computation functions
    #########################################

    # CNL of defective MoS2 is known to be pinned to the
    # sulfur vacancy energy
    def _setCNL0(self) :
        self.CNL_0 = 0.28  # [eV]

    def _compDoS(self, dE) :

        dos_path = "./raw_data/MoS2/"

        #'''
        defect_name = ["0", "0.1", "0.5", "1.0", "1.5", "2.0", "3.0"]
        defect_nums = np.array([float(x) for x in defect_name])

        # Our DOS will be the weighted average of the two nearest
        # defect DOSs.
        defect_smaller = np.max(defect_nums[defect_nums <= self.defect_pct])
        defect_larger  = np.min(defect_nums[defect_nums >= self.defect_pct])
        x = (self.defect_pct - defect_smaller) / (defect_larger - defect_smaller + 1e-5)

        filepath_1 = dos_path + "MoS2_DOS_" + defect_name[np.where(defect_nums == defect_smaller)[0][0]] + ".dat"
        filepath_2 = dos_path + "MoS2_DOS_" + defect_name[np.where(defect_nums == defect_larger )[0][0]] + ".dat"

        # Load DOSs from file.
        E1, D1 = loadDoSFromFile_nosort(filepath_1)
        E2, D2 = loadDoSFromFile_nosort(filepath_2)

        # Resample the DOSs to comparable energy ranges.
        E_all = np.append(E1, E2)
        E_min = np.min(E_all)
        E_max = np.max(E_all)
        self.E0 = np.arange(E_min, E_max, dE)

        # Perform resampling by linear interpolation
        D1 = np.interp(self.E0, E1, D1)
        D2 = np.interp(self.E0, E2, D2)

        # Compute the weighted average of density of states. 
        self.D0 = (1.0-x)*D1 + x*D2
        #'''

        '''
        filepath_1 = dos_path + "MoS2_DOS_0.dat"
        filepath_2 = dos_path + "MoS2_DOS_0.1.dat"

        # Load DOSs from file.
        E1, D1 = loadDoSFromFile_nosort(filepath_1)
        E2, D2 = loadDoSFromFile_nosort(filepath_2)

        # Resample the DOSs to comparable energy ranges.
        E_all = np.append(E1, E2)
        E_min = np.min(E_all)
        E_max = np.max(E_all)
        self.E0 = np.arange(E_min, E_max, dE)

        # Perform resampling by linear interpolation
        D1 = np.interp(self.E0, E1, D1)
        D2 = np.interp(self.E0, E2, D2)

        # Compute the density of states based on linear
        # extrapolation from a very sparse defect density
        #
        # This assumes that the S vacancies interact only
        # with the regular MoS2 states and there are no
        # vacancy-vacancy interactions
        x  = self.defect_pct / 0.1
        self.D0 = D1 + x*(D2-D1)
        '''

        # Set small DOSs to zero
        self.D0[self.D0 < 10] = 0

        # Re-normalize the DOS. We rely on the normalization in 
        # https://doi.org/10.1038/s41586-021-03472-9, which uses
        # supercell of 5x5 unit cells. Our DOS is actually from 
        # https://dx.doi.org/10.1021/acs.jpcc.0c04203?ref=pdf but
        # their normalization is clearly wrong -- the DOS is far
        # too large.
        #self.D0 = 4.6e-2 * self.D0  # [eV^(-1) nm^(-2)]
        
        return self.E0, self.D0