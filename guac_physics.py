####################################
# guac_physics.py
#
# This file contains all the physics
# functions needed for contact
# resistance modeling
####################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import os
from guac_math import *


####################################
# GLOBAL QUANTITIES
####################################


ELECTRON_MASS   = 9.11e-31  # [kg]
ELECTRON_CHARGE = 1.6e-19   # [C ]
PLANCK_CONSTANT = 6.626e-34 # [J / Hz]
RED_PLANCK_CONSTANT = PLANCK_CONSTANT/(2*np.pi)  # [J*s]
EPS_0_SI        = 8.85e-12  # [F / m]
EPS_0_NATURAL   = 55.3      # [e^2 eV^(-1) um^(-1)]


####################################
# THERMAL AVERAGING
####################################


# Returns the thermal expectation of a quantity X.
# The thermal expectation is the average over Fermi-Dirac distribution.
# This thermal average could be very small if X is nearly zero near E_F
# which is the regular Schottky barrier effect.
#
# Inputs:
# - kT    : Thermal voltage [eV]
# - E_F   : Fermi energy    [eV]
# - X_vals: Values of quantity being averaged 
# - E_vals: Values of energy corresponding to X_vals [eV]
# - D_vals: [Optional] can include the semiconductor density of states
#           if needed. If included, returns the thermal average of X*D
#           WITHOUT normalizing for the thermal average of D itself.
#
# Output:
# - X_avg : Thermal expectation value of X
def getThermalExpectation(X_vals, E_vals, kT, E_F, D_vals = None) :

    def getFD_Derivative():
        if (E_F > np.max(E_vals)) or (E_F < np.min(E_vals)) :
            err_out("Fermi energy not in range")
        
        FD_der_vals = scipy.stats.logistic.pdf(E_vals, loc = E_F, scale = kT)
        FD_der_vals = FD_der_vals / np.sum(FD_der_vals) # normalize
        return FD_der_vals

    FD_der_vals = getFD_Derivative()
    
    # Add density of states weighting, if desired
    DX_vals = None
    if D_vals is not None :
        DX_vals = np.multiply(X_vals, D_vals)
    else :
        DX_vals = X_vals
    
    return np.dot(DX_vals, FD_der_vals)


####################################
# ELECTRONIC TRANSPORT
####################################


# Return the average ballistic velocity
# in the +x direction, for a semiconductor
# carrier with energy E. Follows Lundstrom's
# book "Near-equilibrium Transport: Fundamentals
# and Applications."
#
# This function automatically determines 
# whether the particle is an electron or hole
# so it must also take two effective masses.
#
# This is accurate only near the band edges, but
# due to Fermi averaging only the band edges will
# matter for transport purposes.
#
# This function works only for the semiconductor (S)
# and not for the semiconductor-under-metal (X) because
# it assumes sharp band edges. Therefore, this function
# is useful to compute the quantum resistance (which is
# independent of the metal).
#
# Inputs:
# - E  : Particle energies. May be either array or float. [eV]
# - E_V: Valence band maximum energy [eV]
# - E_C: Conduction band minimum energy [eV]
# - M_H: Hole effective mass [me]
# - M_E: Electron effective mass [me]
# - dim: Spatial dimension of the semiconductor (1, 2, or 3)
#
# Outputs:
# - v_x: Average ballistic velocity in the +x-direction [m / s]
def getBallisticSpeed(E, E_V, E_C, M_H, M_E, dim = 2) :
    if E_C < E_V :
        err_out("Conduction band minimum should lie above valence band maximum")
    
    # Consider transport velocity = 0 in the bandgap
    # since no current will be carried there
    dE = None
    M  = None
    if   isinstance(E, (int, float)):
        dE = E-E_C if E > E_C else E_V-E if E < E_V else 0
        M  = M_E   if E > E_C else  M_H
    elif isinstance(E, np.ndarray):
        dE = np.asarray([x-E_C if x > E_C else E_V-x if x < E_V else 0 for x in E])
        M  = np.asarray([M_E   if x > E_C else M_H                     for x in E])
    
    dE = ELECTRON_CHARGE*dE
    M  = ELECTRON_MASS  *M
    v = np.sqrt(2*dE/M)      # Carrier speed
    
    prefactor = 1
    if   dim == 1 :
        prefactor = 1
    elif dim == 2 :
        prefactor = 2/np.pi
    elif dim == 3 :
        prefactor = np.pi/4
    else :
        err_out("Dimension of semiconductor must be 1, 2, or 3")
        
    return prefactor*v


# Return the thermal velocity of carriers in
# a semiconductor with effective mass M.
# This is equivalent to estimating the ballistic
# transport velocity when kinetic energy = kT.
#
# Inputs:
# - kT: Thermal energy [eV]
# - M : Carrier effective mass [me]
#
# Output:
# - v : Thermal velocity [m / s]
def getThermalVelocity(kT, M) :
    return getBallisticSpeed(kT, -10, 0, M, M, dim = 1)


# Use 'mode conservation' to estimate the effective
# number of modes and density of states of electrons
# which can actually cross over the X-S barrier.
#
# This is inspired by Gautam Shine's thesis. 
#
# If M_X(E) and M_S(E) are the number of transport
# modes per energy at energy E, then we set
#
# M(E) = min(M_X, M_S)(E) (similar to current conservation)
#
# D(E) = D_X(E)*(M/M_X)(E) (effective density of states).
#
# Inputs:
# - M_X_vals: Number of modes per energy in the X-region
# - D_X_vals: Density of states in the X-region
# - E_X_vals: Energies corresponding to {M,D}_X_vals
# - M_S_vals: Number of modes per energy in the S-region
# - D_S_vals: Density of states in the S-region
# - E_S_vals: Energies corresponding to {M,D}_S_vals
#
# Outputs:
# - M_vals  : Modes per energy across the interface
# - D_vals  : Density of states in the X-region, but only for
#             carriers which can cross over the interface
# - E_vals  : Corresponding energies
def getInterfaceModesAndDOS(M_X_vals, D_X_vals, E_X_vals, \
                            M_S_vals, D_S_vals, E_S_vals) :
    
    #return M_S_vals, D_S_vals, E_S_vals
    
    # TODO jqin: don't need the rest of this
    #            since Lorentzian broadening is 'automatic'
    #            and should not be considered misalignment.
    #
    #            Should really consider misalignment between
    #            broadened version of M_S(E) and M_X(E).
    
    # Compute M(E) = min(M_X, M_S)(E)
    M_vals, E_vals = getMinFunction(M_X_vals, E_X_vals, M_S_vals, E_S_vals)
    
    # Compute the values of M_X_vals at the new E_vals
    M_X_2 = doResample(M_X_vals, E_X_vals, E_vals, interp_type = "linear")
    
    # Compute D(E) = D_X(E)*(M/M_X)(E)
    D_X_2 = doResample(D_X_vals, E_X_vals, E_vals, interp_type = "linear")
    M_X_2 = np.divide(M_vals, M_X_2)
    D_vals= np.multiply(D_X_2, M_X_2)
    
    return M_vals, D_vals, E_vals
    

####################################
# METAL-INDUCED GAP STATES (MIGS);
# CHARGE NEUTRALITY LEVEL (CNL) 
# RELATED FUNCTIONS
####################################


# Charge neutrality level computation.
# Computes the energy level at which the output has the
# same amount of charge as the input.
#
# Inputs:
# - D_in   : Density of states of original system.
# - E_in   : Energy levels of states of original system.
# - CNL_in : Input charge neutrality level. For semiconductors
#            CNL_in is not unique (it may be anywhere in the bandgap).
#            CNL_in does not have to be one of the entries of E_in;
#            we use interpolation to compute the total charge.
# - D_out  : Density of states of modified system.
# - E_out  : Energy levels of states of modified system.
#
# Output:
# - CNL_out: Output charge neutrality level. This better be unique!
#            If not unique, (i.e., MIGS density is zero), then 
#            presumably interpolation will fail.
def getCNL(D_in, E_in, CNL_in, D_out, E_out):

    # Sanity check that total number of states is preserved
    Q_tot_in  = np.sum(D_in )*getEnergyResolution(E_in )
    Q_tot_out = np.sum(D_out)*getEnergyResolution(E_out)
    Q_tot_err = (Q_tot_out-Q_tot_in)/Q_tot_in*100
    if abs(Q_tot_err) > 0.5 :  # Accept rel. err. less than 0.5%
        err_out("Inputs to CNL computation do not satisfy Sum Rule: Rel Err = " + str(Q_tot_err))
    
    # Compute the functions Q_in(E_in) and Q_out(E_out),
    # the total charges at zero temperature as a function
    # of the Fermi level E_{in, out}.
    Q_in  = scipy.integrate.cumulative_trapezoid(D_in , x = E_in , initial = 0)
    Q_out = scipy.integrate.cumulative_trapezoid(D_out, x = E_out, initial = 0)
    
    # Compute Q(CNL_in), the total amount of electronic charge
    # in the original ("in") material.
    Q = scipy.interpolate.CubicSpline(E_in, Q_in)(CNL_in)
    
    # Compute CNL_out, the charge neutrality level of the output
    # density of states. Again use spline interpolation but
    # "invert" it simply by using E(Q) instead of Q(E).
    CNL_out = scipy.interpolate.CubicSpline(Q_out, E_out)(Q)
    
    return CNL_out


# Get analytical approximation to MIGS density and CNL in bandgap.
#
# Inputs:
# - E_vals : Array of energy values (evenly spaced; strictly increasing)
#            of the "S" material (i.e., semiconductor with NO metal on top).
# - D_vals : Array of DoS values corresponding to E_vals (of the "S" material)
# - G_vals : Array of Lorentzian widths corresponding to the metal-semiconductor
#           interaction strength, as a function of energy E_vals.
# - E_V    : Energy of valence band maximum of "S"
# - E_C    : Energy of conduction band minimum of "S"
# 
# Outputs:
# - E_out  : Array of energy values in the bandgap 
# - D_out  : Estimated MIGS density in the bandgap
# - CNL_est: Estimated CNL based on the MIGS density
def getEstMIGSDoS(E_vals, D_vals, G_vals, E_V, E_C):
    
    if E_V < np.min(E_vals) :
        err_out("Valence band maximum energy out-of-range")
    if E_C > np.max(E_vals) :
        err_out("Conduction band minimum energy out-of-range")
    if E_V >= E_C :
        err_out ("Conduction band minimum energy must exceed valence band maximum energy")
    
    # Estimate the density of states of "X",
    # the semiconductor with metal on top
    D_X, E_X = getNonUnifConv(D_vals, E_vals, G_vals, "Lorentzian")
    
    # Estimate the "averaged" density of states of the
    # valence and conduction bands
    # TODO jqin: think more carefully about this part!!
    D_spline = scipy.interpolate.CubicSpline(E_X, D_X)
    D_V = D_spline(E_V)
    D_C = D_spline(E_C)
    
    # TODO jqin: also think more carefully about this part!!
    #            Should there be some averaging of G as well?
    G_spline = scipy.interpolate.CubicSpline(E_vals, G_vals)
    G_V = G_spline(E_V)
    G_C = G_spline(E_C)
    
    # Inner function to estimate MIGS DoS
    def getMIGS_Analytical(E) :
        if (E < E_V) or (E > E_C) :
            err_out("Analytical MIGS formula only holds inside bandgap")
        migs_C = D_C*(np.arctan(G_C/(E_C - E)) if E != E_C else np.pi/2)
        migs_V = D_V*(np.arctan(G_V/(E - E_V)) if E != E_V else np.pi/2)
        return (migs_C + migs_V) / np.pi
        
    dE = getEnergyResolution(E_vals)
    dE = dE * 0.3  # Use better resolution for analytical MIGS
    
    # Compute estimated analytical MIGS density in bandgap [E_V, E_C]
    E_out = np.arange(E_V, E_C, dE)
    D_out = np.asarray([getMIGS_Analytical(E) for E in E_out])
    
    # Inner function to estimate CNL
    def getCNL_Analytical() :
        migs_ratio = (D_V*G_V) / (D_C*G_C)
        S_CNL      = 1.0 / (1.0 + np.power(migs_ratio, 0.667))
        CNL        = S_CNL*E_V + (1-S_CNL)*E_C
        return CNL
    
    CNL_out = getCNL_Analytical()
    
    return E_out, D_out, CNL_out
    
    
####################################
# SCHOTTKY BARRIER COMPUTATION
####################################


# Compute an analytical estimate of the
# Fermi-level pinning factor
#
# Inputs:
# - D_vals: Density of states of X [eV^(-1) nm^(-2)]
# - E_vals: Energy values corresponding to D_vals [eV]
# - d     : Dipole 'bond' thickness [nm]
# - eps   : Relative permittivity [unitless]
# - E     : Energy at which to estimate pinning factor
#
# Outputs:
# - S     : Estimated pinning factor at energy E
def getEstPinFactor(D_vals, E_vals, d, eps, E):
    
    # Density of states [eV^(-1) nm^(-2)]
    D = np.interp(E, E_vals, D_vals)
    
    # Compute the effect of DOS
    EPS_NM = eps*EPS_0_NATURAL*1e-3 # [e^2 eV^(-1) nm^(-1)]
    n = d*D/EPS_NM
    
    # Compute pinning factor
    S = 1.0/(1.0+n)
    return S
    
    
# Returns the energy level of the Schottky barrier
# for a semiconductor in contact with metal.
#
# In other words, returns CNL + dE, where dE is
# the energy shift of the Fermi level of the 
# semiconductor, relative to the CNL.
#
# Essentially, solves the equation
#
#   E_FM - CNL = dE + (d/eps) \int_{CNL}^{CNL+dE} dE' \rho_S(E').
#
# Inputs:
# - D_vals: Density of states of the metal [eV^(-1) nm^(-2)]
# - E_vals: Corresponding energy levels [eV]
# - CNL   : Charge neutrality level of the semiconductor [eV]
# - E_FM  : Fermi level of the metal [eV]
# - d     : Dipole 'bond' thickness [nm]
# - eps   : Relative permittivity [unitless]
#
# Outputs:
# - E_Sch : Energy level of the Schottky barrier. 
#           This is not the same as the Schottky barrier height.
#           The n-,p-type Schottky barrier heights are E_C - E_Sch
#           and E_Sch - E_V, respectively.
def getSchottkyEnergy(D_vals, E_vals, CNL, E_FM, d, eps):
    
    # Compute \int_{CNL}^{CNL+E} dE' \rho_S(E') as a function of E
    Q = scipy.integrate.cumulative_trapezoid(D_vals, x = E_vals, initial = 0)
    Q = Q - np.interp(CNL, E_vals, Q)
    
    # Compute E + (d/eps) \int_{CNL}^{CNL+E} dE' \rho_S(E') as a function of E
    EPS_NM = eps*EPS_0_NATURAL*1e-3 # [e^2 eV^(-1) nm^(-1)]
    dE_tot = (E_vals - CNL) + d/EPS_NM * Q
    
    # Compute the energy at which dE_tot = E_FM - CNL
    E_Sch = np.interp(E_FM-CNL, dE_tot, E_vals)
    
    return E_Sch


####################################
# CONTACT RESISTANCE COMPUTATION
####################################


# Compute the quantum resistance of
# electron transport in a material
# given its density of states,
# energy-dependent transport velocity,
# and Fermi energy.
#
# Inputs:
# - vX_vals: Transport velocity as a function
#            of energy [m / s]
# - D_vals : Density of states [eV^(-1) nm^(-2)]
# - E_vals : Energy values corresponding to vX_vals
#            and D_vals [eV]
# - E_F    : Fermi energy [eV]
# - kT     : Thermal voltage [eV]
#
# Output:
# - Quantum resistance [Ohm*um]
def getRQ(vX_vals, D_vals, E_vals, E_F, kT):
    
    vX_in = vX_vals*1e6  # Convert vX from [m/s] to [um/s]
    D_in  = D_vals *1e6  # Convert D  from [eV^(-1) nm^(-2)] to [eV^(-1) um^(-2)]
    
    # Compute quantum conductance
    G_Q = 0.5*ELECTRON_CHARGE*getThermalExpectation(vX_in, E_vals, kT, E_F, D_vals = D_in)
    
    # Return quantum resistance
    return 1.0/G_Q


# Compute the quantum resistance of
# electron transport in a material
# given the number of transport modes
# as a function of energy,
# and Fermi energy.
#
# Inputs:
# - M_vals : Number of transport modes as function of energy
# - E_vals : Energy values corresponding to vX_vals
#            and D_vals [eV]
# - E_F    : Fermi energy [eV]
# - kT     : Thermal voltage [eV]
#
# Output:
# - Quantum resistance [Ohm*?] (units depend on # transport modes and dimension of resistor)
def getRQfromModes(M_vals, E_vals, E_F, kT):
    
    # Compute quantum conductance
    G_Q = getThermalExpectation(M_vals, E_vals, kT, E_F, D_vals = None)
    G_Q = 2*ELECTRON_CHARGE**2/PLANCK_CONSTANT*G_Q 
    
    # Return quantum resistance
    return 1.0/G_Q

# Compute the specific contact conductivity
# of a metal-semiconductor interface.
# 
# Inputs:
# - G_vals: Amount of broadening as a function of energy [eV].
#           May be a NumPy array or a constant float.
# - D_vals: Density of states [eV^(-1) nm^(-2)]
# - E_vals: Energy values [eV]
# - E_F   : Fermi energy [eV]
# - kT    : Thermal voltage [eV]
#
# Output:
# - Specific contact conductivity [Ohm^(-1) um^(-2)]
def getGC(G_vals, D_vals, E_vals, E_F, kT):
    
    # Convert broadening from energy [eV] to hopping rate [s^(-1)]
    G_in = None
    if   isinstance(G_vals, (int, float)):
        G_in = G_vals*np.ones(D_vals.shape[0])
    elif isinstance(G_vals, np.ndarray):
        G_in = G_vals
    G_in = G_vals * ELECTRON_CHARGE / PLANCK_CONSTANT
    
    D_in = D_vals*1e6  # Convert D  from [eV^(-1) nm^(-2)] to [eV^(-1) um^(-2)]
    
    # Return specific conductivity
    return 0.5*ELECTRON_CHARGE*getThermalExpectation(G_in, E_vals, kT, E_F, D_vals = D_in)
    

# Compute the "alpha" parameter of Solomon's R_cf model.
#
# Inputs:
# - Rsh: Sheet   resistance [Ohm/sq]
# - RQ : Quantum resistance [Ohm*um]
# 
# Output:
# - Alpha: Rsh/RQ [um^(-1)]
def getAlpha(Rsh, RQ):
    return Rsh/RQ
    
    
# Compute the "beta" parameter of Solomon's R_cf model.
#
# Inputs:
# - gC: Specific contact conductivity [Ohm^(-1) um^(-2)]
# - RQ: Quantum resistance            [Ohm      um     ]
#
# Output:
# - Beta: gC*RQ/2 [um^(-1)]
def getBeta(gC, RQ):
    return gC*RQ/2
    

# Compute the transfer length Lt of Solomon's R_cf model.
# 
# Inputs:
# - Alpha: inverse-length parameter for diffusive transport [um^(-1)]
# - Beta : inverse-length parameter for ballistic transport [um^(-1)]
#
# Output:
# - Lt   : Transfer length [um]
def getTransferLength(Alpha, Beta):
    return np.power(2*Alpha*Beta + Beta*Beta, -0.5)
    # For CNT, Lt = 1/Beta
    
    
# Compute R_cf, the distributed part of contact resistance,
# from Solomon's model.
#
# Inputs:
# - RQ : Quantum resistance [Ohm*um]
# - gC : Specific contact conductivity [Ohm^(-1) um^(-2)]
# - Rsh: Sheet resistance   [Ohm/sq]
# - Lc : Contact length     [um].
#        Contact length may be either float or array.
def getR_cf(RQ, gC, Rsh, Lc):
    Alpha = getAlpha(Rsh, RQ)  # [um^(-1)]
    Beta  = getBeta (gC , RQ)  # [um^(-1)]
    Lt    = getTransferLength(Alpha, Beta)  # [um]
    R     = RQ / (2*Beta*Lt)
    return R/np.tanh(Lc/Lt)
    
