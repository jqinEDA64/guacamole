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
#           without normalizing for the thermal average of D itself.
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
    Q_tot_in  = np.sum(D_in )
    Q_tot_out = np.sum(D_out)
    Q_tot_err = (Q_tot_out-Q_tot_in)/Q_tot_in*100
    if abs(Q_tot_err) > 0.5 :  # Accept rel. err. less than 0.5%
        err_out("Inputs to CNL computation do not satisfy Sum Rule")
    
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


# Compute the specific contact resistivity
# of a metal-semiconductor interface.
# TODO jqin
    
