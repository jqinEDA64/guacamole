import numpy as np
import matplotlib.pyplot as plt


# Define numerical constants and conversion factors
Q_CONST   = 1.6e-19        # [C]
EPS_CONST = 8.85e-12       # [F/m]
EPS_NATURAL_CONST = 55.3   # [e^2 eV^(-1) um^(-1)]
EPS_HA_A_CONST = 0.149     # [e^2 Ha^(-1) A^(-1)]
EV_HARTREE_CONST = 27.2    # [eV/Ha]
ANGSTROM_BOHR_CONST = 0.53 # [a0/A]
EV_J_CONST = Q_CONST       # [J/eV]
PI = np.pi                 # []


# Interface class. Generic class which represents metal-semiconductor interfaces.
# Computes quantities of interest, such as vdW gap distance.
#
# NOTE: in this class, the following units are used:
# - Length: Angstrom [A]
# - Energy: Hartree  [Ha]
class Interface :
    
    # Constructor for metal-semiconductor interface
    # Here, k_m and k_s (wavefunction decay constants) are given in A^(-1)
    # rho_m and rho_s (metal and semiconductor valence electron densities) are given in A^(-3)
    # 
    # Following convention, C6_m and C6_s are given in [Ha a0^6] instead of [Ha A^6].
    # Likewise, alpha_m and alpha_s are given in [a0^3] instead of [A^3].
    # The use of Bohr instead of Angstrom in the vdW-related quantities does not matter because the length unit
    # is divided out when computing the Hamaker constant A_ms.
    def __init__(self, k_m, k_s, rho_m, rho_s, C6_m, C6_s, alpha_m, alpha_s):
        self.k_m = k_m                                          # Decay constant in metal [A^(-1)]
        self.k_s = k_s                                          # Decay constant in semiconductor [A^(-1)]
        self.sigma_m = rho_m/(2*k_m)                            # Surface charge on metal (no q multiplier) [A^(-2)]
        self.sigma_s = rho_s/(2*k_s)                            # Surface charge on semiconductor (no q multiplier) [A^(-2)]
        self.A_ms    = 0.58*np.sqrt(C6_m*C6_s)/alpha_m/alpha_s  # Hamaker constant [Ha]

    def checkDistSign(self, d):
        if d <= 0 :
            print("ERROR: Negative distance")
            quit()

    # Compute vdW attraction energy at distance d [A]
    # Output: energy per area [Ha/A^2]
    def getVDWEnergy(self, d, damp = True):
        self.checkDistSign(d)
        vdw = -self.A_ms/(12*PI*d*d)
        if not damp :
            return vdw
        damp_dist = 0.2*(1/self.k_m + 1/self.k_s) # TODO what should damp_dist be?
        damp_rate = 20
        damp_denm = 1 + np.exp(damp_rate*(damp_dist-d))
        return vdw/damp_denm  # Add Fermi function to damp the vdW force
    
    # Compute Pauli repulsion energy at distance d [A]
    # Output: energy per area [Ha/A^2]
    def getPauliEnergy(self, d):
        self.checkDistSign(d)
        k_avg = 0.5*(self.k_m + self.k_s)
        S = 2*np.sqrt(self.k_m*self.k_s)*d*np.exp(-k_avg*d)
        return (self.sigma_m + self.sigma_s)*np.sqrt(self.sigma_m*self.sigma_s)*d*S*S/EPS_HA_A_CONST
    
    # Compute total energy at distance d [A]
    # Output: energy per area [Ha/A^2]
    def getEnergy(self, d):
        self.checkDistSign(d)
        return self.getVDWEnergy(d) + self.getPauliEnergy(d)
    
    # Compute derivative of energy at distance d [A]
    # Output: derivative of energy per area [Ha/A^3]
    def getEnergyDerivative(self, d):
        d_eps = 0.01
        return (self.getEnergy(d+0.5*d_eps)-self.getEnergy(d-0.5*d_eps))/d_eps
    
    # Compute vdW gap (arg min of getEnergy()) and return the value in [A]
    # Use Newton's method for minimization
    def getVDWGap(self, tol = 1e-4):

        d_curr = 10*(1/self.k_m + 1/self.k_s)  # Select something large (getPauliEnergy() = 0)
        d_diff = 1e5                           # Relative difference in bond length
        eps    = 1e-8                          # Epsilon to avoid divide-by-zero
        
        while np.abs(d_diff) > tol:
            d_old  = d_curr
            d_curr = d_curr - self.getEnergyDerivative(d_curr)/(np.abs(self.getEnergy(d_curr)) + eps)
            d_diff = 0.5*(d_curr - d_old) / (d_curr + d_old)

        return d_curr
    
    # Plot the interface energy per area as a function of separation
    def plotEnergy(self):
        x_vals = np.linspace(1.5, 5, 50)
        y_VDW = [self.getVDWEnergy(d, True) for d in x_vals]
        y_Pau = [self.getPauliEnergy(d) for d in x_vals]
        y_Tot = [self.getEnergy(d) for d in x_vals]
        plt.plot(x_vals, y_VDW)
        plt.plot(x_vals, y_Pau)
        plt.plot(x_vals, y_Tot)
        plt.show()
    
MS_int = Interface(1,2,0.08,0.005,520,48,558,12)
MS_int.plotEnergy()
d_vdw = MS_int.getVDWGap()
print(d_vdw)

# TODO try graphene-graphene layers