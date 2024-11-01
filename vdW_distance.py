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

    # Approximate method to estimate the wavefunction decy constants
    # from the static polarizabilities
    # Input polarizability "alpha" in Bohr^3.
    # Output decay constant "k" in A^(-1).
    @staticmethod
    def getDecayFromPolarizability(alpha):
        return 7.02*np.power(alpha, -0.333)

    # Factory method to create an interface, eliminating the dependence on decay constants
    # (which are estimated from the static polarizabilities)
    @staticmethod
    def makeInterface(rho_m, rho_s, C6_m, C6_s, alpha_m, alpha_s):
        k_m = Interface.getDecayFromPolarizability(alpha_m)
        k_s = Interface.getDecayFromPolarizability(alpha_s)
        return Interface(k_m, k_s, rho_m, rho_s, C6_m, C6_s, alpha_m, alpha_s)

    def checkDistSign(self, d):
        if d <= 0 :
            print("ERROR: Negative distance")
            quit()

    # Compute vdW attraction energy at distance d [A]
    # Output: energy per area [Ha/A^2]
    def getVDWEnergy(self, d, damp = False):
        self.checkDistSign(d)
        vdw = -self.A_ms/(12*PI*d*d)
        if not damp :
            return vdw
        damp_dist = 0.2*(1/self.k_m + 1/self.k_s) # TODO what should damp_dist be?
        damp_rate = 20
        damp_denm = 1 + np.exp(damp_rate*(damp_dist-d))
        return vdw/damp_denm  # Add Fermi function to damp the vdW force
    
    # Compute wavefunction overlap at distance d [A]
    # Output wavefunction overlap []
    def getOverlap(self, d):
        k_avg = 0.5*(self.k_m + self.k_s)
        S = 0
        if self.k_m == self.k_s :
            S = np.exp(-k_avg*d)*d
        else :
            S = np.exp(-k_avg*d)*2*np.sinh((self.k_m-self.k_s)*d/2)/(self.k_m-self.k_s)
        return 2*np.sqrt(self.k_m*self.k_s)*S

    # Compute Pauli repulsion energy at distance d [A]
    # Output: energy per area [Ha/A^2]
    def getPauliEnergy(self, d):
        self.checkDistSign(d)
        S = self.getOverlap(d)
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
    def getVDWGap(self, tol = 1e-3):

        d_curr = 10*(1/self.k_m + 1/self.k_s)  # Select something large (getPauliEnergy() = 0)

        if self.getEnergyDerivative(d_curr) <= 0 :
            print("ERROR: Energy derivative should be positive for initialized d")
            quit()

        N = (int)(d_curr/tol)
        x_vals = np.linspace(d_curr, tol, N)
        y_vals = [self.getEnergy(x) for x in x_vals]

        i = 1
        while i < N :
            if y_vals[i] < y_vals[i+1] and y_vals[i] < y_vals[i-1] :
                return x_vals[i]
            i = i + 1
        
        print("ERROR: No optimum distance found")
        quit()

        return 0
    
    # Plot the interface energy per area as a function of separation
    def __plotEnergy(self, d_min, d_max):
        x_vals = np.linspace(d_min, d_max, 100)
        y_VDW = [self.getVDWEnergy(d, True) for d in x_vals]
        y_Pau = [self.getPauliEnergy(d) for d in x_vals]
        y_Tot = [self.getEnergy(d) for d in x_vals]
        plt.plot(x_vals, y_VDW, color = "black", ls = "--", label = "vdW Energy")
        plt.plot(x_vals, y_Pau, color = "black", ls = "-.", label = "Pauli Energy")
        plt.plot(x_vals, y_Tot, color = "black", ls = "-" , label = "Total Energy")
        plt.xlabel("Interface separation [$\\AA$]")
        plt.ylabel("Interfacial energy [Ha/$\\AA$^2]")
        return plt

    def plotEnergy(self, extratext = ""):
        d_VDW = self.getVDWGap()
        plt = self.__plotEnergy(d_VDW*0.7, d_VDW*2)
        label = "$d^*=$" + str(round(d_VDW,2)) + " $\\AA$"
        plt.scatter(d_VDW, self.getEnergy(d_VDW), color = "red", zorder = 100, label = label)
        plt.legend()

        title = "Interface energy vs interface separation"
        if len(extratext) :
            title += ":\n" + extratext
        plt.title(title)
        plt.show()

# AtomAtom class. Generic class which represents vdW interaction between atoms.
# Computes quantities of interest, such as vdW gap distance.
#
# NOTE: in this class, the following units are used:
# - Length: Angstrom [A]
# - Energy: Hartree  [Ha]
#
# We still use "m" and "s" to refer to the different atoms, although this may
# not be "metal" or "semiconductor" atoms.
class AtomAtom(Interface) :
    
    # Constructor for metal-semiconductor interface
    # Here, k_m and k_s (wavefunction decay constants) are given in A^(-1)
    # rho_m and rho_s (metal and semiconductor valence electron densities) are given in A^(-3)
    # 
    # Following convention, C6_m and C6_s are given in [Ha a0^6] instead of [Ha A^6].
    # Likewise, alpha_m and alpha_s are given in [a0^3] instead of [A^3].
    # The use of Bohr instead of Angstrom in the vdW-related quantities does not matter because the length unit
    # is divided out when computing the Hamaker constant A_ms.
    def __init__(self, k_m, k_s, rho_m, rho_s, C6_m, C6_s, alpha_m, alpha_s):
        super().__init__(self, k_m, k_s, rho_m, rho_s, C6_m, C6_s, alpha_m, alpha_s)

    # Compute vdW attraction energy at distance d [A]
    # Output: energy per area [Ha/A^2]
    def getVDWEnergy(self, d, damp = False):
        # TODO change this!!
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
        # TODO change this!
        self.checkDistSign(d)
        k_avg = 0.5*(self.k_m + self.k_s)
        S = 2*np.sqrt(self.k_m*self.k_s)*d*np.exp(-k_avg*d)
        return (self.sigma_m + self.sigma_s)*np.sqrt(self.sigma_m*self.sigma_s)*d*S*S/EPS_HA_A_CONST
    
    # Plot the interface energy per area as a function of separation
    def __plotEnergy(self, d_min, d_max):
        x_vals = np.linspace(d_min, d_max, 100)
        y_VDW = [self.getVDWEnergy(d, True) for d in x_vals]
        y_Pau = [self.getPauliEnergy(d) for d in x_vals]
        y_Tot = [self.getEnergy(d) for d in x_vals]
        plt.plot(x_vals, y_VDW, color = "black", ls = "--", label = "vdW Energy")
        plt.plot(x_vals, y_Pau, color = "black", ls = "-.", label = "Pauli Energy")
        plt.plot(x_vals, y_Tot, color = "black", ls = "-" , label = "Total Energy")
        plt.xlabel("Inter-atom separation [$\\AA$]")
        plt.ylabel("Interaction energy [Ha/$\\AA$^2]")
        return plt

    def plotEnergy(self, extratext = ""):
        d_VDW = self.getVDWGap()
        plt = self.__plotEnergy(d_VDW*0.7, d_VDW*2)
        label = "$d^*=$" + str(round(d_VDW,2)) + " $\\AA$"
        plt.scatter(d_VDW, self.getEnergy(d_VDW), color = "red", zorder = 100, label = label)
        plt.legend()

        title = "Atom-atom energy vs inter-atom separation"
        if len(extratext) :
            title += ":\n" + extratext
        plt.title(title)
        plt.show()