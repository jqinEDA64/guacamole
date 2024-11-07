import numpy as np
import matplotlib.pyplot as plt


# Define numerical constants and conversion factors
PI                  = np.pi      # []
Q_CONST             = 1.6e-19    # [C]
EPS_CONST           = 8.85e-12   # [F/m]
EPS_NATURAL_CONST   = 55.3       # [e^2 eV^(-1) um^(-1)]
EPS_HA_A_CONST      = 0.149      # [e^2 Ha^(-1) A^(-1)]
EV_HARTREE_CONST    = 27.2       # [eV/Ha]
ANGSTROM_BOHR_CONST = 1.89       # [a0/A]
EV_J_CONST          = Q_CONST    # [J/eV]
PLANCK_CONST        = 6.63e-34   # [J s]
RED_PLANCK_CONST    = \
    PLANCK_CONST/(2*PI)          # [J s]
ELECTRON_MASS       = 9.11e-31   # [kg]


class Surface :
    # Constructor for surface of a metal or semiconductor or insulator
    # - k (wavefunction decay constant) is given in [A^(-1)]
    # - Sigma (valence electron density per area) are given in [A^(-2)]
    # 
    # Following convention, 
    # - C6 is given in [Ha a0^6] instead of [Ha A^6].
    # - Static atomic polarizability alpha is given in [a0^3] instead of [A^3].
    #
    # The use of Bohr instead of Angstrom in the vdW-related quantities does not matter because the length unit
    # is divided out when computing the Hamaker constant A_ms.
    def __init__(self, k, sigma, C6, alpha, name):
        self.k     = k
        self.sigma = sigma
        self.C6    = C6
        self.alpha = alpha
        self.name  = name
        self.Na    = 0      # Per-area density of atoms on the surface [A^(-2)]

    # Assuming the metal wavefunction goes as exp(-kz) outside the metal
    # Compute the decay constant, k, in terms of the 
    # - Metal's workfunction [eV]
    # - Electron effective mass [me]
    # - Carrier density [A^(-3)]
    @staticmethod
    def compMetalDecayConst(W, m, rho):

        c = 2*m*ELECTRON_MASS*Q_CONST/RED_PLANCK_CONST**2  # Standard SI units

        # Simplest model first
        W_SI   = W                                 # Workfunction in [V]
        rho_SI = rho*1e30                          # Carrier density in [m^(-3)]
        mWhSq  = 0.5*c*W_SI                        # mW/hbar^2 in SI units

        K_sq   = mWhSq*(1 + np.sqrt(1 + 2*W_SI**(-1)*mWhSq**(-1)*3*rho_SI*Q_CONST/(8*EPS_CONST)))
        k      = 1e-10*np.sqrt(K_sq)
        return k

    # Factory method which creates a metal surface. The inputs are
    # - Workfunction, W [eV]
    # - Carrier effective mass, m [me]
    # - Carrier density, rho [A^(-3)]
    # - Atomic C6 coefficient, C6 [Ha a0^6]
    # - Atomic polarizability, alpha [a0^3]
    @staticmethod
    def makeMetalSurface(W, m, rho, C6, alpha, name):
        k = Surface.compMetalDecayConst(W, m, rho)
        sigma = rho/(2*k)
        return Surface(k, sigma, C6, alpha, name)
    
    # Factory method which creates a metal surface from its
    # density, molar mass, resistivity, and valence. The inputs are
    # - Workfunction, W [eV]
    # - 
    # TODO
    
    # Factory method which creates a low-D material surface. The inputs are
    # - Wavefunction decay constant, k [A^(-1)]
    # - Valence charge density per area, sigma [A^(-2)]
    # - Atomic C6 coefficient, C6 [Ha a0^6]
    # - Atomic polarizability, alpha [a0^3]
    @staticmethod
    def make2DSurface(k, sigma, C6, alpha, name):
        return Surface(k, sigma, C6, alpha, name)

    # Prints the value of k (wavefunction decay constant) and its inverse k^(-1).
    def printDecayConst(self):
        k = self.k
        print(self.name + " has k = " + str(round(k,2)) + " [A^(-1)] or k^(-1) = " + str(round(k**(-1),2)) + " [A]")

    # Prints relative value of vdW attraction strength. 
    # Essentially C6 divided by free-atom volume.
    def printVDWStrength(self):
        C6 = self.C6
        a  = self.alpha
        print(self.name + " has normalized C6/(a0^2) = " + str(round(C6/a/a,2)) + " [Ha]")

    # Set the area density of atoms on the surface of this solid.
    # - Area density of atoms on surface, Na [A^(-2)]
    def setAreaDensity(self, Na):
        self.Na = Na

# Interface class. Generic class which represents metal-semiconductor interfaces.
# Computes quantities of interest, such as vdW gap distance.
class Interface :
    
    # Constructor for vdW interface between two materials
    # Pass two Surface objects, s1 and s2, to the constructor.
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2
        self.A_ms = 0.58*np.sqrt(self.s1.C6*self.s2.C6)/(self.s1.alpha*self.s2.alpha)
        #self.A_ms = PI**2*self.s1.Na**(3/2)*self.s2.Na**(3/2)*ANGSTROM_BOHR_CONST**(-6)*np.sqrt(self.s1.C6*self.s2.C6)

    # Factory method to create an interface, eliminating the dependence on decay constants
    # (which are estimated from the static polarizabilities)
    @staticmethod
    def makeInterface(s1, s2):
        return Interface(s1, s2)

    def checkDistSign(self, d):
        if d <= 0 :
            print("ERROR: Negative distance")
            quit()

    # Compute vdW attraction energy at distance d [A]
    # Output: energy per area [Ha/A^2]
    def getVDWEnergy(self, d, damp = False):
        self.checkDistSign(d)

        ##################################
        # Temporary vdW energy for testing
        ##################################
        #C6 = np.sqrt(self.s1.C6*self.s2.C6)
        #C  = C6/(self.s1.alpha*self.s2.alpha)**0.333
        #return -C*ANGSTROM_BOHR_CONST**(-4)/np.sqrt(d**2 + 1)**6

        k1 = self.s1.k
        k2 = self.s2.k

        vdw = -self.A_ms/(12*PI*d*d)
        if not damp :
            return vdw
        damp_dist = 0.2*(1/k1 + 1/k2) # TODO what should damp_dist be?
        damp_rate = 20
        damp_denm = 1 + np.exp(damp_rate*(damp_dist-d))
        return vdw/damp_denm  # Add Fermi function to damp the vdW force
    
    # Compute wavefunction overlap at distance d [A]
    # Output wavefunction overlap []
    def getOverlap(self, d):

        k1 = self.s1.k
        k2 = self.s2.k

        k_avg = 0.5*(k1 + k2)
        #S = np.exp(-k_avg*d)*d
        S = 0
        if k1 == k2 :
            S = np.exp(-k_avg*d)*d
        else :
            S = np.exp(-k_avg*d)*2*np.sinh((k1-k2)*d/2)/(k1-k2)
        return 2*np.sqrt(k1*k2)*S

    # Compute Pauli repulsion energy at distance d [A]
    # Output: energy per area [Ha/A^2]
    def getPauliEnergy(self, d):
        self.checkDistSign(d)
        S = self.getOverlap(d)

        sigma1 = self.s1.sigma
        sigma2 = self.s2.sigma

        return (sigma1 + sigma2)*np.sqrt(sigma1*sigma2)*d*S*S/EPS_HA_A_CONST
    
    # Compute total energy at distance d [A]
    # Output: energy per area [Ha/A^2]
    def getEnergy(self, d):
        self.checkDistSign(d)
        return self.getVDWEnergy(d) + self.getPauliEnergy(d)
    
    # Compute vdW gap (arg min of getEnergy()) and return the value in [A]
    def getVDWGap(self, tol = 1e-3):

        k1 = self.s1.k
        k2 = self.s2.k
        d_curr = 10*(1/k1 + 1/k2)  # Select something large (getPauliEnergy() = 0)

        N = (int)(d_curr/tol)
        x_vals = np.linspace(d_curr, tol, N)          # Note d_curr >> tol (x_vals is in decreasing order).
        y_vals = [self.getEnergy(x) for x in x_vals]

        # Naive, brute-force method which computes the first minimum obtained 
        # when iterating over decreasing distance, d.
        i = 1
        while i < N :
            if y_vals[i] < y_vals[i+1] and y_vals[i] < y_vals[i-1] :
                return x_vals[i]
            i = i + 1
        
        print("ERROR: No optimum distance found")
        quit()

        return 0
    
    # Plot the interface energy per area as a function of separation
    def _plotEnergy(self, d_min, d_max, d_opt):

        x_vals = np.linspace(d_min, d_max, 100)

        # If the area density of atoms on the second surface is known,
        # report the binding energy per atom on second surface
        isDensityKnown = self.s2.Na > 0
        prefactor = 1000*EV_HARTREE_CONST/self.s2.Na if isDensityKnown else 1
        y_VDW = [prefactor*self.getVDWEnergy(d, True) for d in x_vals]
        y_Pau = [prefactor*self.getPauliEnergy(d) for d in x_vals]
        y_Tot = [prefactor*self.getEnergy(d) for d in x_vals]

        plt.plot(x_vals, y_VDW, color = "black", ls = "--", label = "vdW Energy")
        plt.plot(x_vals, y_Pau, color = "black", ls = "-.", label = "Pauli Energy")
        plt.plot(x_vals, y_Tot, color = "black", ls = "-" , label = "Total Energy")
        plt.xlabel("Interface separation [$\\AA$]")

        if isDensityKnown :
            plt.ylabel("$E_b$ per atom on " + self.s2.name + " surface [meV]")
        else:
            plt.ylabel("Interfacial energy [Ha/$\\AA$^2]")

        # Plot the optimal point
        if d_opt > d_min and d_opt < d_max :
            E = prefactor*self.getEnergy(d_opt)
            label = "$d^*=$" + str(round(d_opt,2)) + " $\\AA$"
            if isDensityKnown :
                label += "\n$E_b=$" + str(round(E,2)) + " meV"
            plt.scatter(d_opt, E, color = "red", zorder = 100, label = label)

        return plt

    def plotEnergy(self, extratext = ""):
        d_VDW = self.getVDWGap()
        plt = self._plotEnergy(d_VDW*0.7, d_VDW*2, d_VDW)
        plt.legend()

        title = "Interface energy vs interface separation"
        if len(extratext) :
            title += ":\n" + extratext
        else :
            title += ":\n" + self.s1.name + " on " + self.s2.name
        plt.title(title)
        plt.show()

    