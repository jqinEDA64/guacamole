import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

 
# NOTE: Original MATLAB code from https://lampz.tugraz.at/~hadley/ss1/bands/tbtable/cnt_files/cnts.html?

# Returns the bandstructure of the CNT.
#
# Inputs:
# - n:       First  chiral number
# - m:       Second chiral number
# - gamma:   Nearest-neighbor hopping energy
# - k_steps: Number of discretization points in k-space
# 
# Outputs:
# - k_vals:  Wavevector values [nm^(-1)]
# - E_p   :  Positive-energy (n-type) energy values
# - E_n   :  Negative-energy (p-type) energy values
def getCNT_bandstructure(n, m, gamma, k_steps) :
    a_1 = np.array([0.5, np.sqrt(3)/2])
    a_2 = np.array([-0.5, np.sqrt(3)/2])

    b_1 = np.array([2*np.pi, 2*np.pi/np.sqrt(3)])
    b_2 = np.array([-2*np.pi, 2*np.pi/np.sqrt(3)])

    d_r = math.gcd(2*n+m, n+2*m)
    N_CNT = 2*(n**2 + m**2 + n*m) / d_r
    N_CNT = (int)(N_CNT)

    t_1 = (2*m+n) / d_r
    t_2 = -(2*n+m)/ d_r
    T   = t_1*a_1 + t_2*a_2

    K_1 = (1/N_CNT) * (-t_2*b_1 + t_1*b_2)
    K_2 = (1/N_CNT) * (m   *b_1 - n  *b_2)
    K_1_norm = np.linalg.norm(K_1)
    K_2_norm = np.linalg.norm(K_2)

    l_2 = np.linspace(-np.pi/np.linalg.norm(T), np.pi/np.linalg.norm(T), k_steps)
    E_p = np.zeros((N_CNT, k_steps))
    E_n = np.zeros((N_CNT, k_steps))

    for l_1 in range(0, N_CNT) :

        k_x = l_1*K_1[0] + l_2*K_2[0]/K_2_norm
        k_y = l_1*K_1[1] + l_2*K_2[1]/K_2_norm
        
        val = gamma*np.sqrt(1+4*np.cos(np.sqrt(3)*k_y/2) * np.cos(k_x/2) + 4*np.cos(k_x/2)**2)
        E_p[l_1, :] =  val
        E_n[l_1, :] = -val
    
    return (1.0/0.246)*l_2, E_p, E_n


# Returns the effective mass of the CNT.
# See https://doi.org/10.1109/ICICDT.2007.4299580 
#
# m = \hbar^2 / (d^2E / dk^2)
#
# Inputs:
# - n:       First  chiral number
# - m:       Second chiral number
# - gamma:   Nearest-neighbor hopping energy
# 
# Outputs:
# - m:       CNT effective mass (in units of m_e)
def getCNT_effectivemass(n, m, gamma) :

    k_steps = int(1e5)
    k_vals, E_p, E_n = getCNT_bandstructure(n, m, gamma, k_steps)

    hbar = 6.58e-16             # Reduced Planck constant [eV s]
    dk   = k_vals[1]-k_vals[0]  # Wavevector spacing [nm^(-1)]

    E_p    = E_p.flatten()
    i      = np.argmin(E_p)
    d2E_dk2= (E_p[i-1]+E_p[i+1]-2*E_p[i])/(dk*dk)  # [eV m^2]
    m      = hbar*hbar/d2E_dk2  # Effective mass [eV s^2 m^(-2)]
    m      = 1.6e-19/9.11e-31*m # Effective mass [m_e]

    return m

# Get bandgap of the CNT, given the bandstructure
#
# Inputs: 
# - E_p, E_n: p- and n-branches of the CNT bandstructure. 
#             Obtained from getCNT_bandstructure().
# 
# Outputs:
# - E_g: Bandgap of the CNT. 
def get_Eg(E_p, E_n) :
    E_p_min = np.min(E_p)
    E_n_max = np.max(E_n)
    E_g     = 0
    if E_p_min > E_n_max :
        E_g = E_p_min-E_n_max

    print("Bandgap = " + str(E_g) + " [eV]")
    return E_g 


# Plot dispersion relation.
#
# Inputs: 
# - k_vals: Momentum values from getCNT_bandstructure().
# - E_p, E_n: Energy values from getCNT_bandstructure().
# - n: First chiral number.
# - m: Second chiral number.
#
# Outputs:
# - None (generates a graph of bandstructure)
def plotE_k(k_vals, E_p, E_n, n, m):
    d_r = math.gcd(2*n+m, n+2*m)
    N_CNT = 2*(n**2 + m**2 + n*m) / d_r
    N_CNT = (int)(N_CNT)

    for l_1 in range(0, N_CNT) :
        plt.plot(k_vals, E_p[l_1, :], color = "red")
        plt.plot(k_vals, E_n[l_1, :], color = "blue")
    plt.xlabel("$k/k_\\text{max}$")
    plt.ylabel("E [eV]")
    plt.title("Bandstructure of (" + str(n) + "," + str(m) + ") CNT\n$\\gamma = $" + str(gamma) + " [eV]")
    plt.show()


# Return the density of states. See Deji Akinwande and Philip Wong's CNT book.
#
# Inputs:
# - n : First  chiral number.
# - m : Second chiral number.
# - t : Hopping energy
# - dE: Desired energy spacing for the density of states
#
# Outputs:
# - E : Energy values for the CNT density of states [eV]
# - D : Density of states for the CNT density of states [eV^(-1) nm^(-1)]
def getCNT_DOS(n, m, t) :

    k_steps = int(1e5)
    Hist_steps = int(k_steps/50)
    k_vals, E_p, E_n = getCNT_bandstructure(n, m, t, k_steps)
    dk = k_vals[1] - k_vals[0]

    data = E_p.flatten()
    data = np.append(data, E_n.flatten())
    
    D, bin_edges = np.histogram(data, bins=Hist_steps)
    dE   = bin_edges[1]-bin_edges[0]
    E    = bin_edges[0:Hist_steps] + dE*0.5

    # Factor of 2   for spin degeneracy
    # Factor of 2Pi for phase space volume correction
    D    = 2 / (2*np.pi) * D * dk / dE

    return E, D
'''
def getCNT_DOS(n, m, t) :

    k_steps = int(1e5)
    Hist_steps = int(k_steps/50)
    k_vals, E_p, E_n = getCNT_bandstructure(n, m, t, k_steps)

    data = E_p.flatten()
    data = np.append(data, E_n.flatten())
    
    D, bin_edges = np.histogram(data, bins=Hist_steps)
    dE   = bin_edges[1]-bin_edges[0]
    E    = bin_edges[0:Hist_steps] + dE*0.5
    D    = 2.46e10 * D * dE

    return E, D
'''


# Plot density of states
'''
Hist_steps = (int)(k_steps/50)
def plotDOS(save = False) :
    data = E_p.flatten()
    data = np.append(data, E_n.flatten())
    D, bin_edges = np.histogram(data, bins=Hist_steps)
    dE   = bin_edges[1]-bin_edges[0]
    E    = bin_edges[0:Hist_steps] + dE*0.5
    D    = 2 * 2.46e10 * D * dE
    plt.plot(E, D)
    plt.xlabel("E [eV]")
    plt.ylabel("Density of states [eV$^{-1}$ m$^{-1}$]")
    plt.title("DOS of (" + str(n) + "," + str(m) + ") CNT\n$\\gamma = $" + str(gamma) + " [eV]")
    plt.show()
    
    if not save :
        return
        
    filepath = "./DoS_CNT_" + str(n) + "_" + str(m) + "_clean.dat"
    df = pd.DataFrame()
    df["E"] = E
    df["DOS"] = D
    
    df.to_csv(filepath, index = False)
    
    print("Wrote DOS to " + filepath)

plotDOS(True)
'''


