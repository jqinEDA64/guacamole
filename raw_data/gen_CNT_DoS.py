import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

 
# NOTE: Original MATLAB code from https://lampz.tugraz.at/~hadley/ss1/bands/tbtable/cnt_files/cnts.html?


k_steps = (int)(2e5)
Hist_steps = (int)(k_steps/50)

# CNT settings here!
n = 13
m = 0

#gamma = 2.74 #(default)

gamma = 3.18

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

# Bandgap
E_p_min = np.min(E_p)
E_n_max = np.max(E_n)
E_g     = 0
if E_p_min > E_n_max :
    E_g = E_p_min-E_n_max
print("Bandgap = " + str(E_g) + " [eV]")


# Plot dispersion relation
def plotE_k():
    k_vals = np.linspace(-1, 1, k_steps)
    for l_1 in range(0, N_CNT) :
        plt.plot(k_vals, E_p[l_1, :], color = "red")
        plt.plot(k_vals, E_n[l_1, :], color = "blue")
    plt.xlabel("$k/k_\\text{max}$")
    plt.ylabel("E [eV]")
    plt.title("Bandstructure of (" + str(n) + "," + str(m) + ") CNT\n$\\gamma = $" + str(gamma) + " [eV]")
    plt.show()


# Plot density of states
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


