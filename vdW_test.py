from vdW_distance import Surface, Interface

# Main test vehicle for plotting the interfacial energy between two surfaces
def plot_Energy(s1, s2) :
    interface = Interface.makeInterface(s1, s2)
    interface.plotEnergy()

# GRAPHENE
k_Graphene = 2.34    # [A^(-1)] (CALIBRATED by matching to C-C distance in graphite)
s_Graphene = 0.19    # [A^(-2)] (counts only half an electron per C atom due 
                     # to pZ-orbital on both sides of graphene sheet)
C6_Graphene= 47.9    # [Ha a0^6] (for atomic Carbon)
a_Graphene = 11.7    # [a0^3] (for atomic Carbon)
Graphene = Surface.make2DSurface(k_Graphene, s_Graphene, C6_Graphene, a_Graphene, "Graphene")

# ALUMINUM
W_Al   = 4.08   # [eV]
m_Al   = 0.97   # [me]
rho_Al = 0.18   # [A^(-3)]
C6_Al  = 520    # [Ha a0^6]
a_Al   = 57.5   # [a0^3]
Aluminum = Surface.makeMetalSurface(W_Al, m_Al, rho_Al, C6_Al, a_Al, "Aluminum")

plot_Energy(Aluminum, Graphene)