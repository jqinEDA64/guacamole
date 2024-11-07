from vdW_distance import Surface, Interface
import matplotlib.pyplot as plt

# Main test vehicle for plotting the interfacial energy between two surfaces
def plot_Energy(s1, s2) :
    interface = Interface.makeInterface(s1, s2)
    interface.plotEnergy()
    #plot = interface._plotEnergy(2, 5, 0)
    #plot.show()
    

# GRAPHENE
k_Graphene = 2.3     # [A^(-1)] (CALIBRATED by matching to C-C distance in graphite)
s_Graphene = 0.19    # [A^(-2)] (counts only half an electron per C atom due 
                     # to pZ-orbital on both sides of graphene sheet)
C6_Graphene= 47.9    # [Ha a0^6] (for atomic Carbon)
a_Graphene = 11.7    # [a0^3] (for atomic Carbon)
Graphene = Surface.make2DSurface(k_Graphene, s_Graphene, C6_Graphene, a_Graphene, "Graphene")
Graphene.setAreaDensity(0.385)
Graphene.printVDWStrength()

# ALUMINUM (all values from tables of constants)
W_Al   = 4.08   # [eV]
m_Al   = 0.97   # [me]
rho_Al = 0.18   # [A^(-3)]
C6_Al  = 520    # [Ha a0^6]
a_Al   = 57.5   # [a0^3]
Aluminum = Surface.makeMetalSurface(W_Al, m_Al, rho_Al, C6_Al, a_Al, "Aluminum")
Aluminum.setAreaDensity(0.19)
Aluminum.printDecayConst()
Aluminum.printVDWStrength()

#SCANDIUM
W_Sc   = 3.53   # [eV]
m_Sc   = 1      # [me]
rho_Sc = 0.12   # [A^(-3)]   # I computed this from https://periodictable.com/Elements/021/data.html
C6_Sc  = 1570   # [Ha a0^6]
a_Sc   = 123    # [a0^3] 
Scandium = Surface.makeMetalSurface(W_Sc, m_Sc, rho_Sc, C6_Sc, a_Sc, "Scandium")
Scandium.setAreaDensity(0.19)
Scandium.printDecayConst()
Scandium.printVDWStrength()

# CHROMIUM
W_Cr   = 4.04   # [eV]
m_Cr   = 1      # [me]
rho_Cr = 0.5    # [A^(-3)]
C6_Cr  = 709    # [Ha a0^6]
a_Cr   = 78     # [a0^3] 
Chromium = Surface.makeMetalSurface(W_Cr, m_Cr, rho_Cr, C6_Cr, a_Cr, "Chromium")
Chromium.setAreaDensity(0.19)
Chromium.printDecayConst()
Chromium.printVDWStrength()

# COPPER
W_Cu   = 4.7    # [eV]
m_Cu   = 1      # [me]
rho_Cu = 0.085  # [A^(-3)]
C6_Cu  = 264    # [Ha a0^6]
a_Cu   = 42     # [a0^3] 
Copper = Surface.makeMetalSurface(W_Cu, m_Cu, rho_Cu, C6_Cu, a_Cu, "Copper")
Copper.setAreaDensity(0.19)
Copper.printDecayConst()
Copper.printVDWStrength()

#plot_Energy(Graphene, Graphene)
plot_Energy(Aluminum, Graphene)
#plot_Energy(Scandium, Graphene)