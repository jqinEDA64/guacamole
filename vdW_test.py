from vdW_distance import Surface, Interface
import matplotlib.pyplot as plt

# Main test vehicle for plotting the interfacial energy between two surfaces
def plot_Energy(s1, s2) :
    interface = Interface.makeInterface(s1, s2)
    interface.plotEnergy()

# Test vehicle for printing the surface properties
def print_Properties(s1, s2) :

    # Construct interface
    interface = Interface.makeInterface(s1, s2)

    # Print banner
    title = s1.name + "-" + s2.name + " interface properties: "
    bar   = '-' * len(title)
    print(bar + "\n" + title + "\n" + bar)

    # Surface-related information
    s1.printDecayConst()
    s1.printVDWStrength()
    s2.printDecayConst()
    s2.printVDWStrength()

    # vdW gap information
    d_vdW = interface.getVDWGap()
    print("vdW gap distance = " + str(round(d_vdW,2)) + " [A]")

    # Binding energy information
    EV_HARTREE_CONST = 27.2 # [eV/Ha]
    print("Binding energy = " + str(round(1000*EV_HARTREE_CONST/s2.Na*interface.getEnergy(d_vdW))) \
          + " [meV] per atom of " + s2.name + " surface")

# GRAPHENE
k_Graphene = 2.3     # [A^(-1)] (CALIBRATED by matching to C-C distance in graphite)
s_Graphene = 0.19    # [A^(-2)] (counts only half an electron per C atom due 
                     # to pZ-orbital on both sides of graphene sheet)
C6_Graphene= 47.9    # [Ha a0^6] (for atomic Carbon)
a_Graphene = 36.7    # [a0^3] (for atomic Carbon)
Graphene = Surface.make2DSurface(k_Graphene, s_Graphene, C6_Graphene, a_Graphene, "Graphene")
Graphene.setAreaDensity(0.385)

# ALUMINUM (all values from tables of constants)
W_Al   = 4.10   # [eV]
m_Al   = 0.97   # [me]
rho_Al = 0.18   # [A^(-3)]
C6_Al  = 520    # [Ha a0^6]
a_Al   = 120.4  # [a0^3]
Aluminum = Surface.makeMetalSurface(W_Al, m_Al, rho_Al, C6_Al, a_Al, "Aluminum")
Aluminum.setAreaDensity(0.19)

#SCANDIUM
W_Sc   = 3.53   # [eV]
m_Sc   = 1      # [me]
rho_Sc = 0.12   # [A^(-3)]   # I computed this from https://periodictable.com/Elements/021/data.html
C6_Sc  = 1570   # [Ha a0^6]
a_Sc   = 183    # [a0^3] 
Scandium = Surface.makeMetalSurface(W_Sc, m_Sc, rho_Sc, C6_Sc, a_Sc, "Scandium")
Scandium.setAreaDensity(0.19)

# CHROMIUM
W_Cr   = 4.04   # [eV]
m_Cr   = 1      # [me]
rho_Cr = 0.5    # [A^(-3)]
C6_Cr  = 709    # [Ha a0^6]
a_Cr   = 108    # [a0^3] 
Chromium = Surface.makeMetalSurface(W_Cr, m_Cr, rho_Cr, C6_Cr, a_Cr, "Chromium")
Chromium.setAreaDensity(0.19)

# COPPER
W_Cu   = 4.85   # [eV]
m_Cu   = 1      # [me]
rho_Cu = 0.085  # [A^(-3)]
C6_Cu  = 264    # [Ha a0^6]
a_Cu   = 76     # [a0^3] 
Copper = Surface.makeMetalSurface(W_Cu, m_Cu, rho_Cu, C6_Cu, a_Cu, "Copper")
Copper.setAreaDensity(0.19)

# SILVER
W_Ag   = 4.74   # [eV]
m_Ag   = 1      # [me]
rho_Ag = 0.059  # [A^(-3)]
C6_Ag  = 341    # [Ha a0^6]
a_Ag   = 113    # [a0^3]
Silver = Surface.makeMetalSurface(W_Ag, m_Ag, rho_Ag, C6_Ag, a_Ag, "Silver")
Silver.setAreaDensity(0.19)

# GOLD 
W_Au   = 5.05   # [eV]
m_Au   = 1      # [me]
rho_Au = 0.059  # [A^(-3)]
C6_Au  = 427    # [Ha a0^6]
a_Au   = 125    # [a0^3]
Gold = Surface.makeMetalSurface(W_Au, m_Au, rho_Au, C6_Au, a_Au, "Gold")
Gold.setAreaDensity(0.19)

# COBALT
W_Co   = 5      # [eV]
m_Co   = 1      # [me]
rho_Co = 0.089  # [A^(-3)]
C6_Co  = 461    # [Ha a0^6]
a_Co   = 93     # [a0^3]
Cobalt = Surface.makeMetalSurface(W_Co, m_Co, rho_Co, C6_Co, a_Co, "Cobalt")
Cobalt.setAreaDensity(0.19)

# PALLADIUM
W_Pd   = 5.09   # [eV]
m_Pd   = 1      # [me]
rho_Pd = 0.27   # [A^(-3)]  # Computed assuming valence = 4 (not convinced)
C6_Pd  = 628    # [Ha a0^6]
a_Pd   = 97     # [a0^3]
Palladium = Surface.makeMetalSurface(W_Pd, m_Pd, rho_Pd, C6_Pd, a_Pd, "Palladium")
Palladium.setAreaDensity(0.19)

# PLATINUM
W_Pt   = 5.68   # [eV]
m_Pt   = 1      # [me]
rho_Pt = 0.13   # [A^(-3)]  # Computed assumin valence = 2
C6_Pt  = 470    # [Ha a0^6]
a_Pt   = 136    # [a0^3]
Platinum = Surface.makeMetalSurface(W_Pt, m_Pt, rho_Pt, C6_Pt, a_Pt, "Platinum")
Platinum.setAreaDensity(0.19)

#plot_Energy(Graphene, Graphene)
#plot_Energy(Copper, Graphene)
#plot_Energy(Scandium, Graphene)

print_Properties(Graphene, Graphene)