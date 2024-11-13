from vdW_distance import Surface, Metal, Semiconductor, Interface
import matplotlib.pyplot as plt

# Main test vehicle for plotting the interfacial energy between two surfaces
def plot_Energy(s1, s2) :
    interface = Interface(s1, s2)
    interface.plotEnergy()

# Test vehicle for printing the surface properties.
# Returns as a string for printing (to terminal or to file)
def print_Properties(s1, s2) :

    out = ""  # Printed output

    # Construct interface
    interface = Interface(s1, s2)

    # Print banner
    title = s1.name + "-" + s2.name + " interface properties:"
    bar   = '-' * len(title)
    out += bar + "\n" + title + "\n" + bar + "\n"

    # Surface-related information
    # TODO reorganize better for metal and semiconductor specializations
    out += s1.printWorkfunction() + "\n"
    out += s1.printDecayConst()  + "\n"
    out += s1.printVDWStrength() + "\n"
    out += s2.printWorkfunction() + "\n"
    out += s2.printDecayConst()  + "\n"
    out += s2.printVDWStrength() + "\n"

    # vdW gap information
    d_vdW = interface.getVDWGap()
    out += "vdW gap distance = " + str(round(d_vdW,2)) + " [A]\n"

    # Binding energy information
    EV_HARTREE_CONST = 27.2 # [eV/Ha]
    out += "Binding energy = " + str(-round(1000*EV_HARTREE_CONST/s2.Na*interface.getEnergy(d_vdW))) \
        + " [meV] per atom of " + s2.name + " surface\n"
    
    # Pushback dipole
    out += "Pushback dipole = " + str(interface.getDipole_Pushback(d_vdW)) + " [V]\n"
    out += "\n"
    
    return out

# GRAPHENE
W_Graphene = 4.9     # [eV]
k_Graphene = 2.3     # [A^(-1)] (CALIBRATED by matching to C-C distance in graphite)
s_Graphene = 0.19    # [A^(-2)] (counts only half an electron per C atom due 
                     # to pZ-orbital on both sides of graphene sheet)
C6_Graphene= 47.9    # [Ha a0^6] (for atomic Carbon)
a_Graphene = 36.7    # [a0^3] (for atomic Carbon)
Graphene = Semiconductor(W_Graphene, W_Graphene, 0, k_Graphene, s_Graphene, C6_Graphene, a_Graphene, "Graphene")
Graphene.setAreaDensity(0.385)

# ALUMINUM (all values from tables of constants)
W_Al   = 4.10   # [eV]
m_Al   = 0.97   # [me]
rho_Al = 0.18   # [A^(-3)]
C6_Al  = 520    # [Ha a0^6]
a_Al   = 120.4  # [a0^3]
Aluminum = Metal(W_Al, m_Al, rho_Al, C6_Al, a_Al, "Aluminum")
Aluminum.setAreaDensity(0.19)

#SCANDIUM
W_Sc   = 3.53   # [eV]
m_Sc   = 1      # [me]
rho_Sc = 0.12   # [A^(-3)]   # I computed this from https://periodictable.com/Elements/021/data.html assuming valence = 2
C6_Sc  = 1570   # [Ha a0^6]
a_Sc   = 183    # [a0^3] 
Scandium = Metal(W_Sc, m_Sc, rho_Sc, C6_Sc, a_Sc, "Scandium")
Scandium.setAreaDensity(0.19)

# CHROMIUM
W_Cr   = 4.04   # [eV]
m_Cr   = 1      # [me]
rho_Cr = 0.16   # [A^(-3)]   # Computed assuming valence = 2
C6_Cr  = 709    # [Ha a0^6]
a_Cr   = 108    # [a0^3] 
Chromium = Metal(W_Cr, m_Cr, rho_Cr, C6_Cr, a_Cr, "Chromium")
Chromium.setAreaDensity(0.19)

# COPPER
W_Cu   = 4.85   # [eV]
m_Cu   = 1      # [me]
#rho_Cu = 0.085  # [A^(-3)]
rho_Cu = 0.93   # [A^(-3)]  # from https://lampz.tugraz.at/~hadley/ss1/materials/thermo/dos2mu.html
C6_Cu  = 264    # [Ha a0^6]
a_Cu   = 76     # [a0^3] 
Copper = Metal(W_Cu, m_Cu, rho_Cu, C6_Cu, a_Cu, "Copper")
Copper.setAreaDensity(0.19)

# SILVER
W_Ag   = 4.74   # [eV]
m_Ag   = 1      # [me]
rho_Ag = 0.059  # [A^(-3)]
C6_Ag  = 341    # [Ha a0^6]
a_Ag   = 113    # [a0^3]
Silver = Metal(W_Ag, m_Ag, rho_Ag, C6_Ag, a_Ag, "Silver")
Silver.setAreaDensity(0.19)

# GOLD 
W_Au   = 5.05   # [eV]
m_Au   = 1      # [me]
rho_Au = 0.059  # [A^(-3)]
C6_Au  = 427    # [Ha a0^6]
a_Au   = 125    # [a0^3]
Gold = Metal(W_Au, m_Au, rho_Au, C6_Au, a_Au, "Gold")
Gold.setAreaDensity(0.19)

# COBALT
W_Co   = 5      # [eV]
m_Co   = 1      # [me]
rho_Co = 0.089  # [A^(-3)]
C6_Co  = 461    # [Ha a0^6]
a_Co   = 93     # [a0^3]
Cobalt = Metal(W_Co, m_Co, rho_Co, C6_Co, a_Co, "Cobalt")
Cobalt.setAreaDensity(0.19)

# PALLADIUM
W_Pd   = 5.09   # [eV]
m_Pd   = 1      # [me]
rho_Pd = 0.14   # [A^(-3)]  # Computed assuming valence = 2
C6_Pd  = 628    # [Ha a0^6]
a_Pd   = 97     # [a0^3]
Palladium = Metal(W_Pd, m_Pd, rho_Pd, C6_Pd, a_Pd, "Palladium")
Palladium.setAreaDensity(0.19)

# PLATINUM
W_Pt   = 5.68   # [eV]
m_Pt   = 1      # [me]
rho_Pt = 0.13   # [A^(-3)]  # Computed assuming valence = 2
C6_Pt  = 470    # [Ha a0^6]
a_Pt   = 136    # [a0^3]
Platinum = Metal(W_Pt, m_Pt, rho_Pt, C6_Pt, a_Pt, "Platinum")
Platinum.setAreaDensity(0.19)

#plot_Energy(Graphene, Graphene)
#plot_Energy(Copper, Graphene)
#plot_Energy(Platinum, Graphene)
#plot_Energy(Gold, Graphene)

f = open("vdW_test_output.txt", "w")
f.write("TEST OF GUACAMOLE MODEL\n\n")
f.write(print_Properties(Graphene, Graphene))
f.write(print_Properties(Aluminum, Graphene))
f.write(print_Properties(Scandium, Graphene))
f.write(print_Properties(Chromium, Graphene))
f.write(print_Properties(Copper, Graphene))
f.write(print_Properties(Silver, Graphene))
f.write(print_Properties(Gold, Graphene))
f.write(print_Properties(Cobalt, Graphene))
f.write(print_Properties(Palladium, Graphene))
f.write(print_Properties(Platinum, Graphene))
f.close()
