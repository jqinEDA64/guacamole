from vdW_distance import Interface

def test_one():
    MS_int = Interface(1,2,0.08,0.005,520,48,558,12)
    MS_int.plotEnergy()

###################
# Carbon parameters
###################

C6_C    = 47.9  # C6 for Carbon [Ha a0^6]
alpha_C = 11.7  # Static polarizability for Carbon [a0^3]

# Graphene has a honeycomb structure with distance 1.42 A between neighboring atoms
# This implies that each atom occupies 2.62 A^2 area. If Carbon has 4 valence 
# electrons per atom, the area density of Carbon valence electrons is 4/(2.62 A^2)
sigma_C = 1.53  # Area density of Carbon valence electrons [A^(-2)]

# What is the decay length of graphene wavefunction, perpendicular to the sheet?
# Not sure but use atomic Carbon as a guide
#
# TODO Current issue is that the estimate of atomic size is too large (and therefore k_C is too small)
#

k_C     = 3.07              # Inverse decay length of wavefunction perpendicular to graphene sheet [A^(-1)]
rho_C   = 2*k_C*sigma_C  # Effective volume charge of graphene [e A^(-3)]

# TODO how to convert the energy scale to binding energy in [eV]?
def test_graphene_graphene():
    CC_int = Interface(k_C, k_C, rho_C, rho_C, C6_C, C6_C, alpha_C, alpha_C)
    CC_int.plotEnergy("Graphene-graphene interaction")

def test_carbon_metal(rho_M, C6_M, alpha_M, interfacename = ""):
    CM_int = Interface.makeInterface(rho_M, rho_C, C6_M, C6_C, alpha_M, alpha_C)
    CM_int.plotEnergy(interfacename)

############
# Test bench
############

#test_carbon_metal(rho_C, C6_C, alpha_C, "Carbon on Carbon")
#test_carbon_metal(0.18, 520, 57.5, "Aluminum on Carbon")
#test_carbon_metal(0.059, 427, 45.4, "Gold on Carbon")
#test_carbon_metal(0.18, 427, 45.4, "Gold on Carbon w/ Aluminum charge density")
#test_carbon_metal(0.12, 1570, 123, "Scandium on Carbon")
#test_carbon_metal(0.0847, 264, 41.7, "Copper on Carbon")
#test_carbon_metal(0.066, 470, 48, "Platinum on Carbon")
#test_carbon_metal()

#test_graphene_graphene()
