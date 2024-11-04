from vdW_distance import Surface, Interface

def test_graphene_graphene() :

    #####################
    # Graphene parameters
    #####################

    k_Graphene = 2.3
    s_Graphene = 0.19    # [A^(-2)] (counts only half an electron per C atom due 
                         # to pZ-orbital on both sides of graphene sheet)
    C6_Graphene= 47.9    # [Ha a0^6] (for atomic Carbon)
    a_Graphene = 11.7    # [a0^3] (for atomic Carbon)

    ##################
    # Graphene surface
    ##################
    graphene = Surface.make2DSurface(k_Graphene, s_Graphene, C6_Graphene, a_Graphene)

    ############################
    # Graphene-graphee interface
    ############################
    gg_interface = Interface.makeInterface(graphene, graphene)
    gg_interface.plotEnergy("Graphene on Graphene")

test_graphene_graphene()

