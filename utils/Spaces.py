#!/usr/bin/env python3
from ngsolve import *

class Spaces:

    def __init__(self,simulation):
        self.simulation = simulation

    def __call__(self,mesh,options):

        order = int(options["order"])

        if self.simulation == "TH-S":
            Q = Periodic(H1(mesh,order=order-1))
            V = Periodic(VectorH1(mesh,order=order,dirichlet="wall|boundary"))
            S = L2(mesh,order=order-1)**3
            X = V*Q*S
            return X

        elif self.simulation == "MCS-S":
            Q = L2(mesh,order=order-1,lowest_order_wb=options["condense"])
            T = L2(mesh,order=order-1)
            E = L2(mesh,order=order)**2
            V = Periodic(HDiv(mesh, order=order,dirichlet="wall|boundary",RT=False))
            S = Periodic(HCurlDiv(mesh,order=order-1,orderinner=order,discontinuous=False))
            X = V*S*Q*T*E
            if options["condense"]:
                S = HCurlDiv(mesh,order=order-1,orderinner=order,discontinuous=True)
                Vhat = Periodic(TangentialFacetFESpace(mesh,order=order-1,dirichlet="wall|boundary|outlet"))
                X = V*S*Q*T*E*Vhat               
            return X

        elif self.simulation == "MCS":
            Q = L2(mesh,order=order-1,lowest_order_wb=options["condense"])
            T = L2(mesh,order=order-1)
            V = Periodic(HDiv(mesh, order=order,dirichlet="wall|boundary",RT=False))
            S = Periodic(HCurlDiv(mesh,order=order-1,orderinner=order,discontinuous=False))
            X = V*S*Q*T
            if options["condense"]:
                S = HCurlDiv(mesh,order=order-1,orderinner=order,discontinuous=True)
                Vhat = Periodic(TangentialFacetFESpace(mesh,order=order-1,dirichlet="wall|boundary|outlet"))
                X = V*S*Q*T*Vhat               
            return X

        elif self.simulation == "SV-S":
            Q = L2(mesh,order=order-1)
            V = Periodic(VectorH1(mesh,order=order,dirichlet="wall|boundary"))
            S = L2(mesh,order=order-1)**2
            if options["condense"]:
                Q.SetCouplingType(ngstd.IntRange(0,Q.ndof),COUPLING_TYPE.WIREBASKET_DOF)
                Q.FinalizeUpdate()
            X = V*Q*S
            return X

        else:
            raise Exception("Space {} not known".format(self.simulation))

