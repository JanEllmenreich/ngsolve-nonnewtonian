#!/usr/bin/env python3
from ngsolve import *
from load_parameters import *
from Spaces import *
from Geometry import *
from IO import *
from CustomSolvers import *
import numpy as np

simulation = "MCS"
if options['constitutive']:
    raise NameError('''Constitutive Relations can not be applied to standard MCS.
                    Set options['constitutive'] to False''')

h = specialcf.mesh_size
n = specialcf.normal(2)

#Define Spaces
X = Spaces(simulation)(mesh,options)
if options["condense"]:
    (u,sigma,p,omega,uhat),(v,tau,q,rho,vhat) = X.TnT()
else:
    (u,sigma,p,omega),(v,tau,q,rho) = X.TnT()

#Print Info
if options["print_info"]:
    PrintInfo(X,simulation,options,nn_model)

#Gridfunction
gfu = GridFunction(X)

#Set boundary conditions
gfu.components[0].Set(uin, definedon=mesh.Boundaries("boundary"))
if options["condense"]:
    gfu.components[4].Set(uin, definedon=mesh.Boundaries("boundary"))
base_lf = CoefficientFunction((0,0))*v*dx

#Test- and Trialfunction
Omega = CoefficientFunction((0,omega,-omega,0),dims=(2,2))
Rho = CoefficientFunction((0,rho,-rho,0),dims=(2,2))
Du = 0.5*(grad(u)+grad(u).trans)
Dgfu = 0.5*(grad(gfu.components[0])+grad(gfu.components[0]).trans)

#Base BilinearForm
base_blf = (InnerProduct(v,div(sigma))+InnerProduct(u,div(tau)))*dx
base_blf += -InnerProduct(sigma*n,n)*InnerProduct(v,n)*dx(element_boundary=True)
base_blf += -InnerProduct(tau*n,n)*InnerProduct(u,n)*dx(element_boundary=True)
base_blf += (div(v)*p + div(u)*q)*dx
base_blf += (InnerProduct(Omega,tau)+InnerProduct(sigma,Rho))*dx
if (options["benchmark"] == "cylinder"):
    pass
else:
    base_blf += 1e-8*p*q*dx
if options["condense"]:
    base_blf += -tang(sigma*n,n)*vhat*dx(element_boundary=True)
    base_blf += -tang(tau*n,n)*uhat*dx(element_boundary=True)
else:
    #Forcing LinearForm
    base_lf += InnerProduct(tang(tau.Trace()*n,n),uin)*ds(definedon="boundary")

#Forcing LinearForm
if options["benchmark"] == "unit":
    base_lf += -nn_model.force*v*dx(bonus_intorder=8)
if options["benchmark"] == "periodic":
    base_lf += options["dp"]*v[0]*dx

#Set Newtonian solution as initial solution
a = BilinearForm(X,condense=bool(options["condense"]))
a += base_blf
a += 0.5/options["nu"]*InnerProduct(sigma,tau)*dx

f = LinearForm(X)
f += base_lf

with TaskManager():
    f.Assemble()
    a.Assemble()

    f.vec.data -= a.mat*gfu.vec
    if options["condense"]:
        f.vec.data += a.harmonic_extension_trans*f.vec
        gfu.vec.data += a.mat.Inverse(freedofs=X.FreeDofs(coupling=True),inverse=options["inverse"])*f.vec
        gfu.vec.data += a.inner_solve*f.vec
        gfu.vec.data += a.harmonic_extension*gfu.vec
    else:
        gfu.vec.data += a.mat.Inverse(freedofs=X.FreeDofs(),inverse=options["inverse"])*f.vec


## Nonlinear BLF
a_newton = BilinearForm(X,condense=bool(options["condense"]))
a_newton += base_blf - base_lf
a_newton += 1/nn_model.law(Du,sigma)*InnerProduct(sigma,tau)*dx(bonus_intorder=10)
a_fixpoint = BilinearForm(X,condense=bool(options["condense"]))
a_fixpoint += base_blf - base_lf
a_fixpoint += 1/nn_model.law(Dgfu,gfu.components[1])*InnerProduct(sigma,tau)*dx(bonus_intorder=10)

#Start Nonlinear solving process
info = {}
if not options["benchmark"] == "unit":
    info = solver.solve(a_newton,a_fixpoint,gfu,f.vec)

##Plot output
data = [gfu.components[0],
            gfu.components[1],
            Dgfu,
            gfu.components[2],
            div(gfu.components[0])]
data.append(nn_model.law(data[2],data[1]))

if options["draw_ngsolve"]:
    DrawNgsolve(data,mesh,nn_model,options)

L2_Error = CalcError(data,mesh,nn_model,options)

#####Write Outputs and Plots##############
if options["write_output"]:
    filename = "data/{}_{}_{}".format(options["benchmark"],simulation,nn_model)
    WriteOutput(options,
                L2_Error,
                info,
                filename)

if options["draw_matplotlib"]:
    DrawMatplotlib(data,mesh,nn_model,options,simulation)

if options["vtk_output"]:
    filename = "data/vtk_{}_{}_{}".format(options["benchmark"],simulation,nn_model)
    ExportVTK(data,mesh,filename)



