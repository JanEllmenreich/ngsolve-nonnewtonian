#!/usr/bin/env python3
from load_parameters import *
from ngsolve import *
from Spaces import *
from Geometry import *
from IO import *
from CustomSolvers import *
import numpy as np

simulation = "MCS-S"

h = specialcf.mesh_size
n = specialcf.normal(2)

#Define Spaces
X = Spaces(simulation)(mesh,options)

#Test- and Trialfunctions
if options["condense"]:
    (u,sigma,p,omega,eps,uhat),(v,tau,q,rho,phi,vhat) = X.TnT()
else:
    (u,sigma,p,omega,eps),(v,tau,q,rho,phi) = X.TnT()
Eps = CoefficientFunction((eps[0],eps[1],eps[1],-eps[0]),dims=(2,2))
Phi = CoefficientFunction((phi[0],phi[1],phi[1],-phi[0]),dims=(2,2))
Omega = CoefficientFunction((0,omega,-omega,0),dims=(2,2))
Rho = CoefficientFunction((0,rho,-rho,0),dims=(2,2))

#Print Info
if options["print_info"]:
    PrintInfo(X,simulation,options,nn_model)

#Gridfunction
gfu = GridFunction(X)
gfuEps = CoefficientFunction((gfu.components[4][0],gfu.components[4][1],gfu.components[4][1],-gfu.components[4][0]),dims=(2,2))

#Set boundary conditions
gfu.components[0].Set(uin, definedon=mesh.Boundaries("boundary"))
if options["condense"]:
    gfu.components[5].Set(uin, definedon=mesh.Boundaries("boundary"))
base_lf = CoefficientFunction((0,0))*v*dx

#Base BilinearForm
base_blf = InnerProduct(Eps,tau)*dx
base_blf += (InnerProduct(v,div(sigma))+InnerProduct(u,div(tau)))*dx
base_blf += -InnerProduct(sigma*n,n)*InnerProduct(v,n)*dx(element_boundary=True)
base_blf += -InnerProduct(tau*n,n)*InnerProduct(u,n)*dx(element_boundary=True)
base_blf += (div(v)*p + div(u)*q)*dx
base_blf += (InnerProduct(Omega,tau)+InnerProduct(sigma,Rho))*dx
if (options["benchmark"] == "cylinder"):
    pass
else:
    base_blf += -1e-8*p*q*dx
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
a = BilinearForm(X,condense=options["condense"])
a += InnerProduct(sigma,Phi)*dx-2*options["nu"]*InnerProduct(Eps,Phi)*dx
a += base_blf

f = LinearForm(X)
f += base_lf

with TaskManager():
    f.Assemble()
    a.Assemble()
    f.vec.data -= a.mat*gfu.vec
    if options["condense"]:
        f.vec.data += a.harmonic_extension_trans * f.vec
        gfu.vec.data += a.mat.Inverse(freedofs=X.FreeDofs(coupling=True),inverse=options["inverse"])*f.vec
        gfu.vec.data += a.inner_solve*f.vec
        gfu.vec.data += a.harmonic_extension*gfu.vec
    else:
        gfu.vec.data += a.mat.Inverse(freedofs=X.FreeDofs(),inverse=options["inverse"])*f.vec
#################################################

#Nonlinear BilinearForm
a_newton = BilinearForm(X,condense=options["condense"])#),printelmat=True)#,elmatev=True)
a_newton += base_blf - base_lf
a_fixpoint = BilinearForm(X,condense=options["condense"])
a_fixpoint += base_blf - base_lf

if options["constitutive"] and (options['solver'] == 'newton'):
    a_newton += InnerProduct(nn_model.constitutive(Eps-options['kappa']*sigma,sigma-options['kappa']*Eps),Phi)*dx(bonus_intorder=10)
else:
    a_newton += 1/nn_model.law(Eps,sigma)*InnerProduct(sigma,Phi)*dx(bonus_intorder=10) - InnerProduct(Eps,Phi)*dx
    a_fixpoint += 1/nn_model.law(gfuEps,gfu.components[1])*InnerProduct(sigma,Phi)*dx(bonus_intorder=10) - InnerProduct(Eps,Phi)*dx


#Start Nonlinear solving process
info = {}
if not options["benchmark"] == "unit":
     info = solver.solve(a_newton,a_fixpoint,gfu,f.vec)

##Plot output
data = [gfu.components[0],
            gfu.components[1],
            gfuEps,
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





