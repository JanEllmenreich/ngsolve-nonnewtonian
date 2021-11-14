#!/usr/bin/env python3
from ngsolve import *
from load_parameters import *
from Spaces import *
from Geometry import *
from IO import *
from CustomSolvers import *
import numpy as np

simulation = "SV-S"
mesh = Barycentric_Refinement(mesh,options)

h = specialcf.mesh_size
n = specialcf.normal(2)

#Define Spaces
X = Spaces(simulation)(mesh,options)

#Test- and Trialfunctions
(u,p,sigma),(v,q,tau) = X.TnT()


if options["print_info"]:
    PrintInfo(X,simulation,options,nn_model)

#Gridfunction and Residual vector
gfu = GridFunction(X)
gfu.components[0].Set(uin, definedon=mesh.Boundaries("boundary"))
base_lf = CoefficientFunction((0,0))*v*dx

#Define tensors
Du = 0.5*(grad(u)+grad(u).trans)
Dv = 0.5*(grad(v)+grad(v).trans)
Sigma = CoefficientFunction((sigma[0],sigma[1],sigma[1],-sigma[0]),dims=(2,2))
Tau = CoefficientFunction((tau[0],tau[1],tau[1],-tau[0]),dims=(2,2))
gfuDu = 0.5*(grad(gfu.components[0])+grad(gfu.components[0]).trans)
gfuSigma = CoefficientFunction((gfu.components[2][0],gfu.components[2][1],gfu.components[2][1],-gfu.components[2][0]),
                               dims=(2,2))

###Set Newtonian solution as starting point###
##Base BLF##
base_blf =  (InnerProduct(Sigma,Dv)-(div(u)*q+div(v)*p))*dx
base_blf += -1e-8*p*q*dx


#Forcing LinearForm
if options["benchmark"] == "unit":
    base_lf += nn_model.force*v*dx(bonus_intorder=8)
if options["benchmark"] == "periodic":
    base_lf += -options["dp"]*v[0]*dx

###Set Newtonian solution as starting point###
a = BilinearForm(X,condense=bool(options["condense"]))
a += (InnerProduct(Du,Tau)-0.5/options["nu"]*InnerProduct(Sigma,Tau))*dx
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
        gfu.vec.data += a.mat.Inverse(freedofs=X.FreeDofs(),inverse=options["inverse"]) * f.vec
#################################################

a_newton = BilinearForm(X,condense=bool(options["condense"]))
a_newton += base_blf - base_lf
a_fixpoint = BilinearForm(X,condense=bool(options["condense"]))
a_fixpoint += base_blf - base_lf


if options["constitutive"] and (options['solver'] == 'newton'):
        a_newton += -InnerProduct(nn_model.constitutive(Du-options['kappa']*Sigma,Sigma-options['kappa']*Du),Tau)*dx(bonus_intorder=10)
else:
    a_newton += InnerProduct(Du,Tau)*dx-1/nn_model.law(Du,Sigma)*InnerProduct(Sigma,Tau)*dx(bonus_intorder=10)
    a_fixpoint += InnerProduct(Du,Tau)*dx-1/nn_model.law(gfuDu,gfuSigma)*InnerProduct(Sigma,Tau)*dx(bonus_intorder=10)
  
#Start Nonlinear solving process
info = {}
if not options["benchmark"] == "unit":
    info = solver.solve(a_newton,a_fixpoint,gfu,f.vec)

##Plot output
data = [gfu.components[0],
            gfuSigma,
            gfuDu,
            gfu.components[1],
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
