#!/usr/bin/env python3
import sys
sys.path.insert(1,'./utils')
sys.path.insert(1,'./data')
import numpy as np
import pandas as pd
from ngsolve import *
import NonNewtonianModels
from Geometry import *
from CustomSolvers import *
ngsolve.ngsglobals.msg_level = 0

#Load parameters
options = dict(pd.read_csv('config.csv',index_col=0).loc["options",:])
options["condense"] = bool(options["condense"])
options["order"] = int(options["order"])

#Set Newtonian model for unit benchmark
if options['benchmark'] == 'unit':
        options['model'] = 'Newtonian'

#Set NGSolve threads
SetNumThreads(int(options['threads'])) 

#Setup Solver
solver = Solvers(options)


#Load Nonnewtonian/Newtonian model
nn_model = getattr(NonNewtonianModels,options["model"])(options)

#Create mesh
mesh = mesh_list[options["benchmark"]](options,nn_model)

#Choose correct Boundary condition
uin = CoefficientFunction( (0,0) )
if options["benchmark"] == "cylinder":
        uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
if options["benchmark"] == "cavity":
        4*options['u_b']*x*(options['L']-x)/(options['L']**2)
        uin = CoefficientFunction( (16*options['u_b']*x**2*(options['L']-x)**2/(options['L']**4),0) )
                

