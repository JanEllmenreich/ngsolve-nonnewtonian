#!/usr/bin/env python3
import sys
sys.path.insert(1,'./utils')
import subprocess
import argparse
from itertools import product
import numpy as np
import pandas as pd

options = { #Geometry
            "H"          : 2,                       #Channel Height for periodic and square
            "L"          : 1,                       #Channel Length for periodic and square
            "benchmark"  : "unit",                  #Select periodic, cavity, cylinder or unit

            #Flow properties
            "nu"     : 1.0,                         #Kinematic viscosity
            "dp"     : -2.0,                        #Pressure drop for periodic benchmark
            "u_b"    : 1.0,                         #Boundary velocity

            #FEM properties
            "structured"        : False,            #Structured mesh or not
            "hmax"              : 0.2,              #Element size
            "order"             : 2,                #Polynomial order std FEM
            "condense"          : True,             #Apply static condensation - apply with parser
            "constitutive"      : False,            #Use Implicit constitutive relation, 
                                                    #gets regularized
                                                    #does not work with standard MCS

            #Solver options
            "threads"    : 4,                       #Set number of threads
            "solver"     : 'newton',                #Newton or Fixpoint
            "it_max"     : 200,                     #Max Iterations
            "tol_min"    : 1e-8,                    #Stopping toleranceq
            "switch"     : False,                   #Switch from Fixpoint to Newton
            "tol_switch" : 1e-4,                    #Switch tolerance from Fixedpoint to Newton
            "damp_fix"   : 1.0,                     #Damping factor
            "damp_newton": 1.0,                     #Damping factor
            "printrates" : True,                    #Print solver information
            "linesearch" : 'none',                  #Linesearch backtracking, 2divison or none
            "inverse"    : 'pardiso',               #Direct solver pardiso or umfpack
              
            #Nonnewtonian parameters
            "model"         : "Powerlaw",                 #Choose between Powerlaw, Bingham, Ellis
            "r"             : 1.4,                        #Powerlaw exponent r > 1
            "tau_y"         : 1.0,                          #Yield stress
            "kappa"         : 1e-8,                       #Regularization parameter
            "law"           : "G1",                       #Regularized law G1-G2

            #Output options
            "draw_ngsolve"    : True,              #Draw netgen solution
            "draw_matplotlib" : True,              #Draw velocity profile matplotlib
            "resolution_plot" : 501,                #Resolution Channelheight
            "write_output"    : False,               #Write output to file
            "print_info"      : False,              #Print to terminal simulation info
            "vtk_output"      : False               #Export VTK
        }

## Parse arguments ##
nn_ = {'Powerlaw':1, 'Bingham':2}
parser = argparse.ArgumentParser(description='Choose FEM and Anaysis')
parser.add_argument('bash', help='Select python3 or netgen', type=str)
parser.add_argument('benchmark', help='Enter a benchmark', type=str)
parser.add_argument('fem', help='Enter a Fem combination (1-15)', type=int)
parser.add_argument('-struct', help="Structured Mesh", action='store_true',default=options['structured'])
parser.add_argument('-nn', help='Enter a Nonnewtonian model (1-3)', type=int,default=nn_[options['model']])
parser.add_argument('-hmax',nargs='*',help='Add a list of mesh sizes',type=float,default=[options['hmax']])
parser.add_argument('-kappa',nargs='*',help='Add a list of regularization parameters',type=float,default=[options['kappa']])
parser.add_argument('-thr',nargs='?',help='Set number of threads',type=int,default=options['threads'])
arguments = parser.parse_args()
if arguments.fem > 31:
    parser.error("""Fem has to be smaller than 32
                    1 ... MCS
                    2 ... MCS-S
                    3 ... MCS + MCS-S 
                    4 ... TH-S
                    5 ... TH-S + MCS
                    8 ... SV-S
                    etc...
                    The rest is a combination of above binaries""")
if arguments.nn > 7:
    parser.error("""nn_model has to be smaller than 4
                    1 ... Powerlaw
                    2 ... Bingham
                    3 ... Powerlaw + Bingham
                    etc...
                    The rest is a combination of above binaries""")

options['threads'] = arguments.thr
options['benchmark'] = arguments.benchmark
options['structured'] = arguments.struct

fem = np.flip(np.array([int(e) for e in "{0:05b}".format(arguments.fem)],dtype=bool))
fem_ = ['MCS','MCS-S','TH-S','SV-S']
fem_ = np.array([key for key,value in zip(fem_,fem) if value])

nn = np.flip(np.array([int(e) for e in "{0:03b}".format(arguments.nn)],dtype=bool))
nn_ = np.array([key for key,value in zip(nn_,nn) if value])
simulations = list(product(fem_,nn_,arguments.hmax,arguments.kappa))

## Start all simulations ##
for sim in simulations:
    options['model'] = sim[1]
    options['hmax'] = sim[2]
    options['kappa'] = sim[3]
    fd = pd.DataFrame(options,index=['options'])
    fd.to_csv('config.csv',header=True,index=True,mode="w")
    process = subprocess.run("{} {}.py".format(arguments.bash,sim[0]), shell=True)

