#!/usr/bin/env python3
from ngsolve import *
import pandas as pd
import numpy as np
import copy

#Draw all quantities in NGSolve
def DrawNgsolve(data,mesh,nn_model,options):
    Draw(data[0],mesh,"Velocity")
    Draw(data[1],mesh,"Stress")
    Draw(data[2],mesh,"Strain")
    Draw(data[3],mesh,"Pressure")
    Draw(data[4],mesh,"Divergence")
    Draw(data[5],mesh,"Viscosity")
    if (options["benchmark"] == "unit") or (options["benchmark"] == "periodic"):
        Draw(nn_model.exact_u,mesh,"Exact_u")
        Draw(nn_model.exact_Du,mesh,"Exact_Du")
        Draw(nn_model.exact_sigma,mesh,"Exact_sigma")

#Draw velocity profiles in NGSolve
def DrawMatplotlib(data,mesh,nn_model,options,simulation):
    if not (options["benchmark"] == "cylinder"):
        import matplotlib.pyplot as plt
        import matplotlib
        import warnings
        warnings.filterwarnings("ignore")
        matplotlib.use("GTK3Agg")
        matplotlib.rcParams['text.usetex'] = True

        ##Determine flow in middle of channel
        H = options["H"]; L = options["L"]
        y_channel = np.linspace(-H/2,H/2,options["resolution_plot"])
        mip = mesh(L/2,y_channel)
        #Result velocity - Newtonian, simulation and Exact solution
        out_array = np.array([
                1/(2*options["nu"]) * (-options["dp"]) * H*H/4*(1-4*np.power(y_channel,2)/(H*H)),
                np.array(data[0][0](mip)).reshape(options["resolution_plot"]),
                nn_model.draw(y_channel)])
        H = options["H"]
        y_channel = np.linspace(-H/2,H/2,options["resolution_plot"])
        fig, ax = plt.subplots(figsize=(9,7))
        if options["benchmark"] == "periodic":
            ax.plot(out_array[0,:],y_channel,ls="--",color='blue',label=r"newtonian")
            ax.plot(out_array[2,:],y_channel,ls="-",color='k',label=r"{}".format(nn_model))
        elif options["benchmark"] == "unit":
            ax.plot(out_array[2,:],y_channel,ls="-",color='k',label=r"{}".format('Exact'))
        ax.plot(out_array[1,:],y_channel,ls=":",color='red',label=r"simulation")
        ax.grid(True, which="both",linewidth=0.5)
        ax.set_xlabel(r'$u$',fontsize=20)
        ax.set_ylabel(r'$y$',fontsize=20)
        ax.set_ylim(-H/2-H/12,H/2+H/12)
        ax.set_title(r"{}".format(simulation))
        ax.legend()
        plt.show()

#Calculate and print L2-Norms 
def CalcError(data,mesh,nn_model,options):    
    L2_Error = {}
    if (options["benchmark"] == "unit") or (options["benchmark"] == "periodic"):
        diff_u = nn_model.exact_u - data[0]
        diff_sigma = nn_model.exact_sigma - data[1]
        diff_Du = nn_model.exact_Du - data[2]
        diff_p = nn_model.pressure - data[3]

        L2_Error["Velocity"] = np.sqrt(Integrate(InnerProduct(diff_u,diff_u),mesh))
        L2_Error["Stress"]   = np.sqrt(Integrate(InnerProduct(diff_sigma,diff_sigma),mesh))
        L2_Error["Strain rate"]   = np.sqrt(Integrate(InnerProduct(diff_Du,diff_Du),mesh))
        L2_Error["Pressure"]   = np.sqrt(Integrate(InnerProduct(diff_p,diff_p),mesh))

    for key in L2_Error:
        print("L2-Norm {} = {}".format(key,L2_Error[key]))
    print("")

    return L2_Error

#Write Output to filename
def WriteOutput(options,L2_Error,info,filename):
    from os import path
    output = copy.deepcopy(options)
    remove = ["benchmark","switch","tol_switch","printrates","draw_ngsolve","draw_matplotlib",
              "resolution_plot","write_output","print_info","vtk_output"]
    if options['model'] == 'Powerlaw':
        remove.extend(['tau_y','law','kappa'])
    elif options['model'] == 'Bingham':
        remove.extend(['r'])
    else:
        remove.extend(['tau_y','law','kappa','r'])
    for k in remove:
        output.pop(k,None)
    for key in L2_Error:
        output[key] = L2_Error[key]
    for key in info:
        output[key] = info[key]

    fileexist = path.isfile("{}.csv".format(filename))
    df = pd.DataFrame(output,index=['output'])
    if fileexist:
        df.to_csv("{}.csv".format(filename),index=False,mode='a',header=False)
        df = pd.read_csv("{}.csv".format(filename))
    else:
        df.to_csv("{}.csv".format(filename),index=False,mode='w',header=True)

#Print simulation Info
def PrintInfo(fes,simulation,options,nn_model):
    dof_types= {}
    for i in range(fes.ndof):
        ctype = fes.CouplingType(i)
        if ctype in dof_types.keys():
            dof_types[ctype] += 1
        else: dof_types[ctype] = 1

    print(
    """
    Simulation: {}
    NonnewtonianModel: {}
    Benchmark: {}
    Threads: {}
    Height: {}, Length: {}, h: {}
    Pressure drop: {}
    Polynomial order: {}
    Static condensation: {}
    Number of unknowns in space: {}
    Number of unknowns w/o dirichlet: {}""".format(
        simulation, nn_model,options["benchmark"],options['threads'],options["H"],options["L"],
        options["hmax"],options["dp"],options["order"],
        options["condense"],
        fes.ndof,np.sum(np.array(fes.FreeDofs())),dof_types
        )
    )
    print("    Type of DoF:")
    for e in dof_types:
        print("         {}: {}".format(e,dof_types[e]))
    print('')

def ExportVTK(data,mesh,filename):
    # VTKOutput object
    vtk = VTKOutput(ma=mesh,
                coefs=data,
                names = ["Velocity","Stress","Strains","Pressure","Divergence","Viscosity"],
                filename=filename,
                subdivision=3)
    # Exporting the results:
    vtk.Do()

