#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from itertools import cycle, product
from mpltools import annotation
import matplotlib.ticker as ticker

matplotlib.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
    "font.sans-serif": "Computer Modern Sans Serif"})

sim_type = "periodic"
kappa = 1e-8
structured = False
constitutive = False
tau = 0.2
r = 1.4
tol = 1e-8
hmax = np.exp2(-6)

savefig = True
simulations = { "MCS" : [True,'darkorange','-',r'$\mathcal{MCS}$','o'],
                "MCS-S" : [True,'red','-.',r'$\mathcal{MCS}\text{-}\mathcal{S}$','v'],
                "TH-S" : [True,'green',':',r'$\mathcal{TH}\text{-}\mathcal{S}$','s'],
                "SV-S" : [True,'blue','--',r'$\mathcal{SV}\text{-}\mathcal{S}$',"^"]}

nn_tags = {"Bingham" : True,
           "Powerlaw" : False,
           "Newtonian" : False}

active = [[i,j] for i,j in list(product(simulations,nn_tags)) if simulations[i][0] and nn_tags[j]]


csv_files = ["../{}_{}_{}.csv".format(sim_type,i,j) for i,j in active]
colors = [simulations[i][1] for i,j in active]
lines = [simulations[i][2] for i,j in active]
labels = [simulations[i][3] for i,j in active]
markers = [simulations[i][4] for i,j in active]




# ax_labels = ["\\boldsymbol{u}_h - \\boldsymbol{u}","\\boldsymbol{\\tau}_h - \\boldsymbol{\\tau}",
#             "\\boldsymbol{\\varepsilon}(\\boldsymbol{u}_h) - \\boldsymbol{\\varepsilon}(\\boldsymbol{u})", "p_h - p"]#
ax_labels = ["\\boldsymbol{u}_{\\kappa,h} - \\boldsymbol{u}","\\boldsymbol{\\tau}_{\\kappa,h} - \\boldsymbol{\\tau}",
              "\\boldsymbol{\\varepsilon}(\\boldsymbol{u}_{\\kappa,h}) - \\boldsymbol{\\varepsilon}(\\boldsymbol{u})", 
              "p_{\\kappa,h} - p"]#
ax_orientation = ["Velocity","Stress","Strain rate","Pressure"]
cm = 1/2.54  # centimeters in inches
fig, ax = plt.subplots(2,2,num=1,figsize=(12*cm,12*cm),dpi=400)
ax = ax.reshape(ax.size)
for i in range(ax.size):
    ax[i].set_axisbelow(True)
    ax[i].grid(True,linewidth=0.5)
    ax[i].set_xlabel(r'$h$',fontsize=12,labelpad=-1); ax[i].set_title(r"$\| {{{}}} \|$".format(ax_labels[i]),fontsize=12,x=0.51,y=1.01)
    ax[i].tick_params(axis='both', which='major', labelsize=10) 
    ax[i].set(xlim=(1e-3,0.5))
    #ax[i].set(xlim=(1e-9,10))
    #ax[i].set_box_aspect(1)


# ax[0].set(ylim=(1e-9,1e-1))
# ax[1].set(ylim=(1e-5,1e-0))
# ax[2].set(ylim=(1e-6,1e-0))
# ax[3].set(ylim=(1e-6,1e-0))
ax[0].set(ylim=(1e-7,1e-3))
ax[1].set(ylim=(1e-5,1e-1))
ax[2].set(ylim=(1e-5,1e-2))
ax[3].set(ylim=(1e-5,1e-1))
# ax[0].set(ylim=(1e-7,1e-3))
# ax[1].set(ylim=(1e-3,1e-0))
# ax[2].set(ylim=(1e-5,1e-2))
# ax[3].set(ylim=(1e-4,1e-0))

# =============================================================================
# box = ax.get_position()
# ax.set_position([box.x0, box.y0,
#                   box.width, box.height * 0.9])
# =============================================================================


for i in range(len(csv_files)):
    #Load parameters
    fd = pd.read_csv(csv_files[i])
    if sim_type=="unit":
        fd = fd.loc[(fd["structured"] == structured)]
    elif sim_type=="periodic":
        if nn_tags["Bingham"]:
            fd = fd.loc[(fd['tau_y'] == tau) & (fd['constitutive'] == constitutive) & (fd['kappa']==kappa) & 
                        (fd['model']=='Bingham')& (fd['converged?']==True) & (fd['structured']== structured) ]
            fd = fd.sort_values(by='hmax',ascending = False)
            # fd = fd.loc[(fd['tau_y'] == tau) & (fd['constitutive'] == constitutive) & (fd['hmax']==hmax) & 
            #             (fd['model']=='Bingham')& (fd['converged?']==True) & (fd['structured']== structured) ]            
            # fd = fd.sort_values(by='kappa',ascending = False)
        if nn_tags["Powerlaw"]:
            fd = fd.loc[(fd['r'] == r) & (fd['model']=='Powerlaw') & (fd['converged?']==True) 
                        & (fd['structured']== structured) & (fd['tol_min']==tol) ]
    #Plot
    # np.set_printoptions(precision=4, formatter={'all':lambda x: "$%.5e$" % x + ' &'})
    # formatter={'all':lambda x: 'int: '+str(-x)}
    # print(np.array([fd['Velocity'],fd['Stress'],fd['Strain rate'],fd['Pressure']]).T)

    for j in range(ax.size):
        ax[j].loglog(fd['hmax'],fd[ax_orientation[j]],lw=1,marker=markers[i],markersize=2,ls=lines[i],color=colors[i],label=labels[i])
        #ax[j].loglog(fd['kappa'],fd[ax_orientation[j]],lw=1,marker=markers[i],markersize=3,ls=lines[i],color=colors[i],label=labels[i])


for j in range(ax.size):
    ax[j].yaxis.set_major_locator(ticker.LogLocator(base=10.0,numticks=200))
    ax[j].yaxis.set_minor_locator(ticker.LogLocator(subs='all',base=10.0,numticks=200))
    ax[j].xaxis.set_major_locator(ticker.LogLocator(base=10.0,numticks=8))
    #ax[j].xaxis.set_minor_locator(ticker.LogLocator(subs='all',base=10.0,numticks=200))


fig.tight_layout()
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels,bbox_to_anchor=(0.5,1.02), loc='center',ncol=4,fontsize=12)


if sim_type == "unit":
    annotation.slope_marker((0.04, 1e-7), (3, 1), ax=ax[0],size_frac=0.15,pad_frac=0.1,text_kwargs={'color': 'black','fontsize' : 10},
                            poly_kwargs={'facecolor': (0, 0, 0)})
    annotation.slope_marker((0.039, 1e-5), (2, 1), ax=ax[1],size_frac=0.15,pad_frac=0.1,text_kwargs={'color': 'black','fontsize' : 10},
                            poly_kwargs={'facecolor': (0, 0, 0)})
    annotation.slope_marker((0.04, 1e-5), (2, 1), ax=ax[2],size_frac=0.15,pad_frac=0.1,text_kwargs={'color': 'black','fontsize' : 10},
                            poly_kwargs={'facecolor': (0, 0, 0)})
    annotation.slope_marker((0.04, 1e-5), (2, 1), ax=ax[3],size_frac=0.15,pad_frac=0.1,text_kwargs={'color': 'black','fontsize' : 10},
                            poly_kwargs={'facecolor': (0, 0, 0)})
else:
    if nn_tags["Powerlaw"]:
        annotation.slope_marker((0.04, 1e-7), (3, 1), ax=ax[0],size_frac=0.15,pad_frac=0.1,text_kwargs={'color': 'black'},
                                poly_kwargs={'facecolor': (0, 0, 0)})
        annotation.slope_marker((0.025, 1e-3), (2, 1), ax=ax[0],size_frac=0.15,pad_frac=0.1,invert=True,text_kwargs={'color': 'black'},
                                poly_kwargs={'facecolor': (0, 0, 0)})
        # annotation.slope_marker((0.04, 1e-3), (1.4, 1), ax=ax[1],size_frac=0.15,pad_frac=0.1,text_kwargs={'color': 'black'},
        #                         poly_kwargs={'facecolor': (0, 0, 0)})
        # annotation.slope_marker((0.04, 1e-5), (1.4, 1), ax=ax[3],size_frac=0.15,pad_frac=0.1,text_kwargs={'color': 'black'},
        #                         poly_kwargs={'facecolor': (0, 0, 0)})
        annotation.slope_marker((0.04, 1e-4), (2, 1), ax=ax[2],size_frac=0.15,pad_frac=0.1,text_kwargs={'color': 'black'},
                                poly_kwargs={'facecolor': (0, 0, 0)})
    
if savefig:
    if sim_type=="unit":
        fig.savefig(r"unittest.png",transparent=False, bbox_inches='tight', pad_inches=0.05) 
    else:
        if nn_tags["Bingham"]:
            #fig.savefig(r"Convergence_{}_tau{}_hmax{}.png".format(active[0][1],tau,hmax),transparent=False, bbox_inches='tight', pad_inches=0.05) 
            #fig.savefig(r"Constitutive_{}_tau{}_kappa{}.png".format(active[0][1],tau,kappa),transparent=False, bbox_inches='tight', pad_inches=0.05)
            fig.savefig(r"{}_tau{}_kappa{}.png".format(active[0][1],tau,kappa),transparent=False, bbox_inches='tight', pad_inches=0.05)
        if nn_tags["Powerlaw"]:
            fig.savefig(r"{}_r{}.png".format(active[0][1],r),transparent=False, bbox_inches='tight', pad_inches=0.05) 


     

