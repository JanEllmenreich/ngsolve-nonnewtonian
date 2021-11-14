#!/usr/bin/env python3
from ngsolve import *
import numpy as np


class Solvers:

    def __init__(self,options):

        self.linesearch = {"l2" : self.L2Linesearch,
                           "backtracking": self.BacktrackingLinesearch,
                           "2division": self.DivisionLinesearch,
                           "none": self.NoLinesearch}                       
        self.condense = options["condense"]
        self.options = options

    def solve(self,a_newton,a_fixpoint,gfu,res):
        info = {}
        if self.options["solver"] == "newton":
            a = a_newton
            damping = self.options["damp_newton"]
        else:
            a = a_fixpoint
            damping = self.options["damp_fix"]

        du = gfu.vec.CreateVector()
        fes = gfu.space

        with TaskManager():
            for it in range(1,self.options["it_max"]+1):
                a.AssembleLinearization(gfu.vec)
                a.Apply(gfu.vec, res)

                if self.condense:
                    res.data += a.harmonic_extension_trans *res
                    du.data = a.mat.Inverse(freedofs=fes.FreeDofs(coupling=True),inverse=self.options["inverse"])*res
                    du.data += a.inner_solve*res
                    du.data += a.harmonic_extension*du
                else:
                    du.data = a.mat.Inverse(fes.FreeDofs(),inverse=self.options["inverse"]) * res

                ##Update Solution
                if self.options['solver'] == 'newton':
                    self.linesearch[self.options["linesearch"]](gfu,a,res,du,fes,damping)
                else:
                    gfu.vec.data -= damping*du

                #stopping criteria
                stopcritval = sqrt(abs(InnerProduct(du,res)))

                #Print rates
                if self.options["printrates"]:
                    print ("Iteration = {}, err = {}".format(it,stopcritval))

                #Switch from Fixed Point to Newton
                if self.options["switch"] and stopcritval < self.options["tol_switch"]:
                    a = a_newton
                    damping = self.options["damp_newton"]
                    self.options["switch"] = False

                #Stopping criterion
                if stopcritval < self.options["tol_min"]:
                    print("{} Converged at iteration {} of {}\n".format(self.options["solver"],it,self.options["it_max"]))
                    info["converged?"] = True; info["it"] = it
                    return info
        print("{} did not converge in {} iterations.".format(self.options["solver"],self.options["it_max"]))
        print("Last residual: {}".format(stopcritval))
        info["converged?"] = False; info["it"] = self.options["it_max"]
        return info

    def L2Linesearch(self,gfu,a,res,du,fes,damping,n=1):
        lambdas = np.array([0,damping])
        residuals = np.zeros(3)
        rhs = gfu.vec.CreateVector()
        for i in range(n):
            #Calculate residuals
            rhs.data = gfu.vec - lambdas[0]*du
            a.Apply(rhs, res)
            residuals[0] = np.sum(np.square(res.FV().NumPy()[fes.FreeDofs()]))
            rhs.data = gfu.vec - 0.5*(lambdas[0]+lambdas[1])*du
            a.Apply(rhs, res)
            residuals[1] = np.sum(np.square(res.FV().NumPy()[fes.FreeDofs()]))
            rhs.data = gfu.vec - lambdas[1]*du
            a.Apply(rhs, res)
            residuals[2] = np.sum(np.square(res.FV().NumPy()[fes.FreeDofs()]))
            #print("Residuals: {} {} {}".format(residuals[0],residuals[1],residuals[2]))

            delta_lam = lambdas[1] - lambdas[0]
            Delta_lam1 = (3*residuals[2] - 4*residuals[1] +   residuals[0])/ delta_lam
            Delta_lam2 = (  residuals[2] - 4*residuals[1] + 3*residuals[0])/ delta_lam

            lambdas[0] = lambdas[1]
            lambdas[1] -= (Delta_lam1*delta_lam)/(Delta_lam1 - Delta_lam2)
            
        #print("Lambda L2 Linesearch: {}".format(lambdas[1]))
        gfu.vec.data -= lambdas[1] * du

    def BacktrackingLinesearch(self,gfu,a,res,du,fes,dampfactor):
        #Setup lambdas and residuals
        lambdas = dampfactor*np.ones(3)
        residuals = np.zeros(3)
        lambda_min = 1e-5
        alpha = 1e-4

        #Calculate first residual
        residuals[0] = 0.5*np.sum(np.square(res.FV().NumPy()[fes.FreeDofs()]))

        #Check if step descents enough
        gfu.vec.data -= lambdas[0]*du
        a.Apply(gfu.vec, res)
        residuals[1] = 0.5*np.sum(np.square(res.FV().NumPy()[fes.FreeDofs()]))
        if residuals[1] < residuals[0]*(1-2*alpha*lambdas[0]):
            #print("Residuals[1] = {}, Residuals[0] = {}".format(residuals[1],residuals[0]))
            #print("Normal interpolation")
            pass
        else:
            #print("Quadratic interpolation")
            lambdas[2] = residuals[0]/(residuals[1] + residuals[0])
            if lambdas[2] < 0.1: lambdas[2] = 0.1
            gfu.vec.data -= (lambdas[2]-lambdas[0])*du
            a.Apply(gfu.vec, res)
            residuals[2] = 0.5*np.sum(np.square(res.FV().NumPy()[fes.FreeDofs()]))

            if residuals[2] < residuals[0]*(1-2*alpha*lambdas[0]):
                pass
            else:
                #print("Cubic interpolation")

                #Cubic interpolation
                while residuals[2] > residuals[0]*(1-2*alpha*lambdas[2]):
                    cubic_mat = 1/(lambdas[2]-lambdas[1])*np.array([[1/lambdas[2]**2,-1/lambdas[1]**2],
                                                                    [-lambdas[1]/lambdas[2]**2,lambdas[2]/lambdas[1]**2]])
                    cubic_vec = np.array([residuals[2]-residuals[0]*(1-2*lambdas[2]),
                                          residuals[1]-residuals[0]*(1-2*lambdas[1])])
                    coef = cubic_mat.dot(cubic_vec)
                    # coef =  np.array([(residuals[2]*2*(lambdas[2]-1)+residuals[0]*2*(lambdas[2]+1))/lambdas[2]**3,
                                  # (residuals[2]*(3-2*lambdas[2]-residuals[0]*(4*lambdas[2]+3)))/lambdas[2]**2 ])
                    lambdas[1] = lambdas[2]; residuals[1] = residuals[2]
                    lambdas[2] = (-coef[1]+np.sqrt(np.square(coef[1])+6*coef[0]*residuals[0]))/(3*coef[0])
                    if lambdas[2]>0.5*lambdas[1]: lambdas[2] = 0.5*lambdas[1]
                    elif lambdas[2] < 0.1*lambdas[1]: lambdas[2] = 0.1*lambdas[1]
                    gfu.vec.data -= (lambdas[2]-lambdas[1])*du
                    a.Apply(gfu.vec, res)
                    residuals[2] = 0.5*np.sum(np.square(res.FV().NumPy()[fes.FreeDofs()]))
                    if lambdas[2] < lambda_min:
                        break

    def DivisionLinesearch(self,gfu,a,res,du,fes,dampfactor):
        #Setup lambdas and residuals
        lambdas = dampfactor*np.ones(2)
        residuals = np.zeros(2)
        alpha = 1e-4
        lambda_min = 1e-3

        #Calculate first residual
        residuals[0] = 0.5*np.sum(np.square(res.FV().NumPy()[fes.FreeDofs()]))

        #Check if step descents enough
        gfu.vec.data -= lambdas[1]*du
        a.Apply(gfu.vec, res)
        residuals[1] = 0.5*np.sum(np.square(res.FV().NumPy()[fes.FreeDofs()]))
        while residuals[1] > residuals[0]*(1-2*alpha*lambdas[0]):
            lambdas[0] = lambdas[1]; lambdas[1] *= 0.5
            print(lambdas[1])
            gfu.vec.data -= (lambdas[1]-lambdas[0])*du
            residuals[1] = 0.5*np.sum(np.square(res.FV().NumPy()[fes.FreeDofs()]))
            if lambdas[1] < lambda_min:
                break



    def NoLinesearch(self,gfu,a,res,du,fes,dampfactor):
        gfu.vec.data -= dampfactor*du




