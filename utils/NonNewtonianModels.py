#!/usr/bin/env python3
import numpy as np
from ngsolve import *

class Newtonian:

    def __init__(self,options):
        self.options = options
        #Solution and viscosity
        psi = (x*(x-options["L"])*(y-options["H"]/2)*(y+options["H"]/2))**2
        u_xe = psi.Diff(y)
        u_ye = -psi.Diff(x)
        pressure = x**5+y**5-(options["L"]**5)/6

        #Calculate exact force
        grad_ue = -2*options["nu"]*0.5*(self.cust_grad((u_xe,u_ye)) + self.cust_grad((u_xe,u_ye)).trans)
        self.force = CoefficientFunction((grad_ue[0,0].Diff(x)+grad_ue[0,1].Diff(y),grad_ue[1,0].Diff(x)+grad_ue[1,1].Diff(y)))
        self.force += CoefficientFunction((pressure.Diff(x),pressure.Diff(y)))
        self.exact_u = CoefficientFunction((u_xe,u_ye))
        self.exact_Du = 0.5*(self.cust_grad((u_xe,u_ye)) + self.cust_grad((u_xe,u_ye)).trans)
        self.exact_sigma = 2*options["nu"]*0.5*(self.cust_grad((u_xe,u_ye)) + self.cust_grad((u_xe,u_ye)).trans)
        self.pressure = pressure

    def cust_grad(self,arr):
        return CoefficientFunction((arr[0].Diff(x),arr[0].Diff(y),arr[1].Diff(x),arr[1].Diff(y)),dims=(2,2))

    def law(self,Du,S):
        return CoefficientFunction((self.options["nu"]))

    def draw(self,y_channel):
        u_ex = 4*(self.options["L"]/2)**4*y_channel*(y_channel-self.options["H"]/2)*(y_channel+self.options["H"]/2)
        return u_ex
    
    def __str__(self):
        return "Newtonian"

class NonNewtonian:
    def __init__(self,options,exact_Du,exact_u):
        self.options = options
        tau_xy = self.options["dp"]*y
        self.exact_sigma = CoefficientFunction((0,tau_xy,tau_xy,0),dims=(2,2))
        self.exact_Du = exact_Du
        self.exact_u = exact_u
        self.pressure = CoefficientFunction((0))

class Powerlaw(NonNewtonian):
    def __init__(self,options):
        self.r = options["r"]
        self.K = options["nu"]
        Du = IfPos(y,
                -(1/self.K*options["dp"]*(-y))**(1/(self.r-1)),
                 (1/self.K*options["dp"]*(y))**(1/(self.r-1)))
        exact_Du = 0.5*CoefficientFunction((0,Du,Du,0),dims=(2,2))
        exact_u = CoefficientFunction(
            ((1/self.K*(-options["dp"]))**(1/(self.r-1))*(self.r-1)/self.r*(options["H"]/2)**(self.r/(self.r-1))*(1-(2*IfPos(y,y,-y)/options["H"])**(self.r/(self.r-1))),
            0))
        super().__init__(options,exact_Du,exact_u)

 
    def draw(self,y_channel):
        return (1/self.K*(-self.options["dp"]))**(1/(self.r-1))*(self.r-1)/self.r*(self.options["H"]/2)**(self.r/(self.r-1))*(1-(2*np.abs(y_channel)/self.options["H"])**(self.r/(self.r-1)))

    def law(self,Du,S):
        abs_Du = 0.5*InnerProduct(Du,Du)
        return 2*self.K*(4*abs_Du)**((self.r-2)/2)

    def constitutive(self,Du,S):
        abs_Du = 0.5*InnerProduct(Du,Du)
        return S - 2*self.K*(4*abs_Du)**((self.r-2)/2)*Du

    def __str__(self):
        return "Powerlaw"

class Bingham(NonNewtonian):

    def __init__(self,options):
        self.tau_y = options["tau_y"]
        self.kappa = options["kappa"]
        self.reg_law = options["law"]

        y_bot = self.tau_y/options["dp"]
        y_top = -self.tau_y/options["dp"]
        exact_u = CoefficientFunction((IfPos(y+np.abs(y_bot),IfPos(y-np.abs(y_top),
        0.5/options["nu"]*(-options["dp"])*(options["H"]**2/4-y**2)-self.tau_y/options["nu"]*(-y+0.5*options["H"]),
        0.5/options["nu"]*(-options["dp"])*(options["H"]**2/4-np.square(y_bot))-self.tau_y/options["nu"]*(y_bot+0.5*options["H"])),
        0.5/options["nu"]*(-options["dp"])*(options["H"]**2/4-y**2)-self.tau_y/options["nu"]*(y+0.5*options["H"])),0))

        Du = IfPos(y+np.abs(y_bot),IfPos(y-np.abs(y_top),
        (options["dp"]*y+self.tau_y)/options["nu"],
        0),
        (options["dp"]*y-self.tau_y)/options["nu"])
        exact_Du = 0.5*CoefficientFunction((0,Du,Du,0),dims=(2,2))
        super().__init__(options,exact_Du,exact_u)

    def draw(self,y_channel):
        u_ex = np.linspace(-self.options["H"]/2,self.options["H"]/2,y_channel.size)
        y_bot = self.tau_y/self.options["dp"]
        y_top = -self.tau_y/self.options["dp"]
        mask_bot = y_channel<y_bot
        mask_top = y_channel>y_top
        u_ex[:] = 0.5/self.options["nu"]*(-self.options["dp"])*(self.options["H"]**2/4-np.square(y_bot))-self.tau_y/self.options["nu"]*(y_bot+0.5*self.options["H"])
        u_ex[mask_bot] = 0.5/self.options["nu"]*(-self.options["dp"])*(self.options["H"]**2/4-np.square(y_channel[mask_bot]))-self.tau_y/self.options["nu"]*(y_channel[mask_bot]+0.5*self.options["H"])
        u_ex[mask_top] = 0.5/self.options["nu"]*(-self.options["dp"])*(self.options["H"]**2/4-np.square(y_channel[mask_top]))-self.tau_y/self.options["nu"]*(-y_channel[mask_top]+0.5*self.options["H"])
        return u_ex

    def law(self,Du,S):
        abs_Du = sqrt(self.kappa**2 + 2*InnerProduct(Du,Du))
        nn_laws = {
        "G1":
            2*(self.options["nu"] + self.tau_y/abs_Du),
        "G2":
            2*self.options["nu"] + self.tau_y/sqrt(0.5*InnerProduct(Du,Du))*(1-exp(-2*sqrt(0.5*InnerProduct(Du,Du))/self.kappa))           
        }
        return nn_laws[self.reg_law]

    def constitutive(self,Du,S):
        abs_Du = sqrt(0.5*InnerProduct(Du,Du))
        return abs_Du*S - (self.tau_y + 2*self.options["nu"]*abs_Du)*Du 

    def __str__(self):
        return "Bingham"

