#Python packages are imported
import numpy as np
from scipy import optimize
import sympy as sm
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

#To contruct the OLG model we define a numerical class

class NumericalmodelclassOLG():

    def __init__(self, **kwargs):
        #Within the class the attributes of the objects are initialized
         # self refers to the variables and paremeters from the "setup(self)"
        self.par_init()
        self.par_new(kwargs)
        self.func_model()


    def par_init(self):
        #The parameters and variables are being defined
         # self refers to the variables and paremeters from the "setup(self)"

        #Parameter values
        self.alpha = 1/3        #The share of capital in Cobb-douglas 
        self.n = 0.01           #Constant growth rate of population
        self.rho = 0.25         #Time preference parameter
        self.tau = 0.25         #Share of wages to  contribution
        self.A = 20             #Technology 

        #The variables of the transition curve is defined
            #self refers to the variables and paremeters from the "setup(self)" 
        self.k_N = 1000         #Amount of grids on the transition diagram
        self.k_max = 20         #The maximum level of capital
        self.k_min = 1e-6       #The minimum level of capital


    def par_new(self, kwargs):
        #self refers to the variables and paremeters from the "setup(self)"
        for key, value in kwargs.items():
            setattr(self, key, value)


    def func_model(self):
        #The lower bound of capital and consumption is defined as marginally above zero
        epsilon = 1e-10

        self.ss = 0
        #Defining the utility function 
        self.u = lambda c_t: np.log(np.fmax(c_t,epsilon))

        #Defining the production per capita function and the function of the derivative of production per capita
        self.y = lambda k_t: self.A * np.fmax(k_t,epsilon)**self.alpha
        self.y_p = lambda k_t: self.alpha * self.A * np.fmax(k_t,epsilon)**(self.alpha-1)
    
    
    def opt_firm(self,k_t):
        #w_t: Real wage
        #R_t: Gross real interest rate
        R_t = self.y_p(k_t)
        w_t = self.y(k_t) - self.y_p(k_t) * k_t
        return R_t, w_t
        
    
    def life_u(self, c_t, w_t, R_t1, w_t1,):
        
        #The lifetime utility of consumption is defined for period t and t+1. c_t: Consumption in period t. 
            #w_t: Real wage in period t
            #w_t1: Real wage in period t+1
            # R_t1: Gross real interest rate in period t+1
            #u_life: Lifetime utility
      
        #From the budget constraint we now define:
        #The savings
        s_b = ((1 - self.tau) * w_t - c_t)
        
        #The consumption period t+1
        c_t1 = s_b * R_t1 + (1 + self.n) * self.tau * w_t1
        
        #The utility when young and old
        u_life= self.u(c_t) + (1 / (1 + self.rho)) * self.u(c_t1)
        return u_life
    
    
    def household_opt(self, w_t, R_t1, w_t1):
        #For consumption the upper and lower bound are defined in period t
        c_max = (1 - self.tau) * w_t 
        c_min = 0
        
        #Through bounded minimization for scalar products the optimal savings is found 
        obj = lambda c_t: -self.life_u(c_t, w_t, R_t1, w_t1)
        c_t = optimize.fminbound(obj, c_min, c_max)
        
        #By the budget contraint savings is defined
        s_t = (1 - self.tau) * w_t - c_t
        return s_t
        
        
    def eq(self, k_t1, disp=0):
        #For period t+1 the factor prices for labout and capitla are derived
        R_t1, w_t1 = self.opt_firm(k_t1)
        
        #Optimal savings 
        def obj(k_t):
            _R_t, w_t = self.opt_firm(k_t)               #For period t the factor price of labout is derived
            s_t = self.household_opt(w_t, R_t1, w_t1)    #Optimal savings
            dev = (k_t1 - s_t / (1 + self.n))**2         #The deviation that should be minimized is defined
            return dev
        
        #Upper and lower bound for capital is defined
        k_min = 0
        k_max = self.k_max

        #Through bounded minimization for scalar products the optimal level of capital per capita is found     
        k_t = optimize.fminbound(obj, k_min, k_max, disp=disp)
        return k_t
    
    
    def numericaltransition_curve(self):    
        #we can from the min to max bound of capital per capita, create an empty np linspace linspace
        self.plot_k_t1 = np.linspace(self.k_min, self.k_max, self.k_N)
        
        #In order to find the equilibrium value of k_t, we loop through every value of k_t+1
        self.plot_k_t = np.empty(self.k_N)   #Finding the steady state value capital when k_t is appriximately close to k_t+1
        for i, k_t1 in enumerate(self.plot_k_t1):
            k_t = self.eq(k_t1)
            self.plot_k_t[i] = k_t
            if (np.abs(k_t1 - k_t) < 0.01 and k_t > 0.01 and k_t < 19):
                self.ss = k_t 
            
    
    def plot_numericaltransition_curve(self, ax, **kwargs):
        #The transition curve is given by: 
        ax.plot(self.plot_k_t, self.plot_k_t1, **kwargs)
        ax.set_xlabel("$k_t$")
        ax.set_ylabel("$k_t+1$")
        ax.set_xlim(0,self.k_max)
        ax.set_ylim(0,self.k_max)
        
        
    def plot_numerical45_curve(self, ax, **kwargs):
        #For k_t+1=k_t plotting the 45 degree curve
        ax.plot([self.k_min, self.k_max], [self.k_min, self.k_max], **kwargs)
        