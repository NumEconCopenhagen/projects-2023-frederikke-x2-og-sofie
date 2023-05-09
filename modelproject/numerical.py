# a. Python packages are imported
import numpy as np
from scipy import optimize
import sympy as sm
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# b. The OLG model is contructed as a numerical solution class

class NumericalmodelclassOLG():
    
    #Equation system of the OLG model

    def __init__(self, **kwargs):

        #Within the class the attributes of the objects are initialized
         # self refers to the variables and paremeters from the "setup(self)"

        self.parameters_initial()
        self.parameters_new(kwargs)
        self.functions_model()


    def parameters_initial(self):

        #The parameters and variables are being defined
         # self refers to the variables and paremeters from the "setup(self)"


        # a. Parameter values

       
        self.alpha = 1/3  # 1. The share of capital in Cobb-douglas 

        
        self.rho = 0.25  # 2. Time preference parameter 

        
        self.n = 0.01  # 3. Constant growth rate of population

        
        self.tau = 0.25  # 4. Share of wages to  contribution

        
        self.A = 20  # 5. Technology 


        # b. The variables of the transition curve is defined
            # self refers to the variables and paremeters from the "setup(self)" 

        
        self.kN = 1000  # 1. Amount of grids on the transition diagram

        
        self.kmin = 1e-6  # 2. The minimum level of capital

        
        self.kmax = 20  # 3. The maximum level of capital


    def parameters_new(self, kwargs):

        # self refers to the variables and paremeters from the "setup(self)"

        for key, value in kwargs.items():
            setattr(self, key, value)


    def functions_model(self):

       #Defining functions of the model
            #Args: 
            # self (setup): Parameters from the OLG class
            # c_t (float): Consumption in period t
            # k_t (float): Capital per capita in period t

        #Returns: 
            #u (function): Utility of current consumption
            # y (function): Production per capita
            # y_prime (function): Derivative of production per capita

       
        # a. The lower bound of capital and consumption is defined as marginally above zero
        epsilon = 1e-10

        self.ss = 0
        # b. Defining the utility function 
        self.u = lambda c_t: np.log(np.fmax(c_t,epsilon))

        # c. Defining the production per capita function and the function of the derivative of production per capita
        self.y = lambda k_t: self.A * np.fmax(k_t,epsilon)**self.alpha
        self.y_prime = lambda k_t: self.alpha * self.A * np.fmax(k_t,epsilon)**(self.alpha-1)
    
    
    def optimization_firm(self,k_t):
        
       #Deriving the optimal factor prices in order to solve the firm´s optimization problem
            #Args: 
            # self (setup): Parameters from the OLG class
            #k_t (float): Capital per capita in period t

        #Returns: 
            #  w_t (function): Real wage
            #R_t (function): Gross real interest rate
     
        R_t = self.y_prime(k_t)
        w_t = self.y(k_t) - self.y_prime(k_t) * k_t
        
        return R_t, w_t
        
    
    
    def lifetime_utility(self, c_t, w_t, R_tnext, w_tnext,):
        
        #The lifetime utility of consumption is defined for period t and t+1
            #Args: 
                #self (setup): Parameters from the OLG class
                # c_t (float): Consumption in period t
                # w_t (float): Real wage in period t
                # w_tnext (float): Real wage in period t+1
                # R_tnext (float): Gross real interest rate in period t+1
        #Returns: 
            #u_lifetime (function): Lifetime utility

      
        
        # a. Savings defined from the budget constraint
        saving_b = ((1 - self.tau) * w_t - c_t)
        
        # b. In period t+1 consumption is defined from the budget constraint
        c_tnext = saving_b * R_tnext + (1 + self.n) * self.tau * w_tnext
        
        # c. Lifetime utility when young and old from consumption, respectively
        u_lifetime = self.u(c_t) + (1 / (1 + self.rho)) * self.u(c_tnext)
        
        return u_lifetime
    
    
    def household_optimization(self, w_t, R_tnext, w_tnext):
        
        
        #Finding the optimal saving that maximizes life time utility of households
            #Args:
            #   self (setup): Parameters from the OLG class
            # w_t (float): Real wage in period t
            #  w_tnext (float): Real wage in period t+1
            # R_tnext (float): Gross real interest rate in period t+1
            
        # Returns:
            # s_t (function): Optimal savings in period t
        
       
        
        # a. For consumption the upper and lower bound are defined in period t
        cmax = (1 - self.tau) * w_t 
        cmin = 0
        
        # b. Through bounded minimization for scalar products the optimal savings is found 
        obj = lambda c_t: -self.lifetime_utility(c_t, w_t, R_tnext, w_tnext)
        c_t = optimize.fminbound(obj, cmin, cmax)
        
        # c. By the budget contraint savings is defined
        s_t = (1 - self.tau) * w_t - c_t
        
        return s_t
        
        
    def equilibrium(self, k_t1, disp=0):
        
      
        #Equilibrium of capital per capita is found
         # Args: 
            # self (setup): Parameters from the OLG class
            # k_tnext (float): Capital per capita in period t+1
        # Return:
            # k_t (function): Optimal capital per capita
        
       
        
        # a. For period t+1 the factor prices for labout and capitla are derived
        R_tnext, w_tnext = self.optimization_firm(k_t1)
        
        # b. Optimal savings 
        def obj(k_t):
            
            
            _R_t, w_t = self.optimization_firm(k_t)  # 1. For period t the factor price of labout is derived
            
            
            s_t = self.household_optimization(w_t, R_tnext, w_tnext)  # 2. Optimal savings
            
            
            dev = (k_t1 - s_t / (1 + self.n))**2  # 3. The deviation that should be minimized is defined
            
            return dev
        
        # c. Upper and lower bound for capital is defined
        kmin = 0
        kmax = self.kmax
        
        
        # d. Through bounded minimization for scalar products the optimal level of capital per capita is found     
        k_t = optimize.fminbound(obj, kmin, kmax, disp=disp)
        
        
        return k_t
    
    
    def numericaltransition_curve(self):
        
        
        #Finding optimal capital accumulation 
            # Args: 
                # self (setup): Parameters from the OLG class
        #Return:
            # plot_k_tnext (function): Capital in period t+1
            # plot_k_t (function): Capital in period t
            
      
        
        # a. Creating an empty numpy linspace from the minimum to the maximum bound of capital per capita, representing k_t+1
        self.plot_k_t1 = np.linspace(self.kmin, self.kmax, self.kN)
        
        # b. For each value of k_t+1 loop through in order to find corresponding equilibrium value of k_t 
        
       
        self.plot_k_t = np.empty(self.kN)   # 1. Finding the steady state value capital when k_t is appriximately close to k_t+1
        for i, k_t1 in enumerate(self.plot_k_t1):
            k_t = self.equilibrium(k_t1)
            self.plot_k_t1[i] = k_t
            if (np.abs(k_t1 - k_t) < 0.01 and k_t > 0.01 and k_t < 19):
                self.ss = k_t 
            
    
    def plot_numericaltransition_curve(self, ax, **kwargs):
       
    
        #Plotting the transition curve of capital accumulation 
            # Args: 
                # self (setup): Parameters from the OLG class
                # ax (ndarray?): Subplot axis 
            # Return:
                #transition_curve (function): grapich presentation of optimal capital accumulation
            
        
        ax.plot(self.plot_k_t, self.plot_k_t1, **kwargs)
        
        ax.set_xlabel("$k_t$")
        ax.set_ylabel("$k_t+1$")
        ax.set_xlim(0,self.kmax)
        ax.set_ylim(0,self.kmax)
        
        
    def plot_numerical45_curve(self, ax, **kwargs):
        
        
        #For k_t+1=k_t plotting the 45 degree curve
            # Args: 
                #  self (setup): Parameters from the OLG class
                # ax (ndarray?): Subplot axis 
            # Return:
                # 45 (function): Linear function 
        
       
        
        ax.plot([self.kmin, self.kmax], [self.kmin, self.kmax], **kwargs)
        