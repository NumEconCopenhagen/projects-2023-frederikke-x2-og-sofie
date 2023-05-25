
from types import SimpleNamespace

import numpy as np
from scipy import optimize
from tabulate import tabulate

import pandas as pd 
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

import types

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wFages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        # splitting the function
        if par.sigma == 0 :
           H = np.min(HM,HF)
        elif par.sigma==1:
            H = HM**(1-par.alpha)*HF**par.alpha
        else :
           HM = np.fmax(HM, 1e-07)
           HF = np.fmax(HF, 1e-07)
           inner = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))
           H = np.fmax(inner, 1e-07)**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)

        if par.rho != 1:
            utility = np.where(Q >= 1e-08, Q ** (1 - par.rho) / (1 - par.rho), 1e-08 ** (1 - par.rho) / (1 - par.rho))
        else:
            utility = np.where(Q >= 1e-08, np.log(Q), np.log(1e-08))
    
        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve_cont(self,do_print=False):
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        #Using the optimizer function to minimize the utility function
        from scipy import optimize
        initial_guess = [11, 11, 11, 11] #Our guess      
        objective_function = lambda x: -self.calc_utility(x[0], x[1], x[2], x[3]) #Using the utility function of LM, HM, LF, HF
        constraint1 = ({'type': 'ineq', 'fun': lambda x: 24-x[0]-x[1]}) #Constraint to ensure that the male can not work or be home more than 24 hours
        constraint2  = ({'type': 'ineq', 'fun': lambda x: 24-x[2]-x[3]}) #Constraint to ensure that the female can not work or be home more than 24 hours
        constraints = [constraint1, constraint2] #Making a list of the constraints
        bounds = [(0, 24)]*4 #Bounds to ensure that the time used on each parameter is included in a day of 24 hours (4 parameters)
        
        res = optimize.minimize(objective_function, initial_guess, method='SLSQP', constraints=constraints, bounds=bounds, tol = 1e-08) #Making the optimizer function
    

        opt.LM = res.x[0]
        opt.HM = res.x[1]
        opt.LF = res.x[2]
        opt.HF = res.x[3]

        return opt

        pass    


    def solve_wF_vec(self,discrete=False):
        #used to illustrate question 4

        sol = self.sol
        par = self.par

        for i, w_F in enumerate(par.wF_vec):
            par.wF = w_F
            if discrete:
                opt = self.solve_discrete()
            else:
                opt = self.solve_cont()
            if opt is not None:
                sol.LM_vec[i], sol.HM_vec[i], sol.LF_vec[i], sol.HF_vec[i] = opt.LM, opt.HM, opt.LF, opt.HF
                

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
class NewModelQ5:
    def __init__(self):
        """ The model is setup as the same way for question 1-4, but here child care is added"""
        
        # a. create namespaces
        par = self.par = SimpleNamespace()

        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0
        
        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. child care
        par.gamma = 0.5 
        par.delta = 1.0 

        # nr. of children
        par.N = 5

        # g. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    # Utility calculation
    def calc_utility(self, LM, HM, LF, HF, N):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM * LM + par.wF * LF

        # b. home production
        if par.sigma == 0 :
           H = np.min(HM,HF)
        elif par.sigma==1:
            H = HM**(1-par.alpha)*HF**par.alpha
        else :
           HM = np.fmax(HM, 1e-07)
           HF = np.fmax(HF, 1e-07)
           inner = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))
           H = np.fmax(inner, 1e-07)**(par.sigma/(par.sigma-1))
       
        # c. Child care is added 
        child_care = N ** par.gamma

        # d. total consumption utility with childcare
        Q = C ** par.omega * H ** (1 - par.omega) * child_care ** par.delta
        utility = np.fmax(Q, 1e-8) ** (1 - par.rho) / (1 - par.rho)

        # e. disutility of work
        epsilon_ = 1 + 1 / par.epsilon
        TM = LM + HM
        TF = LF + HF
        disutility = par.nu * (TM ** epsilon_ / epsilon_ + TF ** epsilon_ / epsilon_)

        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() 
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility with child care as an additional variable
        u = self.calc_utility(LM,HM,LF,HF,par.N)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) 
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
    
    def solve(self, do_print=False):
        """ solve model continuously """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # Define the objective function to be maximized with child care, with a solution method
        def objective(x):
            LM, HM, LF, HF = x
            return -self.calc_utility(LM, HM, LF, HF, par.N)
    
        # The constraints are defined
        def constraint(x):
            LM, HM, LF, HF = x
            return np.array([24 - (LM + HM), 24 - (LF + HF)])
    
        # Set the initial guess
        x0 = np.array([11, 11, 11, 11])
        
        # An optimize minimize function is used to max the utility with the constriant
        res = optimize.minimize(objective, x0, method='trust-constr', constraints={'type': 'ineq', 'fun': constraint})
        
        # The objects being max
        LM, HM, LF, HF = res.x
        
        # The solution is saved
        sol.LM = LM
        sol.HM = HM
        sol.LF = LF
        sol.HF = HF
        
        # The solution is printed
        if do_print:
            for k,v in sol.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return sol
      

    def solve_wF_vec(self, discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

        # Calculate log ratios for each wF
        log_ratios = []
        for wF in par.wF_vec:
            par.wF = wF
            results = self.solve(discrete)
            log_ratios.append(np.log(results.HF / results.HM))

        # Fit a linear regression
        X = np.log(np.array(par.wF_vec) / par.wM).reshape(-1, 1)
        y = np.array(log_ratios)
        lin_reg = LinearRegression().fit(X, y)

        # Save the estimated coefficients
        sol.beta0 = lin_reg.intercept_
        sol.beta1 = lin_reg.coef_[0]

    def run_regression(self):
        # Define alpha, sigma, and wF values
        alpha_list = [0.25, 0.5, 0.75]
        sigma_list = [0.5, 1.0, 1.5]
        wF_list = [0.8, 0.9, 1.0, 1.1, 1.2]

        # Create an empty dictionary to store results
        results_dict = {}

        # Loop over all combinations of alpha, sigma, and wF
        for alpha in alpha_list:
            for sigma in sigma_list:
                for wF in wF_list:
                    # Set parameter values
                    self.par.alpha = alpha
                    self.par.sigma = sigma
                    self.par.wF = wF

                    # Solve the model
                    sol = self.solve()

                    # Calculate log ratios
                    log_HF_HM = np.log(sol.HF / sol.HM)
                    log_wF_wM = np.log(self.par.wF / self.par.wM)

                    # Store results in dictionary
                    if (alpha, sigma) not in results_dict:
                        results_dict[(alpha, sigma)] = []
                    results_dict[(alpha, sigma)].append((wF, log_HF_HM, log_wF_wM))

        # Initialize table
        table = []

        # Perform regression for each combination of alpha and sigma
        for alpha in alpha_list:
            for sigma in sigma_list:
                # Initialize arrays for regression
                X = np.empty((0,))
                Y = np.empty((0,))

                # Fill arrays with data
                for wF, log_HF_HM, log_wF_wM in results_dict[(alpha, sigma)]:
                    X = np.append(X, log_wF_wM)
                    Y = np.append(Y, log_HF_HM)

                # Perform linear regression
                A = np.vstack([np.ones(X.size), X]).T
                beta, sse, _, _ = np.linalg.lstsq(A, Y, rcond=None)

                # Add regression results to table
                row = [alpha, sigma, beta[0], beta[1], sse]
                table.append(row)

        df1 = pd.DataFrame(table, columns=["Alpha", "Sigma", "Beta0", "Beta1", "SSE"])

        # Format floating-point numbers in DataFrame
        df1 = df1.round({"Alpha": 2, "Sigma": 1, "Beta0": 4, "Beta1": 4, "SSE": 4})

        # Format and return table
        return df1