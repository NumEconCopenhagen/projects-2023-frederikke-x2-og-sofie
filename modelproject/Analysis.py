from scipy import optimize
import sympy as sm
import numpy as np
from types import SimpleNamespace

class AnalysismodelclassOLG():
    # The OLG model is being coded
    def __init__(self):
        

        self.par = SimpleNamespace()
        self.setup()

    def setup(self):
        par = self.par

        #All the parameters and the variables are being defined

        # a. The variables and parameters used for the utility function 
        par.cy1t = sm.symbols('c_1t')    # 1. Consumption in period t, when young.      
        
        par.co2t = sm.symbols('c_{2t+1}')   # 2. Consumption in period t+1, when old. 
        
        par.rho = sm.symbols('rho')  # 3. The time preference and/or patience  

        
        # b. The variables and parameters used for the budget constraints
        par.r_t = sm.symbols('r_t')  # 1. The interest rate in period t        
        par.r_t1 = sm.symbols('r_{t+1}')    # 2. The interest rate  in period t+1
        
        par.w_t = sm.symbols('w_t')   # 3. The Wage rate in period t       
        par.w_t1 = sm.symbols('w_{t+1}')   # 4. The wage rate in period t+1 
        
        par.d_t1 = sm.symbols('d_{t+1}')   # 5. The benefit in period t+1 
        
        par.n = sm.symbols('n')    # 6. The constant populations growth          
        
        par.tau = sm.symbols('tau')     # 7. The tax rate on wage    
        
        par.s_t = sm.symbols('s_t')    # 8. The rate on savings 
        
        par.lamb = sm.symbols('lambda_t')   # 9. The lagrange multiplier 
        

        # c. The variables and parameters used for the production function
        par.K_t = sm.symbols('K_t')   # 1. Capital in peirod t       
        par.K_t1 = sm.symbols('K_{t+1}')     # 2. Capital in period t+1
        
        par.L_t = sm.symbols('L_t')     # 3. The labour in period t     
        par.L_t1 = sm.symbols('L_{t+1}')     # 4. The labour in period t+1
        
        par.A = sm.symbols('A')    # 5. The constant total factor of productivity         
        
        par.U_t = sm.symbols('U_t')   # 6. Utility in period t       
    
        par.alpha = sm.symbols('alpha')    # 7. The elasticity in the CES function
        
        par.k_t = sm.symbols('k_t')      # 8. Per worker caiptal in period t    
        par.k_t1 = sm.symbols('k_{t+1}')     # 9. Per worker caital in period t+1
        
        par.kss = sm.symbols('k^*')    # 10. Steaty State for caital 
        
        # d. Variables for the Auxiliary regression 
        par.a = sm.symbols('a')
        par.b = sm.symbols('b')
        par.c = sm.symbols('c')

        
    def utilityfunc(self):
        par = self.par
        #'''
        #Defining the utility function

        #Args: 
        #parameters from setup       : see setup(self) for definitions

        #Returns:
        #(sympy function)           : utility function, Ut
        #'''

        return sm.log(par.cy1t)+ sm.log(par.co2t) * 1/(1+par.rho)
    
    
    def budgetconstraint(self):
        par = self.par
        
        #'''
        #Defining the intertemporal budget constraint
        
        #Args: 
        #parameters from setup       : see setup(self) for definition

        #Returns:
        #(sympy function)           : intertemporal budget constraint
        #'''

        # a. Define benefit when old as tau * w_(t+1)
        d_t1 = par.tau * par.w_t1 

        # b. Define period budget constraints as sympy equations
        bc_t1 = sm.Eq(par.cy1t + par.s_t, (1-par.tau) * par.w_t)
        bc_t2 = sm.Eq(par.co2t, par.s_t * (1 + par.r_t1)+ (1 + par.n) * d_t1)

        # c. Solving for savings in the second budget constraint
        bc_t2_s = sm.solve(bc_t2, par.s_t)

        # e. Inserting savings into the first budget constraint
        bc1 = bc_t1.subs(par.s_t, bc_t2_s[0])

        # d. Defining LHS og RHS for budget constraint
        RHS =  sm.solve(bc1, par.w_t*(1 - par.tau))[0]
        LHS = par.w_t * (1 - par.tau)

        return RHS - LHS
           
    
    def euler(self):
        par = self.par
        #'''
        #Finding Euler
        
        #Args: 
        #parameters from setup       : see setup(self) for definition

        #Returns:
        #(sympy function)            : Euler equation
        #'''
        
        
        # a. Setting up the Lagrangian 
        lagrange = self.utility() + par.lamb * self.budgetconstraint()
        
        # b. Finding the first order conditions
        foc1 = sm.Eq(0, sm.diff(lagrange, par.cy1t))
        foc2 = sm.Eq(0, sm.diff(lagrange, par.co2t))
        
        # c. Solving for lambda for the two FOC
        lamb_1 = sm.solve(foc1, par.lamb)[0]
        lamb_2 = sm.solve(foc2, par.lamb)[0]
        
        
        # d. Define Euler
        euler_1 = sm.solve(sm.Eq(lamb_1,lamb_2), par.cy1t)[0]

        # e. Return Euler equation
        return sm.Eq(euler_1, par.cy1t)
    
    
    def optimalsavings(self):
        par = self.par
        #'''
        #Finding the optimal savings function
        
        #Args: 
        #parameters from setup       : see setup(self) for definition

        #Returns:
        #(sympy function)            : optimal savingsfunction
        #'''
        
        # a. Define benefit when old as tau * w_(t+1)
        d_t1 = par.tau * par.w_t1 

        # b. Define period budget constraints 
        bc_t1 = (1-par.tau) * par.w_t - par.s_t 
        bc_t2 = par.s_t * (1 + par.r_t1)+ (1 + par.n) * d_t1

        # c. Substitute budget constraints into Euler
        eul = self.euler()
        sav = (eul.subs(par.cy1t , bc_t1)).subs(par.co2t , bc_t2)
                
        # d. Simplify expression
        saving1 = sm.solve(sav, par.s_t)[0]
        saving2 = sm.collect(saving1, [par.tau])
        saving = sm.collect(saving1, [par.w_t, par.w_t1])
        
        
        # e. Return optimal saving
        return saving       
       

    
    def capitalaccumulation(self):
        par = self.par

        #'''
       # Finding capital accumulation
        
       # Args: 
        #parameters from setup       : see setup(self) for definition

       # Returns:
       # (sympy function)            : capital accumulation
       # '''
        # a. Auxiliary variables
        a = (1 / (1 + (1+par.rho)/(2+par.rho)*((1-par.alpha)/par.alpha) * par.tau))
        b = ((1-par.alpha)*(1-par.tau))/((1+par.n)*(2+par.rho))
        c = par.A * par.k_t**par.alpha
         
        # b. Deriving and displaying the capital accumulation
        kt_00 = par.a * (par.b * par.c)
        kt_01 = ((kt_00.subs(par.a , a)).subs(par.b ,b)).subs(par.c,c)
        k_t = sm.Eq(par.k_t1, kt_01)
        
        print('The capital accumulation is')
                
        return k_t
        
        
        
        
    def steadystate_capital(self):
        par = self.par

       # '''
        #Finding capital in steady state
        
       # Args: 
       # parameters from setup       : see setup(self) for definition

       # Returns:
       # (sympy function)            : steady state for capital
       # '''
        
        # a. Auxiliary variables
        a = (1 / (1 + (1+par.rho)/(2+par.rho)*((1-par.alpha)/par.alpha) * par.tau))
        b = ((1-par.alpha)*(1-par.tau))/((1+par.n)*(2+par.rho))
        c = 1 / (1-par.alpha)
        
        # a. substituting steady state in    
        k_star0 = (a * b * par.A )** par.c
        k_star1 = ((k_star0.subs(par.a , a)).subs(par.b ,b)).subs(par.c,c)
        k_star = sm.Eq(par.kss, k_star1)
        
        print(k_star)
        
        return k_star1