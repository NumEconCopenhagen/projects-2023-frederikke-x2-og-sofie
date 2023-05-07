from scipy import optimize
import sympy as sm
import numpy as np
from types import SimpleNamespace

class OLGmodelClass():
    ''' create the model '''
    def __init__(self):
        

        self.par = SimpleNamespace()
        self.setup()

    def setup(self):
        par = self.par

        '''Define parameters and variables'''

        # a. Utility function
        # i. Consumption when young
        par.c1t = sm.symbols('c_1t')        
        
        # ii. Consumption when old
        par.c2t = sm.symbols('c_{2t+1}')    
        
        # iii. Time preference (patience)
        par.rho = sm.symbols('rho')       

        
        # b. Budget constraints
        # i. Interest rate in period t and t+1
        par.rt = sm.symbols('r_t')          
        par.rt1 = sm.symbols('r_{t+1}')     
        
        # ii. Wage rate in period t and t+1 
        par.wt = sm.symbols('w_t')          
        par.wt1 = sm.symbols('w_{t+1}')     
        
        # iii. Benefit in period t+1 
        par.dt1 = sm.symbols('d_{t+1}')     
        
        # iv. Population growth
        par.n = sm.symbols('n')             
        
        # v. Wage tax rate 
        par.tau = sm.symbols('tau')         
        
        # vi. Savings rate 
        par.st = sm.symbols('s_t')
        
        # vii. Lagrange multiplier
        par.lam = sm.symbols('lambda_t')
        

        # c. Production function
        # i. Capital in period t and t+1
        par.Kt = sm.symbols('K_t')          
        par.Kt1 = sm.symbols('K_{t+1}')     
        
        # ii. Labor in period t and t+1
        par.Lt = sm.symbols('L_t')          
        par.Lt1 = sm.symbols('L_{t+1}')     
        
        # iii. Total factor productivity
        par.A = sm.symbols('A')             
        
        # iv. Utility 
        par.Ut = sm.symbols('U_t')          
        
        # v. CES-elasticity
        par.alpha = sm.symbols('alpha')  
        
        # vi. Capital per worker in period t and t+1
        par.kt = sm.symbols('k_t')          
        par.kt1 = sm.symbols('k_{t+1}')     
        
        # vii. Capital in steady state
        par.kss = sm.symbols('k^*') 
        
        # d. Auxiliary variables
        par.a = sm.symbols('a')
        par.b = sm.symbols('b')
        par.c = sm.symbols('c')

        
    def utility(self):
        par = self.par
        '''
        Defining the utility function

        Args: 
        parameters from setup       : see setup(self) for definitions

        Returns:
        (sympy function)           : utility function, Ut
        '''

        return sm.log(par.c1t)+ sm.log(par.c2t) * 1/(1+par.rho)
    
    
    def budgetconstraint(self):
        par = self.par
        
        '''
        Defining the intertemporal budget constraint
        
        Args: 
        parameters from setup       : see setup(self) for definition

        Returns:
        (sympy function)           : intertemporal budget constraint
        '''

        # a. Define benefit when old as tau * w_(t+1)
        dt1 = par.tau * par.wt1 

        # b. Define period budget constraints as sympy equations
        bc_t1 = sm.Eq(par.c1t + par.st, (1-par.tau) * par.wt)
        bc_t2 = sm.Eq(par.c2t, par.st * (1 + par.rt1)+ (1 + par.n) * dt1)

        # c. Solving for savings in the second budget constraint
        bc_t2_s = sm.solve(bc_t2, par.st)

        # e. Inserting savings into the first budget constraint
        bc1 = bc_t1.subs(par.st, bc_t2_s[0])

        # d. Defining LHS og RHS for budget constraint
        RHS =  sm.solve(bc1, par.wt*(1 - par.tau))[0]
        LHS = par.wt * (1 - par.tau)

        return RHS - LHS
           
    
    def euler(self):
        par = self.par
        '''
        Finding Euler
        
        Args: 
        parameters from setup       : see setup(self) for definition

        Returns:
        (sympy function)            : Euler equation
        '''
        
        
        # a. Setting up the Lagrangian 
        lagrange = self.utility() + par.lam * self.budgetconstraint()
        
        # b. Finding the first order conditions
        foc1 = sm.Eq(0, sm.diff(lagrange, par.c1t))
        foc2 = sm.Eq(0, sm.diff(lagrange, par.c2t))
        
        # c. Solving for lambda for the two FOC
        lamb1 = sm.solve(foc1, par.lam)[0]
        lamb2 = sm.solve(foc2, par.lam)[0]
        
        
        # d. Define Euler
        euler1 = sm.solve(sm.Eq(lamb1,lamb2), par.c1t)[0]

        # e. Return Euler equation
        return sm.Eq(euler1, par.c1t)
    
    
    def optimalsavings(self):
        par = self.par
        '''
        Finding the optimal savings function
        
        Args: 
        parameters from setup       : see setup(self) for definition

        Returns:
        (sympy function)            : optimal savingsfunction
        '''
        
        # a. Define benefit when old as tau * w_(t+1)
        dt1 = par.tau * par.wt1 

        # b. Define period budget constraints 
        bc_t1 = (1-par.tau) * par.wt - par.st 
        bc_t2 = par.st * (1 + par.rt1)+ (1 + par.n) * dt1

        # c. Substitute budget constraints into Euler
        eul = self.euler()
        sav = (eul.subs(par.c1t , bc_t1)).subs(par.c2t , bc_t2)
                
        # d. Simplify expression
        saving1 = sm.solve(sav, par.st)[0]
        saving2 = sm.collect(saving1, [par.tau])
        saving = sm.collect(saving1, [par.wt, par.wt1])
        
        
        # e. Return optimal saving
        return saving       
       

    
    def capitalaccumulation(self):
        par = self.par

        '''
        Finding capital accumulation
        
        Args: 
        parameters from setup       : see setup(self) for definition

        Returns:
        (sympy function)            : capital accumulation
        '''
        # a. Auxiliary variables
        a = (1 / (1 + (1+par.rho)/(2+par.rho)*((1-par.alpha)/par.alpha) * par.tau))
        b = ((1-par.alpha)*(1-par.tau))/((1+par.n)*(2+par.rho))
        c = par.A * par.kt**par.alpha
         
        # b. Deriving and displaying the capital accumulation
        kt_00 = par.a * (par.b * par.c)
        kt_01 = ((kt_00.subs(par.a , a)).subs(par.b ,b)).subs(par.c,c)
        kt = sm.Eq(par.kt1, kt_01)
        
        print('The capital accumulation is')
                
        return kt
        
        
        
        
    def steadystate_capital(self):
        par = self.par

        '''
        Finding capital in steady state
        
        Args: 
        parameters from setup       : see setup(self) for definition

        Returns:
        (sympy function)            : steady state for capital
        '''
        
        # a. Auxiliary variables
        a = (1 / (1 + (1+par.rho)/(2+par.rho)*((1-par.alpha)/par.alpha) * par.tau))
        b = ((1-par.alpha)*(1-par.tau))/((1+par.n)*(2+par.rho))
        c = 1 / (1-par.alpha)
        
        # a. substituting steady state in    
        k_star0 = (a * b * par.A )** par.c
        k_star1 = ((k_star0.subs(par.a , a)).subs(par.b ,b)).subs(par.c,c)
        k_star = sm.Eq(par.kss, k_star1)
        
        display(k_star)
        
        return k_star1