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
        # The utility function is set up
        # self refers to the variables and paremeters from the "setup(self)"
        # Return is a sympy function, which returns the utility function denoted as U_t

        return sm.log(par.cy1t)+ sm.log(par.co2t) * 1/(1+par.rho)
    
    
    def budgetcon(self):
        par = self.par
        
        #The intertemporal budget constraint function is set up
        # self refers to the variables and paremeters from the "setup(self)"
        # Return is a sympy function, which returns the functino for the intertemoral budget constraint, which is solved when 
        # Right hand side (RHS) equals the left hand side (LHS)
        
        d_t1 = par.tau * par.w_t1 # 1. Defining the benefit when old 

        bc_t1 = sm.Eq(par.cy1t + par.s_t, (1-par.tau) * par.w_t) # 2. The sympy equations for the budget constraint when young is defined 
        bc_t2 = sm.Eq(par.co2t, par.s_t * (1 + par.r_t1)+ (1 + par.n) * d_t1) # 3. The sympy equations for the budget constraint when old is defined 

        bc_t2_s = sm.solve(bc_t2, par.s_t) # 4. Saving is solved by using the budget constraint when old using sympy

        # e. Inserting savings into the first budget constraint
        bc1 = bc_t1.subs(par.s_t, bc_t2_s[0]) # 5. The solved savings rate is inserted into the budget constraint when young 

        RHS =  sm.solve(bc1, par.w_t*(1 - par.tau))[0] # 6. The right hand side of the budget constraint is defined 
        LHS = par.w_t * (1 - par.tau) # 7. The left hand side of the budget constraint is defined 

        return RHS - LHS
           
    
    def euler(self):
        par = self.par
        # The equations used to find the Euler equation is found
        # self refers to the variables and paremeters from the "setup(self)"
        # Return is a sympy function, which returns the Euler equation denoted as euler_1
        
        lagrange = self.utilityfunc() + par.lamb * self.budgetcon() # 1. The Lagrangian is definded
        
        foc1 = sm.Eq(0, sm.diff(lagrange, par.cy1t)) # 1. The firs order condition wrt the consumption when young
        foc2 = sm.Eq(0, sm.diff(lagrange, par.co2t)) # 2. The firs order condition wrt the consumption when old
        
        lamb_1 = sm.solve(foc1, par.lamb)[0] # 3. The lambda is solved for in the FOC, when young
        lamb_2 = sm.solve(foc2, par.lamb)[0] # 4. The lambda is solved for in the FOC, when old
        
        euler_1 = sm.solve(sm.Eq(lamb_1,lamb_2), par.cy1t)[0] # 5. The Euler equation is defined 

        return sm.Eq(euler_1, par.cy1t)
    
    
    def optimalsavings(self):
        par = self.par
        # The equations used to find the optimal saving is defined
        # self refers to the variables and paremeters from the "setup(self)"
        # Return is a sympy function, which returns the optimal saving denoted as saving
        
        d_t1 = par.tau * par.w_t1 # 1. Defining the benefit when old 
 
        bc_t1 = (1-par.tau) * par.w_t - par.s_t # 2. Defining the budget constraint when young
        bc_t2 = par.s_t * (1 + par.r_t1)+ (1 + par.n) * d_t1 # 3. Defining the budget constraint when old

        # 4. The budget constraints are substituted into the Euler equation
        eul = self.euler() 
        sav = (eul.subs(par.cy1t , bc_t1)).subs(par.co2t , bc_t2)
                
        # d. The exprestions are being simplified
        saving1 = sm.solve(sav, par.s_t)[0]
        saving2 = sm.collect(saving1, [par.tau])
        saving = sm.collect(saving1, [par.w_t, par.w_t1])
        
        return saving       
       

    
    def capitalaccumulation(self):
        par = self.par

        # The equations used to find the caital accumulation is defined
        # self refers to the variables and paremeters from the "setup(self)"
        # Return is a sympy function, which returns the capital accumulaiton denoted as k_t

        # a. Auxiliary variables are being defined, to make it easier to refer to later
        a = (1 / (1 + (1+par.rho)/(2+par.rho)*((1-par.alpha)/par.alpha) * par.tau))
        b = ((1-par.alpha)*(1-par.tau))/((1+par.n)*(2+par.rho))
        c = par.A * par.k_t**par.alpha
         
        # b. The capital accumulation is derived and dislayed using the auxiliary variables as reference
        kt_00 = par.a * (par.b * par.c)
        kt_01 = ((kt_00.subs(par.a , a)).subs(par.b ,b)).subs(par.c,c)
        k_t = sm.Eq(par.k_t1, kt_01)
        
        print('We find the capital accumulation to be given by')
                
        return k_t
        
        
        
        
    def steadystate_capital(self):
        par = self.par

        # The equations used to find the steady sate for caital is defined
        # self refers to the variables and paremeters from the "setup(self)"
        # Return is a sympy function, which returns the steady state for capital denoted as k_star
        
        # a. Auxiliary variables are being defined, to make it easier to refer to later
        a = (1 / (1 + (1+par.rho)/(2+par.rho)*((1-par.alpha)/par.alpha) * par.tau))
        b = ((1-par.alpha)*(1-par.tau))/((1+par.n)*(2+par.rho))
        c = 1 / (1-par.alpha)
        
        # a. The steady state is substituted in    
        k_star0 = (a * b * par.A )** par.c
        k_star1 = ((k_star0.subs(par.a , a)).subs(par.b ,b)).subs(par.c,c) #Finds the value of the steady state
        k_star = sm.Eq(par.kss, k_star1) #Shows the equation steady state is
        
        print('The steady state is found to be:')
        display(k_star)
        
        return k_star1