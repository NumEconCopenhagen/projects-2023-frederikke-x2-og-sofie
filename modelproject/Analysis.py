from scipy import optimize
from types import SimpleNamespace
import numpy as np
import sympy as sm
from IPython.display import display

class AnalysismodelclassOLG():
    # The OLG model is being coded
    def __init__(self):
        
        self.par = SimpleNamespace()
        self.setup()

    def setup(self):
        par = self.par

        #All the parameters and the variables are being defined

        #The variables and parameters used for the utility function 
        par.c_y1t = sm.symbols('c_1t')           #Consumption in period t, when young.      
        par.c_o2t = sm.symbols('c_{2t+1}')       #Consumption in period t+1, when old. 
        par.rho = sm.symbols('rho')             #The time preference and/or patience  

        #The variables and parameters used for the budget constraints
        par.r_t = sm.symbols('r_t')             #The interest rate in period t        
        par.r_t1 = sm.symbols('r_{t+1}')        #The interest rate  in period t+1
        par.w_t = sm.symbols('w_t')             #The Wage rate in period t       
        par.w_t1 = sm.symbols('w_{t+1}')        #The wage rate in period t+1
        par.s_t = sm.symbols('s_t')             #The rate on savings 
        par.lamb = sm.symbols('lambda_t')       #The lagrange multiplier  
        par.d_t1 = sm.symbols('d_{t+1}')        #The benefit in period t+1         
        par.tau = sm.symbols('tau')             #The tax rate on wage    
        par.n = sm.symbols('n')                 #The constant populations growth  

        #The variables and parameters used for the production function
        par.K_t = sm.symbols('K_t')             #Capital in peirod t       
        par.K_t1 = sm.symbols('K_{t+1}')        #Capital in period t+1
        par.L_t = sm.symbols('L_t')             #The labour in period t     
        par.L_t1 = sm.symbols('L_{t+1}')        #The labour in period t+1
        par.k_t = sm.symbols('k_t')             #Per worker caiptal in period t  
        par.A = sm.symbols('A')                 #The constant total factor of productivity         
        par.U_t = sm.symbols('U_t')             #Utility in period t  
        par.k_t1 = sm.symbols('k_{t+1}')        #Per worker caital in period t+1     
        par.alpha = sm.symbols('alpha')         #The elasticity in the CES function  
        par.kss = sm.symbols('k^*')             #Steaty State for caital 
        
        #Variables for the Auxiliary regression 
        par.a = sm.symbols('a')
        par.b = sm.symbols('b')
        par.c = sm.symbols('c')

        
    def utility_func(self):
        par = self.par
        # The utility function is set up
        # self refers to the variables and paremeters from the "setup(self)"
        # Return is a sympy function, which returns the utility function denoted as U_t
        return sm.log(par.c_y1t)+ sm.log(par.c_o2t) * 1/(1+par.rho)
    
    
    def budget_con(self):
        par = self.par
        #The intertemporal budget constraint function is set up
        # self refers to the variables and paremeters from the "setup(self)"
        # Return is a sympy function, which returns the functino for the intertemoral budget constraint, which is solved when 
        # Right hand side (RHS) equals the left hand side (LHS)
        
        #Defining following equations:
        d_t1 = par.tau * par.w_t1                                               #The benefit when old 
        bc_t1 = sm.Eq(par.c_y1t + par.s_t, (1-par.tau) * par.w_t)                #Budget constraint when young
        bc_t2 = sm.Eq(par.c_o2t, par.s_t * (1 + par.r_t1)+ (1 + par.n) * d_t1)   #The budget constraint when old
        bc_t2_s = sm.solve(bc_t2, par.s_t)                                      #Saving is solved by using the budget constraint when old using sympy

        #Inserting savings into the first budget constraint
        bc1 = bc_t1.subs(par.s_t, bc_t2_s[0])            #The solved savings rate is inserted into the budget constraint when young 
        RHS =  sm.solve(bc1, par.w_t*(1 - par.tau))[0]   #The right hand side of the budget constraint is defined 
        LHS = par.w_t * (1 - par.tau)                    #The left hand side of the budget constraint is defined 
        return RHS - LHS
           
    
    def euler_eq(self):
        par = self.par
        # The equations used to find the Euler equation is found
        # self refers to the variables and paremeters from the "setup(self)"
        # Return is a sympy function, which returns the Euler equation denoted as euler_1
        lagrange = self.utility_func() + par.lamb * self.budget_con()   #The Lagrangian is definded
        foc1 = sm.Eq(0, sm.diff(lagrange, par.c_y1t))                  #The firs order condition wrt the consumption when young
        foc2 = sm.Eq(0, sm.diff(lagrange, par.c_o2t))                  #The firs order condition wrt the consumption when old
        lamb_1 = sm.solve(foc1, par.lamb)[0]                          #The lambda is solved for in the FOC, when young
        lamb_2 = sm.solve(foc2, par.lamb)[0]                          #The lambda is solved for in the FOC, when old
        euler_1 = sm.solve(sm.Eq(lamb_1,lamb_2), par.c_y1t)[0]         #The Euler equation is defined
        return sm.Eq(euler_1, par.c_y1t)
    
    
    def opt_save(self):
        par = self.par
        # The equations used to find the optimal saving is defined
        # self refers to the variables and paremeters from the "setup(self)"
        # Return is a sympy function, which returns the optimal saving denoted as saving
        d_t1 = par.tau * par.w_t1                               #Defining the benefit when old 
        bc_t1 = (1-par.tau) * par.w_t - par.s_t                 #Defining the budget constraint when young
        bc_t2 = par.s_t * (1 + par.r_t1)+ (1 + par.n) * d_t1    #Defining the budget constraint when old

        # 4. The budget constraints are substituted into the Euler equation
        eul = self.euler_eq() 
        sav = (eul.subs(par.c_y1t , bc_t1)).subs(par.c_o2t , bc_t2)
                
        #The exprestions are being simplified
        saving1 = sm.solve(sav, par.s_t)[0]
        saving2 = sm.collect(saving1, [par.tau])
        saving = sm.collect(saving1, [par.w_t, par.w_t1])
        return saving       

    
    def cap_acc(self):
        par = self.par
        # The equations used to find the caital accumulation is defined
        # self refers to the variables and paremeters from the "setup(self)"
        # Return is a sympy function, which returns the capital accumulaiton denoted as k_t

        #Auxiliary variables are being defined, to make it easier to refer to later
        a = (1 / (1 + (1+par.rho)/(2+par.rho)*((1-par.alpha)/par.alpha) * par.tau))
        b = ((1-par.alpha)*(1-par.tau))/((1+par.n)*(2+par.rho))
        c = par.A * par.k_t**par.alpha
         
        #The capital accumulation is derived and dislayed using the auxiliary variables as reference
        kt_00 = par.a * (par.b * par.c)
        kt_01 = ((kt_00.subs(par.a , a)).subs(par.b ,b)).subs(par.c,c)
        k_t = sm.Eq(par.k_t1, kt_01) 
        return k_t
        
        
    def cap_steadystate(self):
        par = self.par
        # The equations used to find the steady sate for caital is defined
        # self refers to the variables and paremeters from the "setup(self)"
        # Return is a sympy function, which returns the steady state for capital denoted as k_star
        
        #Auxiliary variables are being defined, to make it easier to refer to later
        a = (1 / (1 + (1+par.rho)/(2+par.rho)*((1-par.alpha)/par.alpha) * par.tau))
        b = ((1-par.alpha)*(1-par.tau))/((1+par.n)*(2+par.rho))
        c = 1 / (1-par.alpha)
        
        #The steady state is substituted in    
        k_star0 = (a * b * par.A )** par.c
        k_star1 = ((k_star0.subs(par.a , a)).subs(par.b ,b)).subs(par.c,c) #Finds the value of the steady state
        k_star = sm.Eq(par.kss, k_star1) #Shows the equation steady state is
        print('The steady state is found to be:')
        display(k_star)
        return k_star1