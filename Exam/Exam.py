import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import minimize_scalar
from scipy import optimize

class LaborSupplyModel:
    def __init__(self):
        # Declare the symbolic variables
        self.L, self.w, self.tau, self.G, self.alpha, self.nu, self.kappa, self.w_tilde = sp.symbols(
            'L w tau G alpha nu kappa w_tilde'
        )

        # Define the utility function
        self.C = self.kappa + (1 - self.tau) * self.w * self.L
        self.utility = sp.log(self.C**self.alpha * self.G**(1 - self.alpha)) - self.nu * self.L**2 / 2

    def solve_optimal_labor_supply(self):
        # Differentiate utility function with respect to L
        utility_diff = sp.diff(self.utility, self.L)

        # Solve for the optimal labor supply choice
        optimal_L = sp.solve(utility_diff, self.L)

        # Substitute w with (1 - tau) * w in the optimal solution
        optimal_L = optimal_L[0].subs(self.w, self.w_tilde / (1 - self.tau))

        return optimal_L
        

    def display_optimal_labor_supply_equation(self):
        # Get the optimal labor supply choice expression
        optimal_L = self.solve_optimal_labor_supply()

        # Express the optimal labor supply choice in terms of w_tilde
        optimal_L_expr = (-self.kappa + sp.sqrt(self.kappa**2 + 4 * self.alpha / self.nu * self.w_tilde**2)) / (2 * self.w_tilde)

        # Create an equation for the optimal labor supply choice
        optimal_L_eq = sp.Eq(self.L, optimal_L_expr)

        # Print the equation for the optimal labor supply choice
        print("Optimal labor supply choice:")
        sp.pprint(optimal_L_eq)


class LaborSupplyGraph:
    def __init__(self, alpha, kappa, nu, tau):
        self.alpha = alpha
        self.kappa = kappa
        self.nu = nu
        self.tau = tau
    
    def optimal_labor_supply_graph(self, w, G):
        #Finding the modified wage
        tilde_w = (1 - self.tau) * w
        #Finding and return the optimal labor supply 
        return (-self.kappa + np.sqrt(self.kappa**2 + 4 * self.alpha / self.nu * tilde_w**2)) / (2 * tilde_w)
    
    def plot_labor_supply_graph(self, w_range, G_values):
        #Loof over different values of G
        for G in G_values:
            #Finding the optimal labor supply for each wage
            L_star = self.optimal_labor_supply_graph(w_range, G)
            #Plotting the optimal labor supply 
            plt.plot(w_range, L_star, label=f'G = {G}')

        plt.xlabel('Wage (w)')
        plt.ylabel('Optimal Labor Supply (L*)')
        plt.title('Optimal Labor Supply at different wages')
        plt.legend()
        plt.grid(True)
        plt.show()

class LaborSupplyGraphQ3:
    def __init__(self, alpha, kappa, nu):
        self.alpha = alpha
        self.kappa = kappa
        self.nu = nu

    def optimal_labor_supply(self, w, tau):
        #Finding the modified wage
        tilde_w = (1 - tau) * w
        #Finding and return the optimal labor supply 
        return (-self.kappa + np.sqrt(self.kappa**2 + 4 * self.alpha / self.nu * tilde_w**2)) / (2 * tilde_w)

    def government_spending(self, w, tau, L):
        #Finding and returning government spending 
        return tau * w * L

    def worker_utility(self, w, tau, L):
        #Finding modified wage
        tilde_w = (1 - tau) * w
        #Finding consumption level
        C = self.kappa + (1 - tau) * w * L
        #Finding and returning worker utility 
        return np.log(C**self.alpha * (tau * w * L)**(1 - self.alpha)) - self.nu * L**2 / 2

    def plot_implied_values(self, w, tau_values):
        #For each tau value we find the optimal labor supply, government spending and worker utility
        L_values = self.optimal_labor_supply(w, tau_values)
        G_values = self.government_spending(w, tau_values, L_values)
        utility_values = self.worker_utility(w, tau_values, L_values)

#We plot these against tax rates
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(tau_values, L_values)
        plt.xlabel('Tax Rate (tau)')
        plt.ylabel('Labor Supply (L)')
        plt.title('Labor Supply at different tax rates')

        plt.subplot(1, 3, 2)
        plt.plot(tau_values, G_values)
        plt.xlabel('Tax Rate (tau)')
        plt.ylabel('Government Spending (G)')
        plt.title('Government Spending at different tax rates')

        plt.subplot(1, 3, 3)
        plt.plot(tau_values, utility_values)
        plt.xlabel('Tax Rate (tau)')
        plt.ylabel('Worker Utility')
        plt.title('Worker Utility at different tax rates')

        plt.tight_layout()
        plt.show()

class LaborSupplyGraphQ4:
    def __init__(self, alpha, kappa, nu):
        self.alpha = alpha
        self.kappa = kappa
        self.nu = nu

    def optimal_labor_supply(self, w, tau):
        #Finding modified wage
        tilde_w = (1 - tau) * w
        #Finding the discriminant for the optimal labor supply 
        discriminant = self.kappa**2 + 4 * self.alpha / self.nu * tilde_w**2

        #Checking if labor supply is at any point infinite or undefined 
        if discriminant < 0:
            return float('-inf')

        if tilde_w == 0:
            return float('inf')

        #Returning the optimal labor supply
        return (-self.kappa + np.sqrt(discriminant)) / (2 * tilde_w)

    def government_spending(self, w, tau, L):
        #Returning government spending
        return tau * w * L

    def worker_utility(self, w, tau, L):
        #Finding modified wage
        tilde_w = (1 - tau) * w

        #Checking if worker utility is at any point infinite or undefined 
        if L <= 0 or tilde_w == 0:
         return float('-inf')

        if np.isinf(L):
            return float('-inf')

        C = self.kappa + (1 - tau) * w * L

        if C <= 0 or (tau * w * L) <= 0:
            return float('-inf')

        #Returning worker utility 
        return np.log(C**self.alpha * (tau * w * L)**(1 - self.alpha)) - self.nu * L**2 / 2

    def maximize_utility(self, w, tau_range):
        max_utility = float('-inf')
        optimal_tau = None

        #Loop over tax rates
        for tau in tau_range:
            #Finding optimal labor supply and worker utility 
            L = self.optimal_labor_supply(w, tau)
            utility = self.worker_utility(w, tau, L)

            #If there exist a higher utility this is used
            if utility > max_utility:
                max_utility = utility
                optimal_tau = tau

        #Returning the optimal tax rate and maximum utility
        return optimal_tau, max_utility

    def plot_optimal_tax_rate(self, w, tau_range):
        #Finding the optimal tax rate and maximum utility 
        optimal_tau, max_utility = self.maximize_utility(w, tau_range)

        utility_values = []
        for tau in tau_range:
            L = self.optimal_labor_supply(w, tau)
            utility = self.worker_utility(w, tau, L)
            utility_values.append(utility)

        #Plotting the worker utility and tax rates
        plt.plot(tau_range, utility_values)
        plt.xlabel('Tax Rate (tau)')
        plt.ylabel('Worker Utility')
        plt.title('Worker Utility at different tax rates')
        plt.axvline(x=optimal_tau, color='r', linestyle='--', label='Optimal Tax Rate')
        plt.legend()
        plt.grid(True)
        plt.show()

        #Adding line to show optimal tax rates
        print(f"The socially optimal tax rate maximizing worker utility: tau* = {optimal_tau}")
        print(f"The maximum worker utility: U* = {max_utility}")



class LaborSupplyGraphQ5:
    def __init__(self, alpha, kappa, nu, sigma, rho, epsilon):
        self.alpha = alpha
        self.kappa = kappa
        self.nu = nu
        self.sigma = sigma
        self.rho = rho
        self.epsilon = epsilon

    def utility(self, C, G):
        #Finding and returning the worker utility 
        return (((self.alpha * C**((self.sigma - 1) / self.sigma) + (1 - self.alpha) * G**((self.sigma - 1) / self.sigma))**(self.sigma / (1 - self.sigma)))**(1 - self.rho) - 1) / (1 - self.rho) - self.nu * (C**self.epsilon) / (1 + self.epsilon)

    def consumption(self, w, tau, L):
        #Finding and returning the consumption 
        return self.kappa + (1 - tau) * w * L

    def solve_worker_problem(self, w, tau, G):
        #Defining the objective function in order to solve the workers problem
        objective = lambda L: abs(G - tau * w * L * ((1 - tau) * w * L) ** ((self.sigma - 1) / self.sigma))
        #Minimizing the objective function in order to find the optimal labor supply 
        result = minimize_scalar(objective, bounds=(0, 24), method='bounded')

        return result.x

    def solve_optimal_G(self, w, tau):
        #Solving the problem of the worker to find optimal labor supply 
        L_star = self.solve_worker_problem(w, tau, tau * w * self.solve_worker_problem(w, tau, 24))
        #Finding the optimal G
        G = tau * w * L_star * ((1 - tau) * w * L_star) ** ((self.sigma - 1) / self.sigma)
        return G

class LaborSupplyGraphQ6:
    def __init__(self, alpha, kappa, nu, sigma, rho, epsilon):
        self.alpha = alpha
        self.kappa = kappa
        self.nu = nu
        self.sigma = sigma
        self.rho = rho
        self.epsilon = epsilon

    def utility(self, C, G):
        #Finding and returning the worker utility 
        return (((self.alpha * C**((self.sigma - 1) / self.sigma) + (1 - self.alpha) * G**((self.sigma - 1) / self.sigma))**(self.sigma / (1 - self.sigma)))**(1 - self.rho) - 1) / (1 - self.rho) - self.nu * (C**self.epsilon) / (1 + self.epsilon)

    def consumption(self, w, tau, L):
        #Finding and returning the consumption level 
        return self.kappa + (1 - tau) * w * L

    def solve_worker_problem(self, w, tau, G):
        #Defining the objective function in order to solve the workers problem
        objective = lambda L: abs(self.utility(self.consumption(w, tau, L), G))
        #Minimizing the objective function in order to find the optimal labor supply
        result = optimize.minimize_scalar(objective, bounds=(0, 24), method='bounded')
        return result.x

    def solve_optimal_G(self, w, tau):
        def objective(G):
            #Solving the workers problem in order to find the optimal labor supply and consumption
            L = self.solve_worker_problem(w, tau, G)
            C = self.consumption(w, tau, L)

            #The differnce between the given G and and the implied G is found
            return G - tau * w * L * ((1 - tau) * w * L)**((self.sigma - 1) / self.sigma)

        #Using root to find the optimal G
        result = optimize.root_scalar(objective, method='brentq', bracket=(0, 100))
        return result.root
    
    def solve_optimal_tax_rate(self, w):
        def objective(tau):
            #Finding the Optimal G and labor supply for a given tax rate
            G = self.solve_optimal_G(w, tau)
            L_star = self.solve_worker_problem(w, tau, G)

            #Finding the negative of the workers utility as the objective
            return -self.utility(self.consumption(w, tau, L_star), G)
        
        #Minimizing the objective function in order to find the optimal tax rate
        result = optimize.minimize_scalar(objective, bounds=(0, 1), method='bounded')
        optimal_tax_rate = result.x
        optimal_G = self.solve_optimal_G(w, optimal_tax_rate)
        return optimal_tax_rate, optimal_G
