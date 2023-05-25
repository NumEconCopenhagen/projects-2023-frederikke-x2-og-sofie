import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

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
        tilde_w = (1 - self.tau) * w
        return (-self.kappa + np.sqrt(self.kappa**2 + 4 * self.alpha / self.nu * tilde_w**2)) / (2 * tilde_w)
    
    def plot_labor_supply_graph(self, w_range, G_values):
        for G in G_values:
            L_star = self.optimal_labor_supply_graph(w_range, G)
            plt.plot(w_range, L_star, label=f'G = {G}')

        plt.xlabel('Wage (w)')
        plt.ylabel('Optimal Labor Supply (L*)')
        plt.title('Optimal Labor Supply vs. Wage')
        plt.legend()
        plt.grid(True)
        plt.show()

class LaborSupplyGraphQ3:
    def __init__(self, alpha, kappa, nu):
        self.alpha = alpha
        self.kappa = kappa
        self.nu = nu

    def optimal_labor_supply(self, w, tau):
        tilde_w = (1 - tau) * w
        return (-self.kappa + np.sqrt(self.kappa**2 + 4 * self.alpha / self.nu * tilde_w**2)) / (2 * tilde_w)

    def government_spending(self, w, tau, L):
        return tau * w * L

    def worker_utility(self, w, tau, L):
        tilde_w = (1 - tau) * w
        C = self.kappa + (1 - tau) * w * L
        return np.log(C**self.alpha * (tau * w * L)**(1 - self.alpha)) - self.nu * L**2 / 2

    def plot_implied_values(self, w, tau_values):
        L_values = self.optimal_labor_supply(w, tau_values)
        G_values = self.government_spending(w, tau_values, L_values)
        utility_values = self.worker_utility(w, tau_values, L_values)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(tau_values, L_values)
        plt.xlabel('Tax Rate (tau)')
        plt.ylabel('Labor Supply (L)')
        plt.title('Labor Supply vs. Tax Rate')

        plt.subplot(1, 3, 2)
        plt.plot(tau_values, G_values)
        plt.xlabel('Tax Rate (tau)')
        plt.ylabel('Government Spending (G)')
        plt.title('Government Spending vs. Tax Rate')

        plt.subplot(1, 3, 3)
        plt.plot(tau_values, utility_values)
        plt.xlabel('Tax Rate (tau)')
        plt.ylabel('Worker Utility')
        plt.title('Worker Utility vs. Tax Rate')

        plt.tight_layout()
        plt.show()

class LaborSupplyGraphQ4:
    def __init__(self, alpha, kappa, nu):
        self.alpha = alpha
        self.kappa = kappa
        self.nu = nu

    def optimal_labor_supply(self, w, tau):
        tilde_w = (1 - tau) * w
        return (-self.kappa + np.sqrt(self.kappa**2 + 4 * self.alpha / self.nu * tilde_w**2)) / (2 * tilde_w)

    def government_spending(self, w, tau, L):
        return tau * w * L

    def worker_utility(self, w, tau, L):
        tilde_w = (1 - tau) * w
        C = self.kappa + (1 - tau) * w * L
        return np.log(C**self.alpha * (tau * w * L)**(1 - self.alpha)) - self.nu * L**2 / 2

    def maximize_utility(self, w, tau_range):
        max_utility = float('-inf')
        optimal_tau = None

        for tau in tau_range:
            L = self.optimal_labor_supply(w, tau)
            utility = self.worker_utility(w, tau, L)

            if utility > max_utility:
                max_utility = utility
                optimal_tau = tau

        return optimal_tau, max_utility

    def plot_optimal_tax_rate(self, w, tau_range):
        optimal_tau, max_utility = self.maximize_utility(w, tau_range)

        utility_values = []
        for tau in tau_range:
            L = self.optimal_labor_supply(w, tau)
            utility = self.worker_utility(w, tau, L)
            utility_values.append(utility)

        plt.plot(tau_range, utility_values)
        plt.xlabel('Tax Rate (tau)')
        plt.ylabel('Worker Utility')
        plt.title('Worker Utility vs. Tax Rate')
        plt.axvline(x=optimal_tau, color='r', linestyle='--', label='Optimal Tax Rate')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"The socially optimal tax rate maximizing worker utility: tau* = {optimal_tau}")
        print(f"The maximum worker utility: U* = {max_utility}")