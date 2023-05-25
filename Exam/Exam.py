import sympy as sp

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

   