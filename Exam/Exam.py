import math

class UtilityMaximization:
    def __init__(self, alpha=0.5, kappa=1.0, nu=1/(2*16**2), w=1.0, tau=0.3, G=1.0):
        self.alpha = alpha
        self.kappa = kappa
        self.nu = nu
        self.w = w
        self.tau = tau
        self.G = G

    def utility(self, L):
        C = self.kappa + (1 - self.tau) * self.w * L
        utility_value = math.log(C ** self.alpha * self.G ** (1 - self.alpha)) - self.nu * L ** 2 / 2
        return utility_value

    def maximize_utility(self):
        max_utility = float("-inf")
        optimal_labor = None

        for L in range(25):
            utility_value = self.utility(L)
            if utility_value > max_utility:
                max_utility = utility_value
                optimal_labor = L

        return optimal_labor, max_utility
    
    def calculate_optimal_labor(self):
        tw = (1 - self.tau) * self.w
        discriminant = self.kappa ** 2 + 4 * (self.alpha / self.nu) * tw ** 2
        if discriminant >= 0:
            optimal_labor = (-self.kappa + math.sqrt(discriminant)) / (2 * tw)
            return optimal_labor

        return None
   