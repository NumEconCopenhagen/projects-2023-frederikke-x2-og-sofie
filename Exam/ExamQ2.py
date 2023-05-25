import sympy as sp
import numpy as np
from scipy.optimize import minimize_scalar

class HairSalon:
    def __init__(self, kappa, eta, w):
        self.kappa = sp.Symbol('kappa')
        self.eta = sp.Symbol('eta')
        self.w = sp.Symbol('w')
        self.ell = sp.Symbol('ell')
    
    def calculate_profit(self, ell):
        profit = self.kappa * ell**(1 - self.eta) - self.w * ell
        return profit
    
    def calculate_optimal_ell(self):
        optimal_ell_expr = ((1 - self.eta) * self.kappa / self.w) ** (1 / self.eta)
        return optimal_ell_expr

#question 2 

class HairSalonH:
    def __init__(self, rho, eta, wage, iota, sigma_epsilon, R):
        self.rho = rho
        self.eta = eta
        self.wage = wage
        self.iota = iota
        self.sigma_epsilon = sigma_epsilon
        self.R = R

    def calculate_h(self, epsilon_series):
        kappa_series = [1.0]  # Initial kappa
        ell_series = [0]  # Initial ell
        h_value = 0.0

        for t in range(120):
            # Calculate kappa_t based on AR(1) process
            kappa_t = np.exp(self.rho * np.log(kappa_series[-1]) + epsilon_series[t])
            kappa_series.append(kappa_t)

            # Calculate ell_t based on the policy from Question 1
            ell_t = ((1 - self.eta) * kappa_t / self.wage) ** (1 / self.eta)
            ell_series.append(ell_t)

            # Calculate the period profit
            profit = kappa_t * ell_t ** (1 - self.eta) - self.wage * ell_t
            adjustment_cost = self.iota if ell_t != ell_series[-2] else 0
            period_value = profit - adjustment_cost
            discounted_value = period_value * self.R ** (-t)
            h_value += discounted_value

        return h_value

    def calculate_expected_h(self, K):
        h_values = []
        for _ in range(K):
            epsilon_series = np.random.normal(-0.5 * self.sigma_epsilon ** 2, self.sigma_epsilon, size=120)
            h_value = self.calculate_h(epsilon_series)
            h_values.append(h_value)
        return np.mean(h_values)
    
#question 3

class HairSalon3:
    def __init__(self, rho, eta, wage, iota, sigma_epsilon, R, delta):
        self.rho = rho
        self.eta = eta
        self.wage = wage
        self.iota = iota
        self.sigma_epsilon = sigma_epsilon
        self.R = R
        self.delta = delta

    def calculate_h(self, epsilon_series):
        kappa_series = [1.0]  # Initial kappa
        ell_series = [0]  # Initial ell
        h_value = 0.0

        for t in range(120):
            # Calculate kappa_t based on AR(1) process
            kappa_t = np.exp(self.rho * np.log(kappa_series[-1]) + epsilon_series[t])
            kappa_series.append(kappa_t)

            # Calculate ell_t based on the policy with threshold delta
            ell_star = ((1 - self.eta) * kappa_t / self.wage) ** (1 / self.eta)
            if t > 0 and abs(ell_series[-1] - ell_star) > self.delta:
                ell_t = ell_star
            else:
                ell_t = ell_series[-1]
            ell_series.append(ell_t)

            # Calculate the period profit
            profit = kappa_t * ell_t ** (1 - self.eta) - self.wage * ell_t
            adjustment_cost = self.iota if ell_t != ell_series[-2] else 0
            period_value = profit - adjustment_cost
            discounted_value = period_value * self.R ** (-t)
            h_value += discounted_value

        return h_value

    def calculate_expected_h(self, K):
        h_values = []
        for _ in range(K):
            epsilon_series = np.random.normal(-0.5 * self.sigma_epsilon ** 2, self.sigma_epsilon, size=120)
            h_value = self.calculate_h(epsilon_series)
            h_values.append(h_value)
        return np.mean(h_values)
    

#Question 4



class HairSalonoptimal:
    def __init__(self, rho, eta, wage, iota, sigma_epsilon, R):
        self.rho = rho
        self.eta = eta
        self.wage = wage
        self.iota = iota
        self.sigma_epsilon = sigma_epsilon
        self.R = R

    def calculate_h(self, epsilon_series, delta):
        kappa_series = [1.0]  # Initial kappa
        ell_series = [0]  # Initial ell
        h_value = 0.0

        for t in range(120):
            # Calculate kappa_t based on AR(1) process
            kappa_t = np.exp(self.rho * np.log(kappa_series[-1]) + epsilon_series[t])
            kappa_series.append(kappa_t)

            # Calculate ell_t based on the policy with threshold delta
            ell_star = ((1 - self.eta) * kappa_t / self.wage) ** (1 / self.eta)
            if t > 0 and abs(ell_series[-1] - ell_star) > delta:
                ell_t = ell_star
            else:
                ell_t = ell_series[-1]
            ell_series.append(ell_t)

            # Calculate the period profit
            profit = kappa_t * ell_t ** (1 - self.eta) - self.wage * ell_t
            adjustment_cost = self.iota if ell_t != ell_series[-2] else 0
            period_value = profit - adjustment_cost
            discounted_value = period_value * self.R ** (-t)
            h_value += discounted_value

        return h_value

    def objective_function(self, delta):
        K = 1000  # Number of shock series to simulate
        h_values = []

        for _ in range(K):
            epsilon_series = np.random.normal(-0.5 * self.sigma_epsilon ** 2, self.sigma_epsilon, size=120)
            h_value = self.calculate_h(epsilon_series, delta)
            h_values.append(h_value)

        return -np.mean(h_values)  # Negate to maximize the objective function
    

#Question 5

class HairSalonDynamic:
    def __init__(self, base_price):
        self.base_price = base_price

    def calculate_dynamic_price(self, demand_level):
        # Implement your dynamic pricing algorithm here
        # Use the demand level to adjust the base price
        # You can consider other factors like time of year, hairdresser availability, etc.
        # Return the dynamically adjusted price

        # Example: Adjust the base price based on the demand level
        dynamic_price = self.base_price * (1 + demand_level)  # Adjust the price based on demand level

        return dynamic_price

class DemandData:
    def __init__(self, num_periods, rho, sigma_epsilon):
        self.num_periods = num_periods
        self.rho = rho
        self.sigma_epsilon = sigma_epsilon

    def generate_epsilon_series(self):
        epsilon_series = []
        epsilon_t = np.random.normal(-0.5 * self.sigma_epsilon ** 2, self.sigma_epsilon)

        for _ in range(self.num_periods):
            epsilon_series.append(epsilon_t)
            epsilon_t = self.rho * epsilon_t + np.random.normal(-0.5 * self.sigma_epsilon ** 2, self.sigma_epsilon)

        return epsilon_series

# Function to calculate H
def calculate_H(salon, demand_data):
    R = (1 + 0.01) ** (1/12)
    K = 1000  # Number of random shock series

    total_h = 0.0
    for k in range(K):
        h = 0.0
        epsilon_series = demand_data.generate_epsilon_series()  # Generate random shock series
        for t in range(len(epsilon_series)):
            demand_level = epsilon_series[t]
            dynamic_price = salon.calculate_dynamic_price(demand_level)

            # Update h using the dynamic price and other relevant variables
            h += R ** (-t) * dynamic_price

        total_h += h

    H = total_h / K
    return H
