import sympy as sp
import numpy as np

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