import sympy as sp
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

class HairSalonQ1:
    def __init__(self, kappa, eta, w):
        #Setting up variables
        self.kappa = sp.Symbol('kappa')
        self.eta = sp.Symbol('eta')
        self.w = sp.Symbol('w')
        self.ell = sp.Symbol('ell')
    
    def calculate_profit(self, ell):
        #Calculating the profit given af level of input, hairdressers (ell)
        profit = self.kappa * ell**(1 - self.eta) - self.w * ell
        return profit
    
    def calculate_optimal_ell(self):
        #Calculating the optimal level of hairdresser ell that maximizes profit
        optimal_ell_expr = ((1 - self.eta) * self.kappa / self.w) ** (1 / self.eta)
        return optimal_ell_expr

#question 2 

class HairSalonQ2:
    def __init__(self, rho, iota, sigma_epsilon, R, eta, w):
        self.rho = rho
        self.iota = iota
        self.sigma_epsilon = sigma_epsilon
        self.R = R
        self.eta = eta
        self.w = w

    def policy(self, kappa_t):
        return ((1 - self.eta) * kappa_t / self.w) ** (1 / self.eta)

    def simulate_shock_series(self):
        epsilon = np.random.normal(-0.5 * self.sigma_epsilon ** 2, self.sigma_epsilon, size=120)
        kappa = [1]
        ell = [0]

        for t in range(120):
            kappa_t = self.rho * np.log(kappa[t]) + epsilon[t]
            ell_t = self.policy(kappa_t)
            kappa.append(np.exp(kappa_t))
            ell.append(ell_t)

        return kappa, ell

    def calculate_salon_value(self, kappa, ell):
        salon_value = sum([(self.R ** -t) * (kappa[t] * ell[t] ** (1 - self.eta) - self.w * ell[t] -
                                              (ell[t] != ell[t-1]) * self.iota)
                           for t in range(120)])
        return salon_value

    def calculate_ex_ante_value(self, K):
        total_value = 0

        for _ in range(K):
            kappa, ell = self.simulate_shock_series()
            salon_value = self.calculate_salon_value(kappa, ell)
            total_value += salon_value

        H = total_value / K
        return H
    
#question 3

class HairSalonQ3:
    def __init__(self, rho, iota, sigma_epsilon, R, eta, w, delta):
        self.rho = rho
        self.iota = iota
        self.sigma_epsilon = sigma_epsilon
        self.R = R
        self.eta = eta
        self.w = w
        self.delta = delta

    def policy(self, kappa_t):
        return ((1 - self.eta) * kappa_t / self.w) ** (1 / self.eta)

    def simulate_shock_series(self):
        epsilon = np.random.normal(-0.5 * self.sigma_epsilon ** 2, self.sigma_epsilon, size=120)
        kappa = [1]
        ell = [0]

        for t in range(120):
            kappa_t = self.rho * np.log(kappa[t]) + epsilon[t]
            ell_t = self.policy(kappa_t)
            if t > 0 and abs(ell[t-1] - ell_t) > self.delta:
                ell_t = ell_t
            else:
                ell_t = ell[t]
            kappa.append(np.exp(kappa_t))
            ell.append(ell_t)

        return kappa, ell

    def calculate_salon_value(self, kappa, ell):
        salon_value = sum([(self.R ** -t) * (kappa[t] * ell[t] ** (1 - self.eta) - self.w * ell[t] -
                                              (ell[t] != ell[t-1]) * self.iota)
                           for t in range(120)])
        return salon_value

    def calculate_ex_ante_value(self, K):
        total_value = 0

        for _ in range(K):
            kappa, ell = self.simulate_shock_series()
            salon_value = self.calculate_salon_value(kappa, ell)
            total_value += salon_value

        H = total_value / K
        return H
    

#Question 4


class HairSalonoptimalQ4:
    def __init__(self, rho, iota, sigma_epsilon, R, eta, w, delta):
        self.rho = rho
        self.iota = iota
        self.sigma_epsilon = sigma_epsilon
        self.R = R
        self.eta = eta
        self.w = w
        self.delta = delta

    def policy(self, kappa_t):
        return ((1 - self.eta) * kappa_t / self.w) ** (1 / self.eta)

    def simulate_shock_series(self):
        epsilon = np.random.normal(-0.5 * self.sigma_epsilon ** 2, self.sigma_epsilon, size=120)
        kappa = [1]
        ell = [0]

        for t in range(120):
            kappa_t = self.rho * np.log(kappa[t]) + epsilon[t]
            ell_t = self.policy(kappa_t)
            if t > 0 and abs(ell[t-1] - ell_t) > self.delta:
                ell_t = ell_t
            else:
                ell_t = ell[t]
            kappa.append(np.exp(kappa_t))
            ell.append(ell_t)

        return kappa, ell

    def calculate_salon_value(self, kappa, ell):
        salon_value = sum([(self.R ** -t) * (kappa[t] * ell[t] ** (1 - self.eta) - self.w * ell[t] -
                                              (ell[t] != ell[t-1]) * self.iota)
                           for t in range(120)])
        return salon_value

    def calculate_ex_ante_value(self, K):
        total_value = 0

        for _ in range(K):
            kappa, ell = self.simulate_shock_series()
            salon_value = self.calculate_salon_value(kappa, ell)
            total_value += salon_value

        H = total_value / K
        return H
    

#Question 5


class HairSalonDynamicQ5:
    def __init__(self, rho, iota, sigma_epsilon, R, eta, w, delta):
        self.rho = rho
        self.iota = iota
        self.sigma_epsilon = sigma_epsilon
        self.R = R
        self.eta = eta
        self.w = w
        self.delta = delta

    def policy(self, kappa_t):
        return ((1 - self.eta) * kappa_t / self.w) ** (1 / self.eta)

    def dynamic_pricing(self, kappa_t, ell_t, desired_profit_margin):
        if ell_t == 0:
            price = self.w * (1 + desired_profit_margin)
        else:
            price = kappa_t * ell_t ** (-self.eta) * (1 + desired_profit_margin)
        return price

    def simulate_shock_series(self):
        epsilon = np.random.normal(-0.5 * self.sigma_epsilon ** 2, self.sigma_epsilon, size=120)
        kappa = [1]
        ell = [0]

        for t in range(120):
            kappa_t = self.rho * np.log(kappa[t]) + epsilon[t]
            ell_t = self.policy(kappa_t)
            if t > 0 and abs(ell[t-1] - ell_t) > self.delta:
                ell_t = ell_t
            else:
                ell_t = ell[t]
            kappa.append(np.exp(kappa_t))
            ell.append(ell_t)

        return kappa, ell

    def calculate_salon_value(self, kappa, ell, desired_profit_margin):
        salon_value = sum([(self.R ** -t) * (self.dynamic_pricing(kappa[t], ell[t], desired_profit_margin) * ell[t] -
                                              self.w * ell[t] - (ell[t] != ell[t-1]) * self.iota)
                           for t in range(120)])
        return salon_value

    def calculate_ex_ante_value(self, K, desired_profit_margin):
        total_value = 0

        for _ in range(K):
            kappa, ell = self.simulate_shock_series()
            salon_value = self.calculate_salon_value(kappa, ell, desired_profit_margin)
            total_value += salon_value

        H = total_value / K
        return H


