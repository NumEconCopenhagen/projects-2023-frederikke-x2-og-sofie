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

    def policy(self, kappa_t):#calculates the value of the policy variable based on the provided input kappa_t
        # using the given formula
        return ((1 - self.eta) * kappa_t / self.w) ** (1 / self.eta)

    def simulate_shock_series(self):
        epsilon = np.random.normal(-0.5 * self.sigma_epsilon ** 2, self.sigma_epsilon, size=120)#Calculate the ex-post 
        #value of the salon for the generated shock series
        kappa = [1] #inital kappa value
        ell = [0] #initial ell value

        for t in range(120):  # Calculate kappa_t based on AR(1) process with 120 iterations from 0 to 119
            kappa_t = self.rho * np.log(kappa[t]) + epsilon[t] #the value of kappa_t is calculated
            ell_t = self.policy(kappa_t) #ell_t is calculated using the policy, with the calculated kappa_t from abouve
            kappa.append(np.exp(kappa_t)) #the calculated values of kappa_t is appended to this list
            ell.append(ell_t) #the calculated values of ell_t is appended to this list

        return kappa, ell #returns all iterations over kappa and ell lists

    def calculate_salon_value(self, kappa, ell): #calculates the total value of the salon over the 120 iterations
        #by summing the the values of kappa and ell from abouve loop and the constants
        salon_value = sum([(self.R ** -t) * (kappa[t] * ell[t] ** (1 - self.eta) - self.w * ell[t] -
                                              (ell[t] != ell[t-1]) * self.iota)
                           for t in range(120)])
        return salon_value

    def calculate_ex_ante_value(self, K):
        total_value = 0 #inital value

        for _ in range(K): #iterates K loops, where _ is a placeholder 
            #simulate a shock series and returns kappa and ell arrays  
            kappa, ell = self.simulate_shock_series()
            salon_value = self.calculate_salon_value(kappa, ell) #Calculate the value of the salon 
            #using the simulated shock series
            total_value += salon_value #add the salon_value to total value

        H = total_value / K #calculate the average value of the salon 
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

        for t in range(120): # Calculate kappa_t based on AR(1) process with 120 iterations from 0 to 119
            kappa_t = self.rho * np.log(kappa[t]) + epsilon[t] #the value of kappa_t is calculated
            ell_t = self.policy(kappa_t) #ell_t is calculated using the policy, with the calculated kappa_t from abouve
            if t > 0 and abs(ell[t-1] - ell_t) > self.delta: #controls it is not the first period, and if the  the 
                #absolute difference between the previous value of ell and the current value are greater than delta
                ell_t = ell_t #if true the value of ell_t is assinged to itself, because there has been a change in ell
            else:
                ell_t = ell[t] #if false the previous value of ell is assigned to ell_t indication no change
            kappa.append(np.exp(kappa_t)) #the calculated values of kappa_t is appended to this list
            ell.append(ell_t) #the calculated values of ell_t is appended to this list

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


