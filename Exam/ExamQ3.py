import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class GriewankOptimizer:
    def __init__(self, bounds, tolerance, warmup_iters, max_iters):
        self.bounds = bounds
        self.tolerance = tolerance
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.x_star = None
        self.history = []

    def griewank(self, x):
        return self.griewank_(x[0], x[1])

    def griewank_(self, x1, x2):
        A = x1**2 / 4000 + x2**2 / 4000
        B = np.cos(x1 / np.sqrt(1)) * np.cos(x2 / np.sqrt(2))
        return A - B + 1

    def run_optimizer(self, x0):
        result = minimize(self.griewank, x0, method='BFGS', tol=self.tolerance)
        return result.x

    def optimize(self):
        for k in range(self.max_iters):
            x = np.random.uniform(self.bounds[0], self.bounds[1], size=2)

            if k >= self.warmup_iters:
                chi = 0.50 * 2 / (1 + np.exp((k - self.warmup_iters) / 100))
                x_k0 = chi * x + (1 - chi) * self.x_star
                x_ast = self.run_optimizer(x_k0)
            else:
                x_ast = self.run_optimizer(x)

            if k == 0 or self.griewank(x_ast) < self.griewank(self.x_star):
                self.x_star = x_ast

            self.history.append(x_ast)

            if self.griewank(self.x_star) < self.tolerance:
                break

    def plot_initial_guesses(self):
        history = np.array(self.history)
        plt.scatter(history[:, 0], history[:, 1], c='b', label='Effective Initial Guesses')
        plt.scatter(self.x_star[0], self.x_star[1], c='r', label='Global Minimum')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Effective Initial Guesses')
        plt.legend()
        plt.show()