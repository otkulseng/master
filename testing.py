import numpy as np
from scipy.optimize import minimize

class RastriginFuncWithGrad:
    def __init__(self):
        self.grad = None  # To store the gradient after function evaluation

    def func(self, x):
        """
        Computes the Rastrigin function value and its gradient.
        """
        n = len(x)
        A = 10
        value = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
        gradient = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
        self.grad = gradient
        return value

    def jac(self, x):
        """
        Returns the gradient stored during the last function evaluation.
        """
        return self.grad
# Initialize the class instance
rastrigin_func_grad = RastriginFuncWithGrad()

# Define initial guess
x0 = np.array([5.0, 5.0, 5.0, 5.0])  # 4-dimensional Rastrigin function

# Perform optimization using L-BFGS-B method
result = minimize(
    fun=rastrigin_func_grad.func,
    x0=x0,
    method='L-BFGS-B',
    jac=rastrigin_func_grad.jac,
)

# Display the results
print("\nOptimization Result for Rastrigin Function:")
print(result)
