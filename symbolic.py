from sympy import symbols, prod, expand
import math
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator

def newman_approx(n):
    # Symbolic computation of the newman polynomials.
    # Gives out two numpy arrays Q = [q_j] and P = [p_j] representing
    # coefficients in the expansion |x| = P(x) / Q(x).
    # Both P and Q are even, which means that
    # P = p[0] + p[1]x^2 + p[2]x^4 ...
    # Q = q[0] + q[1]x^2 + q[2]x^4 ...
    a = math.exp(-1 / math.sqrt(n))
    x = symbols('x')

    factors = [(a**k + x) for k in range(n)]
    h_x = prod(factors)
    h_minus_x = prod([(a**k - x) for k in range(n)])

    P_x = expand(h_x - h_minus_x)
    Q_x = expand(h_x + h_minus_x)

    coefficients = np.zeros(n + 1)
    total = [Q_x, P_x]
    for i in range(n+1):
        coefficients[i] = total[i % 2].coeff(x, i)

    return coefficients[::2], np.insert(coefficients[1::2], 0, 0)


def generate_functional(coefs, matr: csr_matrix):
    def apply(v):
        out = np.zeros_like(v)
        for c in coefs:
            out += c * v
            v = matr.dot(matr.dot(v))
        return out
    return LinearOperator(
        dtype= matr.dtype,
        shape=matr.shape,
        matvec=apply
    )

def get_sparse_newman_functionals(n: int, matr: csr_matrix):
    Q, P = newman_approx(n)
    return generate_functional(Q, matr), generate_functional(P, matr)


