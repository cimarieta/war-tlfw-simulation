# coding: utf-8
import numpy as np
import eigenvalue as eg
import numpy.linalg as LA
from itertools import chain, izip
from fitness import *
from math import factorial


def comb(n, k):
    if n < k:
        return 0.
    return float(factorial(n))/(factorial(k)*factorial(n-k))


def winning_prob(alpha, k, n):
    """
        Calculates the winning probability of a group with `k` altruists
    """
    return 0.5 + alpha*k/(2*n)


def sel(k, l, n, alpha, c, beta):
    """
        Calculates the average number of groups of type `k`
        that become type `l` by selection forces
    """
    w = fitness(0, c)
    wm = avg_fitness(n, k, c)
    p = 0 if wm == 0 else k*w/(n*wm)
    indiv_term = comb(n, l)*p**l*(1-p)**(n-l)
    group_term = 1-beta+2*beta*winning_prob(alpha, k, n)
    return group_term*indiv_term


def selection_matrix(n, alpha, c, beta):
    """
        Calculates selection matrix
    """
    grid = np.indices((n+1, n+1))
    indices = izip(chain.from_iterable(grid[0]), chain.from_iterable(grid[1]))
    S = np.array([sel(k, l, n, alpha, c, beta) for k, l in indices])
    S = S.reshape(n+1, n+1)

    return S


def mig(k, l, m):
    """
        Calculates probability of a group k become a group of type l
        by migration
    """
    return comb(k, l)*((1-m)**l)*(m**(k-l))


def migration_matrix1(n, m):
    """
        Calculates migration matrix - part 1
    """
    grid = np.indices((n+1, n+1))
    indices = izip(chain.from_iterable(grid[0]), chain.from_iterable(grid[1]))
    M = np.array([mig(i, j, m) for i, j in indices])
    M = M.reshape(n+1, n+1)
    return M


def migration_matrix2(n, m):
    """
        Calculates migration matrix - part 2
    """
    T = np.zeros((n+1, n+1), dtype=float)
    func = lambda k, m: m*k
    T[:, 1] = func(np.arange(n+1), m)
    return T


def migration_matrix(n, m):
    """
        Calculates migration matrix - complete (part1 and part2)
    """
    M = np.identity(n+1) if m==0 else migration_matrix1(n, m)
    T = migration_matrix2(n, m)
    return M+T


def init_selmig(n, m, alpha, c, beta):
    """
        Initializes selection and migration matrices
    """
    S = selection_matrix(n, alpha, c, beta)
    M = migration_matrix(n, m)
    return S, M
