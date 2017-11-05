# coding: utf-8
import numpy as np


def fitness(label, cost):
    """
        Calculates fitness for an altruist (label=0) for a non-altruist
        (label=1)
    """
    fit = 1-(1-label)*cost
    return fit


def avg_fitness(n, k, cost):
    """
        Calculates average fitness for a group with `k` altruists
    """
    freq = float(k)/n
    avg_fit = freq*fitness(0, cost) + (1-freq)*fitness(1, cost)
    return avg_fit


def calc_ref_fitness(cost, n_indiv):
    """
        Calculates fitness reference values
    """
    ref_fitness = [fitness(0, cost)]*(int(n_indiv)+1)
    return np.array(ref_fitness)


def calc_ref_avg_fitness(cost, n_indiv):
    """
        Calculates average fitness reference values
    """
    ref_avg_fitness = [avg_fitness(n_indiv, k, cost) \
                       for k in range(0, int(n_indiv)+1)]
    return np.array(ref_avg_fitness)
