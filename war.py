# coding: utf-8
import numpy as np


def calc_winning_prob(i, j, alpha, n):
    """
        Calculates probability of group with `i` altruists win war
        against group with `j` altruists
    """
    arg1 = 4.*alpha*i/n
    arg2 = 4.*alpha*j/n
    return np.exp(arg1)/(np.exp(arg1)+np.exp(arg2))


def calc_winning_prob_matrix(alpha, n_indiv):
    """
        Calculates winning probability matrix
        (probability that a group with `row_index` altruists
        win war agains a group with `column_index` altruists)
    """
    return np.fromfunction(
        lambda i,j: calc_winning_prob(i, j, alpha, n_indiv),
        (n_indiv+1, n_indiv+1)
    )


