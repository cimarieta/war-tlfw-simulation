# coding: utf-8
import numpy.linalg
import numpy as np


def dom_eig(matrix):
    """
        Calculates dominant eigenvalue of a matrix
        without the first row and the first column
    """
    matrix = np.delete(matrix, 0, 0)
    matrix = np.delete(matrix, 0, 1)

    av = numpy.linalg.eigvals(matrix)
    x = np.argsort(-abs(av))
    return np.real(av[x[0]])


def calc_assoc_vec(matrix):
    """
        Calculates the array associated to the dominant eigenvalue
        of a matrix without the first row and column
    """

    n = matrix.shape[0]-1
    matrix = np.delete(matrix, 0, 0)
    matrix = np.delete(matrix, 0, 1)

    eig_val, eig_vec = numpy.linalg.eig(matrix, left=True, right=False)
    x = np.argsort(-abs(av))
    return np.fabs(avec[:, x[0]])


def maxmin_cols(matrix, n):
    """
        Returns array with maximum and minimum values
        considering the sums of each column
    """
    v = np.sum(matrix, axis=0)
    v = np.sort(v)
    return np.array([v[0], v[n]])
