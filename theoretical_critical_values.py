from selmig import *
import eigenvalue as eg
#from scipy.optimize import brentq

PARAM_NAMES = [
    'n_indiv', 'cost', 'alpha', 'beta', 'migration_rate'
]

# Larger values -> easier for altruists to emerge
POSITIVE_PARAMS = ['alpha', 'beta']

# Larger values -> more difficult for altruists to emerge
NEGATIVE_PARAMS = ['cost', 'migration_rate']


def rho(custom_var, var_name, params):
    """
        Returns (dominant eigenvalue - 1)
    """
    params[var_name] = custom_var
    n = params['n_indiv']
    c = params['cost']
    alpha = params['alpha']
    beta = params['beta']
    m = params['migration_rate']
    S,M = init_selmig(n, m, alpha, c, beta)
    return eg.dom_eig(np.dot(M,S))-1


def critical_value(var_name, params):
    """
        Calculates critical value for a given parameter
        `var_name` under parameters given by `params`
        Finds the critical value for which the dominant eigenvalue
        of the calculated matrix is roughly 1
    """
    a, b = 0., 1.

    fa = rho(a, var_name, params)
    fb = rho(b, var_name, params)

    if(fa*fb>0):
        if(fa>0):
            return int(var_name in NEGATIVE_PARAMS)
        else:
            return int(var_name in POSITIVE_PARAMS)

    precision = 0.00001
    while((b-a) > precision):
        medium_point = (a+b)/2
        fmpt = rho(medium_point, var_name, params)
        if not fmpt:
            break
        if fmpt*fa > 0:
            a = medium_point
        else:
            b = medium_point

    return medium_point


if __name__ == '__main__':
    params = {
        'n_indiv': 26,
        'cost': 0.03,
        'beta': 0.7,
        'alpha': 0.01,
        'migration_rate': 0.01
    }
    print critical_value('migration_rate', params)
