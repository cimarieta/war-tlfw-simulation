# coding: utf-8
from collections import OrderedDict
from dask import delayed, compute
import dask.multiprocessing
from hashlib import md5
from itertools import product
import logging
import numpy as np
import pickle

from agent_simulation import Simulation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


DEFAULT_PARAM_RANGES = OrderedDict((
    ('n_groups', [10000]),
    ('n_indiv', [26]),
    ('n_altruists', [1]),
    ('cost', [0.03, 0.15]),
    ('alpha', [0.2]),
    ('beta', [0.5]),
    ('migration_rate', [0.01]),
    ('mutation_rate', [0.0001])
))


@delayed
def run_sim(params):
    """
        Run a single simulation with parameters given by `params`
    """
    sim = Simulation(params)
    return sim.run()


@delayed
def save_results(results, filename):
    """
        Saves results in `filename` using pickle
    """
    with open(filename, 'wb') as f:
        pickle.dump(results, f)


def get_hashed_params(set_params):
    """
        Returns params hashed using md5
    """
    hash_obj = md5(str(set_params).encode('utf-8'))
    return hash_obj.hexdigest()


def run_multiple_sims(param_ranges, prefix_name=None, n_times=20):
    """
        Run multiple simulations using several parameter sets
        specified by `param_ranges`
    """
    param_names = list(DEFAULT_PARAM_RANGES.keys())
    set_params = OrderedDict()
    for param_name in param_names:
        set_params[param_name] = param_ranges.get(
            param_name,
            DEFAULT_PARAM_RANGES[param_name]
        )

    if prefix_name is None:
        prefix_name = get_hashed_params(set_params)
    set_param_values = list(set_params.values())
    product_params = []
    for param_values in product(*set_param_values):
        params = {
            param_names[i]: param_value \
                for i, param_value in enumerate(param_values)
        }
        product_params.append(params)

    logging.info('Starting to run the simulations...')
    sim_results = [[run_sim(params) for i in range(n_times)] for params in product_params]

    params_filename = 'data/{}_params.pkl'.format(prefix_name)
    save_results(product_params, params_filename)

    results_filename = 'data/{}_results.pkl'.format(prefix_name)
    stored = save_results(sim_results, results_filename)
    stored.compute(get=dask.multiprocessing.get)

    logging.info('Simulations have finished!')
    logger.info('You can check the results in %s'%results_filename)


def main():
    gen_param_ranges = lambda cost, alpha, max_mig: {
        'alpha': [alpha], 'c': [cost], 'beta': np.linspace(0.,1.,11),
        'migration_rate': np.linspace(0.,max_mig,11)
    }
    # cost = 0.03, alpha = 0.1, max_mig = 0.2
    run_multiple_sims(gen_param_ranges(0.03, 0.1, 0.2), prefix_name='upper_left')
    # cost = 0.15, alpha = 0.1, max_mig = 0.05
    run_multiple_sims(gen_param_ranges(0.15, 0.1, 0.05), prefix_name='upper_right')
    # cost = 0.03, alpha = 0.2, max_mig = 0.2
    run_multiple_sims(gen_param_ranges(0.03, 0.2, 0.6), prefix_name='bottom_left')
    # cost = 0.15, alpha = 0.2, max_mig = 0.05
    run_multiple_sims(gen_param_ranges(0.15, 0.2, 0.12), prefix_name='bottom_right')


if __name__ == '__main__':
    main()
