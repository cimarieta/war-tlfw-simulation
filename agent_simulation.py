# coding: utf-8
from itertools import chain, izip
import logging
import numpy as np
import scipy.stats
import time

from fitness import calc_ref_fitness, calc_ref_avg_fitness
from war import calc_winning_prob_matrix

logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PARAMS_NAMES = [
    'n_groups', 'n_individuals', 'prob_altruist',
    'cost_altruist', 'alpha', 'beta', 'migration_rate',
    'mutation_rate'
]


class Simulation():
    def __init__(self, params):
        self.n_groups = params['n_groups']
        self.n_indiv = params['n_indiv']
        self.n_altruists = params['n_altruists']
        self.cost = params['cost']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.pmig = params['migration_rate']
        self.pmut = params['mutation_rate']
        total_indiv = self.n_groups*self.n_indiv
        self.initial_altruist_freq = float(self.n_altruists)/total_indiv
        self.altruists_freq = []
        self.alts_per_group = self.init_alts_per_group()
        self.ref_fitness_values = calc_ref_fitness(self.cost, self.n_indiv)
        self.ref_avg_fitness_values = calc_ref_avg_fitness(self.cost, self.n_indiv)
        self.ref_winning_prob_matrix = calc_winning_prob_matrix(self.alpha, self.n_indiv)


    def init_alts_per_group(self):
        total_indiv = self.n_groups*self.n_indiv
        idx = np.random.choice(total_indiv, self.n_altruists, replace=False)
        with_altruists = idx/self.n_indiv

        alts_count = np.bincount(with_altruists)
        extra_zeros = np.zeros(self.n_groups-len(alts_count))
        alts_count = np.append(alts_count, extra_zeros)
        return alts_count


    def calc_distribution_alts(self):
        self.alts_distrib = np.zeros((self.n_groups, self.n_indiv))
        pos_array = np.arange(self.n_indiv)
        alts_pos = (np.random.sample(pos_array, n_alts) for n_alts in self.alts_per_group)
        for i, list_positions in enumerate(alts_pos):
            self.alts_distrib[i][list_positions] = 1


    def run(self, max_iter=5000, precision=0.01):
        total_indiv = self.n_groups*self.n_indiv
        prev_freq = float(self.n_altruists)/total_indiv
        stop_crit = lambda x: (x > 0.9) if self.initial_altruist_freq < 0.5 \
            else (x < 0.1)

        # For each period, there's war, reproduction, mutation and migration
        for it in xrange(1, max_iter+1):
            self.altruists_freq.append(prev_freq)
            if stop_crit(prev_freq):
                break
            logger.debug('war...')
            self.calc_group_reproduction()
            logger.debug('reproduction...')
            self.calc_indiv_reproduction()
            logger.debug('mutation...')
            self.calc_distribution_alts()
            self.calc_mutation()
            logger.debug('migration...')
            self.calc_migration()
            prev_freq = float(np.sum(np.sum(self.alts_per_group)))/total_indiv

        return prev_freq, it


    def calc_war_results(self):
        # Getting number of groups involved at war
        if self.beta == 1.:
            warrior_groups = self.n_groups
        else:
            warrior_groups = np.random.binomial(self.n_groups, self.beta)
        perm_idx = np.random.permutation(self.n_groups)
        warrior_groups = warrior_groups if warrior_groups%2==0 else warrior_groups-1
        warrior_groups_pairs = izip(
            perm_idx[0: warrior_groups/2],
            perm_idx[warrior_groups/2: warrior_groups]
        )
        for group_pair in warrior_groups_pairs:
            idx1, idx2 = group_pair
            num_alts1 = int(self.alts_per_group[idx1])
            num_alts2 = int(self.alts_per_group[idx2])
            random_num = np.random.random()
            winprob_1vs2 = self.ref_winning_prob_matrix[num_alts1, num_alts2]
            if winprob_1vs2 > random_num:
                loser_id = idx2
                winner_n_alts = num_alts1
            else:
                loser_id = idx1
                winner_n_alts = num_alts2
            # loser dies and winner occupies loser's deme
            self.alts_per_group[loser_id] = winner_n_alts


    def calc_group_reproduction(self):
        if self.beta > 0 and self.n_groups > 1:
            self.calc_war_results()
        else:
            self.alts_per_group = np.array(self.alts_per_group, dtype=np.int)
            alts_count = np.bincount(self.alts_per_group)
            values = np.arange(len(alts_count))
            probs = alts_count*self.ref_avg_fitness_values[values]
            probs = probs/np.sum(probs)
            distw = scipy.stats.rv_discrete(
                name='distw',
                values=(values, probs)
            )
            self.alts_per_group = distw.rvs(size=self.n_groups)


    def calc_indiv_reproduction(self):
        self.alts_per_group = np.array(self.alts_per_group, dtype=np.int)
        fitness_curr_gen = self.ref_fitness_values[self.alts_per_group]
        weighted_fitness = self.alts_per_group*fitness_curr_gen
        avg_fitness_curr_gen = self.ref_avg_fitness_values[self.alts_per_group]
        total_avg_fitness = self.n_indiv*avg_fitness_curr_gen
        prob_altruist = weighted_fitness/total_avg_fitness
        self.alts_per_group = np.random.binomial(self.n_indiv, prob_altruist)


    def calc_mutation(self):
        N = self.n_groups
        n = self.n_indiv
        flips = np.random.binomial(1, self.pmut, size=(N,n))
        self.alts_distrib = (self.alts_distrib+flips)%2
        self.alts_per_group = np.sum(self.alts_distrib, axis=1)


    def calc_migration(self):
        if self.pmig > 0.:
            N = self.n_groups
            n = self.n_indiv

            # `n_migs` stores the number of migrants per group
            n_migs = np.random.binomial(n, self.pmig, N)

            # `migs` stores the migrants themselves
            migs = list(chain.from_iterable([self.alts_distrib[i][0:n_migs[i]] for i in range(N)]))

            # shuffle and then redistribute them among groups
            np.random.shuffle(migs)
            cont = 0
            for i in range(N):
                self.alts_distrib[i][0:n_migs[i]] = migs[cont:cont+n_migs[i]]
                cont += n_migs[i]

            self.alts_per_group = np.sum(self.alts_distrib, axis=1)


if __name__ == '__main__':
    params = {
        'n_groups': 5000,
        'n_indiv': 26,
        'n_altruists': 1,
        'cost': 0.03,
        'alpha': 0.03,
        'beta': 0.7,
        'migration_rate': 0.01,
        'mutation_rate': 0.0001
    }
    start = time.clock()
    sim = Simulation(params)
    print '--> (final altruist frequency, #iterations):', sim.run()
    end = time.clock()
    execution_time = end - start
    print 'Execution Time: %.4f seconds'%execution_time
