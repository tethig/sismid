#!/usr/bin/env python3

"""
Simulate Wright-Fisher population dynamics with selection
"""

# packages
import argparse
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

try:
    import itertools.izip as zip
except ImportError:
    import itertools

## global variables
# alphabet = ['A', 'T', 'G', 'C']
# pop_size = 100 # changed to agree with argparse default
# seq_length = 100
# mutation_rate = 0.0001 # per gen per individual per site
# generations = 500
# lethal = 0.1  # proportion of lethal mutations
# mu, sigma = -0.29, 0.31 #mu=mean sigma=sd of phiX174 normal distribution
# lower, upper = -1.0, 1.0 # clips for truncation
# Acknowledgement: table 2 in https://doi.org/10.1111/j.1558-5646.2012.01691.x
# dfem = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
# notes on (a,b) clipping: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html

## population
# base_haplotype = ''.join(["A" for i in range(seq_length)])
# pop = {}
# pop[base_haplotype] = pop_size
# fitnesses = {} # name changed to fitnesses for clarity (fitness is individual)
# fitnesses[base_haplotype] = 1.0
# history, halcyon = [], []

# mutation
def get_mutation_count():
    mean = mutation_rate * pop_size * seq_length
    return np.random.poisson(mean)

def get_random_haplotype():
    haplotypes = list(pop.keys())
    frequencies = [x/float(pop_size) for x in pop.values()]
    total = sum(frequencies)
    frequencies = [x / total for x in frequencies]
    return np.random.choice(haplotypes, p=frequencies)

def get_mutant(haplotype):
    site = int(rng.integers(low=0, high=seq_length, size=1)) # using updated generator
    possible_mutations = list(alphabet)
    possible_mutations.remove(haplotype[site])
    mutation = np.random.choice(possible_mutations)
    new_haplotype = haplotype[:site] + mutation + haplotype[site+1:]
    return new_haplotype

def get_fitness(haplotype):
    old_fitness = fitnesses[haplotype]
    if rng.random(1) < lethal: # mutation may be lethal
        return 0.0
    else: # remaining chance > neutral change (note: memoryless at mutation level)
        new_fitness = old_fitness + float(dfem.rvs(size=1))
        if new_fitness < 0.0:
            return 0.0
        else:
            return new_fitness

def mutation_event():
    haplotype = get_random_haplotype()
    if pop[haplotype] >= 1: # was ">"
        pop[haplotype] -= 1
        new_haplotype = get_mutant(haplotype)
        if new_haplotype in pop:
            pop[new_haplotype] += 1
        else:
            pop[new_haplotype] = 1
        if new_haplotype not in fitnesses:
            fitnesses[new_haplotype] = get_fitness(haplotype)

def mutation_step():
    mutation_count = get_mutation_count()
    for i in range(mutation_count):
        mutation_event()

# genetic drift and selection
def get_offspring_counts():
    haplotypes = list(pop.keys())
    frequencies = [pop[haplotype]/float(pop_size) for haplotype in haplotypes]
    fitness_list = [fitnesses[haplotype] for haplotype in haplotypes] # name changed to fitness_list
    weights = [x * y for x,y in zip(frequencies, fitness_list)]
    total = sum(weights)
    weights = [x / total for x in weights]
    return list(np.random.multinomial(pop_size, weights))

def offspring_step():
    counts = get_offspring_counts()
    for (haplotype, count) in zip(list(pop.keys()), counts):
        if (count > 0):
            pop[haplotype] = count
        else:
            del pop[haplotype]

# simulate
def time_step():
    mutation_step()
    offspring_step()

def simulate():
    clone_pop = dict(pop)
    history.append(clone_pop)
    clone_fit = dict(fitnesses)
    halcyon.append(clone_fit) # store fitnesses too
    for i in range(generations):
        time_step()
        clone_pop = dict(pop)
        history.append(clone_pop)
        clone_fit = dict(fitnesses)
        halcyon.append(clone_fit)

# plot diversity
def get_distance(seq_a, seq_b):
    diffs = 0
    length = len(seq_a)
    assert len(seq_a) == len(seq_b)
    for chr_a, chr_b in zip(seq_a, seq_b):
        if chr_a != chr_b:
            diffs += 1
    return diffs / float(length)

def get_diversity(population):
    haplotypes = list(population.keys())
    haplotype_count = len(haplotypes)
    diversity = 0
    for i in range(haplotype_count):
        for j in range(haplotype_count):
            haplotype_a = haplotypes[i]
            haplotype_b = haplotypes[j]
            frequency_a = population[haplotype_a] / float(pop_size)
            frequency_b = population[haplotype_b] / float(pop_size)
            frequency_pair = frequency_a * frequency_b
            diversity += frequency_pair * get_distance(haplotype_a, haplotype_b)
    return diversity

def get_diversity_trajectory():
    trajectory = [get_diversity(generation) for generation in history]
    return trajectory

def diversity_plot(xlabel="generation"):
    mpl.rcParams['font.size']=14
    trajectory = get_diversity_trajectory()
    plt.plot(trajectory, "#447CCD")
    plt.ylabel("diversity")
    plt.xlabel(xlabel)

# plot divergence
def get_divergence(population):
    haplotypes = list(population.keys())
    divergence = 0
    for haplotype in haplotypes:
        frequency = population[haplotype] / float(pop_size)
        divergence += frequency * get_distance(base_haplotype, haplotype)
    return divergence

def get_divergence_trajectory():
    trajectory = [get_divergence(generation) for generation in history]
    return trajectory

def divergence_plot(xlabel="generation"):
    mpl.rcParams['font.size']=14
    trajectory = get_divergence_trajectory()
    plt.plot(trajectory, "#447CCD")
    plt.ylabel("divergence")
    plt.xlabel(xlabel)

# plot fitness (new)
def get_mean_fitness(population, fitdict):
    haplotypes = list(population.keys())
    w = 0
    for haplotype in haplotypes:
        frequency = population[haplotype] / float(pop_size)
        w += frequency * fitdict[haplotype]
    return w

def get_mean_fitness_trajectory():
    #print(history[0], "\n", halcyon[0]) # testing initiation
    trajectory = [get_mean_fitness(pgen, fgen) for pgen, fgen in zip(history, halcyon)]
    return trajectory

def fitness_plot(xlabel="generation"):
    mpl.rcParams['font.size'] = 14
    trajectory = get_mean_fitness_trajectory()
    plt.plot(trajectory, "#E76B6B")
    plt.ylabel("mean fitness")
    plt.xlabel(xlabel)

# plot trajectories
def get_frequency(haplotype, generation):
    pop_at_generation = history[generation]
    if haplotype in pop_at_generation:
        return pop_at_generation[haplotype]/float(pop_size)
    else:
        return 0

def get_trajectory(haplotype):
    trajectory = [get_frequency(haplotype, gen) for gen in range(generations)]
    return trajectory

def get_all_haplotypes():
    haplotypes = set()
    for generation in history:
        for haplotype in generation:
            haplotypes.add(haplotype)
    return haplotypes

def stacked_trajectory_plot(xlabel="generation"):
    colors_lighter = ["#A567AF", "#8F69C1", "#8474D1", "#7F85DB", "#7F97DF", "#82A8DD", "#88B5D5", "#8FC0C9", "#97C8BC", "#A1CDAD", "#ACD1A0", "#B9D395", "#C6D38C", "#D3D285", "#DECE81", "#E8C77D", "#EDBB7A", "#EEAB77", "#ED9773", "#EA816F", "#E76B6B"]
    mpl.rcParams['font.size']=18
    haplotypes = get_all_haplotypes()
    trajectories = [get_trajectory(haplotype) for haplotype in haplotypes]
    plt.stackplot(range(generations), trajectories, colors=colors_lighter)
    plt.ylim(0, 1)
    plt.ylabel("frequency")
    plt.xlabel(xlabel)

# plot snp trajectories
def get_snp_frequency(site, generation):
    minor_allele_frequency = 0.0
    pop_at_generation = history[generation]
    for haplotype in pop_at_generation.keys():
        allele = haplotype[site]
        frequency = pop_at_generation[haplotype] / float(pop_size)
        if allele != "A":
            minor_allele_frequency += frequency
    return minor_allele_frequency

def get_snp_trajectory(site):
    trajectory = [get_snp_frequency(site, gen) for gen in range(generations)]
    return trajectory

def get_all_snps():
    snps = set()
    for generation in history:
        for haplotype in generation:
            for site in range(seq_length):
                if haplotype[site] != "A":
                    snps.add(site)
    return snps

def snp_trajectory_plot(xlabel="generation"):
    colors = ["#781C86", "#571EA2", "#462EB9", "#3F47C9", "#3F63CF", "#447CCD", "#4C90C0", "#56A0AE", "#63AC9A", "#72B485", "#83BA70", "#96BD60", "#AABD52", "#BDBB48", "#CEB541", "#DCAB3C", "#E49938", "#E68133", "#E4632E", "#DF4327", "#DB2122"]
    mpl.rcParams['font.size']=18
    snps = get_all_snps()
    trajectories = [get_snp_trajectory(snp) for snp in snps]
    data = []
    for trajectory, color in zip(trajectories, itertools.cycle(colors)):
        data.append(range(generations))
        data.append(trajectory)
        data.append(color)
    plt.plot(*data)
    plt.ylim(0, 1)
    plt.ylabel("frequency")
    plt.xlabel(xlabel)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "run Wright-Fisher simulation with mutation and genetic drift")
    parser.add_argument('--pop_size', type = int, default = 100.0, help = "population size")
    parser.add_argument('--mutation_rate', type = float, default = 0.0001, help = "mutation rate")
    parser.add_argument('--seq_length', type = int, default = 100, help = "sequence length")
    parser.add_argument('--generations', type = int, default = 500, help = "generations")
    parser.add_argument('--lethal', type = float, default = 0.1, help = "chance mutation is lethal")
    parser.add_argument('--lower', type = float, default = -1.0, help = "lower limit on selection co-efficient")
    parser.add_argument('--upper', type = float, default = 1.0, help = "upper limit on selection co-efficient")
    parser.add_argument('--summary', action = "store_true", default = False, help = "don't plot trajectories")

    params = parser.parse_args()
    pop_size = params.pop_size
    mutation_rate = params.mutation_rate
    seq_length = params.seq_length
    generations = params.generations
    lethal = params.lethal
    lower = params.lower
    upper = params.upper

    # initialize
    rng = np.random.default_rng()  # create an instance of a Generator
    alphabet = ['A', 'T', 'G', 'C']
    mu, sigma = -0.29, 0.31
    dfem = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    base_haplotype = ''.join(["A" for i in range(seq_length)])
    pop = {}
    pop[base_haplotype] = pop_size
    fitnesses = {}  # name changed to fitnesses for clarity (fitness is individual)
    fitnesses[base_haplotype] = 1.0
    history, halcyon = [], []

    # simulate
    simulate()

    if params.summary:
        parameters = (pop_size, mutation_rate, seq_length, generations, lethal, lower, upper)
        par_as_str = '_'.join(([str(x) for x in parameters]))
        par_as_header = "# pop_size: {}, mut_rate: {}, seq_len: {}, gens: {}, lethal: {}, lower: {}, upper: {}, mu: -0.29, sigma: 0.31".format(*parameters)

        plt.figure(num=None, figsize=(14, 7.5), dpi=80, facecolor='w', edgecolor='k')
        plt.subplot2grid((2, 1), (0, 0))
        stacked_trajectory_plot(xlabel="")
        plt.subplot2grid((2, 1), (1, 0))
        snp_trajectory_plot()
        plt.savefig(par_as_str + "_haplo_snp.png")

        plt.figure(num=None, figsize=(12, 7.5), dpi=80, facecolor='w', edgecolor='k')
        plt.subplot2grid((3, 1), (0, 0))
        diversity_plot()
        plt.subplot2grid((3, 1), (1, 0))
        divergence_plot()
        plt.subplot2grid((3, 1), (2, 0))
        fitness_plot()
        plt.savefig(par_as_str + "_variables.png")

        with(open(par_as_str + "_haplorecord.txt", "w")) as recordsheet:
            print(par_as_header, file=recordsheet)
            print("haplotype\tfitness\tfirstgen\tduration\t", "\t".join("gen"+str(x+1) for x in range(generations)), file=recordsheet)

            haplotypes = get_all_haplotypes()
            for haplotype in haplotypes:
                traj = get_trajectory(haplotype)
                not_zeroes = [x != 0 for x in traj]
                living_gen = sum(not_zeroes)
                if True in not_zeroes:
                    first_gen = not_zeroes.index(True) + 1
                else:
                    first_gen = 'NA'
                print(haplotype, "\t", get_fitness(haplotype), "\t", first_gen, "\t",
                    living_gen, "\t", "\t".join([str(x * pop_size) for x in traj]), file=recordsheet)

        with(open(par_as_str + "_reduced.txt", "w")) as summarysheet:
            print(par_as_header, file=summarysheet)
            print("measure\tbaseline\t", "\t".join("gen"+str(x+1) for x in range(generations)), file=summarysheet)

            trajectory = get_diversity_trajectory()
            print("diversity\t", "\t".join([str(x) for x in trajectory]), file=summarysheet)
            trajectory = get_divergence_trajectory()
            print("divergence\t", "\t".join([str(x) for x in trajectory]), file=summarysheet)
            trajectory = get_mean_fitness_trajectory()
            print("meanfitness\t", "\t".join([str(x) for x in trajectory]), file=summarysheet)
        
    else:
        plt.figure(num=None, figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')
        plt.subplot2grid((4, 2), (0, 0), colspan=2)
        stacked_trajectory_plot(xlabel="")
        plt.subplot2grid((4, 2), (1, 0), colspan=2)
        snp_trajectory_plot(xlabel="")
        plt.subplot2grid((4, 2), (2, 0), colspan=2)
        fitness_plot(xlabel="")
        plt.subplot2grid((4, 2), (3, 0))
        diversity_plot()
        plt.subplot2grid((4, 2), (3, 1))
        divergence_plot()
        plt.show()
