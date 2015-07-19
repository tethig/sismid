# packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

# global variables
pop_size = 50
seq_length = 100
alphabet = ['A', 'T', 'G', 'C']
mutation_rate = 0.0001 # per gen per individual per site
generations = 500

# population
base_haplotype = ''.join(["A" for i in range(seq_length)])
pop = {}
pop[base_haplotype] = pop_size
history = []

# mutation
def get_mutation_count():
    mean = mutation_rate * pop_size * seq_length
    return np.random.poisson(mean)

def get_random_haplotype():
    haplotypes = pop.keys() 
    frequencies = [x/float(pop_size) for x in pop.values()]
    return np.random.choice(haplotypes, p=frequencies)

def get_mutant(haplotype):
    site = np.random.randint(seq_length)
    possible_mutations = list(alphabet)
    possible_mutations.remove(haplotype[site])
    mutation = np.random.choice(possible_mutations)
    new_haplotype = haplotype[:site] + mutation + haplotype[site+1:]
    return new_haplotype

def mutation_event():
    haplotype = get_random_haplotype()
    if pop[haplotype] > 1:
        pop[haplotype] -= 1
        new_haplotype = get_mutant(haplotype)
        if new_haplotype in pop:
            pop[new_haplotype] += 1
        else:
            pop[new_haplotype] = 1

def mutation_step():
    mutation_count = get_mutation_count()
    for i in range(mutation_count):
        mutation_event()

# genetic drift
def get_offspring_counts():
    haplotypes = pop.keys() 
    frequencies = [x/float(pop_size) for x in pop.values()]
    return list(np.random.multinomial(pop_size, frequencies))

def offspring_step():
    counts = get_offspring_counts()
    for (haplotype, count) in zip(pop.keys(), counts):
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
    for i in range(generations):
        time_step()
        clone_pop = dict(pop)
        history.append(clone_pop)

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
    haplotypes = population.keys()
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
    plt.plot(trajectory)    
    plt.ylabel("diversity")
    plt.xlabel(xlabel)
    
# plot divergence
def get_divergence(population):
    haplotypes = population.keys()
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
    plt.plot(trajectory)
    plt.ylabel("divergence")
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

def stacked_trajectory_plot():
	colors = ["#781C86", "#571EA2", "#462EB9", "#3F47C9", "#3F63CF", "#447CCD", "#4C90C0", "#56A0AE", "#63AC9A", "#72B485", "#83BA70", "#96BD60", "#AABD52", "#BDBB48", "#CEB541", "#DCAB3C", "#E49938", "#E68133", "#E4632E", "#DF4327", "#DB2122"]
	mpl.rcParams['font.size']=18
	haplotypes = get_all_haplotypes()
	trajectories = [get_trajectory(haplotype) for haplotype in haplotypes]
	plt.stackplot(range(generations), trajectories, colors=colors)
	plt.ylim(0, 1)
	plt.ylabel("frequency")
	plt.xlabel("generation")

if __name__=="__main__":
	parser = argparse.ArgumentParser(description = "run wright-fisher simulation with mutation and genetic drift")
	parser.add_argument('--pop_size', type = int, default = 50.0, help = "population size")
	parser.add_argument('--mutation_rate', type = float, default = 0.0001, help = "mutation rate")
	parser.add_argument('--seq_length', type = int, default = 100, help = "sequence length")
	parser.add_argument('--generations', type = int, default = 500, help = "generations")
	parser.add_argument('--no_hap', action = "store_true", default = False, help = "don't plot haplotypes")		

	params = parser.parse_args()
	pop_size = params.pop_size
	mutation_rate = params.mutation_rate
	seq_length = params.seq_length
	generations = params.generations		
	
	simulate()

	plt.figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')
	if params.no_hap:
		plt.subplot2grid((2,1), (0,0))
		diversity_plot()
		plt.subplot2grid((2,1), (1,0))
		divergence_plot()
	else:	
		plt.subplot2grid((2,2), (0,0), colspan=2)
		stacked_trajectory_plot()
		plt.subplot2grid((2,2), (1,0))
		diversity_plot()
		plt.subplot2grid((2,2), (1,1))
		divergence_plot()
	plt.show()		
