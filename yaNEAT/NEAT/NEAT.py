from dataclasses import dataclass, field
from bisect import bisect_left
import numpy as np
from typing import List, Dict, Set, Tuple


@dataclass
class Gene:
    """A gene is a combination of a structural innovation and some changeable attributes."""
    innovation: int
    weight: float
    activation: str
    active: bool = field(default=True)


@dataclass
class Innovation:
    """Source-destination pair."""
    source: int
    destination: int


@dataclass
class Genome:
    """Represents the genotype for a network. It is essentially a linearized representation of a graph."""
    genes: Dict[int, Gene]
    connections: Set[Tuple[int, int]]
    nodes: List[int]

    def insert_node(self, node):
        i = bisect_left(self.nodes, node)
        if i >= len(self.nodes) - 1 or self.nodes[i + 1] != node:
            self.nodes.insert(i, node)

    def innovation(self):
        return max(self.genes.keys(), default=0)


class NEATContext:
    """A manager for common information needed by NEAT."""
    def __init__(self, mutation_prob, activations, innovations=None):
        self.mutation_prob = mutation_prob
        self.activations = activations

        if innovations:
            self.innovations = innovations
        else:
            self.innovations = []

    def reproduce(self, parent1: Genome, parent2: Genome, fitness1, fitness2):
        """Reproduce two genomes."""
        child = Genome({}, set(), [])
        for i in range(max(parent1.innovation(), parent2.innovation())):
            gene = None
            if i in parent1.genes and i in parent2.genes:
                if fitness1 > fitness2 or np.random.rand() > 0.5:
                    # inherit from parent1
                    gene = parent1.genes[i]
                else:
                    # inherit from parent2
                    gene = parent2.genes[i]
            else:
                # disjoint/excess gene
                if i in parent1.genes:
                    if fitness1 > fitness2 or np.random.rand() > 0.5:
                        gene = parent1.genes[i]
                elif i in parent2.genes:
                    if fitness1 < fitness2 or np.random.rand() > 0.5:
                        gene = parent2.genes[i]

            if gene:
                innov = self.innovations[gene.innovation]
                child.genes[i] = gene
                child.connections.add((innov.source, innov.destination))
                child.insert_node(innov.source)
                child.insert_node(innov.destination)

        # mutate
        if np.random.rand() < self.mutation_prob:
            if np.random.rand() < 0.5:
                self.mutate_add_node(child)
            else:
                self.mutate_add_connection(child)

        return child

    def mutate_add_node(self, genome: Genome):
        """Add a random node to the genome."""
        k = max(genome.nodes, default=0) + 1  # id for next node

        # TODO this is some garbage fix it
        # select randomly from connections
        i, j = next(iter(genome.connections))

        # disable connection
        innov, oldgene = [(innov, gene) for innov, gene in genome.genes.items()
                          if self.innovations[innov].source == i and self.innovations[innov].destination == j][0]
        genome.genes[innov].active = False
        genome.connections.remove((self.innovations[innov].source, self.innovations[innov].destination))

        # add connection gene from source to new
        innov1 = Innovation(i, k)
        self.innovations.append(innov1)

        genome.genes[len(self.innovations)] = Gene(len(self.innovations), 1., np.random.choice(self.activations))
        genome.connections.add((i, k))

        # add connection gene from new to destination
        innov2 = Innovation(k, j)
        self.innovations.append(innov2)

        genome.genes[len(self.innovations)] = Gene(len(self.innovations), oldgene.weight, oldgene.activation)
        genome.connections.add((k, j))

        # track new node
        genome.insert_node(k)

    def mutate_add_connection(self, genome: Genome):
        """Add a random connection to the genome."""
        # TODO fix edge case where there are no connections left to make, and clean up selection code
        # find next structural mutation
        if len(genome.nodes) == 0:
            raise Exception('genome must have at least 1 node to add connection')

        assert len(genome.nodes) >= 1, 'length error on genome {}'.format(genome)
        i, j = np.random.choice(genome.nodes, size=2)
        while (i, j) in genome.connections:
            i, j = np.random.choice(genome.nodes, size=2)

        innov = Innovation(i, j)
        self.innovations.append(innov)

        # TODO make sure that activation matches the activation of any previous connection to this destination
        genome.genes[len(self.innovations)] = Gene(len(self.innovations), np.random.uniform(),
                                                   np.random.choice(self.activations))
        genome.connections.add((i, j))


def compatibility_distance(g1: Genome, g2: Genome, c1, c2, c3):
    """Compute a distance measure between two genomes. Specifically, compute a linear combination of the number of
    disjoint genes, the number of excess genes, and the average difference of weights, including disabled genes."""
    D, W = 0, 0.
    E = abs(g1.innovation() - g2.innovation())
    N = max(len(g1.genes), len(g2.genes))
    for i in range(max(g1.innovation(), g2.innovation())):
        if i in g1 and i in g2:
            E += g1.genes[i].weight - g2.genes[i].weight
        elif i in g1 or i in g2:
            D += 1
    return c1 * E / N + c2 * D / N + c3 * W


def assign_species(representative_genomes, genomes, c1, c2, c3, threshold):
    """Given genomes that are representative of a species, assign each new genome to the first species that is close
    enough or assign to a new species, if no existing one is close enough."""
    assignments = [] * len(representative_genomes)
    for genome in genomes:
        for i, repgenome in enumerate(representative_genomes):
            distance = compatibility_distance(genome, repgenome, c1, c2, c3)
            if distance < threshold:
                assignments[i].append(genome)
                break
        else:
            # no matching genome; speciate
            assignments.append([genome])


def assign_offspring_counts(all_species, total_offspring, fitness_fn):
    """Assign an offspring count to each species and compute the adjusted fitness for each genome. Specifically, give
    each species an amount proportional to the sum of the species' adjusted fitnesses."""
    fitnesses = []
    for species in all_species:
        fitnesses = []
        for organism in species:
            fitness = fitness_fn(organism)
            fitness /= len(species)
            fitnesses.append(fitness)
    total_fitnesses = [sum(species_fitnesses) for species_fitnesses in fitnesses]
    total_fitness = sum(total_fitnesses)
    offspring_counts = [int(total_offspring * species_fitness / total_fitness) for species_fitness in total_fitnesses]
    return fitnesses, offspring_counts


def assign_next_generation_pairings(fit_species, offspring_counts):
    """Given the fit species of each species, assign random pairings for the next generation."""
    return [[np.random.choice(species, 2, replace=False) for _ in range(offspring)]
            for species, offspring in zip(fit_species, offspring_counts)]