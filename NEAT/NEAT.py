import random
import copy
from graphviz import Digraph
import uuid
from abc import ABC, abstractmethod

class Evaluator(ABC):

    def __init__(self, population_size, input_size, output_size, mutation_rate, perturbating_probability, add_connection_mutation_chance, add_node_mutation_chance, compatibility_distance_threshold, c1, c2, c3):

        self.historical_marker = HistoricalMarker()

        self.mutation_rate = mutation_rate
        self.perturbating_probability = perturbating_probability
        self.add_connection_mutation_chance = add_connection_mutation_chance
        self.add_node_mutation_chance = add_node_mutation_chance
        self.compatibility_distance_threshold = compatibility_distance_threshold
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.genomes = []
        self.species = []
        self.population_size = population_size

        self.genome_fitness = {} # Dictionnary (key: genome, value: fitness)
        self.genome_species = {} # Dictionnary (key: genome, value: species)

        self.fittest_genome = None
        self.highest_fitness = 0
        self.generation_number = 0

        self.next_generation_genomes = []
        # Create the first generation
        self.create_first_generation(input_size,output_size)

    @abstractmethod
    def evaluate_genome(self,genome):
        pass

    def create_first_generation(self,input_size,output_size):
        genome = Genome({},{})
        for i in range(input_size):
            genome.add_node_gene(NodeGene("INPUT",self.historical_marker))
        for i in range(output_size):
            genome.add_node_gene(NodeGene("OUTPUT",self.historical_marker))
        for node_1 in list(genome.node_genes.values()):
            for node_2 in list(genome.node_genes.values()):
                if node_1.type == "INPUT" and node_2.type == "OUTPUT":
                    new_connection = ConnectionGene(node_1.innovation_number,node_2.innovation_number, random.uniform(-1,1),True,self.historical_marker)
                    genome.add_connection_gene(new_connection)
        self.genomes.append(genome)
        while len(self.genomes) < self.population_size:
            self.genomes.append(genome.copy())

    def evaluate(self):
        self.generation_number += 1
        print("{:=^50}".format(" Generation {} ".format(self.generation_number)))
        # Reset the Evaluator
        self.reset()
        # Place genome into species
        self.place_genomes_into_species()
        # Remove dormant species
        self.remove_dormant_species()
        # Evaluate genome and assign fitness
        self.evaluate_genomes()
        # Put best genome of each generation into the next generation
        self.start_next_generation()
        # Breed the rest of the genome
        self.complete_next_generation()
        # Display some information about each generation
        self.generation_info_display()

    def reset(self):
        for species in self.species:
            species.reset()
        self.genome_fitness.clear()
        self.genome_species.clear()
        self.next_generation_genomes.clear()

    def place_genomes_into_species(self):

        for genome in self.genomes:
            found_species = False
            for species in self.species:
                if Genome.compatibility_distance(species.mascot, genome, self.c1,self.c2,self.c3) < self.compatibility_distance_threshold:
                    species.add_genome(genome)
                    self.genome_species[genome] = species
                    found_species = True
                    break
            if not(found_species):
                # If nothing is found, create a new species
                new_species = Species(genome)
                self.species.append(new_species)
                self.genome_species[genome] = new_species

    def remove_dormant_species(self):
        for species in self.species:
            if not(species):
                self.species.remove(species)

    def evaluate_genomes(self):

        for genome in self.genomes:
            genome_species = self.genome_species[genome]
            fitness = self.evaluate_genome(genome)
            adjusted_fitness = fitness / len(genome_species.genomes)

            self.genome_fitness[genome] = adjusted_fitness
            genome_species.add_adjusted_fitness(adjusted_fitness)
            genome_species.add_genome_fitness(genome, adjusted_fitness)

            if adjusted_fitness > self.highest_fitness:
                self.highest_fitness = adjusted_fitness
                self.fittest_genome = genome

    def start_next_generation(self):
        for species in self.species:
            fittest_genome = max(species.genome_fitness, key=species.genome_fitness.get)
            self.next_generation_genomes.append(fittest_genome)

    def complete_next_generation(self):

        while len(self.next_generation_genomes) < self.population_size:
            species = self.get_random_species_biased_adjusted_fitness()

            genome_1 = self.get_random_genome_biased_adjusted_fitness(species)
            genome_2 = self.get_random_genome_biased_adjusted_fitness(species)

            if self.genome_fitness[genome_1] >= self.genome_fitness[genome_2]:
                child = Genome.crossover(genome_1,genome_2)
            else:
                child = Genome.crossover(genome_2,genome_1)

            if random.random() < self.mutation_rate:
                child.mutate(self.perturbating_probability)

            if random.random() < self.add_connection_mutation_chance:
                child.add_connection_gene_mutation(self.historical_marker)

            if random.random() < self.add_node_mutation_chance:
                child.add_node_gene_mutation(self.historical_marker)

            self.next_generation_genomes.append(child)

        self.genomes = self.next_generation_genomes
        self.next_generation_genomes = []

    def get_random_species_biased_adjusted_fitness(self):
        total_fitness = 0
        for species in self.species:
            total_fitness += species.total_adjusted_fitness
        goal_fitness = random.uniform(0,total_fitness)
        count_fitness = 0
        for species in self.species:
            count_fitness += species.total_adjusted_fitness
            if count_fitness >= goal_fitness:
                return species

    def get_random_genome_biased_adjusted_fitness(self,species):
        total_fitness = 0
        for genome in species.genomes:
            total_fitness += species.genome_fitness[genome]
        goal_fitness = random.uniform(0,total_fitness)
        count_fitness = 0
        for genome in species.genomes:
            count_fitness += species.genome_fitness[genome]
            if count_fitness >= goal_fitness:
                return genome

    def generation_info_display(self):
        pass
        # Add the display of the fittest number each 5/10 generations
        # Add the number of species, highest fitness

class Species():

    def __init__(self,genome):
        self.genomes = [genome]
        self.mascot = genome
        self.genome_fitness = {}
        self.total_adjusted_fitness = 0

    def add_genome(self,genome):
        self.genomes.append(genome)

    def add_adjusted_fitness(self, adjusted_fitness):
        self.total_adjusted_fitness += adjusted_fitness

    def add_genome_fitness(self, genome, adjusted_fitness):
        self.genome_fitness[genome] = adjusted_fitness

    def reset(self):
        self.mascot = random.choice(self.genomes)
        self.genomes.clear()
        self.genome_fitness.clear()
        self.total_adjusted_fitness = 0


class Genome():

    def __init__(self, connection_genes, node_genes):
        self.connection_genes = connection_genes
        self.node_genes = node_genes

    def __repr__(self):
        return "Number of node genes : {}. Number of connection genes : {}.".format(len(self.node_genes),len(self.connection_genes))

    def copy(self):
        return copy.deepcopy(self)

    def add_node_gene(self,node):
        self.node_genes[node.innovation_number] = node

    def add_connection_gene(self,connection):
        self.connection_genes[connection.innovation_number] = connection

    def add_connection_gene_mutation(self,historical_marker):

        connection_valid = False
        count = 0
        # Notice that if after 3 trials, not valid connection is found, then there might not be any valid new connection (for instance at the beginnin, all the node are linked are no connection can be added)
        while not(connection_valid) and count<=3:
            count+=1
            # Set connection_valid to True, this value is changed later if the connection is not valiid
            connection_valid = True
            # Randomly select two nodes
            node_1 = random.choice(list(self.node_genes.values()))
            node_2 = random.choice(list(self.node_genes.values()))
            # Check if the connection is between a node and himself
            if node_1.innovation_number == node_2.innovation_number:
                connection_valid = False
            # Check if the nodes are both INPUT or both OUTPUT
            if (node_1.type=="INPUT" and node_2.type=="INPUT") or (node_1.type=="OUTPUT" and node_2.type=="OUTPUT"):
                connection_valid = False
            # Check if the connection already exists
            for connection in self.connection_genes.values():
                if (connection.in_node==node_1.innovation_number and connection.out_node==node_2.innovation_number) or (connection.in_node==node_2.innovation_number and connection.out_node==node_1.innovation_number):
                    connection_valid = False

        # Find the correct direction of the connection
        if (node_1.type=="HIDDEN" and node_2.type =="INPUT") or (node_1.type=="OUTPUT" and node_2.type=="HIDDEN") or (node_1.type=="OUTPUT" and node_2.type =="INPUT"):
            node_1, node_2 = node_2, node_1

        # Generate a random weight
        weight = random.gauss(0,0.3)
        # Create the new connection
        new_connection = ConnectionGene(node_1.innovation_number,node_2.innovation_number,weight, True, historical_marker)
        # Add the connection to the list
        self.add_connection_gene(new_connection)

    def add_node_gene_mutation(self,historical_marker):
        # Choose a random connection
        old_connection = random.choice(list(self.connection_genes.values()))
        # Deactivate the old connection
        old_connection.disable()
        # Create the new node
        new_node = NodeGene("HIDDEN",historical_marker)
        self.add_node_gene(new_node)
        # Create the two new connections

        new_connection_1 = ConnectionGene(old_connection.in_node, new_node.innovation_number, 1, True, historical_marker)
        new_connection_2 = ConnectionGene(new_node.innovation_number,old_connection.out_node, old_connection.weight, True, historical_marker)
        # Add the two connection
        self.add_connection_gene(new_connection_1)
        self.add_connection_gene(new_connection_2)

    def mutate(self,perturbating_probability):
        for connection in list(self.connection_genes.values()):
            connection.mutate_weight(perturbating_probability)

    def crossover(parent1,parent2):
        # Notice that the parent1 is the most fit parent.
        child = Genome({},{})
        for node in parent1.node_genes.values():
            child.add_node_gene(node.copy())

        for connection in parent1.connection_genes.values():
            if connection.innovation_number in list(parent2.connection_genes.keys()):
                parent = random.choice([parent1,parent2])
                child.add_connection_gene(parent.connection_genes[connection.innovation_number].copy())
            else:
                # Disjoint or excess gene
                child.add_connection_gene(connection.copy())
        return child

    def print_genome(self):
        id = str(uuid.uuid4())
        graph = Digraph(name=id, filename=id+".gv", directory="models/", format="pdf")
        graph.attr(rankdir="LR")
        graph.attr(splines="line")
        graph.attr(ranksep = "1.2")
        graph.attr(nodesep = "0.2")

        for layer_type in ["INPUT","OUTPUT"]:
            with graph.subgraph(name='cluster_'+layer_type) as cluster:
                cluster.attr(style='filled')
                cluster.attr(color='lightgrey')
                cluster.node_attr.update(color='white')
                for node in self.node_genes.values():
                    if node.type==layer_type:
                        cluster.node(str(node.innovation_number))
                cluster.attr(label=layer_type)

        with graph.subgraph(name='cluster_hidden') as cluster_hidden:
            # cluster_hidden.attr(style='filled')
            cluster_hidden.attr(color='grey')
            for node in self.node_genes.values():
                if node.type=="HIDDEN":
                    cluster_hidden.node(str(node.innovation_number))
            cluster_hidden.attr(label='HIDDEN')

        for connection in self.connection_genes.values():
            if connection.expressed:
                graph.edge(str(connection.in_node),str(connection.out_node), label=str(connection.innovation_number))

        graph.view()

    def compatibility_distance(genome_1,genome_2,c1,c2,c3):
        E = Genome.count_excess_genes(genome_1,genome_2)
        D = Genome.count_disjoint_genes(genome_1,genome_2)
        N = Genome.count_larger_gene_number(genome_1,genome_2)
        W = Genome.calculate_average_weight_matching_genes(genome_1,genome_2)
        return c1*E/N + c2*D/N + c3*W

    def calculate_average_weight_matching_genes(genome_1,genome_2):
        weight_differences_sum = 0
        matching_number = 0
        for innovation_number in genome_1.connection_genes.keys():
            if innovation_number in list(genome_2.connection_genes.keys()):
                weight_differences_sum += abs(genome_1.connection_genes[innovation_number].weight - genome_2.connection_genes[innovation_number].weight)
                matching_number += 1
        return weight_differences_sum /matching_number

    def count_matching_genes(genome_1,genome_2):
        matching_genes_number = 0
        # Count the number of matching nodes
        for innovation_number in genome_1.node_genes.keys():
            if innovation_number in list(genome_2.node_genes.keys()):
                matching_genes_number +=1
        # Count the number of mathcing connections
        for innovation_number in genome_1.connection_genes.keys():
            if innovation_number in list(genome_2.connection_genes.keys()):
                matching_genes_number +=1
        return matching_genes_number

    def count_disjoint_genes(genome_1,genome_2):
        disjoint_genes_number = 0
        max_innovation_number_1 = max(genome_1.connection_genes.keys())
        max_innovation_number_2 = max(genome_2.connection_genes.keys())
        smallest_max_innovation_number = min(max_innovation_number_1,max_innovation_number_2)
        for innovation_number in range(1,smallest_max_innovation_number+1):
            # Count the number of disjoint nodes
            if innovation_number in list(genome_1.node_genes.keys()) and not(innovation_number in list(genome_2.node_genes.keys())):
                disjoint_genes_number += 1
            if not(innovation_number in list(genome_1.node_genes.keys())) and innovation_number in list(genome_2.node_genes.keys()):
                disjoint_genes_number += 1
            # Count the number of disjoint connection
            if innovation_number in list(genome_1.connection_genes.keys()) and not(innovation_number in list(genome_2.connection_genes.keys())):
                disjoint_genes_number += 1
            if not(innovation_number in list(genome_1.connection_genes.keys())) and innovation_number in list(genome_2.connection_genes.keys()):
                disjoint_genes_number += 1
        return disjoint_genes_number

    def count_excess_genes(genome_1,genome_2):
        excess_genes_number = 0
        max_innovation_number_1 = max(genome_1.connection_genes.keys())
        max_innovation_number_2 = max(genome_2.connection_genes.keys())
        smallest_max_innovation_number = min(max_innovation_number_1,max_innovation_number_2)
        greatest_max_innovation_number = max(max_innovation_number_1,max_innovation_number_2)
        for innovation_number in range(1,greatest_max_innovation_number+1):
            # Count the number of excess nodes
            if innovation_number in list(genome_1.node_genes.keys()) and not(innovation_number in list(genome_2.node_genes.keys()))  and innovation_number > smallest_max_innovation_number:
                excess_genes_number += 1
            if not(innovation_number in list(genome_1.node_genes.keys())) and innovation_number in list(genome_2.node_genes.keys())  and innovation_number > smallest_max_innovation_number:
                excess_genes_number += 1
            # Count the number of excess connection
            if innovation_number in list(genome_1.connection_genes.keys()) and not(innovation_number in list(genome_2.connection_genes.keys()))  and innovation_number > smallest_max_innovation_number:
                excess_genes_number += 1
            if not(innovation_number in list(genome_1.connection_genes.keys())) and innovation_number in list(genome_2.connection_genes.keys())  and innovation_number > smallest_max_innovation_number:
                excess_genes_number += 1
        return excess_genes_number

    def count_larger_gene_number(genome_1,genome_2):
        gene_number_1 = len(genome_1.node_genes) + len(genome_1.connection_genes)
        gene_number_2 = len(genome_2.node_genes) + len(genome_2.connection_genes)
        return max(gene_number_1,gene_number_2)



class ConnectionGene():

    def __init__(self,in_node, out_node, weight, expressed, historical_marker):

        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.expressed = expressed
        self.innovation_number = historical_marker.get_innovation_number(self)
    
    def enable(self):
        self.expressed = True

    def disable(self):
        self.expressed = False

    def copy(self):
        return copy.deepcopy(self)

    def mutate_weight(self, perturbating_probability):
        if random.random() < perturbating_probability:
            self.weight +=  random.gauss(0,0.3)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<the value of the standard deviation can be changed 
        else:
            self.weight = random.uniform(-1,1)


class NodeGene():

    def __init__(self, type, historical_marker):
        self.type = type # The type should be either "INPUT"/"HIDDEN"/"OUTPUT"
        self.innovation_number = historical_marker.get_innovation_number(self)

    def copy(self):
        return copy.deepcopy(self)

class HistoricalMarker():

    def __init__(self):
        self.existing_connections = {}
        self.existing_nodes = {}
        self.connection_innovation_number = 0
        self.node_innovation_number = 0

    def __repr__(self):
        return "Number of node : {}. Number of connection : {}.".format(len(self.existing_nodes),len(self.existing_connections))

    def get_innovation_number(self,new_gene):
        if new_gene.__class__.__name__ == "ConnectionGene":
            new_connection = new_gene
            for connection in self.existing_connections.values():
                if connection.in_node==new_connection.in_node and connection.out_node==new_connection.out_node:
                    return connection.innovation_number
            self.connection_innovation_number += 1
            self.existing_connections[self.connection_innovation_number] = new_connection
            return self.connection_innovation_number
        elif new_gene.__class__.__name__ == "NodeGene":
            self.node_innovation_number += 1
            self.existing_nodes[self.node_innovation_number] = new_gene
            return self.node_innovation_number

if __name__ == "__main__":

    historical_marker = HistoricalMarker()

    mutation_rate = 0.8
    probabilty_perturbating = 0.9
    standard_deviation_weight_perturbation = 0.08

    add_node_mutation_chance = 0.1
    add_connection_mutation_chance = 0.1

    compatibility_distance_threshold = 10

    c1 = 1
    c2 = 1
    c3 = 0.4

# TODO : - could merge the function calculating the average weight of matching genes, the excess genes and the disjoint genes
#        - change the historical marker to do a counter
#        - create the first generation (generate the first simple neuron and create child from this neuron) 