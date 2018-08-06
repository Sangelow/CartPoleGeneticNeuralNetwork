import random
import copy
from graphviz import Digraph
import uuid

def compatibility_distance(genome_1,genome_2,c1,c2,c3):
    E = count_excess_genes(genome_1,genome_2)
    D = count_disjoint_genes(genome_1,genome_2)
    N = count_larger_gene_number(genome_1,genome_2)
    W = calculate_average_weight_matching_genes(genome_1,genome_2)
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

    def add_connection_gene_mutation(self):

        connection_valid = False

        while not(connection_valid):
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
        new_connection = ConnectionGene(node_1.innovation_number,node_2.innovation_number,weight, True)
        # Add the connection to the list
        self.add_connection_gene(new_connection)

    def add_node_gene_mutation(self):
        # Choose a random connection
        old_connection = random.choice(list(self.connection_genes.values()))
        # Deactivate the old connection
        old_connection.disable()
        # Create the new node
        new_node = NodeGene("HIDDEN")
        self.add_node_gene(new_node)
        # Create the two new connections

        new_connection_1 = ConnectionGene(old_connection.in_node, new_node.innovation_number, 1, True)
        new_connection_2 = ConnectionGene(new_node.innovation_number,old_connection.out_node, old_connection.weight, True)
        # Add the two connection
        self.add_connection_gene(new_connection_1)
        self.add_connection_gene(new_connection_2)

    def weight_mutation(self):
        global weight_mutation_chance
        if random.random() < weight_mutation_chance:
            for connection in list(self.connection_genes.values()):
                connection.mutate_weight()

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
    



class ConnectionGene():

    def __init__(self,in_node=int(),out_node = int(), weight = float(), expressed = True):
        # CHECK IF THE CONNECTION ALREADY EXISTS IN HistoricalMarker <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        global historical_marker

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

    def mutate_weight(self):
        global probabilty_perturbating
        global standard_deviation_weight_perturbation
        if random.random() < probabilty_perturbating:
            self.weight +=  random.gauss(0,standard_deviation_weight_perturbation)
        else:
            self.weight = random.uniform(-1,1)


class NodeGene():

    def __init__(self, type=str()):
        global historical_marker
        self.type = type # The type should be either "INPUT"/"HIDDEN"/"OUTPUT"
        self.innovation_number = historical_marker.get_innovation_number(self)

    def copy(self):
        return copy.deepcopy(self)

class HistoricalMarker():

    def __init__(self):
        self.existing_connections = {}
        self.existing_nodes = {}
        self.innovation_number = 0

    def get_innovation_number(self,new_gene):
        if new_gene.__class__.__name__ == "ConnectionGene":
            new_connection = new_gene
            for connection in self.existing_connections.values():
                if connection.in_node==new_connection.in_node and connection.out_node==new_connection.out_node:
                    return connection.innovation_number
            self.innovation_number += 1
            self.existing_connections[self.innovation_number] = new_connection
            return self.innovation_number
        elif new_gene.__class__.__name__ == "NodeGene":
            self.innovation_number += 1
            self.existing_nodes[self.innovation_number] = new_gene
            return self.innovation_number

if __name__ == "__main__":

    historical_marker = HistoricalMarker()

    weight_mutation_chance = 0.8
    probabilty_perturbating = 0.9
    standard_deviation_weight_perturbation = 0.08

    genome_1 = Genome({},{})
    for i in range(1,4):
        genome_1.add_node_gene(NodeGene("INPUT"))
    genome_1.add_node_gene(NodeGene("OUTPUT"))
    genome_1.add_node_gene(NodeGene("HIDDEN"))

    genome_2 = genome_1.copy()
    genome_2.add_node_gene(NodeGene("HIDDEN"))

    genome_1.add_connection_gene(ConnectionGene(1,4,1,True))
    genome_1.add_connection_gene(ConnectionGene(2,4,1,False))
    genome_1.add_connection_gene(ConnectionGene(3,4,1,True))
    genome_1.add_connection_gene(ConnectionGene(2,5,1,True))
    genome_1.add_connection_gene(ConnectionGene(5,4,1,True))

    

    genome_2.add_connection_gene(ConnectionGene(1,4,1,True))
    genome_2.add_connection_gene(ConnectionGene(2,4,1,False))
    genome_2.add_connection_gene(ConnectionGene(3,4,1,True))
    genome_2.add_connection_gene(ConnectionGene(2,5,1,True))
    genome_2.add_connection_gene(ConnectionGene(5,4,1,False))
    genome_2.add_connection_gene(ConnectionGene(5,6,1,True))
    genome_2.add_connection_gene(ConnectionGene(6,4,1,True))

    genome_1.add_connection_gene(ConnectionGene(1,5,1,True))

    genome_2.add_connection_gene(ConnectionGene(3,5,1,True))
    genome_2.add_connection_gene(ConnectionGene(1,6,1,True))

    # genome_1.print_genome()
    # genome_2.print_genome()

    print("Number of matching gennes : {}.".format(count_matching_genes(genome_1,genome_2)))
    print("Number of disjoint genes : {}.".format(count_disjoint_genes(genome_1,genome_2)))
    print("Number of disjoint genes : {}.".format(count_excess_genes(genome_1,genome_2)))
    print(compatibility_distance(genome_1,genome_2,1,1,1))
    # genome_3 = Genome.crossover(genome_2,genome_1)

    # genome_1.add_node_gene_mutatio()
    # genome_1.print_genome()

    # genome_1.add_connection_gene_mutation()
    # genome_1.print_genome()

    # genome_1.weight_mutation()
    # genome_1.print_genome()
