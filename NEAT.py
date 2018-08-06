import random
import copy
from graphviz import Digraph
import uuid

class Genome():

    def __init__(self, connection_genes, node_genes):
        self.connection_genes = connection_genes
        self.node_genes = node_genes

    def __repr__(self):
        return "Number of node genes : {}. Number of connection genes : {}.".format(len(self.node_genes),len(self.connection_genes))

    def copy(self):
        return copy.deepcopy(self)

    def add_node_gene(self,node):
        self.node_genes[node.id] = node

    def add_connection_gene(self,connection):
        self.connection_genes[connection.innovation_number] = connection

    def add_connection_gene_mutation(self):

        global innovation_number_generator

        connection_valid = False

        while not(connection_valid):
            # Set connection_valid to True, this value is changed later if the connection is not valiid
            connection_valid = True
            # Randomly select two nodes
            node_1 = random.choice(list(self.node_genes.values()))
            node_2 = random.choice(list(self.node_genes.values()))
            # Check if the connection is between a node and himself
            if node_1.id == node_2.id:
                connection_valid = False
            # Check if the nodes are both INPUT or both OUTPUT
            if (node_1.type=="INPUT" and node_2.type=="INPUT") or (node_1.type=="OUTPUT" and node_2.type=="OUTPUT"):
                connection_valid = False
            # Check if the connection already exists
            for connection in self.connection_genes.values():
                if (connection.in_node==node_1.id and connection.out_node==node_2.id) or (connection.in_node==node_2.id and connection.out_node==node_1.id):
                    connection_valid = False

        # Find the correct direction of the connection
        if (node_1.type=="HIDDEN" and node_2.type =="INPUT") or (node_1.type=="OUTPUT" and node_2.type=="HIDDEN") or (node_1.type=="OUTPUT" and node_2.type =="INPUT"):
            node_1, node_2 = node_2, node_1

        # Generate a random weight
        weight = 2*random.random()-1
        # Create the new connection
        new_connection = ConnectionGene(node_1.id,node_2.id,weight, True,innovation_number_generator.get_innovation_number())
        # Add the connection to the list
        self.add_connection_gene(new_connection)

    def add_node_gene_mutation(self):
        global innovation_number_generator
        # Choose a random connection
        old_connection = random.choice(list(self.connection_genes.values()))
        # Deactivate the old connection
        old_connection.disable()
        # Create the new node
        new_node = NodeGene("HIDDEN",len(self.node_genes)+1) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ID MIGHT NOT BE RIGHT
        self.add_node_gene(new_node)
        # Create the two new connections

        new_connection_1 = ConnectionGene(old_connection.in_node, new_node.id, 1, True,innovation_number_generator.get_innovation_number())
        new_connection_2 = ConnectionGene(new_node.id,old_connection.out_node, old_connection.weight, True, innovation_number_generator.get_innovation_number())
        # Add the two connection
        self.add_connection_gene(new_connection_1)
        self.add_connection_gene(new_connection_2)

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
                        cluster.node(str(node.id))
                cluster.attr(label=layer_type)

        with graph.subgraph(name='cluster_hidden') as cluster_hidden:
            # cluster_hidden.attr(style='filled')
            cluster_hidden.attr(color='grey')
            for node in self.node_genes.values():
                if node.type=="HIDDEN":
                    cluster_hidden.node(str(node.id))
            cluster_hidden.attr(label='HIDDEN')

        for connection in self.connection_genes.values():
            if connection.expressed:
                graph.edge(str(connection.in_node),str(connection.out_node), label=str(connection.innovation_number))

        graph.view()
    



class ConnectionGene():

    def __init__(self,in_node=int(),out_node = int(), weight = float(), expressed = True, innovation_number=int()):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.expressed = expressed
        self.innovation_number = innovation_number
    
    def enable(self):
        self.expressed = True

    def disable(self):
        self.expressed = False

    def copy(self):
        return copy.deepcopy(self)

class NodeGene():

    def __init__(self, type=str(), id=int()):
        self.type = type # The type should be either "INPUT"/"HIDDEN"/"OUTPUT"
        self.id = id

    def copy(self):
        return copy.deepcopy(self)

class InnovationNumberGenerator():

    def __init__(self):
        self.innovation_number = 10

    def get_innovation_number(self):
        self.innovation_number += 1
        return self.innovation_number

if __name__ == "__main__":

    innovation_number_generator = InnovationNumberGenerator()

    genome_1 = Genome({},{})
    for i in range(1,4):
        genome_1.add_node_gene(NodeGene("INPUT",i))
    genome_1.add_node_gene(NodeGene("OUTPUT",4))
    genome_1.add_node_gene(NodeGene("HIDDEN",5))

    genome_1.add_connection_gene(ConnectionGene(1,4,1,True,1))
    genome_1.add_connection_gene(ConnectionGene(2,4,1,False,2))
    genome_1.add_connection_gene(ConnectionGene(3,4,1,True,3))
    genome_1.add_connection_gene(ConnectionGene(2,5,1,True,4))
    genome_1.add_connection_gene(ConnectionGene(5,4,1,True,5))
    genome_1.add_connection_gene(ConnectionGene(1,5,1,True,8))

    print(genome_1)

    genome_2 = Genome({},{})
    for i in range(1,4):
        genome_2.add_node_gene(NodeGene("INPUT",i))
    genome_2.add_node_gene(NodeGene("OUTPUT",4))
    genome_2.add_node_gene(NodeGene("HIDDEN",5))
    genome_2.add_node_gene(NodeGene("HIDDEN",6))

    genome_2.add_connection_gene(ConnectionGene(1,4,1,True,1))
    genome_2.add_connection_gene(ConnectionGene(2,4,1,False,2))
    genome_2.add_connection_gene(ConnectionGene(3,4,1,True,3))
    genome_2.add_connection_gene(ConnectionGene(2,5,1,True,4))
    genome_2.add_connection_gene(ConnectionGene(5,4,1,False,5))
    genome_2.add_connection_gene(ConnectionGene(5,6,1,True,6))
    genome_2.add_connection_gene(ConnectionGene(6,4,1,True,7))
    genome_2.add_connection_gene(ConnectionGene(3,5,1,True,9))
    genome_2.add_connection_gene(ConnectionGene(1,6,1,True,10))

    print(genome_2)

    genome_3 = Genome.crossover(genome_2,genome_1)

    genome_1.print_genome()
    # genome_2.print_genome()
    # genome_3.print_genome()

    #genome_1.add_node_gene_mutation()
    #genome_1.print_genome()

    genome_1.add_connection_gene_mutation()
    genome_1.print_genome()
