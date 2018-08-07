from NEAT import Evaluator

class TestEvaluator(Evaluator):

    def evaluate_genome(self,genome):
        pass


if __name__ == "__main__":

    generation_size = 100
    input_size = 2
    output_size = 2
    
    mutation_rate = 0.8
    perturbating_probability = 0.9
    add_connection_mutation_chance = 0.1
    add_node_mutation_chance = 0.1

    compatibility_distance_threshold = 3
    c1, c2, c3 = 1, 1, 0.4 


    evaluator = TestEvaluator(generation_size, input_size, output_size, mutation_rate, perturbating_probability, add_connection_mutation_chance, add_node_mutation_chance, compatibility_distance_threshold, c1, c2, c3)
    for i  in range(10):
        evaluator.evaluate()