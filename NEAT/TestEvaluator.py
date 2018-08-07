from NEAT import Evaluator
import gym
import gym.spaces

class TestEvaluator(Evaluator):

    def __init__(self,env,population_size, input_size, output_size, mutation_rate, perturbating_probability, add_connection_mutation_chance, add_node_mutation_chance, compatibility_distance_threshold, c1, c2, c3):
        super().__init__(population_size, input_size, output_size, mutation_rate, perturbating_probability, add_connection_mutation_chance, add_node_mutation_chance, compatibility_distance_threshold, c1, c2, c3)
        self.env = env

    def evaluate_genome(self,genome):
        state = self.env.reset()
        fitness = 0
        for t in range(600):
            print("Test")
            action = genome.act(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            fitness += reward
            if done:
                break
        return fitness

if __name__ == "__main__":

    env_name = 'CartPole-v1' 
    env = gym.make(env_name)

    generation_size = 15
    input_size = 4
    output_size = 1
    
    mutation_rate = 0.8
    perturbating_probability = 0.9
    add_connection_mutation_chance = 0.1
    add_node_mutation_chance = 0.1

    compatibility_distance_threshold = 3
    c1, c2, c3 = 1, 1, 0.4 

    evaluator = TestEvaluator(env,generation_size, input_size, output_size, mutation_rate, perturbating_probability, add_connection_mutation_chance, add_node_mutation_chance, compatibility_distance_threshold, c1, c2, c3)
    for i  in range(10):
        evaluator.evaluate()