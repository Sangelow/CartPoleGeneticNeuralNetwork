import gym
import gym.spaces
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import random
import os
import pickle
import matplotlib.pyplot as plt
import uuid

plt.style.use('seaborn-darkgrid')
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.rc('font', size=12)
plt.rc('legend', fontsize=11, frameon=True, fancybox=True,framealpha=0.5,facecolor="w")

class Generation:

    def __init__(self,env,size,parent_number,mutation_rate,parents=None):
        self.env = env
        self.size = size
        self.parent_number = parent_number
        self.mutation_rate = mutation_rate
        self.parents = parents

        self.population = []
        self.generate_population()
        self.find_best_individuals()
        # self.best_individual.replay()

    def generate_population(self):
        if self.parents != None:
            self.population += self.parents
        for i in range(self.size-len(self.population)):
            # Choose parents only if they exists
            if self.parents != None:
                individual_1 = random.choice(self.parents)
                individual_2 = random.choice(self.parents)
                while mom.id == dad.id:
                    individual_2 = random.choice(self.parents)
            self.population.append(Individual(self.env,self.mutation_rate,self.parents))

    def find_best_individuals(self):
        self.population.sort(key=lambda indiv : indiv.fitness, reverse=True)
        self.best_individuals = self.population[:self.parent_number]

    def crossover(self):
        pass


class Individual:

    def __init__(self,env,mutation_rate,parents=None):
        self.env = env
        self.id = uuid.uuid4()
        self.mutation_rate = mutation_rate
        self.parents = parents
        self.state_size = self.env.observation_space.shape[0] 
        self.hidden_layer_size = 24
        self.action_size = self.env.action_space.n
        self.fitness = 0

        self.create_weights()
        self.build_model()
        # self.find_layers_shape()
        self.calculate_fitness()

    def create_weights(self):
        if self.parents == None:
            self.random_weights()
        else:
            self.crossover_parents()
            self.mutate()

    def random_weights(self):
        weights_input = np.random.randn(self.state_size, self.hidden_layer_size).astype(np.float32) * np.sqrt(2.0/(self.state_size))
        biases_input = np.zeros([self.hidden_layer_size]).astype(np.float32)
        input_layer = [weights_input, biases_input] 

        weights_hidden = np.random.randn(self.hidden_layer_size, self.hidden_layer_size).astype(np.float32) * np.sqrt(2.0/(self.hidden_layer_size))
        biases_hidden = np.zeros([self.hidden_layer_size]).astype(np.float32)
        hidden_layer = [weights_hidden, biases_hidden]

        weights_output = np.random.randn(self.hidden_layer_size, self.action_size).astype(np.float32) * np.sqrt(2.0/(self.hidden_layer_size))
        biases_output = np.zeros([self.action_size]).astype(np.float32)
        output_layer = [weights_output, biases_output]

        self.weights = [input_layer,hidden_layer,output_layer]

    def crossover_parents():
        for i in range(len(self.parents[0].weights)):
            weights_choice = np.random.choice([True, False], size=self.parents[0].weights[i][0].shape)
            biases_choice = np.random.choice([True, False], size=self.parents[0].weights[i][1].shape)

            layer_weights = np.where(weights_choice, self.parents[0].weights[i][0], self.parents[1].weights[i][0])
            layer_biases = np.where(biases_choice, self.parents[0].weights[i][1], self.parents[1].weights[i][1])

            self.weights.append([layer_weights,layer_biases])

    def mutate(self):
        for i in range(len(self.weights)):
            weight_layer = self.weights[i][0]
            biaise_layer = self.weights[i][1]
            weight = np.add(weight_layer, np.random.standard_normal(weight_layer.shape) * self.mutation_rate)
            biase = np.add(biaise_layer, np.random.standard_normal(biaise_layer.shape) * self.mutation_rate)
            self.weights.append([weight,biase])

    def build_model(self):
        self.model = Sequential()

        # Input layer (it has the size of the state space)
        self.model.add(Dense(self.hidden_layer_size,
            name="Input",
            input_dim=self.state_size,
            activation='relu',
            use_bias=True,
            weights=self.weights[0]))

        # Hidden layer
        self.model.add(Dense(self.hidden_layer_size,
            name="Hidden",
            activation='relu',
            use_bias=True,
            weights=self.weights[1]))

        # Creation of the output layer (it has the size of the action set)
        self.model.add(Dense(self.action_size,
            name="Output",
            activation='linear',
            use_bias=True,
            weights=self.weights[2]))

    def act(self, state):
        """Chose an action given the current state."""
        # Predict (model chosen action)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def find_layers_shape(self):
        print("=== {:^50} ===".format("Shapes of weights"))
        print("The weights of a layer is a list containing the weights and the biases.")
        for layer in self.model.layers:
            weights_shape = layer.get_weights()[0].shape
            biases_shape  =layer.get_weights()[1].shape
            print("Layer : {}; Shape of weights : {}; Shape of biases : {}.".format(layer.name, weights_shape, biases_shape))

    def calculate_fitness(self):
        self.fitness = 0
        for _ in range(5):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            for time_t in range(100000):
                # self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.fitness += reward
                next_state = np.reshape(next_state, [1, self.state_size])
                state = next_state
                if done:
                    break
        self.fitness /= 5

    def replay(self):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        for time_t in range(100000):
            self.env.render()
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            next_state = np.reshape(next_state, [1, self.state_size])
            state = next_state
            if done:
                break

    def save(self):
        file_name = "saved_models/model_{}".format(len(os.listdir("saved_models/"))+1)
        with open(file_name,"wb") as f:
            pickle.dump(self.weights, f, pickle.HIGHEST_PROTOCOL)

    def load(self,file_name):
        with open("saved_models/"+file_name, "rb") as f:
            loaded_weights = pickle.load(f)
            # Now you can use the dump object as the original one  
            self.weights= loaded_weights

if __name__ == "__main__":
    # initialize gym environment and the agent
    env_name = 'CartPole-v1' 
    env = gym.make(env_name)

    generation_size = 20
    mutation_rate = 5/100
    parent_number = int(0.2 * generation_size)

    generation_number = 1

    print("""{:^20}""".format("Generation : 1"))
    generation = Generation(env,generation_size,parent_number,mutation_rate,None)
    best_individuals = [generation.best_individual]
    print("""Best individual fitness : {}""".format(best_individuals[-1].fitness))

    while best_individuals[-1].fitness < 500:
        generation_number += 1
        print("""{:^20}""".format("Generation : {}".format(generation_number)))
        generation = Generation(env,generation_size,parent_number,mutation_rate,best_individuals[-1])
        best_individuals.append(generation.best_individual)
        print("""Best individual fitness : {}""".format(best_individuals[-1].fitness))

    best_individuals[-1].save()
    best_individuals[-1].replay()
    env.close()

# TODO :    - try to select a number of best individual and then to do crossover with them (for each weight and biaises choose randomly between mom and dad)
#           - create a random array with the size of the weights, if element <= 0 ==> choose mom element, else choose dad element
#           - find best individual using a distribution that is more likely to choose an inidividual with a high fitness
