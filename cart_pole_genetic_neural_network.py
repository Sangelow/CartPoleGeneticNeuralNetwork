import gym
import gym.spaces
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import random
import os
import pickle
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.rc('font', size=12)
plt.rc('legend', fontsize=11, frameon=True, fancybox=True,framealpha=0.5,facecolor="w")

class Generation:

    def __init__(self,env,size,mutation_rate,parent=None):
        self.size = size
        self.env = env
        self.mutation_rate = mutation_rate
        self.parent = parent
        self.population = []
        self.generate_population()
        self.find_best_individual()
        # self.best_individual.replay()

    def generate_population(self):
        if self.parent != None:
            self.population.append(self.parent)
        for i in range(self.size-len(self.population)):
            self.population.append(Individual(self.env,self.mutation_rate,self.parent))

    def find_best_individual(self):
        self.best_individual = None
        self.best_individual_fitness = 0
        for indiviudal in self.population:
            if indiviudal.fitness > self.best_individual_fitness:
                self.best_individual = indiviudal
                self.best_individual_fitness = indiviudal.fitness


class Individual:

    def __init__(self,env,mutation_rate,parent=None):
        self.env = env
        self.mutation_rate = mutation_rate
        self.parent = parent
        self.state_size = self.env.observation_space.shape[0] 
        self.hidden_layer_size = 24
        self.action_size = self.env.action_space.n
        self.fitness = 0

        self.create_weights()
        self.build_model()
        # self.find_layers_shape()
        self.calculate_fitness()

    def create_weights(self):
        if self.parent == None:
            self.random_weights()
        else:
            self.mutate_parent()

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

    def mutate_parent(self):
        self.weights = []
        for i in range(len(self.parent.weights)):
            parent_layer_weight = self.parent.weights[i][0]
            parent_layer_biaise = self.parent.weights[i][1]
            weight = np.add(parent_layer_weight, np.random.standard_normal(parent_layer_weight.shape) * self.mutation_rate)
            biase = np.add(parent_layer_biaise, np.random.standard_normal(parent_layer_biaise.shape) * self.mutation_rate)
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
        file_name = "saved_models/model_{}".format(len(os.listdir("saved_models/")))
        with open(file_name,"wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # initialize gym environment and the agent
    env_name = 'CartPole-v1' 
    env = gym.make(env_name)

    generation_size = 20
    mutation_rate = 5/100

    print("""{:^20}""".format("Generation : 1"))
    generations = [Generation(env,generation_size,mutation_rate,None)]
    best_individuals = [generations[-1].best_individual]
    print("""Best individual fitness : {}""".format(best_individuals[-1].fitness))

    while best_individuals[-1].fitness < 500:
        print("""{:^20}""".format("Generation : {}".format(len(generations)+1)))
        generations.append(Generation(env,generation_size,mutation_rate,best_individuals[-1]))
        best_individuals.append(generations[-1].best_individual)
        print("""Best individual fitness : {}""".format(best_individuals[-1].fitness))

    best_individuals[-1].replay()
    env.close()

# TODO :    - change the mutation/breeding method (try to do the crossover + mutation)

