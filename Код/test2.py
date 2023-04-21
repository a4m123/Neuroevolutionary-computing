# NEAT algorithm to find the most suitable neural network to solve MNIST dataset

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import pickle
import gzip
import os

from neat import nn, population, statistics, parallel

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

# Define the activation function
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-4.9 * x))

# Define the fitness function
def eval_fitness(genomes):
    for g in genomes:
        net = nn.create_feed_forward_phenotype(g)
        error = 0.0
        for xi, xo in zip(train_set[0], train_set[1]):
            output = net.serial_activate(xi)
            error += (output[0] - xo) ** 2
        g.fitness = -error

# Create the population, which is the top-level object for a NEAT run.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
pop = population.Population(config_path)

# Add a stdout reporter to show progress in the terminal.
pop.add_reporter(statistics.StatisticsReporter())
pop.add_reporter(statistics.StdOutReporter(True))
pop.add_reporter(statistics.Checkpointer(5))

# Run for up to 300 generations.
winner = pop.run(eval_fitness, 300)

# Display the winning genome.
print('Number of evaluations: {0}'.format(pop.total_evaluations))

# Show output of the most fit genome against training data.
print('Output:')
winner_net = nn.create_feed_forward_phenotype(winner)
for xi, xo in zip(train_set[0], train_set[1]):
    output = winner_net.serial_activate(xi)
    print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

# Save the winner.
with open('winner-feedforward', 'wb') as f:
    pickle.dump(winner, f)





