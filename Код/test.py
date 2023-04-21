import neat

# Define the problem
def eval_genome(genome, config):
    # Create a neural network from the genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Evaluate the neural network's performance
    fitness = 0.0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        fitness -= (output[0] - xo[0]) ** 2

    # Set the genome's fitness score
    genome.fitness = fitness

# Create a NEAT configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

# Create a population
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Run the NEAT algorithm for 50 generations
winner = p.run(eval_genomes, 50)

# Create a neural network from the winning genome
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

# Test the neural network
print(winner_net.activate([0, 0])) # Output should be close to 0
print(winner_net.activate([0, 1])) # Output should be close to 1
print(winner_net.activate([1, 0])) # Output should be close to 1
print(winner_net.activate([1, 1])) # Output should be close to 0
