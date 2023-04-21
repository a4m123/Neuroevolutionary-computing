# pyright: reportMissingImports=false
import neat
import numpy as np
#from tensorflow.keras.datasets import mnist
import keras

#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#train_images = train_images.reshape((len(train_images), 28*28)) / 255
#test_images = test_images.reshape((len(test_images), 28*28)) / 255
#train_labels = keras.utils.to_categorical(train_labels, 10)
#test_labels = keras.utils.to_categorical(test_labels, 10)

# load data from cancer2.dt file
train_images = np.loadtxt('A:/Профиль/Rab Table/Учёба/2/нейроэволюционные/Методические/data_to_use/cancer1.dt', delimiter=',', usecols=range(0, 9))

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0.0
        for xi, xo in zip(train_images, train_labels):
            output = net.activate(xi)
            fitness -= (output[xo] - 1) ** 2
        genome.fitness = fitness

def main():
    config_path = 'A:/Профиль/Rab Table/Учёба/2/Neuroevolutionary-computing/Код/config-mnist'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    population = neat.Population(config)
    winner = population.run(eval_genomes, n = 3)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    correct = 0
    for xi, xo in zip(test_images, test_labels):
        output = winner_net.activate(xi)
        if np.argmax(output) == xo:
            correct += 1
    print('Accuracy:', correct / len(test_labels))

if __name__ == "__main__":
    main()