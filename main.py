import pyximport; pyximport.install()

from torch import multiprocessing as mp

from Coach import Coach
from NNetWrapper import NNetWrapper as nn
#from connect4.Connect4Game import Connect4Game as Game
from othello.OthelloGame import OthelloGame as Game
from utils import *

args = dotdict({
    'run_name': 'othello_16x16_4c',#connect4_hardcore
    'workers': mp.cpu_count() - 1,
    'startIter': 1,
    'numIters': 100, #1000
    'process_batch_size': 64,#128
    'train_batch_size': 512,
    'train_steps_per_iteration': 500,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 4*64*(mp.cpu_count()-1), #4*128*(mp.cpu_count()-1)
    'numItersForTrainExamplesHistory': 200,
    'symmetricSamples': False,
    'numMCTSSims': 200,#50
    'numFastSims': 5,
    'probFastSim': 0.75,
    'tempThreshold': 10,
    'temp': 1,
    'compareWithRandom': False,
    'arenaCompareRandom': 500,
    'arenaCompare': 200,#500
    'arenaTemp': 0.1,
    'arenaMCTS': True, #False
    'randomCompareFreq': 1,
    'compareWithPast': True,
    'pastCompareFreq': 5,
    'expertValueWeight': dotdict({
        'start': 0,
        'end': 0,
        'iterations': 35
    }),
    'cpuct': 1.1,#3
    'checkpoint': 'checkpoint',
    'data': 'data',
})

if __name__ == "__main__":
    g = Game(16) #16
    nnet = nn(g)
    c = Coach(g, nnet, args)
    c.learn()
