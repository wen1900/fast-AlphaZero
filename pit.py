import pyximport; pyximport.install()

import Arena
from MCTS import MCTS
from GenericPlayers import *
#from connect4.Connect4Game import Connect4Game as Game, display
#from connect4.Connect4Players import *
from othello.OthelloGame import OthelloGame as Game, display
from othello.OthelloPlayers import *
from NNetWrapper import NNetWrapper as NNet
import numpy as np
from utils import *


"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
if __name__ == '__main__':

    g = Game(16)

    # all players
    rp = RandomPlayer(g).play
    #gp = OneStepLookaheadConnect4Player(g).play
    hp = HumanOthelloPlayer(g).play

    # nnet players
    n1 = NNet(g)
    n1.load_checkpoint('./elo_estimate/', 'iteration-0200-mcts100.pkl')#('./checkpoint/', 'iteration-0200.pkl')
    args1 = dotdict({'numMCTSSims': 1600, 'cpuct': 1.0}) #'numMCTSSims': 50, 'cpuct': 1.0
    mcts1 = MCTS(g, n1, args1)

    def n1p(x, turn):
        if turn <= 2:
            mcts1.reset()
        temp = 1 if turn <= 10 else 0
        policy = mcts1.getActionProb(x, temp=temp)
        if sum(policy) > 1:
            print('Multiple optins ->', policy)
        return np.random.choice(len(policy), p=policy)

############################################################################################################
    n2 = NNet(g)
    n2.load_checkpoint('./elo_estimate/', 'best-iteration-self-1920-mcts200.pkl')#('./checkpoint/', 'iteration-0100.pkl')
    args2 = dotdict({'numMCTSSims': 1600, 'cpuct': 1.0}) #'numMCTSSims': 50, 'cpuct': 1.0
    mcts2 = MCTS(g, n2, args2)

    def n2p(x, turn):
        if turn <= 2:
            mcts2.reset()
        temp = 1 if turn <= 10 else 0
        policy = mcts2.getActionProb(x, temp=temp)
        return np.random.choice(len(policy), p=policy)

    arena = Arena.Arena(n1p, n2p, g, display=display)

############################################################################################################
    #arena = Arena.Arena(n1p, hp, g, display=display)
    print(arena.playGames(20, verbose=True))
