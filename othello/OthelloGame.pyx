# cython: language_level=3

from __future__ import print_function
import numpy as np
from .OthelloLogic import Board
from Game import Game
import sys
sys.path.append('..')


class OthelloGame(Game):
    def __init__(self, n):
        self.n = n 

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.asarray(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.n), action % self.n)
        b.execute_move(move, player)
        return (np.asarray(b.pieces), -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n*x+y] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.countDiff(player) > 0:
            return 1
        return -1

    def getStone(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        Black, White = b.countStone(player)
        winstone = b.countDiff(player)
        return Black, White , winstone

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)


def display(board):
    n = board.shape[0]

    for y in range(n):
        if y == 0:
            print(f"c\\r|{y:02d} |",end="")# output: {c\r|00 |}
        elif y == n-1:
            print(f"{y:02d}  |",end="")
        else:
            print(f"{y:02d} |",end="")#print(y, "|", end="")
    print("")
    print(" -----------------------")
    for y in range(n):
        print(f"{y:02d}", "|", end="")    # print the row #
        for x in range(n):
            piece = board[y][x]    # get the piece to print
            if piece == 1:
                print("\u25CB   ", end="")#b ○
            elif piece == -1:
                print("\u25CF   ", end="")#W ●
            else:
                if x == n:
                    print("-", end="")
                else:
                    print("-   ", end="")
        print("|")
        print("")

    #print("   -----------------------")
