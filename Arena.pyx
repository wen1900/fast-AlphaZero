# cython: language_level=3

from pytorch_classification.utils import Bar, AverageMeter
import time


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1 #1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer +
                             1](self.game.getCanonicalForm(board, curPlayer), it)

            valids = self.game.getValidMoves(
                self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                print()
                print(action)
                print(valids)
                print()
                assert valids[action] > 0
            if verbose:
                print(int(action / self.game.n), int(action % self.game.n))
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert(self.display)
            black, white, winstone = self.game.getStone(board, 1)
            print("Game over: Turn ", str(it), "Result ",
                  str(self.game.getGameEnded(board, 1)))
            print("Black: ", black, "White: ", white, "WinStone: ", winstone)
            self.display(board)
        return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Winrate: {wr}%% | Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps, maxeps=maxeps, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td,
                wr=int(100*(oneWon+0.5*draws)/(oneWon+twoWon+draws)))
            bar.next()

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Winrate: {wr}%% | Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps, maxeps=maxeps, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td,
                wr=int(100*(oneWon+0.5*draws)/(oneWon+twoWon+draws)))
            bar.next()

        bar.update()
        bar.finish()

        return oneWon, twoWon, draws
    def playGames_elo(self, num, elo_A, elo_B, verbose=False): #計算elo，第一次對局(同時更新 新舊模型)
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for i in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
                score_A = 1
            elif gameResult == -1:
                twoWon += 1
                score_A = 0
            else:
                draws += 1
                score_A = 0.5
            # bookkeeping + plot progress

            expected_A = 1 / (1 + 10 ** ((elo_B - elo_A) / 400))
            expected_B = 1 / (1 + 10 ** ((elo_A - elo_B) / 400))
            K = 32
            elo_A = elo_A + K * (score_A - expected_A)
            elo_B = elo_B + K * ((1 - score_A) - expected_B)

            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Winrate: {wr}%% | Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps, maxeps=maxeps, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td,
                wr=int(100*(oneWon+0.5*draws)/(oneWon+twoWon+draws)))
            bar.next()

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
                score_A = 1
            elif gameResult == 1:
                twoWon += 1
                score_A = 0
            else:
                draws += 1
                score_A = 0.5
            # bookkeeping + plot progress

            expected_A = 1 / (1 + 10 ** ((elo_B - elo_A) / 400))
            expected_B = 1 / (1 + 10 ** ((elo_A - elo_B) / 400))
            K = 32
            elo_A = elo_A + K * (score_A - expected_A)
            elo_B = elo_B + K * ((1 - score_A) - expected_B)

            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Winrate: {wr}%% | Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps, maxeps=maxeps, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td,
                wr=int(100*(oneWon+0.5*draws)/(oneWon+twoWon+draws)))
            bar.next()

        elo = elo_A
        prev_elo = elo_B
        bar.update()
        bar.finish()

        return oneWon, twoWon, draws, elo, prev_elo

