import pyximport; pyximport.install()

from pathlib import Path
import pprint
from glob import glob
from utils import *
from NNetWrapper import NNetWrapper as nn
#from connect4.Connect4Game import Connect4Game as Game
from othello.OthelloGame import OthelloGame as Game
from GenericPlayers import *
from MCTS import MCTS
from Arena import Arena
import numpy as np
import choix
import matplotlib.pyplot as plt
import csv
import os

args = dotdict({
    'arenaCompare': 10,
    'arenaTemp': 0,
    'temp': 1,
    'tempThreshold': 10,
    # use zero if no montecarlo
    'numMCTSSims': 1000, #50
    'cpuct': 1.1,#4
    'playRandom': False,
})

if __name__ == '__main__':
    print('Args:')
    pprint.pprint(args)
    if not Path('elo_estimate').exists():
        Path('elo_estimate').mkdir()
    print('Beginning elo estimate')
    networks = sorted(glob('elo_estimate/*'),reverse=False) #將資料夾中的模型按正序排列(數字由小到大)
    model_count = len(networks) + int(args.playRandom)

    if model_count <= 2:
        print(
            "Too few models for elo estimate. Please add models to the elo_estimate/ directory")
        exit()

    total_games = model_count - 1
    total_games *= args.arenaCompare
    print(
        f'Estimate {model_count} different models elo grade with total games {total_games}')
    #win_matrix = np.zeros((model_count, model_count))

    g = Game(16)
    nnet1 = nn(g)
    nnet2 = nn(g)

    file2 = Path(networks[0])
    base_model_names = []
    model_names = []
    pit_winrate = []   
    pit_games = []


    for i in range(model_count - 1):
        file1 = Path(networks[i+1])  
        print(f'{file1.stem} vs {file2.stem}')
        model_names.append(file1.stem)  # 記錄模型名稱
        nnet1.load_checkpoint(folder='elo_estimate', filename=file1.name)
        if args.numMCTSSims <= 0:
            p1 = NNPlayer(g, nnet1, args.arenaTemp,
                            args.tempThreshold).play
        else:
            mcts1 = MCTS(g, nnet1, args)

            def p1(x, turn):
                if turn <= 2:
                    mcts1.reset()
                temp = args.temp if turn <= args.tempThreshold else args.arenaTemp
                policy = mcts1.getActionProb(x, temp=temp)
                return np.random.choice(len(policy), p=policy)
        if file2.name != 'random':
            nnet2.load_checkpoint(folder='elo_estimate', filename=file2.name)
            if args.numMCTSSims <= 0:
                p2 = NNPlayer(g, nnet1, args.arenaTemp,
                                args.tempThreshold).play
            else:
                mcts2 = MCTS(g, nnet2, args)

                def p2(x, turn):
                    if turn <= 2:
                        mcts2.reset()
                    temp = args.temp if turn <= args.tempThreshold else args.arenaTemp
                    policy = mcts2.getActionProb(x, temp=temp)
                    return np.random.choice(len(policy), p=policy)
        else:
            p2 = RandomPlayer(g).play
        arena = Arena(p1, p2, g)
        p1wins, p2wins, draws = arena.playGames(args.arenaCompare)

        win_rate = float(p1wins + 0.5 * draws) / args.arenaCompare
        total = args.arenaCompare

        if win_rate <= 0.6 and win_rate >= 0.4:
            p1wins, p2wins, draws = arena.playGames(30)
            win_rate = float(p1wins + win_rate*args.arenaCompare) / args.arenaCompare + 30
            total += 30
            
        base_model_names.append(file2.stem)
        pit_winrate.append(win_rate)
        pit_games.append(total)
        
        

        #win_matrix[i, j] = p1wins + 0.5*draws
        #win_matrix[j, i] = p2wins + 0.5*draws
        print(f'wins: {p1wins}, ties: {draws}, losses:{p2wins}\n')


    #print("\nWin Matrix(row beat column):")
    #print(win_matrix)

    csv_filename = 'pit_winrate.csv'
    file_exists = os.path.exists(csv_filename)

    with open(csv_filename, mode='a', newline='') as file:  # 'a' 模式是追加
        writer = csv.writer(file)
        
        # 如果檔案不存在，寫入標題
        if not file_exists:
            writer.writerow(["Model", "Model", "Winrate", "total_games"])  
        
        # 追加新數據
        for model, base_model, score, total in zip(model_names, base_model_names, pit_winrate, pit_games):
            writer.writerow([model, base_model, score, total])
    '''
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            params = choix.ilsr_pairwise_dense(win_matrix)
        print("\nRankings:")
        for i, player in enumerate(np.argsort(params)[::-1]):
            name = 'random' if args.playRandom and player == model_count - \
                1 else Path(networks[player]).stem
            print(f"{i+1}. {name} with {params[player]:0.2f} rating")
        print(
            "\n(Rating Diff, Winrate) -> (0.5, 62%), (1, 73%), (2, 88%), (3, 95%), (5, 99%)")
    except Exception:
        print("\nNot Enough data to calculate rankings")
    '''
