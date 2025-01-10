from MCTS import MCTS
from SelfPlayAgent import SelfPlayAgent
import torch
from pathlib import Path
from glob import glob
from torch import multiprocessing as mp
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from tensorboardX import SummaryWriter
from Arena import Arena
from GenericPlayers import RandomPlayer, NNPlayer
from pytorch_classification.utils import Bar, AverageMeter
from queue import Empty
from time import time
import numpy as np
from math import ceil
import os


class Coach:
    def __init__(self, game, nnet, args):
        np.random.seed()
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game) #創建一個 pnet（神經網路的副本），用於網路比較時使用
        self.args = args # 將傳入的參數保存到當前對象中
        self.elp = 0 # 設置一開始的elo為0

        # 查找已存在的訓練網路模型檔案並排序，確定訓練的起始迭代
        networks = sorted(glob(self.args.checkpoint+'/*'))
        self.args.startIter = len(networks)

        # 若未找到任何檔案，則初始化一個基本的神經網路並儲存為 'iteration-0000.pkl'
        if self.args.startIter == 0:
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename='iteration-0000.pkl')
            self.args.startIter = 1

        # 載入最近的神經網路模型以繼續訓練
        self.nnet.load_checkpoint(
            folder=self.args.checkpoint, filename=f'iteration-{(self.args.startIter-1):04d}.pkl')

        self.agents = [] # 用於儲存代理對象
        self.input_tensors = [] # 用於儲存神經網路輸入張量
        self.policy_tensors = [] # 用於儲存策略輸出張量
        self.value_tensors = [] # 用於儲存價值輸出張量
        self.batch_ready = [] # 用於管理批次處理的同步標誌

        # 佇列，用於多處理間的數據傳輸
        self.ready_queue = mp.Queue() # 用於通知 batch 處理就緒
        self.file_queue = mp.Queue() # 用於輸出文件的保存
        self.result_queue = mp.Queue() # 用於存儲訓練的結果數據

        # 共享變數，用於追蹤完成的代理和已玩遊戲數量
        self.completed = mp.Value('i', 0) # 追蹤完成的遊戲代理數
        self.games_played = mp.Value('i', 0) # 追蹤總共完成的遊戲數量

        # 建立記錄器，用於追蹤訓練過程的摘要
        if self.args.run_name != '':
            self.writer = SummaryWriter(log_dir='runs/'+self.args.run_name)
        else:
            self.writer = SummaryWriter()
        self.args.expertValueWeight.current = self.args.expertValueWeight.start # 設定當前 expertValueWeight (專家價值權重) 為其初始值

    def learn(self):
        print('Because of batching, it can take a long time before any games finish.')
        for i in range(self.args.startIter, self.args.numIters + 1):
            print(f'------ITER {i}------')
            self.generateSelfPlayAgents() # 1. 生成自對弈代理 (Agent)
            self.processSelfPlayBatches() # 2. 處理自對弈批次，以產生樣本數據
            self.saveIterationSamples(i) # 3. 保存當前迭代的數據樣本
            self.processGameResults(i) # 4. 處理並更新遊戲結果
            self.killSelfPlayAgents() # 5. 停止所有自對弈代理
            self.train(i) # 6. 使用收集的數據進行神經網路訓練

            # 隨機對弈比較: 若啟用且達到隨機比較頻率時，與隨機策略比較
            if self.args.compareWithRandom and (i-1) % self.args.randomCompareFreq == 0:
                if i == 1:
                    print(
                        'Note: Comparisons with Random do not use monte carlo tree search.')
                self.compareToRandom(i)

            # 歷史策略比較: 若啟用且達到歷史比較頻率時，與過去迭代的策略比較
            if self.args.compareWithPast and (i - 1) % self.args.pastCompareFreq == 0:
                self.compareToPast(i)
            
            # 更新 expertValueWeight (專家價值權重) 的當前值
            z = self.args.expertValueWeight
            self.args.expertValueWeight.current = min(
                i, z.iterations)/z.iterations * (z.end - z.start) + z.start
            print()
        self.writer.close()

    def generateSelfPlayAgents(self):
        self.ready_queue = mp.Queue() # 初始化一個新的佇列，用來存儲準備好的 agent
        boardx, boardy = self.game.getBoardSize() # 取得棋盤的寬和高，作為輸入張量的大小
        
        # 遍歷設定的 worker 數量，為每個 worker 初始化相關資源
        for i in range(self.args.workers):
            # 建立一個大小為 [批處理大小, 棋盤寬, 棋盤高] 的輸入張量，初始值為零
            # 使用 pin_memory 和 share_memory_ 來優化記憶體使用，使跨進程共享更有效
            self.input_tensors.append(torch.zeros(
                [self.args.process_batch_size, boardx, boardy]))
            self.input_tensors[i].pin_memory()
            self.input_tensors[i].share_memory_()

            # 建立一個用於儲存行動機率的 policy 張量，大小為 [批處理大小, 行動數量]
            # 同樣使用 pin_memory 和 share_memory_ 優化記憶體傳輸和共享
            self.policy_tensors.append(torch.zeros(
                [self.args.process_batch_size, self.game.getActionSize()]))
            self.policy_tensors[i].pin_memory()
            self.policy_tensors[i].share_memory_()

            # 建立一個用於儲存當前狀態評估值的 value 張量，大小為 [批處理大小, 1]
            # 同樣進行記憶體優化以加快跨進程的傳輸
            self.value_tensors.append(torch.zeros(
                [self.args.process_batch_size, 1]))
            self.value_tensors[i].pin_memory()
            self.value_tensors[i].share_memory_()
            self.batch_ready.append(mp.Event()) # 為每個 worker 創建一個事件，用來通知批處理是否已準備好

            # 初始化一個 SelfPlayAgent，並傳入其所需的資源，包括 unique ID、棋盤張量和共享變數等
            self.agents.append(
                SelfPlayAgent(i, self.game, self.ready_queue, self.batch_ready[i],
                              self.input_tensors[i], self.policy_tensors[i], self.value_tensors[i], self.file_queue,
                              self.result_queue, self.completed, self.games_played, self.args))
            self.agents[i].start()

    def processSelfPlayBatches(self):
        sample_time = AverageMeter() # 初始化計時器，用於計算生成樣本的平均時間
        bar = Bar('Generating Samples', max=self.args.gamesPerIteration) # 初始化進度條，顯示目前樣本生成的進度
        end = time() # 記錄開始時間，用於計算處理時間
 
        n = 0 # 已完成遊戲數量的暫存值
        while self.completed.value != self.args.workers: # 當所有的工作執行緒尚未完成時，持續處理
            try:
                id = self.ready_queue.get(timeout=1) # 從 ready_queue 中取得準備好的執行緒 ID，設置超時為 1 秒
                self.policy, self.value = self.nnet.process(
                    self.input_tensors[id]) # 使用神經網路進行推理，計算策略 (policy) 和價值 (value)
                
                # 將計算出的策略和價值複製到對應的共享張量中
                self.policy_tensors[id].copy_(self.policy)
                self.value_tensors[id].copy_(self.value)

                self.batch_ready[id].set() # 設置對應執行緒的準備事件為完成，通知該執行緒可以繼續執行
            except Empty:
                pass # 如果 1 秒內沒有資料從隊列中取出，跳過此次迭代
            size = self.games_played.value # 獲取目前完成的遊戲數量
            if size > n: # 如果有新的遊戲完成，更新計時器和進度條
                sample_time.update((time() - end) / (size - n), size - n) # 更新平均處理時間

                # 更新暫存值 n 並重置計時器
                n = size
                end = time()

            # 更新進度條的後綴資訊，顯示樣本生成進度、平均時間、總時間和預計完成時間
            bar.suffix = f'({size}/{self.args.gamesPerIteration}) Sample Time: {sample_time.avg:.3f}s | Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'
            bar.goto(size)# 調整進度條的位置到目前完成的遊戲數量
        
        # 所有遊戲完成後更新並關閉進度條
        bar.update()
        bar.finish()
        print()

    def killSelfPlayAgents(self):
        for i in range(self.args.workers): # 停止並清理所有自對弈代理
            self.agents[i].join() # 等待每個自對弈代理的執行緒結束
            del self.input_tensors[0] # 刪除對應的張量和事件（清理資源）
            del self.policy_tensors[0]
            del self.value_tensors[0]
            del self.batch_ready[0]
        self.agents = [] # 清空代理的列表
        self.input_tensors = [] # 清空輸入張量、策略張量、價值張量及批處理完成事件列表
        self.policy_tensors = []
        self.value_tensors = []
        self.batch_ready = [] 
        self.ready_queue = mp.Queue() # 重置準備隊列為新的空隊列
        self.completed = mp.Value('i', 0) # 重置完成和遊戲計數的共享變量
        self.games_played = mp.Value('i', 0)

    def saveIterationSamples(self, iteration):
        # 取得目前 file_queue 中的樣本數量
        num_samples = self.file_queue.qsize()
        print(f'Saving {num_samples} samples')
        boardx, boardy = self.game.getBoardSize() # 獲取遊戲棋盤的尺寸（例如 16x16）

        # 建立張量，用於存儲樣本的棋盤狀態、策略以及價值
        data_tensor = torch.zeros([num_samples, boardx, boardy])
        policy_tensor = torch.zeros([num_samples, self.game.getActionSize()])
        value_tensor = torch.zeros([num_samples, 1])

        # 將 file_queue 中的數據提取並存入對應的張量
        for i in range(num_samples):
            data, policy, value = self.file_queue.get() # 從隊列中取出樣本
            data_tensor[i] = torch.from_numpy(data) # 將 numpy 數據轉為張量並存入
            policy_tensor[i] = torch.tensor(policy) # 將策略存入策略張量
            value_tensor[i, 0] = value # 將價值存入價值張量

        # 確保保存樣本的目錄存在，若不存在則創建
        os.makedirs(self.args.data, exist_ok=True)

        # 將樣本的棋盤數據、策略和價值保存到檔案
        torch.save(
            data_tensor, f'{self.args.data}/iteration-{iteration:04d}-data.pkl') # 保存棋盤數據
        torch.save(policy_tensor,
                   f'{self.args.data}/iteration-{iteration:04d}-policy.pkl') # 保存策略數據
        torch.save(
            value_tensor, f'{self.args.data}/iteration-{iteration:04d}-value.pkl') # 保存價值數據

        # 釋放記憶體，清理張量
        del data_tensor
        del policy_tensor
        del value_tensor

    def processGameResults(self, iteration):
        num_games = self.result_queue.qsize() # 計算已完成的對局數量
        p1wins = 0 # 玩家1的勝利次數
        p2wins = 0 # 玩家2的勝利次數
        draws = 0 # 平局次數

        # 從結果隊列中逐個提取對局結果，並進行分類統計
        for _ in range(num_games):
            winner = self.result_queue.get() # 獲取一局比賽的勝者
            if winner == 1: # 玩家1獲勝
                p1wins += 1
            elif winner == -1: # 玩家2獲勝
                p2wins += 1
            else: # 平局
                draws += 1
        self.writer.add_scalar('win_rate/p1 vs p2',
                               (p1wins+0.5*draws)/num_games, iteration) # 計算並記錄玩家1的勝率（含平局的0.5分貢獻），保存到 TensorBoard
        self.writer.add_scalar('win_rate/draws', draws/num_games, iteration) # 記錄平局比例，保存到 TensorBoard

    def train(self, iteration):
        datasets = [] # 用於存放多個迭代樣本的數據集列表
        #currentHistorySize = self.args.numItersForTrainExamplesHistory

        # 確定當前訓練樣本的歷史大小範圍
        currentHistorySize = min(
            max(4, (iteration + 4)//2), # 動態計算歷史樣本大小，至少為4，隨迭代增長
            self.args.numItersForTrainExamplesHistory) # 不超過設定的歷史樣本最大值
        # 加載歷史迭代數據到數據集
        for i in range(max(1, iteration - currentHistorySize), iteration + 1):

            # 載入當前迭代範圍內的數據、策略和價值張量
            data_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-data.pkl') # 棋盤數據
            policy_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-policy.pkl') # 策略數據
            value_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-value.pkl') # 價值數據
            datasets.append(TensorDataset(
                data_tensor, policy_tensor, value_tensor)) # 將三種張量打包成一個TensorDataset並添加到列表中

        dataset = ConcatDataset(datasets) # 將所有歷史數據集連接為一個大數據集

        # 創建數據加載器，用於分批次讀取數據
        dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                num_workers=self.args.workers, pin_memory=True) # pin_memory 提升數據傳輸速度

        # 使用神經網路訓練數據
        l_pi, l_v = self.nnet.train(
            dataloader, self.args.train_steps_per_iteration)
        # 使用 TensorBoard 記錄loss
        self.writer.add_scalar('loss/policy', l_pi, iteration) # 記錄策略loss
        self.writer.add_scalar('loss/value', l_v, iteration) # 記錄價值loss
        self.writer.add_scalar('loss/total', l_pi + l_v, iteration) # 記錄總loss

        # 保存當前迭代的模型參數檔案
        self.nnet.save_checkpoint(
            folder=self.args.checkpoint, filename=f'iteration-{iteration:04d}.pkl')

        # 清理記憶體中的數據加載器和數據集，釋放資源
        del dataloader
        del dataset
        del datasets

    def compareToPast(self, iteration):
        past = max(0, iteration-self.args.pastCompareFreq)
        self.pnet.load_checkpoint(folder=self.args.checkpoint,
                                  filename=f'iteration-{past:04d}.pkl')
        print(f'PITTING AGAINST ITERATION {past}')
        #在對戰時更改MCTS模擬次數
        args_pit = self.args
        args_pit.numMCTSSims = 1000
        if(self.args.arenaMCTS):
            pplayer = MCTS(self.game, self.pnet, args_pit)
            nplayer = MCTS(self.game, self.nnet, args_pit)

            def playpplayer(x, turn):
                if turn <= 2:
                    pplayer.reset()
                temp = self.args.temp if turn <= self.args.tempThreshold else self.args.arenaTemp
                policy = pplayer.getActionProb(x, temp=temp)
                return np.random.choice(len(policy), p=policy)

            def playnplayer(x, turn):
                if turn <= 2:
                    nplayer.reset()
                temp = self.args.temp if turn <= self.args.tempThreshold else self.args.arenaTemp
                policy = nplayer.getActionProb(x, temp=temp)
                return np.random.choice(len(policy), p=policy)

            arena = Arena(playnplayer, playpplayer, self.game)
        else:
            pplayer = NNPlayer(self.game, self.pnet, self.args.arenaTemp)
            nplayer = NNPlayer(self.game, self.nnet, self.args.arenaTemp)

            arena = Arena(nplayer.play, pplayer.play, self.game)
        nwins, pwins, draws = arena.playGames(self.args.arenaCompare)

        pwin_rate = pwins/(pwins + nwins + draws)
        elo = np.log(1/pwin_rate - 1) + self.elp
        self.elp = elo
        print(f'NEW/PAST WINS : {nwins} / {pwins} ; DRAWS : {draws} ; ELO : {elo}\n')
        self.writer.add_scalar(
            'win_rate/past', float(nwins + 0.5 * draws) / (pwins + nwins + draws), iteration)

    def compareToRandom(self, iteration):
        r = RandomPlayer(self.game)
        nnplayer = NNPlayer(self.game, self.nnet, self.args.arenaTemp)
        print('PITTING AGAINST RANDOM')

        arena = Arena(nnplayer.play, r.play, self.game)
        nwins, pwins, draws = arena.playGames(self.args.arenaCompareRandom)

        print(f'NEW/RANDOM WINS : {nwins} / {pwins} ; DRAWS : {draws}\n')
        self.writer.add_scalar(
            'win_rate/random', float(nwins + 0.5 * draws) / (pwins + nwins + draws), iteration)
