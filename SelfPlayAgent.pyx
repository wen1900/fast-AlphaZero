# cython: language_level=3
import numpy as np
import torch
import torch.multiprocessing as mp

from MCTS import MCTS


class SelfPlayAgent(mp.Process):

    def __init__(self, id, game, ready_queue, batch_ready, batch_tensor, policy_tensor, value_tensor, output_queue,
             result_queue, complete_count, games_played, args):

        # 繼承 multiprocessing.Process 的初始化方法
        super().__init__()
        self.id = id # 當前執行緒或進程的唯一識別碼
        self.game = game # 遊戲物件，包含遊戲的邏輯與規則
        self.ready_queue = ready_queue # 用於通知主程序準備好數據的隊列
        self.batch_ready = batch_ready # 批次準備完成的事件物件，用於進程間同步
        self.batch_tensor = batch_tensor # 存放當前批次遊戲數據的張量
        self.batch_size = self.batch_tensor.shape[0] # 批次大小，表示同時處理的遊戲數量
        self.policy_tensor = policy_tensor # 存放當前批次策略輸出的張量
        self.value_tensor = value_tensor # 存放當前批次價值輸出的張量
        self.output_queue = output_queue # 用於保存批次結果的輸出隊列
        self.result_queue = result_queue # 用於保存遊戲結果的隊列
        self.games = [] # 儲存每個遊戲的棋盤狀態
        self.canonical = [] # 儲存每個遊戲的規範化棋盤表示（根據當前玩家進行轉換）
        self.histories = [] # 儲存遊戲過程中的歷史狀態
        self.player = [] # 記錄當前每個遊戲的玩家，1 表示玩家 1，-1 表示玩家 2
        self.turn = [] # 記錄遊戲進行到的回合數
        self.mcts = [] # 為每個遊戲初始化 MCTS（蒙特卡洛樹搜索）物件
        self.games_played = games_played # 記錄已完成遊戲的共享變數
        self.complete_count = complete_count # 記錄已完成進程數的共享變數
        self.args = args # 全局參數設定
        self.valid = torch.zeros_like(self.policy_tensor) # 用於存放合法行動的掩碼張量
        self.fast = False # 用於快速模式的標記位（如果需要快速處理）
        # 初始化批次內每個遊戲的初始狀態
        for _ in range(self.batch_size):
            self.games.append(self.game.getInitBoard()) # 取得每個遊戲的初始棋盤
            self.histories.append([]) # 初始化每個遊戲的歷史為空
            self.player.append(1) # 初始設定每個遊戲的玩家為 1
            self.turn.append(1) # 初始設定每個遊戲的回合數為 1
            self.mcts.append(MCTS(self.game, None, self.args)) # 為每個遊戲初始化一個 MCTS 實例
            self.canonical.append(None) # 初始化規範化棋盤為 None

    def run(self):
        # 設定隨機種子，確保每個執行緒的隨機性是獨立的
        np.random.seed()
        while self.games_played.value < self.args.gamesPerIteration: # 主迴圈：當完成的遊戲數小於設定的目標遊戲數時，持續執行
            # 1. 生成每個遊戲的規範化棋盤表示（根據當前玩家進行轉換）
            self.generateCanonical()
            # 2. 隨機決定是否進行快速模擬（根據設定的概率 `probFastSim`）
            self.fast = np.random.random_sample() < self.args.probFastSim
            if self.fast: # 如果是快速模擬模式，執行預設的快速模擬次數
                for i in range(self.args.numFastSims):
                    self.generateBatch() # 生成批次遊戲數據
                    self.processBatch() # 處理當前批次的數據（如進行策略或價值預測）
            else:
                # 如果是完整模擬模式，執行完整的 MCTS 模擬次數
                for i in range(self.args.numMCTSSims):
                    self.generateBatch() # 生成批次遊戲數據
                    self.processBatch() # 處理當前批次的數據

            # 3. 更新遊戲的狀態並執行下一步（將模擬結果應用到遊戲）
            self.playMoves()
        
        # 當所有遊戲完成後，標記當前執行緒已完成
        with self.complete_count.get_lock(): # 使用鎖保護共享變數
            self.complete_count.value += 1
        
        # 關閉與當前執行緒相關的輸出佇列
        self.output_queue.close()
        self.output_queue.join_thread()

    def generateBatch(self):
        # 遍歷當前執行緒的所有遊戲，為每個遊戲生成要處理的葉節點
        for i in range(self.batch_size):
            # 使用 MCTS 演算法尋找需要處理的葉節點
            # `findLeafToProcess` 方法會返回對應棋盤的規範化狀態
            # 第二個參數 True 表示該方法會考慮擴展節點並返回葉節點棋盤
            board = self.mcts[i].findLeafToProcess(self.canonical[i], True)

            # 如果找到有效的棋盤狀態，則將其轉換為張量格式並存入批次張量
            if board is not None:
                self.batch_tensor[i] = torch.from_numpy(board)
        
        # 將當前執行緒的 ID 放入 ready_queue，通知主程式該執行緒已準備好處理該批次
        self.ready_queue.put(self.id)

    def processBatch(self):
        # 等待主程式完成批次的策略與價值計算
        # `self.batch_ready` 是多執行緒的事件，當主程式完成計算後會觸發該事件
        self.batch_ready.wait()
        self.batch_ready.clear() # 清除事件標誌，準備進行下一次的批次處理
        # 遍歷該批次中的每個遊戲
        for i in range(self.batch_size):
            # 將神經網路返回的策略與價值結果應用到對應的 MCTS 節點中
            # `self.policy_tensor[i]` 是該遊戲的策略（可能性分佈）
            # `self.value_tensor[i][0]` 是該遊戲的價值（對當前玩家有利的評估值）
            self.mcts[i].processResults(
                self.policy_tensor[i].data.numpy(), self.value_tensor[i][0]) # 將 PyTorch 張量轉換為 NumPy 陣列， 取得價值張量的第一個值（標量）

    def playMoves(self):
        for i in range(self.batch_size):
            temp = int(self.turn[i] < self.args.tempThreshold)
            policy = self.mcts[i].getExpertProb(
                self.canonical[i], temp, not self.fast)
            action = np.random.choice(len(policy), p=policy)
            if not self.fast:
                self.histories[i].append((self.canonical[i], self.mcts[i].getExpertProb(self.canonical[i], prune=True),
                                          self.mcts[i].getExpertValue(self.canonical[i]), self.player[i]))
            self.games[i], self.player[i] = self.game.getNextState(
                self.games[i], self.player[i], action)
            self.turn[i] += 1
            winner = self.game.getGameEnded(self.games[i], 1)
            if winner != 0:
                self.result_queue.put(winner)
                lock = self.games_played.get_lock()
                lock.acquire()
                if self.games_played.value < self.args.gamesPerIteration:
                    self.games_played.value += 1
                    lock.release()
                    for hist in self.histories[i]:
                        if self.args.symmetricSamples:
                            sym = self.game.getSymmetries(hist[0], hist[1])
                            for b, p in sym:
                                self.output_queue.put((b, p,
                                                       winner *
                                                       hist[3] *
                                                       (1 - self.args.expertValueWeight.current)
                                                       + self.args.expertValueWeight.current * hist[2]))
                        else:
                            self.output_queue.put((hist[0], hist[1],
                                                   winner *
                                                   hist[3] *
                                                   (1 - self.args.expertValueWeight.current)
                                                   + self.args.expertValueWeight.current * hist[2]))
                    self.games[i] = self.game.getInitBoard()
                    self.histories[i] = []
                    self.player[i] = 1
                    self.turn[i] = 1
                    self.mcts[i] = MCTS(self.game, None, self.args)
                else:
                    lock.release()

    def generateCanonical(self):
        for i in range(self.batch_size):
            self.canonical[i] = self.game.getCanonicalForm(
                self.games[i], self.player[i])
