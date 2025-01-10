# cython: language_level=3
# cython: linetrace=True
# cython: profile=True
# cython: binding=True

import math
import numpy as np

EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.reset()

    def reset(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper) 儲存(狀態,動作)的Q值
        self.Nsa = {}  # stores #times edge s,a was visited 儲存(狀態,動作)被訪問的次數
        self.Ns = {}  # stores #times board s was visited 儲存(狀態)被訪問的次數
        self.Ps = {}  # stores initial policy (returned by neural net) 儲存(狀態)的初始策略

        self.Es = {}  # stores game.getGameEnded ended for board s 儲存(狀態)的遊戲結果--是否結束
        self.Vs = {}  # stores game.getValidMoves for board s 儲存(狀態)的合法動作集合

        self.mode = 'leaf' #初始化搜尋模式為leaf
        self.path = [] #初始化搜尋的路徑
        self.v = 0 #初始化狀態值

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        執行 numMCTSSims 次 MCTS 模擬從給定的 canonicalBoard 開始。

        參數:
        canonicalBoard: 當前盤面的標準化表示。
        temp: 控制探索的溫度參數，決定策略的確定性。

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
            probs: 動作策略向量，表示各動作選擇的機率，機率大小與 Nsa[(s,a)]^(1/temp) 成比例。
        """

        # 執行指定次數的蒙地卡羅樹搜索模擬
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (
            s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())] # 計算每個動作的拜訪次數 (counts)，若動作無拜訪次數則設為 0

        # 當溫度參數 temp 為 0，選擇訪問次數最多的動作，確保策略確定
        if temp == 0:
            bestA = np.argmax(counts) # 選擇訪問次數最多的動作
            probs = [0] * len(counts)
            probs[bestA] = 1 # 設定唯一確定的動作
            return probs

        try: # 當溫度大於 0，根據溫度計算機率，增加隨機性
            counts = [x ** (1. / temp) for x in counts]
            probs = [x / float(sum(counts)) for x in counts]
            return probs
        except OverflowError as err: # 若計算過程中溢出 (通常由於數值過大導致)，則選擇訪問次數最多的動作
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

    def getExpertProb(self, canonicalBoard, temp=1, prune=False):
        """
        根據模擬的訪問次數，生成策略分布 (動作選擇機率)。
        
        參數:
            canonicalBoard: 當前的局面，使用「標準」格式表示 (從當前玩家的視角)。
            temp: 控制溫度參數 (溫度越低越偏向高訪問次數的動作)。
            prune: 若為 True，則對策略進行剪枝以排除較不重要的選擇。
        
        返回:
            probs: 一個策略向量，表示每個動作的選擇機率。
        """
        s = self.game.stringRepresentation(canonicalBoard) # 將局面表示轉為字串格式，方便索引

        counts = [self.Nsa[(s, a)] if (
            s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())] # 計算每個動作的訪問次數，若尚無紀錄則設為 0

        if prune:# 若啟用剪枝模式 (prune=True)，嘗試移除較不重要的選項以提升策略的效率
            
            # 找出訪問次數最高的動作及其對應的上置信界 (UCB) 值
            bestA = np.argmax(counts) 
            u_max = self.Qsa[(s, bestA)] + self.args.cpuct * \
                self.Ps[s][bestA] * math.sqrt(self.Ns[s]) / (counts[bestA] + 1)
            
            for a in range(self.game.getActionSize()):
                if a == bestA: # 跳過最佳動作和未訪問過的動作
                    continue
                if counts[a] <= 0:
                    continue
                
                # 根據指定動作的期望訪問次數 (desired) 進行削減
                desired = math.ceil(math.sqrt(2*self.Ps[s][a]*self.Ns[s]))
                u_const = self.Qsa[(s, a)] + self.args.cpuct * \
                    self.Ps[s][a] * math.sqrt(self.Ns[s])
                
                # 若計算出的 u_const/counts[a] 小於 u_max，削減其訪問次數以便於排除
                for _ in range(desired):
                    if counts[a] <= 0:
                        break
                    if u_const / counts[a] < u_max:
                        counts[a] -= 1

        if temp == 0: # 使用溫度參數 temp 調整策略分布
            # 若 temp=0，則選擇訪問次數最多的動作 (完全確定的策略)
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        try:# 若 temp 不為 0，對訪問次數進行 (1/temp) 次方並歸一化
            counts = [x ** (1. / temp) for x in counts]
            probs = [x / float(sum(counts)) for x in counts]
            return probs
        except OverflowError as err:# 若計算中出現數值溢出錯誤 (例如 temp 過小)，退而選擇訪問次數最高的動作
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

    def getExpertValue(self, canonicalBoard):
        """
        計算該局面的最佳估值 (根據 MCTS 的評估值 Q)。
        
        參數:
            canonicalBoard: 當前的局面，使用「標準」格式表示 (從當前玩家的視角)。
        
        返回:
            values: 該局面中最大 Q 值的動作的評估值。
        """
        s = self.game.stringRepresentation(canonicalBoard) # 將局面表示轉為字串格式
        values = [self.Qsa[(s, a)] if (
            s, a) in self.Qsa else 0 for a in range(self.game.getActionSize())] # 構建 Q 值列表，若無紀錄則設為 0
        return np.max(values) # 返回 Q 值的最大值，表示該局面的最佳評估值

    def processResults(self, pi, value):
        """
        處理模擬結果，並將新結果傳播回 MCTS 樹。
        當 mode 為 'leaf' 時，設定葉節點的策略分布；隨後，執行反向傳播以更新沿途各節點的 Q 值與 N 值。

        參數:
            pi: 策略向量，表示每個動作的建議選擇機率。
            value: 葉節點的評估值，表示該節點對當前玩家的預期勝率。
        """
        if self.mode == 'leaf': 
            s = self.path.pop()[0] # 從路徑中取出最後一個節點的狀態表示 s。
            self.Ps[s] = pi # 將策略向量 pi 存入 Ps 字典，供未來擴展此狀態使用。
            self.Ps[s] = self.Ps[s] * self.Vs[s]  # masking invalid moves | 將策略向量與合法動作遮罩 Vs[s] 相乘，過濾掉不合法的動作。
            sum_Ps_s = np.sum(self.Ps[s]) # 計算策略向量的和 (已遮罩非法動作)
            if sum_Ps_s > 0: # 如果策略向量和大於 0，則重新正規化，使所有合法動作的機率總和為 1。
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                # 若所有合法動作均被屏蔽 (和為 0)，則讓所有合法動作均分機率以避免錯誤。
                # 若此情況頻繁發生，可能表示神經網路架構問題或過擬合等訓練異常。
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + self.Vs[s]
                self.Ps[s] /= np.sum(self.Ps[s]) # 將均分的策略重新正規化

            self.Ns[s] = 0 # 將該狀態的訪問次數初始化為 0。
            self.v = -value # 將評估值取負，以適應反向傳播 (從對手角度評估)。

        # 反向傳播: 更新沿途的節點 Q 值與訪問次數。
        self.path.reverse() # 反轉路徑，以便從葉節點往根節點方向傳播
        for s, a in self.path: 
            if (s, a) in self.Qsa: # 若 Q 值已存在，使用新的評估值更新 Q 值並累加訪問次數。
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                    self.Qsa[(s, a)] + self.v) / (self.Nsa[(s, a)] + 1)
                self.Nsa[(s, a)] += 1

            else: # 若 Q 值不存在，初始化該 Q 值並設定訪問次數為 1。
                self.Qsa[(s, a)] = self.v
                self.Nsa[(s, a)] = 1

            self.Ns[s] += 1 # 更新狀態 s 的總訪問次數。
            self.v *= -1 # 評估值取負，用於下一節點的反向傳播 (對手的視角)。
        self.path = [] # 清空路徑，以便於下一次搜索。

    def findLeafToProcess(self, canonicalBoard, isRoot):
        """
        此函式在 MCTS 中負責尋找葉節點 (leaf node) 來擴展樹，並進行模擬。
        如果找到終端節點 (terminal node)，則立即返回；否則繼續沿著 UCB 算法選擇最佳行動。

        參數:
            canonicalBoard: 標準化的遊戲盤面。
            isRoot: 是否為當前搜索的根節點 (用來特殊處理探索策略)。

        返回:
            canonicalBoard: 如果找到葉節點，返回該節點的盤面；若為終端節點則返回 None。
        """
        s = self.game.stringRepresentation(canonicalBoard) # 將盤面轉換為字串，作為唯一標識。

        if s not in self.Es: # 若該狀態未在遊戲結果字典 Es 中，則判斷該狀態是否為終端節點。
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0: # 如果該狀態為終端節點 (勝負已分)，則設定模式為 'terminal' 並儲存該節點的值。
            # terminal node
            self.mode = 'terminal' # 將模式設為終端。
            self.v = -self.Es[s] # 設定當前狀態值為遊戲結果的負值 (對手的視角)。
            return None # 返回 None，代表已找到終端節點。

        if s not in self.Ps: # 若該狀態不在初始策略字典 Ps 中，則將其視為葉節點，準備擴展樹。
            # leaf node
            self.Vs[s] = self.game.getValidMoves(canonicalBoard, 1) # 取得合法動作。
            self.mode = 'leaf' # 設定模式為 'leaf'。
            self.path.append((s, None)) # 將當前狀態記錄在路徑中 (動作為 None)。
            return canonicalBoard # 返回該葉節點的盤面。

        # 從此處開始探索子節點，選擇 UCB 分數最高的動作。
        valids = self.Vs[s] # 取得該狀態下的合法動作。
        cur_best = -float('inf') # 初始化當前最佳 UCB 分數為負無窮大。
        best_act = -1 # 初始化最佳動作為 -1 (無效動作)。

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()): #self.n*self.n + 1 = 257
            if valids[a]: # 檢查動作是否合法。
                if (s, a) in self.Qsa: # 如果該動作有已計算的 Q 值。
                    # prioritize under explored options.
                    if isRoot and self.Nsa[(s, a)] < math.sqrt(2*self.Ps[s][a]*self.Ns[s]): # 如果是根節點，並且該動作的訪問次數不足，則優先探索該動作。
                        best_act = a # 直接選擇該動作作為最佳動作。
                        break # 結束搜尋，進入下一步。

                    # 計算該動作的 UCB 分數。
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, a)])
                else:
                    # 若該動作尚無 Q 值，則設 Q 為 0 並計算 UCB 分數。
                    u = self.args.cpuct * \
                        self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                # 若此 UCB 分數高於目前最佳分數，則更新最佳動作與分數。
                if u > cur_best:
                    cur_best = u
                    best_act = a

        # 根據選出的最佳動作取得下一狀態和下一位玩家。
        a = best_act # 確定要執行的最佳動作。
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a) # 獲取新狀態與玩家。
        next_s = self.game.getCanonicalForm(next_s, next_player) # 標準化新狀態。
        self.path.append((s, a)) # 將當前狀態與動作記錄在路徑中 (供反向傳播使用)。
        return self.findLeafToProcess(next_s, False) # 遞迴尋找葉節點，直到找到為止。

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        執行一次 MCTS (蒙地卡羅樹搜索) 遍歷，直至找到葉節點。
        每個節點選擇具有最大上置信界 (UCB) 的動作。

        當找到葉節點時，呼叫神經網絡以獲得初始策略 P 和評估值 v。
        若葉節點是終端狀態，則將結果回傳並沿路徑向上傳播。更新 Ns, Nsa, Qsa 的值。

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
            v: 當前盤面的負評估值 (代表對手視角的價值)
        """

        s = self.game.stringRepresentation(canonicalBoard) #將盤面轉換為字串

        if s not in self.Es: #如果盤面狀態不在Es(遊戲結果)中
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1) #將遊戲結果存入Es
        if self.Es[s] != 0: #如果遊戲已結束 0:未結束 1:player1贏 -1:player2贏
            # terminal node
            return -self.Es[s] #回傳遊戲結果的負值，在樹狀搜索中，反映對手的視角。返回負值可以幫助算法在評估節點時考慮到對手的最佳策略。

        # 若該盤面尚未進行過搜索，則為葉節點，需用神經網絡計算策略 P 和評估值 v
        if s not in self.Ps: #如果盤面狀態不在Ps(初始策略)中
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard) # 利用神經網絡預測策略 (pi) 和評估值 (v)
            valids = self.game.getValidMoves(canonicalBoard, 1) # 獲取該盤面所有合法的移動
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves | 遮蔽非法的動作 (使其策略值為 0)
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0: # 正規化策略 (使所有值的總和為 1)
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                # 若所有動作均被遮蔽，則平分所有合法的動作機率
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            # 將合法移動保存到 Vs 中以便重用，並初始化盤面訪問次數 Ns
            self.Vs[s] = valids
            self.Ns[s] = 0

            # 返回當前盤面評估值的負值，表示對手的視角
            return -v

        # 若盤面已存在，則選擇具有最大上置信界 (UCB) 的動作
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound | 遍歷每個可能的動作，計算 UCB 值並選擇最大的動作
        for a in range(self.game.getActionSize()):
            if valids[a]: # 只考慮合法的動作
                if (s, a) in self.Qsa:
                    # 若動作已有 Q 值，則使用上置信界公式計算 UCB
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, a)])
                else:
                    # 若無 Q 值，則只使用先驗概率 (Ps) 和 Ns 計算 UCB
                    u = self.args.cpuct * \
                        self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        # 選擇具有最大 UCB 的動作作為最佳動作
        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        # 對下一個狀態進行遞歸搜索
        v = self.search(next_s)

        # 根據搜索結果更新 Qsa 和 Nsa
        if (s, a) in self.Qsa:
            # 平均更新 Q 值，考慮當前 Q 值和新評估值 v
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            # 若該動作尚無 Q 值，則初始化 Q 值和訪問次數
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        # 增加該盤面狀態的總訪問次數
        self.Ns[s] += 1

        # 返回當前評估值的負值，表示對手的視角
        return -v
