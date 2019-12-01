import random
import numpy as np

class NNet:
    def __init__(self, game):
        self.board_size = game.board_size

    def predict(self, board):
        pi = [random.random() for i in range(3 * self.board_size ** 2)]
        pi = np.array(pi)
        pi = pi / sum(pi)
        return pi, random.random() - 0.5

# 两个问题：
# 为什么所有棋子（特别是黑棋）都计算成将箭放到皇后起点为概率最大值
# 为什么黑棋明明在帮白棋下(演员) —> 有可能黑棋的v给反了，导致黑棋输时UCT最大
# 温度超参数是不是应该随着棋局进行改变？
# 5*5 开始时可走步数：260 第四步以后可走的步数就基本小于100

