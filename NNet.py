import random

class NNet:
    def __init__(self, game):
        self.board_size = game.board_size

    def predict(self, board):
        return [random.random() for i in range(3 * self.board_size ** 2)], random.random() - 0.5

# 两个问题：
# 为什么所有棋子都计算成将箭放到皇后起始点为概率最大值
# 为什么黑棋明明在帮白棋下 —> 有可能黑棋的v给反了，导致黑棋输时UCT最大
