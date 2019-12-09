import numpy as np

import matplotlib.pyplot as plt
from Game import Game

BLACK = -2
WHITE = 2
EMPTY = 0
ARROW = 1
#epoch = 5
path = 'board.txt'
board_size = 5

class PrintBoard:

    def __init__(self, game):
        self.game = game
        self.board = game.board
        self.board_size = game.board_size

    def print_board(self, board, epoch):
        fo = open("board.txt", "w")
        fo.write("\n")
        fo.close()
        white_chess_x = []
        white_chess_y = []
        black_chess_x = []
        black_chess_y = []
        arrow_x = []
        arrow_y = []

        for i in range(board_size):
            for j in range(board_size):
                if board[i][j] == WHITE:
                    white_chess_x.append(j)
                    white_chess_y.append(board_size - i)
                if board[i][j] == BLACK:
                    black_chess_x.append(j)
                    black_chess_y.append(board_size - i)
                if board[i][j] == ARROW:
                    arrow_x.append(j)
                    arrow_y.append(board_size - i)
        # print(board)
#        print(np.array2string(board, separator=', '))

        plt.style.use('Solarize_Light2')
        plt.subplot(4, 5, epoch)
#        print(plt.style.available)  # 显示背景风格种类

        ax = plt.scatter(white_chess_x, white_chess_y, c='white', s=100, marker='s')
        ax = plt.scatter(black_chess_x, black_chess_y, c='black', s=100, marker='s')
        ax = plt.scatter(arrow_x, arrow_y, c='blue', s=100, marker='x')
        plt.axis('equal')  # 设置坐标为相等长度
        plt.grid()
        plt.xlim((-1, 5))
        plt.ylim((0, 6))

        epoch += 1
        '''
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(path, 'a') as f:
            f.write('\n\n')
            f.write(np.array2string(board, separator=', '))
        '''
    def print(self, game_num):
        plt.suptitle('The ' + str(game_num) + 'th game')
        plt.savefig('E:\\PyCharm_workspaces\\NeuqAmazonGame\\figure' + str(game_num) + 'th_game.png')
        plt.show()


