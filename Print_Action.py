from Game import Game
WHITE = 2
import math
import pandas as pd
import numpy as np

class  Print_Action:
    def __init__(self, game):
        self.game = game
        self.board_size = game.board_size

    def print_action(self,board,WHITE, pi):
        pro,legal_actions =self.game.get_valid_actions(board,WHITE,pi)
        action = []
        probaility = []
        for a in legal_actions:
            p = 0
            for i in [0,1,2]:
                if pi[a[i] + i * self.game.board_size ** 2] == 0:
                    break
                p += math.log(pi[a[i] + i*self.board_size**2])
            action.append((a[0],a[1],a[2]))
            probaility.append(p)
        df = pd.DataFrame({'action':action,'probaility':probaility})
        df = df.sort_index(by=['probaility'],ascending=False)
        print(df[0:5])