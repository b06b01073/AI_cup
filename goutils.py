import govars
from GoEnv import Go

import numpy as np

def move_encode(move):
    if move == govars.PASS:
        one_hot = np.zeros((govars.ACTION_SPACE))
        one_hot[govars.PASS - 1] = 1
        return govars.PASS, one_hot

    coord = move[2:4]
    move_1d = (ord(coord[0]) - ord('a')) + (ord(coord[1]) - ord('a')) * 19
    one_hot = np.zeros((govars.ACTION_SPACE))
    one_hot[move_1d] = 1
    return move_1d, one_hot


def debug_game(game):
    go_env = Go()
    last_move = 'W'
    # print(game)
    for idx, move in enumerate(game):
        if move[0] == last_move:
            # print('pass')
            go_move, _ = move_encode(govars.PASS)
            go_env.make_move(go_move)
        
        go_move, _ = move_encode(move) # go_move is for the go env, moves[idx] is the one hot vector
        # coord = move[2:4]
        # print(move, (ord(coord[0]) - ord('a')), (ord(coord[1]) - ord('a')))
        go_env.make_move(go_move)
        # go_env.render()
        last_move = move[0]


def pad_board(state):
    return np.pad(state, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
