import govars
from GoEnv import Go
import gogame

import random
import numpy as np

import torch


from tqdm import tqdm

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


def move_decode_char(action1d):
    # decode the 1d move to the 2d char coord
    action2d = action1d // govars.SIZE, action1d % govars.SIZE
    return chr(ord('a') + action2d[0]) + chr(ord('a') + action2d[1])

def move_decode(action1d):
    return action1d // govars.SIZE, action1d % govars.SIZE

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



def move_2d_encode(move_1d, state_size=govars.SIZE):

    coord_0 = move_1d % state_size
    coord_1 = move_1d // state_size

    return (coord_0, coord_1)


def one_hot_decode(one_hot):
    return np.argmax(one_hot)


def flip_board(state):
    flip = np.random.random()
    if flip > 0.5:
        temp = state[govars.BLACK].copy()
        state[govars.BLACK] = state[govars.WHITE]
        state[govars.WHITE] = temp

        new_turn = (int(np.max(state[govars.TURN_CHNL])) + 1) % 2
        state[govars.TURN_CHNL] = new_turn

    return state

def board_augment(state, move):
    # note that the first channel of state is the feature channel, so we need to flip along the second channel, and rotate along the (1, 2) channels
    state_size = state.shape[1]
    
    move = one_hot_decode(move)

    move_2d = move_2d_encode(move, state_size)
    move_1d = np.zeros((govars.ACTION_SPACE,))

    # move_board = np.zeros((govars.SIZE, govars.SIZE))
    move_board = np.zeros((state_size, state_size))
    if move != govars.PASS:
        move_board[move_2d[1], move_2d[0]] = 1
        move_1d[-1] = 1


    # flip the board with 0.5 prob
    flip = random.random() > 0.5 # 0.5 to filp
    if flip:
        state = np.flip(state, 2)
        move_board = np.flip(move_board, 1)


    # rotate the board
    rotate_times = random.randint(a=0, b=3) # counterclockwise rotate 90 * rotate_deg deg
    state = np.rot90(state, rotate_times, axes=(1, 2))
    move_board = np.rot90(move_board, rotate_times, axes=(0, 1))

    move_1d[:-1] = move_board.flatten()
    return state, np.argmax(move_1d)


def test_time_predict(board, net, device):
    preds = torch.zeros((govars.ACTION_SPACE-1,)).to(device)
    rotate_k = [0, 1, 2, 3] # rotate degree
    flip = [False, True]

    augments = [(k, f) for k in rotate_k for f in flip]

    for (rotate_times, f) in augments:
        # augmentation and prediction
        augmented_board = board
        if f:
            augmented_board = torch.flip(augmented_board, dims=(2,))
        augmented_board = torch.rot90(augmented_board, k=rotate_times, dims=(1, 2))

        augmented_board = augmented_board.unsqueeze(dim=0)
        augmented_preds = net(augmented_board).squeeze()[:-1] # discard the PASS move

        
        augmented_preds = torch.softmax(augmented_preds, dim=0).view(govars.SIZE, govars.SIZE)

        # restore the prediction to the original coord system
        # note that it "have" to be done in the reverse order
        augmented_preds = torch.rot90(augmented_preds,k=-rotate_times, dims=(0, 1))
        if f:
            augmented_preds = torch.flip(augmented_preds, dims=(1,))

        augmented_preds = augmented_preds.flatten()
        preds += augmented_preds
    return preds / 8


def test_time_style(board, net, device):
    preds = torch.zeros((govars.STYLE_CAT,)).to(device)
    rotate_k = [0, 1, 2, 3] # rotate degree
    flip = [False, True]
    


    augments = [(k, f) for k in rotate_k for f in flip]
    for (rotate_times, f) in augments:
        augmented_board = board
        if f:
            augmented_board = torch.flip(augmented_board, dims=(2,))
        augmented_board = torch.rot90(augmented_board, k=rotate_times, dims=(2, 3))

        augmented_preds = net(augmented_board).squeeze()


        augmented_preds = torch.softmax(augmented_preds, dim=0)

        augmented_preds = augmented_preds.flatten()
        preds += augmented_preds
    return preds / len(augments)
        

def mask_moves(pred):
    pred[govars.PASS] = float('-inf') # mask the pass move
    return pred

def is_overlapped(rectangle1, rectangle2):
    x1, y1, width1, height1 = rectangle1
    x2, y2, width2, height2 = rectangle2

    if (x1 + width1 <= x2 or x2 + width2 <= x1 or y1 + height1 <= y2 or y2 + height2 <= y1):
        return False 
    else:
        return True  
    

def zero_board(board, top_left_y, top_left_x, cropped_h, cropped_w):
    board[govars.BLACK, top_left_y:top_left_y+cropped_h, top_left_x:top_left_x+cropped_w] = 0
    board[govars.WHITE, top_left_y:top_left_y+cropped_h, top_left_x:top_left_x+cropped_w] = 0
    board[govars.INVD_CHNL, top_left_y:top_left_y+cropped_h, top_left_x:top_left_x+cropped_w] = 0
    return board


def crop_board(board):
    last_move_row, last_move_col = np.where(board[-1] == 1)
    inhibit_size = govars.INHIBIT_CROP_SIZE
    inhibit_top_left_y, inhibit_top_left_x = last_move_row - inhibit_size // 2, last_move_col - inhibit_size // 2 


    cropped_h = np.random.randint(low=1, high=govars.CROP_SIZE + 1)
    cropped_w = np.random.randint(low=1, high=govars.CROP_SIZE + 1)

    while True:
        crop_top_left_y, crop_top_left_x = np.random.randint(low=0, high=govars.SIZE), np.random.randint(low=0, high=govars.SIZE)

        if crop_top_left_y + cropped_h - 1 > govars.SIZE - 1 or crop_top_left_x+ cropped_w - 1 > govars.SIZE - 1:
            # out of bound
            continue 

        if not is_overlapped((inhibit_top_left_x, inhibit_top_left_y, inhibit_size, inhibit_size), (crop_top_left_x, crop_top_left_y, cropped_w, cropped_h)):
            board = zero_board(board, crop_top_left_y, crop_top_left_x, cropped_h, cropped_w)
            break

    return board


def is_too_close(move2d, board):
    last_move_row, last_move_col = np.where(board[-1] == 1)
    inhibit_size = govars.INHIBIT_MOVE_SIZE
    inhibit_top_left_y, inhibit_top_left_x = last_move_row - inhibit_size // 2, last_move_col - inhibit_size // 2 
    inhibit_bottom_right_y, inhibit_bottom_right_x = last_move_row + inhibit_size // 2, last_move_col + inhibit_size // 2

    y, x = move2d

    if y < inhibit_top_left_y or x < inhibit_top_left_x or y > inhibit_bottom_right_y or x > inhibit_bottom_right_x:
        # outside the box
        return False

    return True

def random_moves(board):
    total_moves = np.count_nonzero(board[govars.BLACK]) + np.count_nonzero(board[govars.WHITE])

    max_random_moves = 10
    random_moves = np.random.randint(low=1, high=max_random_moves + 1)


    turn = gogame.turn(board)
    for r in range(random_moves):
        random_move1d = gogame.random_action(board)
        random_move2d =  random_move1d // govars.SIZE, random_move1d % govars.SIZE
        if random_move1d == govars.PASS or is_too_close(random_move2d, board):
            continue

        
        board[turn, random_move2d[0], random_move2d[1]] = 1
        board[govars.INVD_CHNL, random_move2d[0], random_move2d[1]] = 1
        turn = (turn + 1) % 2

    return board

def strong_augment(board, crop, rand_move):
    board = np.copy(board)

    if crop:
        board = crop_board(board)
    if rand_move:
        board = random_moves(board)
        
    board = gogame.random_symmetry(board)

    return board

def crop_move_as_center(game_features, region_size):
    pad_size = region_size
    half_crop_size = region_size // 2
    last_move_r, last_move_c = np.where(game_features[-1] == 1)

    

    padded_game_features = np.pad(game_features, pad_width=((0, 0), (pad_size, pad_size), (pad_size, pad_size)), constant_values=0)

    last_move_r += pad_size
    last_move_c += pad_size

    start_x = int(last_move_c - half_crop_size)
    end_x = int(last_move_c + half_crop_size + 1)

    start_y = int(last_move_r - half_crop_size)
    end_y = int(last_move_r + half_crop_size + 1)

    cropped_game_features = padded_game_features[:, start_y:end_y, start_x:end_x]


    return cropped_game_features.copy()


def pre_augmentation(games, labels, region_size=govars.PADDED_SIZE):
    game_features = []
    augmented_labels = []

    

    for i in range(len(games)):
        game = games[i]
        label = labels[i]
        sym_games = gogame.all_symmetries(game) # 8 times
        
        sym_len = len(sym_games)

        for i in range(sym_len):
            game = sym_games[i]
            flip = game.copy()
            temp = flip[govars.BLACK]
            flip[govars.BLACK] = flip[govars.WHITE]
            flip[govars.WHITE] = temp
            flip[govars.TURN_CHNL] = (flip[govars.TURN_CHNL] + 1) % 2
            sym_games.append(flip)

        game_features.append(sym_games)
        augmented_labels += [label for _ in range(len(sym_games))]





    
    game_features = np.array(game_features)
    game_features = game_features.reshape((-1, govars.FEAT_CHNLS, region_size, region_size))

    augmented_labels = np.array(augmented_labels)

    return game_features.copy(), augmented_labels.copy()