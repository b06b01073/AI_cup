import numpy as np
from scipy import ndimage
from sklearn import preprocessing

import state_utils, govars

"""
The state of the game is a numpy array
* Are values are either 0 or 1

* Shape [NUM_CHNLS, SIZE, SIZE]

the number represent the channel in the feature plan
0 - Black pieces
1 - White pieces
2 - Turn (0 - black, 1 - white)
3 - Invalid moves (including ko-protection)
4 - Previous move was a pass
5 - Game over
"""


def init_state(size):
    # return initial board (numpy board)
    state = np.zeros((govars.NUM_CHNLS, size, size))
    return state


def batch_init_state(batch_size, board_size):
    # return initial board (numpy board)
    batch_state = np.zeros((batch_size, govars.NUM_CHNLS, board_size, board_size))
    return batch_state


def next_state(state, action1d, canonical=False):
    # Deep copy the state to modify
    state = np.copy(state)

    # Initialize basic variables
    board_shape = state.shape[1:]
    pass_idx = np.prod(board_shape)
    passed = action1d == pass_idx
    action2d = action1d // board_shape[0], action1d % board_shape[1]

    player = turn(state)
    previously_passed = prev_player_passed(state)
    ko_protect = None

    if passed:
        # We passed
        state[govars.PASS_CHNL] = 1
        if previously_passed:
            # Game ended
            state[govars.DONE_CHNL] = 1
    else:
        # Move was not pass
        state[govars.PASS_CHNL] = 0

        # Assert move is valid
        assert state[govars.INVD_CHNL, action2d[0], action2d[1]] == 0, ("Invalid move", action2d)

        # Add piece
        state[player, action2d[0], action2d[1]] = 1

        # Get adjacent location and check whether the piece will be surrounded by opponent's piece
        adj_locs, surrounded = state_utils.adj_data(state, action2d, player)

        # Update pieces
        killed_groups = state_utils.update_pieces(state, adj_locs, player)

        # If only killed one group, and that one group was one piece, and piece set is surrounded,
        # activate ko protection
        if len(killed_groups) == 1 and surrounded:
            killed_group = killed_groups[0]
            if len(killed_group) == 1:
                ko_protect = killed_group[0]

    # Update invalid moves, we ignore the invalid moves for now
    state[govars.INVD_CHNL] = state_utils.compute_invalid_moves(state, player, ko_protect)

    # Switch turn
    state_utils.set_turn(state)

    if canonical:
        # Set canonical form
        state = canonical_form(state)

    return state



def invalid_moves(state):
    # return a fixed size binary vector
    if game_ended(state):
        return np.zeros(action_size(state))
    return np.append(state[govars.INVD_CHNL].flatten(), 0)


def valid_moves(state):
    return 1 - invalid_moves(state)




def action_size(state=None, board_size: int = None):
    # return number of actions
    if state is not None:
        m, n = state.shape[1:]
    elif board_size is not None:
        m, n = board_size, board_size
    else:
        raise RuntimeError('No argument passed')
    return m * n + 1


def prev_player_passed(state):
    return np.max(state[govars.PASS_CHNL] == 1) == 1



def game_ended(state):
    """
    :param state:
    :return: 0/1 = game not ended / game ended respectively
    """
    m, n = state.shape[1:]
    return int(np.count_nonzero(state[govars.DONE_CHNL] == 1) == m * n)


def batch_game_ended(batch_state):
    """
    :param batch_state:
    :return: 0/1 = game not ended / game ended respectively
    """
    return np.max(batch_state[:, govars.DONE_CHNL], axis=(1, 2))


def winning(state, komi=0):
    black_area, white_area = areas(state)
    area_difference = black_area - white_area
    komi_correction = area_difference - komi

    return np.sign(komi_correction)



def turn(state):
    """
    :param state:
    :return: Who's turn it is (govars.BLACK/govars.WHITE)
    """
    return int(np.max(state[govars.TURN_CHNL]))



def liberties(state: np.ndarray):
    blacks = state[govars.BLACK]
    whites = state[govars.WHITE]
    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)

    liberty_list = []
    for player_pieces in [blacks, whites]:
        liberties = ndimage.binary_dilation(player_pieces, state_utils.surround_struct)
        liberties *= (1 - all_pieces).astype(np.bool)
        liberty_list.append(liberties)

    return liberty_list[0], liberty_list[1]


def num_liberties(state: np.ndarray):
    black_liberties, white_liberties = liberties(state)
    black_liberties = np.count_nonzero(black_liberties)
    white_liberties = np.count_nonzero(white_liberties)

    return black_liberties, white_liberties


def areas(state):
    '''
    Return black area, white area
    '''

    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)
    empties = 1 - all_pieces

    empty_labels, num_empty_areas = ndimage.measurements.label(empties)

    black_area, white_area = np.sum(state[govars.BLACK]), np.sum(state[govars.WHITE])
    for label in range(1, num_empty_areas + 1):
        empty_area = empty_labels == label
        neighbors = ndimage.binary_dilation(empty_area)
        black_claim = False
        white_claim = False
        if (state[govars.BLACK] * neighbors > 0).any():
            black_claim = True
        if (state[govars.WHITE] * neighbors > 0).any():
            white_claim = True
        if black_claim and not white_claim:
            black_area += np.sum(empty_area)
        elif white_claim and not black_claim:
            white_area += np.sum(empty_area)

    return black_area, white_area


def canonical_form(state):
    state = np.copy(state)
    if turn(state) == govars.WHITE:
        channels = np.arange(govars.NUM_CHNLS)
        channels[govars.BLACK] = govars.WHITE
        channels[govars.WHITE] = govars.BLACK
        state = state[channels]
        state_utils.set_turn(state)
    return state

def random_symmetry(image):
    """
    Returns a random symmetry of the image
    :param image: A (C, BOARD_SIZE, BOARD_SIZE) numpy array, where C is any number
    :return:
    """
    orientation = np.random.randint(0, 8)

    if (orientation >> 0) % 2:
        # Horizontal flip
        image = np.flip(image, 2)
    if (orientation >> 1) % 2:
        # Vertical flip
        image = np.flip(image, 1)
    if (orientation >> 2) % 2:
        # Rotate 90 degrees
        image = np.rot90(image, axes=(1, 2))

    return image


def all_symmetries(image):
    """
    :param image: A (C, BOARD_SIZE, BOARD_SIZE) numpy array, where C is any number
    :return: All 8 orientations that are symmetrical in a Go game over the 2nd and 3rd axes
    (i.e. rotations, flipping and combos of them)
    """
    symmetries = []

    for i in range(8):
        x = image
        if (i >> 0) % 2:
            # Horizontal flip
            x = np.flip(x, 2)
        if (i >> 1) % 2:
            # Vertical flip
            x = np.flip(x, 1)
        if (i >> 2) % 2:
            # Rotation 90 degrees
            x = np.rot90(x, axes=(1, 2))
        symmetries.append(x)

    return symmetries


def random_weighted_action(move_weights):
    """
    Assumes all invalid moves have weight 0
    Action is 1D
    Expected shape is (NUM OF MOVES, )
    """
    move_weights = preprocessing.normalize(move_weights[np.newaxis], norm='l1')
    return np.random.choice(np.arange(len(move_weights[0])), p=move_weights[0])


def random_action(state):
    """
    Assumed to be (NUM_CHNLS, BOARD_SIZE, BOARD_SIZE)
    Action is 1D
    """
    invalid_moves = state[govars.INVD_CHNL].flatten()
    invalid_moves = np.append(invalid_moves, 0)
    move_weights = 1 - invalid_moves

    return random_weighted_action(move_weights)


def str(state):
    board_str = ''

    size = state.shape[1]
    board_str += '\t'
    for i in range(size):
        board_str += '{}'.format(i).ljust(2, ' ')
    board_str += '\n'
    for i in range(size):
        board_str += '{}\t'.format(i)
        for j in range(size):
            if state[0, i, j] == 1:
                board_str += '○'
                if j != size - 1:
                    if i == 0 or i == size - 1:
                        board_str += '═'
                    else:
                        board_str += '─'
            elif state[1, i, j] == 1:
                board_str += '●'
                if j != size - 1:
                    if i == 0 or i == size - 1:
                        board_str += '═'
                    else:
                        board_str += '─'
            else:
                if i == 0:
                    if j == 0:
                        board_str += '╔═'
                    elif j == size - 1:
                        board_str += '╗'
                    else:
                        board_str += '╤═'
                elif i == size - 1:
                    if j == 0:
                        board_str += '╚═'
                    elif j == size - 1:
                        board_str += '╝'
                    else:
                        board_str += '╧═'
                else:
                    if j == 0:
                        board_str += '╟─'
                    elif j == size - 1:
                        board_str += '╢'
                    else:
                        board_str += '┼─'
        board_str += '\n'

    black_area, white_area = areas(state)
    done = game_ended(state)
    ppp = prev_player_passed(state)
    t = turn(state)
    if done:
        game_state = 'END'
    elif ppp:
        game_state = 'PASSED'
    else:
        game_state = 'ONGOING'
    board_str += '\tTurn: {}, Game State (ONGOING|PASSED|END): {}\n'.format('BLACK' if t == 0 else 'WHITE', game_state)
    board_str += '\tBlack Area: {}, White Area: {}\n'.format(int(black_area), int(white_area))
    return board_str
