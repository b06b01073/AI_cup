ANYONE = None
NOONE = -1

SIZE = 19
PADDED_W = 1
PADDED_SIZE = 19 + PADDED_W * 2
BOARD_SIZE = SIZE * SIZE
ACTION_SPACE = SIZE * SIZE + 1 
PASS = SIZE * SIZE

BLACK = 0
WHITE = 1
TURN_CHNL = 2 # if state[TURN_CHNL] == WHITE, then it is white to play
INVD_CHNL = 3 
PASS_CHNL = 4
DONE_CHNL = 5

NUM_CHNLS = 6

LAST_MOVE_PLANES = 4

FEAT_CHNLS = 10 # (black, white, turn, invalid, recent moves * 8, empty, ones)