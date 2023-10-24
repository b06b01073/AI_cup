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

ORI_FEAT_CHANNEL = 6 # (black, white, turn, invalid, empty, ones)

LAST_MOVE_PLANES = 4
CAPTURE_PLANES = 4
SELF_ATARI_PLANES = 4
LIBERTY_PLANES = 8

FEAT_CHNLS = ORI_FEAT_CHANNEL + LAST_MOVE_PLANES # (black, white, turn, invalid, recent moves * 8, empty, ones)

STYLE_CAT = 3