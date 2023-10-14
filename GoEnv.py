import numpy as np
import gogame
import govars

class Go:
    def __init__(self, size=govars.SIZE):
        self.game_state = self.reset_game(size=size)
        self.recent_moves = np.zeros((govars.LAST_MOVE_PLANES, govars.SIZE, govars.SIZE))
        self.empty_plane = np.ones((govars.SIZE, govars.SIZE))
        self.invalid_move = np.zeros((govars.SIZE, govars.SIZE))
        self.one_plane = np.zeros((govars.SIZE, govars.SIZE))

    def reset_game(self, size):
        self.recent_moves = np.zeros((govars.SIZE, govars.SIZE))
        self.empty_plane = np.ones((govars.SIZE, govars.SIZE))
        self.invalid_move = np.zeros((govars.SIZE, govars.SIZE))
        self.one_plane = np.zeros((govars.SIZE, govars.SIZE))
        return gogame.init_state(size=size)
    

    def build_game_feature(self, action1d):
        game_state = np.copy(self.game_state)

        # build the plane of empty
        black = game_state[govars.BLACK].astype(np.int32)
        white = game_state[govars.WHITE].astype(np.int32)
        self.empty_plane = 1 - np.bitwise_or(black, white)
        # print(self.empty_plane)



        # build the planes of recent moves
        last_move = np.zeros((govars.SIZE, govars.SIZE))
        if action1d != govars.PASS:
            action2d = action1d // govars.SIZE, action1d % govars.SIZE
            last_move[action2d] = 1
        self.recent_moves[:-1] = self.recent_moves[1:] 
        self.recent_moves[-1] = last_move
        # print(self.recent_moves)


        
        
        # TODO: build the planes of liberty

        # TODO: Capture size

        # TODO: Self-atari size




    def make_move(self, action1d):
        self.game_state = gogame.next_state(self.game_state, action1d)


        self.build_game_feature(action1d)

        return np.copy(self.game_state)
    
    def render(self):
        return print(gogame.str(self.game_state))
    
    def game_features(self):
        '''Do feature extraction here
        '''
        game_feature = np.stack((
            self.game_state[govars.BLACK],
            self.game_state[govars.WHITE],
            self.game_state[govars.TURN_CHNL],
            self.game_state[govars.INVD_CHNL],
            self.empty_plane,
            self.one_plane,
        ), axis=0)

        game_feature = np.concatenate((game_feature, self.recent_moves), axis=0)


        return np.copy(game_feature)
    

    def set_state(self, state):
        # note that this doesn't set the game feature, use this will caution
        self.game_state = np.copy(state)
    
    def get_state(self):
        return np.copy(self.game_state)
    
