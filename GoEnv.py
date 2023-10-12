import numpy as np
import gogame
import govars

class Go:
    def __init__(self, size=govars.SIZE):
        self.game_state = self.reset_game(size=size)

    def reset_game(self, size):
        return gogame.init_state(size=size)
    
    def make_move(self, pos):
        self.game_state = gogame.next_state(self.game_state, pos)

        return np.copy(self.game_state)
    
    def render(self):
        return print(gogame.str(self.game_state))
    
    def game_features(self):
        '''Do feature extraction here
        '''
        return np.copy(self.game_state)
    
    def get_state(self):
        return np.copy(self.game_state)
    
