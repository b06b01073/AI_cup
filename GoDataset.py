from torch.utils.data import Dataset, DataLoader, Subset
import GoParser
from GoEnv import Go
import govars
import goutils

import numpy as np

class GoDataset(Dataset):
    def __init__(self, path):
        self.games, self.file_names = GoParser.file_parser(path)

    def __len__(self):
        return len(self.games)
    
    def __getitem__(self, idx):
        game = self.games[idx].split(',')
        go_env = Go()
        game_states = []
        moves = []

        # goutils.debug_game(game)

        last_move = 'W'
        try:
            for move_id, move in enumerate(game):
                if move[0] == last_move:
                    # there's an in-between pass move
                    go_move, move_onehot = goutils.move_encode(govars.PASS)
                    game_states.append(goutils.pad_board(go_env.get_state()))
                    # game_states.append(go_env.get_state())
                    moves.append(move_onehot)
                    go_env.make_move(go_move)
                    

                go_move, move_onehot = goutils.move_encode(move) # go_move is for the go env, moves[move_id] is the one hot vector

                # got to make sure the state is pushed to the list before state update
                # print(np.pad(go_env.get_state(), (0, govars.PADDED_W, govars.PADDED_W)))
                game_states.append(goutils.pad_board(go_env.get_state())) 
                # game_states.append(go_env.get_state())
                moves.append(move_onehot)
                go_env.make_move(go_move)
                last_move = move[0]
        except:
            pass
        
        return np.array(game_states, dtype=np.float32), np.array(moves, dtype=np.float32)


def get_loader(path, split=0.9):
    dataset = GoDataset(path)
    train_len = int(len(dataset) * split)
    train_set = Subset(dataset, range(0, train_len))
    val_set = Subset(dataset, range(train_len, len(dataset)))

    return DataLoader(train_set, batch_size=1, shuffle=False, num_workers=6), DataLoader(val_set, batch_size=1, shuffle=False, num_workers=6)


if __name__ == '__main__':
    dataset = GoDataset('./dataset/training/dan_train.csv')
    c = dataset[0]