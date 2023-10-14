from torch.utils.data import Dataset, DataLoader, Subset
import GoParser
from GoEnv import Go
import govars
import goutils
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

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

        

        # we use the try block here since there are invalid moves in the dataset
        try:
            last_move = 'W'
            for move in game:
                if move[0] == last_move:
                    # there's an in-between pass move
                    go_move, move_onehot = goutils.move_encode(govars.PASS)
                    game_states.append(goutils.pad_board(go_env.game_features()))
                    # game_states.append(go_env.game_features())
                    moves.append(move_onehot)
                    go_env.make_move(go_move)
                    

                go_move, move_onehot = goutils.move_encode(move) # go_move is for the go env, moves[move_id] is the one hot vector

                # got to make sure the state is pushed to the list before state update
                game_states.append(goutils.pad_board(go_env.game_features())) 
                moves.append(move_onehot)
                go_env.make_move(go_move)

                last_move = move[0]
        except:
            pass
        
        return np.array(game_states, dtype=np.float32), np.array(moves, dtype=np.float32)


def get_loader(path, split):
    dataset = GoDataset(path)
    train_len = int(len(dataset) * split)
    train_set = Subset(dataset, range(0, train_len))
    val_set = Subset(dataset, range(train_len, len(dataset)))

    return DataLoader(train_set, batch_size=1, shuffle=False, num_workers=10), DataLoader(val_set, batch_size=1, shuffle=False, num_workers=10)


if __name__ == '__main__':
    dataset = GoDataset('./dataset/training/dan_train.csv')
    states, moves = dataset[0]
    for (state, move) in zip(states, moves):
        print(f'move: {move}')
        print(f'state: {state}')