from torch.utils.data import Dataset, DataLoader
import GoParser
from GoEnv import Go
import govars
import goutils
import sys
import numpy as np
import gogame

np.set_printoptions(threshold=sys.maxsize)

class GoDataset(Dataset):
    def __init__(self, games, augment):
        self.games = games
        self.augment = augment

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

                # handle the PASS scenario
                if move[0] == last_move:
                    # there's an in-between pass move
                    go_move, move_onehot = goutils.move_encode(govars.PASS)
                    game_features = go_env.game_features()

                    if self.augment:
                        sym_game_features, sym_move_onehot = gogame.random_symmetry(game_features, move_onehot)
                        game_states.append(goutils.pad_board(sym_game_features))
                        moves.append(sym_move_onehot)
                    else:
                        game_states.append(goutils.pad_board(game_features))
                        moves.append(move_onehot)

                    go_env.make_move(go_move)
                    

                go_move, move_onehot = goutils.move_encode(move) # go_move is for the go env, moves[move_id] is the one hot vector

                # got to make sure the state is pushed to the list before state update
                game_features = go_env.game_features()
                if self.augment:
                    sym_game_features, sym_move_onehot = gogame.random_symmetry(game_features, move_onehot)
                    game_states.append(goutils.pad_board(sym_game_features))
                    moves.append(sym_move_onehot)
                else:
                    game_states.append(goutils.pad_board(game_features))
                    moves.append(move_onehot)

                go_env.make_move(go_move)

                last_move = move[0]
        except:
            pass
        
        return np.array(game_states, dtype=np.float32), np.array(moves, dtype=np.float32)


def get_loader(path, split):
    games = GoParser.file_parser(path)
    train_len = int(len(games) * split)
    train_games = games[:train_len]
    val_games = games[train_len:]
    train_dataset = GoDataset(train_games, augment=True)
    val_dataset = GoDataset(val_games, augment=False)

    return DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=12), DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=12)


if __name__ == '__main__':
    dataset = GoDataset('./dataset/training/dan_train.csv')
    states, moves = dataset[0]
    for (state, move) in zip(states, moves):
        print(f'move: \n{goutils.move_2d_encode(np.argmax(move))}')
        print(f'state:\n {gogame.str(state[:, 1:-1, 1:-1])}')