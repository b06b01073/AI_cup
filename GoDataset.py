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

        

        # the invalid moves are replaced by a PASS move
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
    
        return np.array(game_states, dtype=np.float32), np.array(moves, dtype=np.float32)


class StyleDataset(Dataset):
    def __init__(self, labels, games, augment):
        self.labels = labels
        self.games = games
        self.augment = augment

    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, idx):
        game = self.games[idx].split(',')
        go_env = Go()

        last_move = 'W'
        for move in game:

            # handle the PASS scenario
            if move[0] == last_move:
                # there's an in-between pass move
                go_move, _ = goutils.move_encode(govars.PASS)
                go_env.make_move(go_move)
                

            go_move, _ = goutils.move_encode(move) # go_move is for the go env, moves[move_id] is the one hot vector

            go_env.make_move(go_move)
            last_move = move[0]
        
        game_features = go_env.game_features().astype(np.float32) # last 2 position


        label_onehot = np.zeros((3, ), dtype=np.float32) # get the label
        label_onehot[self.labels[idx] - 1] = 1

        if self.augment:
            # it seems like we can swap the black and white channels etc
            sym_game_features = gogame.random_symmetry(game_features)
            sym_game_features = goutils.flip_board(sym_game_features)
            return goutils.pad_board(sym_game_features), label_onehot
        else:
            return goutils.pad_board(game_features), label_onehot
            

def get_loader(path, split, bootstrap=False):
    games = GoParser.file_parser(path)
    train_len = int(len(games) * split)
    train_games = games[:train_len]
    if bootstrap:
        train_games = bootstrap(train_games)

    val_games = games[train_len:]
    train_dataset = GoDataset(train_games, augment=True)
    test_dataset = GoDataset(val_games, augment=False)

    return DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=12), DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=12)


def style_loader(path, split):
    labels, games = GoParser.style_parser(path)
    train_len = int(len(games) * split)
    train_labels, train_games = labels[:train_len], games[:train_len]
    test_labels, test_games = labels[train_len:], games[train_len:]

    train_dataset = StyleDataset(train_labels, train_games, augment=True)
    test_dataset = StyleDataset(test_labels, test_games, augment=False)

    return DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=6), DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=6)


def bootstrap(games, num_samples):
    # note that the games "need to be split already"
    bootstrap_games = []
    for i in range(num_samples):
        game_index = np.random.randint(0, len(games))
        bootstrap_games.append(games[game_index])

    return bootstrap_games


if __name__ == '__main__':
    # dataset = GoDataset('./dataset/training/dan_train.csv')
    # states, moves = dataset[0]
    # for (state, move) in zip(states, moves):
        # print(f'move: \n{goutils.move_2d_encode(np.argmax(move))}')
        # print(f'state:\n {gogame.str(state[:, 1:-1, 1:-1])}')
    labels, games = GoParser.style_parser('./dataset/training/play_style_train.csv')
    dataset = StyleDataset(labels, games, augment=False)
    print(dataset[0])