from torch.utils.data import Dataset, DataLoader
import GoParser
from GoEnv import Go
import govars
import goutils
import sys
import numpy as np
import gogame
from tqdm import tqdm

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
                    sym_game_features, sym_move = gogame.random_symmetry(game_features, move_onehot)
                    game_states.append(goutils.pad_board(sym_game_features))
                    moves.append(sym_move)
                else:
                    game_states.append(goutils.pad_board(game_features))
                    moves.append(go_move)

                go_env.make_move(go_move)
                

            go_move, move_onehot = goutils.move_encode(move) # go_move is for the go env, moves[move_id] is the one hot vector

            # got to make sure the state is pushed to the list before state update
            game_features = go_env.game_features()
            if self.augment:
                sym_game_features, sym_move = gogame.random_symmetry(game_features, move_onehot)
                game_states.append(goutils.pad_board(sym_game_features))
                moves.append(sym_move)
            else:
                game_states.append(goutils.pad_board(game_features))
                moves.append(go_move)

            go_env.make_move(go_move)

            last_move = move[0]
    
        return np.array(game_states, dtype=np.float32), np.array(moves, dtype=np.int_)


class StyleDataset(Dataset):
    def __init__(self, labels, games, augment):
        self.labels = labels
        self.games = games
        self.augment = augment

    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, idx):
        game = self.games[idx]
        label = self.labels[idx]

        if self.augment:
            return gogame.random_symmetry(game), label
        else:
            return game, label


class GoMatchDataset(Dataset):
    def __init__(self, labels, games, crop=True, rand_move=True, weak_augment=None, strong_augment=None, unlabeled_size=None, train=False):
        self.labels = labels
        self.games = games 
        self.weak_augment = weak_augment
        self.strong_augment = strong_augment
        self.unlabeled_size = unlabeled_size
        self.train = train
        self.crop = crop
        self.rand_move = rand_move

        print(f'Crop board: {self.crop}, Rand move: {self.rand_move}')
    

    def __len__(self):
        return len(self.games)
    

    def __getitem__(self, idx):
        game = self.games[idx].split(',')
        label = self.labels[idx] - 1
        
        go_env = Go()
        game_len = len(game)

        

        if self.train:
            unlabeled_indices = np.random.choice(game_len, size=self.unlabeled_size, replace=True)

            unlabeled_count = np.bincount(unlabeled_indices)

            unlabeled_states = []


        last_move = 'W'
        for idx, move in enumerate(game):

            # handle the PASS scenario
            if move[0] == last_move:
                # there's an in-between pass move
                go_move, _ = goutils.move_encode(govars.PASS)
                go_env.make_move(go_move)
                

            go_move, _ = goutils.move_encode(move) # go_move is for the go env, moves[move_id] is the one hot vector

            go_env.make_move(go_move)
            last_move = move[0]

            
            # push unlabeled data into unlabeled_states
            if self.train:
                if idx in unlabeled_indices:
                    for _ in range(unlabeled_count[idx]):
                        unlabeled_states.append(go_env.game_features())
        
        labeled_state = go_env.game_features()


        # test set
        if not self.train:
            return goutils.pad_board(labeled_state), label
        
        # train set
        labeled_state = goutils.pad_board(self.weak_augment(labeled_state))

        weak_augmented_states = []
        strong_augmented_states = []
        for unlabeled_state in unlabeled_states:
            weak_augmented_state = goutils.pad_board(self.weak_augment(unlabeled_state))
            strong_augmented_state = goutils.pad_board(self.strong_augment(unlabeled_state, self.crop, self.rand_move))

            weak_augmented_states.append(weak_augmented_state)
            strong_augmented_states.append(strong_augmented_state)
        

        return labeled_state, label, np.array(weak_augmented_states), np.array(strong_augmented_states)



def go_match_loader(path, split, unlabeled_size, batch_size, crop, rand_move):
    labels, games = GoParser.style_parser(path)

    indices = np.arange(len(games))
    np.random.shuffle(indices) # inplace op
    
    train_len = int(len(games) * split)
    train_indices = indices[:train_len]
    test_indices = indices[train_len:]

    assert np.intersect1d(train_indices, test_indices).size == 0 # just to make sure :)

    train_labels, train_games = np.array(labels)[train_indices], np.array(games)[train_indices]
    test_labels, test_games = np.array(labels)[test_indices], np.array(games)[test_indices]

    # label_dist = np.bincount(labels)
    # train_label_dist = np.bincount(train_labels)
    # print(f'label dist: {label_dist}, train_label_dist: {train_label_dist}')


    train_dataset = GoMatchDataset(
        train_labels,
        train_games, 
        weak_augment=gogame.random_symmetry,
        strong_augment=goutils.strong_augment,
        crop=crop,
        rand_move=rand_move,
        unlabeled_size=unlabeled_size, 
        train=True
    )

    test_dataset = GoMatchDataset(
        test_labels, 
        test_games, 
        train=False
    )

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6), DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)


def get_loader(path, split, bootstrap=False):
    games = GoParser.file_parser(path)
    train_len = int(len(games) * split)
    train_games = games[:train_len]
    if bootstrap:
        train_games = bootstrap(train_games)

    val_games = games[train_len:]
    train_dataset = GoDataset(train_games, augment=True)
    test_dataset = GoDataset(val_games, augment=False)

    return DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=12), DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=12)



if __name__ == '__main__':
    # dataset = GoDataset('./dataset/training/dan_train.csv')
    # states, moves = dataset[0]
    # for (state, move) in zip(states, moves):
        # print(f'move: \n{goutils.move_2d_encode(np.argmax(move))}')
        # print(f'state:\n {gogame.str(state[:, 1:-1, 1:-1])}')

    labels, games = GoParser.style_parser('./dataset/training/play_style_train.csv')
    dataset = StyleDataset(labels, games, augment=False)
    
    games = []
    labels = []
    

    for game, label in tqdm(dataset):
        games.append(game)
        labels.append(label)

    np.save('dataset/training/games.npy' , np.array(games))
    np.save('dataset/training/labels.npy', np.array(labels))