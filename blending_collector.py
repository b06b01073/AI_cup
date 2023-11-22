from argparse import ArgumentParser
import numpy as np
import GoParser
import torch
import gogame
from tqdm import tqdm
import goutils
from GoEnv import Go
import govars

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def fetch_features(games):
    game_features = []
    for game in tqdm(games):
        game = game.split(',')
        go_env = Go()

        last_move = 'W'

        for move in game:
            if move == '':
                break

            # handle the PASS scenario
            if move[0] == last_move:
                # there's an in-between pass move
                go_move, _ = goutils.move_encode(govars.PASS)
                go_env.make_move(go_move)
                

            go_move, _ = goutils.move_encode(move) # go_move is for the go env, moves[move_id] is the one hot vector

            go_env.make_move(go_move)
            last_move = move[0]
        try:
            game_features.append(goutils.crop_move_as_center(go_env.game_features(), region_size=13))
        except:
            print(game)
    return np.array(game_features)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--test_file', '-t', type=str, default='dataset/testing/play_style_test_public.csv')
    parser.add_argument('--output', type=str)

    args = parser.parse_args()
    games, file_names = GoParser.style_test_parser(args.test_file)

    game_features = fetch_features(games) 
    sym_game_features = []

    for game_feature in game_features:
        sym_games = np.array(gogame.all_symmetries(game_feature))
        sym_game_features.append(sym_games)

    blender = torch.load(args.model)
    blender.eval()
    blender.device = device

    with torch.no_grad():
        sym_game_features = np.array(sym_game_features)
        preds = blender.pred_proba_tta(sym_game_features)
        class_preds = torch.argmax(preds, dim=1)

    with open(args.output, 'w') as f:
        for pred, file_name in tqdm(zip(class_preds, file_names)):
            f.write(f'{file_name},{pred+1}\n')