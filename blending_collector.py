from argparse import ArgumentParser
import numpy as np
import GoParser
import torch
import gogame
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--test_file', '-t', type=str, default='dataset/testing/play_style_test_public.csv')
    parser.add_argument('--output', type=str)

    args = parser.parse_args()
    games, file_names = GoParser.file_test_parser(args.test_file)

    game_features = np.load('dataset/testing/play_style_13.npy')
    sym_game_features = []

    for game_feature in game_features:
        sym_games = np.array(gogame.all_symmetries(game_feature))
        sym_game_features.append(sym_games)

    blender = torch.load(args.model)
    blender.eval()

    with torch.no_grad():
        sym_game_features = np.array(sym_game_features)
        preds = blender.pred_proba_tta(sym_game_features)
        class_preds = torch.argmax(preds, dim=1)

    with open(args.output, 'w') as f:
        for pred, file_name in tqdm(zip(class_preds, file_names)):
            f.write(f'{file_name},{pred+1}\n')