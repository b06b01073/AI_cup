from argparse import ArgumentParser
import torch
import GoParser
from GoEnv import Go 
import govars
import goutils
from tqdm import tqdm
import torch
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(net, game_feature, tta):
    with torch.no_grad():
        last_position = torch.from_numpy(game_feature).unsqueeze(dim=0).to(device)


        if tta:
            pred = goutils.test_time_style(last_position, net, device)
        else:
            pred = net(last_position).squeeze()
            pred = torch.softmax(pred, dim=0)
        

        return pred

def fetch_features(games):
    game_features = []
    for game in tqdm(games):
        game = game.split(',')
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
        game_features.append(goutils.pad_board(go_env.game_features()))
    return game_features

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--ensemble_path', '-p', type=str, default='model_params/ensemble')
    parser.add_argument('--test_file', '-t', type=str)
    parser.add_argument('--tta', action='store_false')
    parser.add_argument('--output', type=str)

    args = parser.parse_args()
    ensemble_preds = None
    games, file_names = GoParser.file_test_parser(args.test_file)
    game_features = fetch_features(games)

    with open(args.output, 'w') as f:
       
        for path in os.listdir(args.ensemble_path):
            net = torch.load(os.path.join(args.ensemble_path, path)).to(device)
            net.eval()

            preds = []
            for feature in tqdm(game_features, dynamic_ncols=True):
                preds.append(predict(net, feature, args.tta))

            if ensemble_preds is None:
                ensemble_preds = torch.zeros((len(preds), govars.STYLE_CAT)).to(device)

            ensemble_preds += torch.stack(preds)

        ensemble_preds /= len(os.listdir(args.ensemble_path)) # not necessary

        ensemble_preds = torch.argmax(ensemble_preds, dim=1)

        for ensemble_pred, file_name in zip(ensemble_preds, file_names):                
            f.write(f'{file_name},{ensemble_pred+1}\n')    
                    



                



    