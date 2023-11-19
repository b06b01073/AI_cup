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
import torch.nn.functional as F
import gogame

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(net, game_feature, tta):
    with torch.no_grad():
        if tta:
            sym_games = np.array(gogame.all_symmetries(game_feature))
            last_position = torch.tensor(sym_games).to(device)
            pred = net(last_position)
            pred = F.softmax(pred, dim=1)
            pred = torch.mean(pred, dim=0)
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
        game_features.append(goutils.crop_move_as_center(go_env.game_features()))
    return np.array(game_features)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--ensemble_path', '-p', type=str, default='model_params/ensemble')
    parser.add_argument('--test_file', '-t', type=str)
    parser.add_argument('--tta', action='store_false')
    parser.add_argument('--output', type=str)

    args = parser.parse_args()
    ensemble_preds = 0
    games, file_names = GoParser.file_test_parser(args.test_file)

    # run this block for the first time
    # game_features = fetch_features(games) 
    # np.save('dataset/testing/play_style_11.npy', game_features)
    
    # game_features = np.load('dataset/testing/play_style_11.npy')

    with open(args.output, 'w') as f:
       
        for path in os.listdir(args.ensemble_path):
            net = torch.load(os.path.join(args.ensemble_path, path)).to(device)
            net.eval()

            preds = []
            for feature in tqdm(game_features, dynamic_ncols=True):
                preds.append(predict(net, feature, args.tta))

            ensemble_preds += torch.stack(preds)

        ensemble_preds /= len(os.listdir(args.ensemble_path)) # not necessary

        ensemble_preds = torch.argmax(ensemble_preds, dim=1)

        for ensemble_pred, file_name in zip(ensemble_preds, file_names):                
            f.write(f'{file_name},{ensemble_pred+1}\n')    
                    



                



    