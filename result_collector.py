from argparse import ArgumentParser
import torch
import GoParser
from GoEnv import Go 
import govars
import goutils
from tqdm import tqdm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict(net, game_feature, visualize):
    last_position = goutils.pad_board(game_feature)
    last_position = torch.from_numpy(last_position).to(device)

    with torch.no_grad():
        # Do TTA here
        pred = goutils.test_time_predict(last_position, net, device)
        # top_moves = torch.topk(pred, k=5).indices

        # top_moves_coord = [adjust_coor(goutils.move_decode_char(top_move)) for top_move in top_moves]

        if visualize:
            go_env.render()
            # print(top_moves_coord)
            input('next?')

        # f.write(f'file')
        return pred

def adjust_coor(coord):
    # it's kinda akward, I built the coord system in a wrong way, the correct move should be mirrored along the top-left to the bottem-right diagonal, the adjustment is to swap the coodinate
    return coord[1] + coord[0]

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model_paths', '-m', nargs='+',type=str)
    parser.add_argument('--test_file', '-t', type=str)
    parser.add_argument('--visualize', '-v', action='store_true')
    parser.add_argument('--output', '-o', type=str)

    args = parser.parse_args()

    games, file_names, pred_player = GoParser.file_test_parser(args.test_file)
    games = games
    file_name = file_names

    # ensemble actually shows no improvment currently
    preds = [[file_name, torch.zeros((govars.ACTION_SPACE-1,)).to(device)] for file_name in file_names]

    for model_path in args.model_paths:
        net = torch.load(model_path)
        net.to(device)
        net.eval()
        for idx, (game, pred_player) in enumerate(tqdm(zip(games, pred_player), desc=model_path)):
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

            if last_move == pred_player:
                # the last move of the game is a PASS move from non pred_player
                print('pass')
                go_move, _ = goutils.move_encode(govars.PASS)
                go_env.make_move(go_move)

            pred = predict(net, go_env.game_features(), args.visualize)
            
            preds[idx][1] += pred



    with open(args.output, 'w') as f:
        for file_name, pred in preds:
            top_moves = torch.topk(pred, k=5).indices

            top_moves_coord = [adjust_coor(goutils.move_decode_char(top_move)) for top_move in top_moves]

            
            f.write(f'{file_name},{top_moves_coord[0]},{top_moves_coord[1]},{top_moves_coord[2]},{top_moves_coord[3]},{top_moves_coord[4]}\n')


            



    