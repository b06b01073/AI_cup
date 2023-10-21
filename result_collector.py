from argparse import ArgumentParser
import torch
import GoParser
from GoEnv import Go 
import govars
import goutils
from tqdm import tqdm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict(net, game_feature, file_name, visualize):
    last_position = goutils.pad_board(game_feature)
    last_position = torch.from_numpy(last_position).to(device)

    with torch.no_grad():
        # Do TTA here
        pred = goutils.test_time_predict(last_position, net, device)
        top_moves = torch.topk(pred, k=5).indices

        top_moves_coord = [adjust_coor(goutils.move_decode_char(top_move)) for top_move in top_moves]

        if visualize:
            go_env.render()
            print(top_moves_coord)
            input('next?')

        print(f'{file_name},{top_moves_coord[0]},{top_moves_coord[1]},{top_moves_coord[2]},{top_moves_coord[3]},{top_moves_coord[4]}')
        # f.write(f'file')

def adjust_coor(coord):
    # it's kinda akward, I built the coord system in a wrong way, the correct move should be mirrored along the top-left to the bottem-right diagonal, the adjustment is to swap the coodinate
    return coord[1] + coord[0]

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model_path', '-m',type=str)
    parser.add_argument('--test_file', '-t', type=str)
    parser.add_argument('--visualize', '-v', action='store_true')
    parser.add_argument('--nag', '-n', action='store_true')

    args = parser.parse_args()
    net = torch.load(args.model_path).to(device)
    net.eval()

    games, file_names = GoParser.file_test_parser(args.test_file)

    net = torch.load(args.model_path)
    for game, file_name in zip(games, file_names):
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

            if args.nag:
                predict(net, go_env.game_features(), file_name, True)

            go_env.make_move(go_move)
            last_move = move[0]

        predict(net, go_env.game_features(), file_name, args.visualize)
        
        



        



    