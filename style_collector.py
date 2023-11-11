from argparse import ArgumentParser
import torch
import GoParser
from GoEnv import Go 
import govars
import goutils
from tqdm import tqdm
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(net, game_feature, file_name, visualize, tta):
    last_position = goutils.pad_board(game_feature)
    last_position = torch.from_numpy(last_position).unsqueeze(dim=0).to(device)

    if tta:
        pred = goutils.test_time_style(last_position, net, device)
    else:
        pred = net(last_position).squeeze()
        pred = torch.softmax(pred, dim=0)
        
    style = torch.argmax(pred).item() + 1

    if visualize:
        go_env.render()
        print(style)
        input('next?')

    return f'{file_name},{style}'
    # f.write(f'file')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model_path', '-m',type=str)
    parser.add_argument('--test_file', '-t', type=str)
    parser.add_argument('--visualize', '-v', action='store_true')
    parser.add_argument('--tta', action='store_false')
    parser.add_argument('--output', type=str)

    args = parser.parse_args()
    net = torch.load(args.model_path).to(device)
    net.eval()

    games, file_names = GoParser.file_test_parser(args.test_file)

    with open(args.output, 'w') as f:
        for game, file_name in tqdm(zip(games, file_names)):
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

            pred = predict(net, go_env.game_features(), file_name, args.visualize, args.tta)
            f.write(f'{pred}\n')
            
            



            



    