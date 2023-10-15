from argparse import ArgumentParser
import torch
import GoParser
from GoEnv import Go 
import govars
import goutils
from tqdm import tqdm


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--task', type=str)

    args = parser.parse_args()
    # net = torch.load(args.model_path).to(device)

    games, file_names = GoParser.file_test_parser(args.test_file)

    for game, file_name in tqdm(zip(games, file_names)):
        game = game.split(',')
        go_env = Go()

        last_move = 'W'

        try:
            for move in game:

                # handle the PASS scenario
                if move[0] == last_move:
                    # there's an in-between pass move
                    go_move, _ = goutils.move_encode(govars.PASS)
                    go_env.make_move(go_move)
                    

                go_move, _ = goutils.move_encode(move) # go_move is for the go env, moves[move_id] is the one hot vector

                go_env.make_move(go_move)
                last_move = move[0]
        except:
            print(file_name)
        
        # last_position = goutils.pad_board(go_env.game_features())
        # last_position = torch.from_numpy(last_position).unsqueeze(dim=0).to(device)

        # with torch.no_grad():
        #     pred = net(last_position)
        #     pred = torch.softmax(pred, dim=1)
        # print(pred)
        


        