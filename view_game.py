from argparse import ArgumentParser
from GoEnv import Go
import goutils
import govars

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_path', '-f', type=str)
    parser.add_argument('--entry', '-e', type=str)

    args = parser.parse_args()
    
    df = open(args.file_path).read().splitlines()
    games = [i.split(',', 2) for i in df]
    for game in games:
        if game[0] == args.entry:
            game = game[2].split(',')
            go_env = Go()
            try:
                last_move = 'W'
                for move in game:
                    print(game)

                    # handle the PASS scenario
                    if move[0] == last_move:
                        # there's an in-between pass move
                        go_move, move_onehot = goutils.move_encode(govars.PASS)

                        print('pass')
                        input()
                        go_env.make_move(go_move)
                        go_env.render()

                    go_move, move_onehot = goutils.move_encode(move) # go_move is for the go env, moves[move_id] is the one hot vector

                    print(move)
                    input()
                    go_env.make_move(go_move)
                    go_env.render()

                    last_move = move[0]
            except:
                print('invalid move')