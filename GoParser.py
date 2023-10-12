from tqdm import tqdm
from GoEnv import Go
import goutils
import govars

def file_parser(path, sanitize=True):
    df = open(path).read().splitlines()
    games = [i.split(',', 2)[-1] for i in df]
    file_names = [i.split(',')[0] for i in df]

    return games, file_names 


def game_parser(game):
    pass
