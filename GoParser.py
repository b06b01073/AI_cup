def file_parser(path):
    df = open(path).read().splitlines()
    games = [i.split(',', 2)[-1] for i in df]
    return games
