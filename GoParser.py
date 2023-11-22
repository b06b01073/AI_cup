def file_parser(path):
    df = open(path).read().splitlines()
    games = [i.split(',', 2)[-1] for i in df]
    return games

def style_parser(path):
    df = open(path).read().splitlines()
    records = [i.split(',', 2) for i in df]
    labels = [int(r[1]) for r in records]
    games = [i.split(',', 2)[-1] for i in df]

    return labels, games


def style_test_parser(path):
    df = open(path).read().splitlines()
    games = [i.split(',', 1)[-1] for i in df]
    
    return games

def file_test_parser(path):
    df = open(path).read().splitlines()
    games = [i.split(',', 2)[-1] for i in df]
    pred_player = [i.split(',', 2)[1] for i in df]
    file_names = [i.split(',', 2)[0] for i in df]
    return games, file_names, pred_player

if __name__ == '__main__':
    style_parser('./dataset/training/play_style_train.csv')

    style_test_parser('./dataset/testing/play_style_test_public.csv')
    file_test_parser('./dataset/testing/dan_test_public.csv')
