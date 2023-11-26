from blender import BlendingClassifier, MetaLearner
import numpy as np
import goutils
from baseline_model import ResNet
from tqdm import tqdm
from argparse import ArgumentParser
import gogame
import govars

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_estimators', '-e', type=int, default=5)
    parser.add_argument('--region_size', type=int, default=13)
    parser.add_argument('--bagging', action='store_true')
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()


    games, labels = np.load('dataset/training/games.npy'), np.load('dataset/training/labels.npy')
    games = np.array([goutils.crop_move_as_center(game, args.region_size) for game in tqdm(games, desc='cropping', dynamic_ncols=True)])

    split = int(len(games) * 0.1)

    # test : val : train = 1 : 1 : 8 
    test_X, test_y = games[:split], labels[:split]
    val_X, val_y = games[split:2*split], labels[split:2*split]
    train_X, train_y = games[2*split:], labels[2*split:]

    train_X, train_y = goutils.pre_augmentation(train_X, train_y)
    val_X, val_y = goutils.pre_augmentation(val_X, val_y)

    estimators = [ResNet(in_channels=govars.FEAT_CHNLS, num_layers=3, region_size=args.region_size) for _ in range(args.num_estimators)]

    clf = BlendingClassifier(
        estimators,
    )


    clf.fit(
        train_X,
        train_y, 
        val_X,
        val_y,
        test_X,
        test_y,
        bagging=args.bagging,
        save_path=args.save_path,
    )



    sym_test_X = []
    for X in test_X:
        sym_X = gogame.all_symmetries(X)

        sym_test_X.append(sym_X)

    sym_test_X = np.array(sym_test_X)

    print(clf.acc_score(test_X, test_y))
    print(clf.acc_score(sym_test_X, test_y, tta=True))
    print(clf.eval_estimators(test_X, test_y))
    # print(clf.acc_score(sym_test_X, test_y, bf_tta=True))