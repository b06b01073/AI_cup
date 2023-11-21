from blender import BlendingClassifier, MetaLearner
import numpy as np
import goutils
from baseline_model import ResNet
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_estimators', '-e', type=int, default=5)
    parser.add_argument('--region_size', type=int, default=13)
    args = parser.parse_args()


    games, labels = np.load('dataset/training/games.npy'), np.load('dataset/training/labels.npy')
    games = np.array([goutils.crop_move_as_center(game, args.region_size) for game in tqdm(games, desc='cropping', dynamic_ncols=True)])

    split = int(len(games) * 0.1)

    # test : val : train = 1 : 1 : 8 
    test_X, test_y = games[:split], labels[:split]
    val_X, val_y = games[split:2*split], labels[split:2*split]
    train_X, train_y = games[2*split:], labels[2*split:]

    train_X, train_y = goutils.pre_augmentation(train_X, train_y, region_size=args.region_size)

    estimators = [ResNet(num_layers=1, region_size=args.region_size) for _ in range(args.num_estimators)]

    final_estimator = MetaLearner(
        in_features=args.num_estimators*3,
        out_features=3,
        hidden_dim=256,
    )

    clf = BlendingClassifier(
        estimators,
        final_estimator # final_estima
    )


    clf.fit(
        train_X,
        train_y, 
        val_X,
        val_y,
        test_X,
        test_y
    )


    print(clf.acc_score(test_X, test_y))