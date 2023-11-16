from argparse import ArgumentParser
import os
import torch
import GoDataset
import govars
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from torchvision import models
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def train(dataset, net, optimizer, loss_func):
    net.train()

    correct_preds = 0
    total_preds = 0

    for states, target in dataset:
        states = states.to(device)
        target = target.to(device)

        preds = net(states) 

        optimizer.zero_grad()

        loss = loss_func(preds, target)
        loss.backward()
        optimizer.step()

        predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)
        # Compare the predicted classes to the target labels
        correct_preds += torch.sum(predicted_classes == target).item()

        total_preds += target.shape[0]


    return correct_preds / total_preds

def test(dataset, net, e):
    net.eval()
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for states, target in tqdm(dataset, desc=f'epoch {e}'):
            states = states.to(device)
            target = target.to(device)

            preds = net(states) 

            predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)
            # Compare the predicted classes to the target labels
            correct_preds += torch.sum(predicted_classes == target).item()

            total_preds += target.shape[0]


    return correct_preds / total_preds


def init_net(model):
    print('Training ResNet')
    # net = ResNet(num_layers=20)
    net = models.resnet18(weights=None)
    new_conv1 = torch.nn.Conv2d(govars.FEAT_CHNLS, 64, kernel_size=7, stride=1, padding=3)
    net.conv1 = new_conv1
    in_features = net.fc.in_features
    net.fc = torch.nn.Linear(in_features, govars.STYLE_CAT)


    return net

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--games_path', type=str, default='./dataset/training/games.npy')
    parser.add_argument('--labels_path', type=str, default='./dataset/training/labels.npy')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--lr', type=float, default=1e-2)

    parser.add_argument('--epoch', '-e', type=int, default=150)
    parser.add_argument('--num_class', '-c', default=3, type=int)
    parser.add_argument('--drop', default=0, type=float)
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--label_smoothing', '--ls', default=0, type=float)
    parser.add_argument('--pretrained', '--pt', type=str)
    parser.add_argument('--split', '-s', type=float, default=0.9)
    parser.add_argument('--save_dir', '--sd', type=str, default='./model_params')
    parser.add_argument('--task', '-t', type=str, default='style')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_false')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--baggings', type=int, default=10)
    parser.add_argument('--bagging_portion', type=float, default=0.9)


    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, 'bagging')):
        os.mkdir(os.path.join(args.save_dir, 'bagging'))



    games, labels = np.load(args.games_path), np.load(args.labels_path)
    kfold = StratifiedKFold(n_splits=args.folds)
    
    # optimizer = optim.Adam(net.parameters(), lr=args.eta_start)

    for fold, (train_indices, val_indices) in enumerate(kfold.split(games, labels)):
        train_set = GoDataset.StyleDataset(
            labels=labels[train_indices],
            games=games[train_indices],
            augment=True
        )
        
        val_set = GoDataset.StyleDataset(
            labels=labels[val_indices],
            games=games[val_indices],
            augment=False
        )

        val_loader = DataLoader(
            dataset=val_set,
            batch_size=args.batch_size,
            num_workers=6
        )

        
        for b in range(args.baggings):
            bagging_indices = np.random.choice(range(len(train_set)), int(len(train_set) * args.bagging_portion))

            bagging_subset = Subset(train_set, bagging_indices)
            bagging_loader = DataLoader(
                bagging_subset, 
                batch_size=args.batch_size,
                shuffle=True
            )


            net = init_net(args.model)
            net = net.to(device)
            loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            optimizer = optim.SGD(
                net.parameters(), 
                lr=args.lr, 
                weight_decay=args.weight_decay,
                momentum=args.momentum,
                nesterov=args.nesterov
            ) 
            lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epoch)
            best_acc = 0
            for e in range(args.epoch):
                train_acc = train(bagging_loader, net, optimizer, loss_func)
                test_acc = test(val_loader, net, e)
                lr_scheduler.step()

                print(f'training acc: {train_acc:.4f}, testing acc: {test_acc:.4f}')

                if test_acc >= best_acc:
                    best_acc = test_acc
                    torch.save(net, os.path.join(args.save_dir, 'bagging', f'{args.model}_{args.task}_fold_{fold}_bagging_{b}.pth'))
                    print(f'saving new model with test_acc: {test_acc:.6f}')

