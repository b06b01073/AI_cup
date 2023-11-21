import torch
from argparse import ArgumentParser
import torch.nn as nn
import numpy as np
import GoDataset 
from torch.utils.data import DataLoader
import torch.optim as optim
import goutils
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def fit(net, train_loader, test_loader, quiet, epochs=20):
    net.to(device)
    optimizer = optim.Adam(
        net.parameters(), 
        lr=1e-4, 
        # weight_decay=1e-4,
    ) 
    loss_func = nn.CrossEntropyLoss()

    training_steps = 0
    best_test_acc = 0
    accs = []

    for epoch in range(epochs):

        train_corrects = 0
        train_total = 0

        for X, y in train_loader:
            training_steps += 1
            net.train()
            X, y = X.to(device), y.to(device)
            # print(X.shape)
            pred_y = net(X)
            optimizer.zero_grad()
            loss = loss_func(pred_y, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                preds_proba = torch.softmax(pred_y, dim=1)
                predicted_classes = torch.argmax(preds_proba, dim=1)
                # Compare the predicted classes to the target labels
                train_corrects += torch.sum(predicted_classes == y).item()

                train_total += y.shape[0]

            if training_steps % 50 == 0:
                net.eval()
                test_corrects = 0
                test_total = 0

                with torch.no_grad():
                    for X, y in test_loader:
                        X = X.to(device)
                        y = y.to(device)

                        preds = net(X) 
                        preds_proba = torch.softmax(preds, dim=1)
                        predicted_classes = torch.argmax(preds_proba, dim=1)
                        # Compare the predicted classes to the y labels
                        test_corrects += torch.sum(predicted_classes == y).item()

                        test_total += y.shape[0]
                best_test_acc = max(best_test_acc, test_corrects / test_total)
                accs.append(test_corrects / test_total)
                if not quiet:
                    print(f'epoch: {epoch}, step: {training_steps}, train acc: {train_corrects / train_total:.4f}, test acc: {test_corrects / test_total:.4f}')
                    
    print(best_test_acc, accs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='model_params/0_10_kyu.pth')
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--unfreeze', type=int, default=1)

    args = parser.parse_args()

    net = torch.load(args.model)
    for param in net.parameters():
        param.requires_grad = False
    net.heads = nn.Sequential(
        nn.Linear(768, 3),
    )

    for i in range(args.unfreeze):
        layer_index = -(i + 1)
        for param in net.encoder.layers[layer_index].parameters():
            param.requires_grad = True


    # for param in net.encoder.layers[-2].parameters():
    #     param.requires_grad = True
    # for param in net.encoder.layers[-1].parameters():
    #     param.requires_grad = True

    # print(net.heads.requires_grad_)
    
    games, labels = np.load('./dataset/training/games.npy'), np.load('./dataset/training/labels.npy')

    train_games, test_games, train_labels, test_labels = train_test_split(games, labels, test_size=0.1)

    if not args.quiet:
        print(games.shape)

    train_games, train_labels = goutils.pre_augmentation(train_games, train_labels)


    train_set = GoDataset.StyleDataset(
        labels=train_labels,
        games=train_games,
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=512,
        num_workers=6,
        shuffle=True
    )
    
    val_set = GoDataset.StyleDataset(
        labels=test_labels,
        games=test_games,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=512,
        num_workers=6
    )
    
    if not args.quiet:
        print(len(train_set), len(val_set))
    fit(net, train_loader, val_loader, args.quiet)

    


    
