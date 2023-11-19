from argparse import ArgumentParser
import os
import torch
import GoDataset
import govars
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
import numpy as np
from torch.utils.data import DataLoader, Subset
import goutils
from torchvision.models.vision_transformer import VisionTransformer as ViT
from baseline_model import MLP, ResNet


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(dataset, val_dataset, net, optimizer, loss_func, epochs):
    '''
        dataset(Dataloader)
        net(nn)
    '''

    pbar = tqdm(range(epochs), dynamic_ncols=True, leave=True)
    for _ in pbar:
        correct_preds = 0
        total_preds = 0
        net.train()
        for states, targets in dataset:
            states = states.to(device)
            targets = targets.to(device)

            preds = net(states) 

            optimizer.zero_grad()

            loss = loss_func(preds, targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds_proba = torch.softmax(preds, dim=1)
                predicted_classes = torch.argmax(preds_proba, dim=1)
                # Compare the predicted classes to the target labels
                correct_preds += torch.sum(predicted_classes == targets).item()

                total_preds += targets.shape[0]

        if val_dataset is not None:
            test_acc, test_proba = test(val_dataset, net)
        else:
            test_acc, test_proba = -1, -1

        pbar.set_description(f'train acc: {correct_preds / total_preds:.4f}, test acc: {test_acc:.4f}')


    return correct_preds / total_preds, test_acc, test_proba # return the acc of the last epoch



def test(dataset, net):
    '''
        dataset(Dataloader)
        net(nn)
    '''
    net.eval()
    correct_preds = 0
    total_preds = 0
    total_preds_proba = torch.empty(0).to(device)

    with torch.no_grad():
        for states, target in dataset:
            states = states.to(device)
            target = target.to(device)

            preds = net(states) 
            preds_proba = torch.softmax(preds, dim=1)
            predicted_classes = torch.argmax(preds_proba, dim=1)
            # Compare the predicted classes to the target labels
            correct_preds += torch.sum(predicted_classes == target).item()

            total_preds += target.shape[0]

            total_preds_proba = torch.concat((total_preds_proba, preds_proba), dim=0)


    return correct_preds / total_preds, total_preds_proba


def eval_ensemble(dataset, preds_proba):
    '''
        dataset(StyleDataset)
    '''
    total_preds = len(dataset)
    correct_preds = 0
    preds = torch.argmax(preds_proba, dim=1).to('cpu').numpy()

    for data, pred in zip(dataset, preds):
        if pred == data[1]: 
            correct_preds += 1

    return correct_preds / total_preds

def init_net(model):
    if 'resnet' in model:
        print(f'Training {model}')
        if model == 'resnet18':
            net = models.resnet18(weights=None)
        elif model == 'resnet34':
            net = models.resnet34(weights=None)
        elif model == 'resnet50':
            net = models.resnet50(weights=None)

        new_conv1 = nn.Conv2d(govars.FEAT_CHNLS, 64, kernel_size=7, stride=1, padding=3)
        net.conv1 = new_conv1
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, govars.STYLE_CAT)
    elif model == 'vit':
        net = ViT(
            image_size=9,
            patch_size=3,
            num_classes=3,
            num_heads=8,
            num_layers=4,
            hidden_dim=768,
            mlp_dim=768,
            dropout=0.1,
            in_channels=govars.FEAT_CHNLS,
        )
    elif model == 'mlp':
        net = MLP(
            input_dim=govars.FEAT_CHNLS * govars.REGION_SIZE * govars.REGION_SIZE,
            hidden_dim=128,
            output_dim=3
        )
    elif model=='RESNET':
        net = ResNet(num_layers=1)

    return net

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--games_path', type=str, default='./dataset/training/games.npy')
    parser.add_argument('--labels_path', type=str, default='./dataset/training/labels.npy')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--weight_decay', '--wd', default=0, type=float)
    parser.add_argument('--label_smoothing', '--ls', default=0, type=float)
    parser.add_argument('--save_dir', '--sd', type=str, default='./model_params')
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--baggings', type=int, default=15)
    parser.add_argument('--bagging_portion', type=float, default=1)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--full', action='store_true')
    


    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)



    games, labels = np.load(args.games_path), np.load(args.labels_path)
    games = np.array([goutils.crop_move_as_center(game) for game in games])
    games, labels = goutils.pre_augmentation(games, labels)

    if not args.full:
        test_len = int(len(games) * 0.1)

        train_set = GoDataset.StyleDataset(
            labels=labels[test_len:],
            games=games[test_len:],
        )
        
        val_set = GoDataset.StyleDataset(
            labels=labels[:test_len],
            games=games[:test_len],
        )

        val_loader = DataLoader(
            dataset=val_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    else:
        print('using full training set')
        train_set = GoDataset.StyleDataset(
            labels=labels,
            games=games,
        )
        val_loader = None


    if not args.full:
        individual_perf = []
        ensemble_preds_prob = torch.zeros((len(val_set), govars.STYLE_CAT,)).to(device)

    for b in range(args.baggings):
        bagging_indices = np.random.choice(range(len(train_set)), int(len(train_set) * args.bagging_portion))

        bagging_subset = Subset(train_set, bagging_indices)
        bagging_loader = DataLoader(
            bagging_subset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )


        net = init_net(args.model)
        net = net.to(device)
        loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = optim.Adam(
            net.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay,
        ) 


        # training and testing loop
        train_acc, test_acc, preds_proba = train(bagging_loader, val_loader, net, optimizer, loss_func, args.epoch)


        # print(f'saving new model with test acc: {test_acc:.6f}, train acc: {train_acc:.6f}')
        torch.save(net, os.path.join(args.save_dir, f'{args.model}_bagging_{b}_wd_{args.weight_decay}_port_{args.bagging_portion}.pth'))

        if not args.full:
            ensemble_preds_prob += preds_proba


            individual_perf.append(test_acc)

    if not args.full:
        ensemble_preds_prob /= args.baggings # just to normalize (not necessary)
        ensemble_acc = eval_ensemble(val_set, ensemble_preds_prob)

        print('Individual performance', individual_perf)
        print(f'ensemble acc: {ensemble_acc}')
