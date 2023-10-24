from argparse import ArgumentParser
import os
import torch
import GoDataset
import govars
from torchvision.models.vision_transformer import VisionTransformer as ViT
from My_ViT import ViT as MViT
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from baseline_model import ResNet
from torchvision import models
from baseline_model import ResNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def train(dataset, net, optimizer, loss_func):
    net.train()

    correct_preds = 0
    total_preds = 0
    acc_interval = int(len(dataset) * 0.1)

    for iter, (states, target) in enumerate(dataset):
        states = states.to(device)
        target = target.to(device)

        preds = net(states) 

        optimizer.zero_grad()

        loss = loss_func(preds, target)
        loss.backward()
        optimizer.step()

        predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)
        target_index = torch.argmax(target, dim=1)
        # Compare the predicted classes to the target labels
        correct_preds += torch.sum(predicted_classes == target_index).item()

        total_preds += target.shape[0]

        if iter % acc_interval == 0 and iter != 0:
            print(f'Accumulated training accuracy [{100 * iter / len(dataset):.2f}%]: {correct_preds / total_preds:.4f}')

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
            target_index = torch.argmax(target, dim=1)
            # Compare the predicted classes to the target labels
            correct_preds += torch.sum(predicted_classes == target_index).item()

            total_preds += target.shape[0]


    return correct_preds / total_preds

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='./dataset/training/play_style_train.csv')
    parser.add_argument('--model', type=str, default='ViT')
    parser.add_argument('--eta_start', type=float, default=3e-2)

    parser.add_argument('--epoch', '-e', type=int, default=200)
    parser.add_argument('--patch_size', '-p', default=7, type=int)
    parser.add_argument('--embedded_dim', '-d', default=768, type=int)
    parser.add_argument('--encoder_layer', '-l', default=4, type=int)
    parser.add_argument('--num_class', '-c', default=3, type=int)
    parser.add_argument('--num_head', '-nh', default=8, type=int)
    parser.add_argument('--drop', default=0, type=float)
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--label_smoothing', '--ls', default=0, type=float)
    parser.add_argument('--pretrained', '--pt', type=str)
    parser.add_argument('--split', '-s', type=float, default=0.9)
    parser.add_argument('--save_dir', '--sd', type=str, default='./model_params')
    parser.add_argument('--task', '-t', type=str, default='style')
    parser.add_argument('--T_max', type=int, default=5)
    parser.add_argument('--resnet', action='store_true')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_false')


    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    

    path = args.path
    train_set, val_set = GoDataset.style_loader(path, args.split)
    if args.pretrained is not None:
        print(f'loading pretrained model from {args.pretrained}')
        net = torch.load(args.pretrained)
    if args.resnet:
        print('Training ResNet')
        # net = ResNet(num_layers=20)
        net = models.resnet18(weights=None)
        new_conv1 = torch.nn.Conv2d(10, 64, kernel_size=7, stride=1, padding=3)
        net.conv1 = new_conv1
        in_features = net.fc.in_features
        net.fc = torch.nn.Linear(in_features, args.num_class)
    else:
        print('Training ViT')
        net = MViT(
            img_shape=(govars.FEAT_CHNLS, govars.PADDED_SIZE, govars.PADDED_SIZE),
            patch_size=args.patch_size,
            num_class=args.num_class,
            num_head=args.num_head,
            encoder_layers=args.encoder_layer,
            embedded_dim=args.embedded_dim,
            drop=args.drop
        )

    net = net.to(device)

    loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(net.parameters(), lr=args.eta_start, momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.weight_decay) 
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20)
    best_acc = 0

    # optimizer = optim.Adam(net.parameters(), lr=args.eta_start)

    for e in range(args.epoch):
        train_acc = train(train_set, net, optimizer, loss_func)
        test_acc = test(val_set, net, e)

        print(f'training acc: {train_acc:.4f}, testing acc: {test_acc:.4f}')
        print(f'learning rate: {scheduler.get_last_lr()}')
        scheduler.step()

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(net, os.path.join(args.save_dir, f'{args.model}_{args.patch_size}_{args.encoder_layer}_{args.task}.pth'))
            print(f'saving new model with test_acc: {test_acc:.6f}')


    with open('./style_result.txt', 'a') as f:
        f.write(f'l: {args.encoder_layer}, acc: {best_acc}\n')