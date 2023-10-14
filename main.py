from argparse import ArgumentParser
import GoDataset
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
import govars
from torchvision.models.vision_transformer import VisionTransformer as ViT
from torch.optim.lr_scheduler import CosineAnnealingLR

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on {device}')

loss_func = nn.CrossEntropyLoss()

def batch_topk_hit(preds, label_index, k=5):
    preds = torch.softmax(preds, dim=1)
    _, topk_indices = preds.topk(k, dim=-1) # output (batch, k)

    # Check if the true label_index is in the top-k predicted labels for each example
    batch_size, pred_size = preds.shape

    correct = 0

    for i in range(batch_size):
        if label_index[i] in topk_indices[i]:
            correct += 1

    return correct

def train(dataset, net, optimizer):
    net.train()

    correct_preds = 0
    total_preds = 0
    acc_interval = int(len(dataset) * 0.05)
    top5_hit = 0 

    for iter, (states, target) in enumerate(dataset):
        states = states.squeeze(dim=0)
        target = target.squeeze(dim=0)

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
        top5_hit += batch_topk_hit(preds, target_index)

        total_preds += target.shape[0]

        if iter % acc_interval == 0 and iter != 0:
            print(f'Accumulate training accuracy [{100 * iter / len(dataset):.2f}%]: top1: {correct_preds / total_preds:.4f}, top5: {top5_hit / total_preds:.4f}')

    return correct_preds / total_preds

def test(dataset, net, e):
    net.eval()
    correct_preds = 0
    total_preds = 0
    top5_hit = 0
    with torch.no_grad():
        for states, target in tqdm(dataset, desc=f'epoch {e}'):

            states = states.squeeze(dim=0)
            target = target.squeeze(dim=0)

            states = states.to(device)
            target = target.to(device)

            preds = net(states) 

            predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)
            target_index = torch.argmax(target, dim=1)
            # Compare the predicted classes to the target labels
            correct_preds += torch.sum(predicted_classes == target_index).item()
            top5_hit += batch_topk_hit(preds, target_index)

            total_preds += target.shape[0]


    return correct_preds / total_preds, top5_hit / total_preds


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='./dataset/training/dan_train.csv')
    parser.add_argument('--model', type=str, default='ViT')
    parser.add_argument('--eta_start', type=float, default=1e-3)

    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--patch_size', '-p', default=7, type=int)
    parser.add_argument('--embedded_dim', '-d', default=384, type=int)
    parser.add_argument('--encoder_layer', '-l', default=6, type=int)
    parser.add_argument('--num_class', '-c', default=362, type=int)
    parser.add_argument('--num_head', '-nh', default=8, type=int)
    parser.add_argument('--drop', default=0, type=float)
    parser.add_argument('--weight_decay', '--wd', default=0, type=float)
    parser.add_argument('--label_smoothing', '--ls', default=0, type=float)
    parser.add_argument('--pretrained', '--pt', type=str)
    parser.add_argument('--split', '-s', type=float, default=0.9)
    parser.add_argument('--save_dir', '--sd', type=str, default='./model_params')
    parser.add_argument('--task', '-t', type=str, default='dan')
    parser.add_argument('--T_max', type=int, default=5)
    parser.add_argument('--eta_min', type=float, default=1e-6)


    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    

    path = args.path
    train_set, val_set = GoDataset.get_loader(path, args.split)
    if args.pretrained is not None:
        print(f'loading pretrained model from {args.pretrained}')
        net = torch.load(args.pretrained)
    else:
        net = ViT(
            image_size=govars.PADDED_SIZE,
            patch_size=args.patch_size,
            num_classes=args.num_class,
            num_heads=args.num_head,
            num_layers=args.encoder_layer,
            hidden_dim=args.embedded_dim,
            mlp_dim=args.embedded_dim,
            in_channels=govars.FEAT_CHNLS
        ).to(device)


    optimizer = optim.Adam(net.parameters(), lr=args.eta_start) 
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min, verbose=True)
    best_acc = 0
    best_ai_cup_score = 0

    for e in range(args.epoch):
        train_acc = train(train_set, net, optimizer)
        test_acc, top5_acc = test(val_set, net, e)
        scheduler.step()

        print(f'training acc: {train_acc:.4f}, testing acc: {test_acc:.4f}, test top5: {top5_acc:.4f}')

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(net, os.path.join(args.save_dir, f'{args.model}_{args.encoder_layer}_{args.task}.pth'))
            print(f'saving new model with test_acc: {test_acc:.6f}')
        
        test_ai_cup_score = 0.25 * test_acc + 0.1 * top5_acc
        if test_ai_cup_score >= best_ai_cup_score:
            best_ai_cup_score = test_ai_cup_score
            torch.save(net, os.path.join(args.save_dir, f'ai_{args.model}_{args.encoder_layer}_{args.task}.pth'))
            print(f'saving new model with best score: {test_acc:.6f}')


    with open('./result.txt', 'a') as f:
        f.write(f'l: {args.encoder_layer}, acc: {best_acc}, ai_cup_score: {best_ai_cup_score}\n')