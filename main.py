from argparse import ArgumentParser
import GoDataset
import baseline_model
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
import govars
from torchvision.models.vision_transformer import VisionTransformer as ViT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on {device}')

loss_func = nn.CrossEntropyLoss()

def train(dataset, net, optimizer, e):
    net.train()

    correct_preds = 0
    total_preds = 0
    acc_interval = 1000

    for iter, (states, target) in enumerate(tqdm(dataset, desc=f'epoch {e}')):
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
        total_preds += target.shape[0]

    return correct_preds / total_preds

def test(dataset, net, e):
    net.eval()
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for iter, (states, target) in enumerate(tqdm(dataset, desc=f'epoch {e}')):

            states = states.squeeze(dim=0)
            target = target.squeeze(dim=0)

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
    parser.add_argument('--path', type=str, default='./dataset/training/dan_train.csv')
    parser.add_argument('--model', type=str, default='ViT')
    parser.add_argument('--lr', type=float, default=1e-4)

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
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--split', '-s', type=float, default=0.9)


    args = parser.parse_args()
    path = args.path
    train_set, val_set = GoDataset.get_loader(path, args.split)
    # net = model.get_model(args.model).to(device)
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


    optimizer = optim.Adam(net.parameters(), lr=args.lr) # Adam somehow works poorly
    best_acc = 0
    patience_count = 0

    for e in range(args.epoch):
        train_acc = train(train_set, net, optimizer, e)
        test_acc = test(val_set, net, e)
        
        print(f'training acc: {train_acc:.4f}, testing acc: {test_acc:.4f}')

        if test_acc >= best_acc:
            best_acc = test_acc
            patience_count = 0
            torch.save(net, f'{args.model}.pth')
            print(f'saving new model with test_acc: {test_acc}')
        else:
            patience_count += 1

        if patience_count >= args.patience:
            break
