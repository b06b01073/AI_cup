from argparse import ArgumentParser
from GoMatch import GoMatch
import GoDataset
import torch.optim as optim
import torch



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='resnet')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--unlabeled_size', type=int, default=10)
    parser.add_argument('--path', '-p', type=str, default='./dataset/training/play_style_train.csv')
    parser.add_argument('--split', '-s', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--nesterov', action='store_false')
    parser.add_argument('--unsupervised_coef', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='model_params')
    parser.add_argument('--file_name', type=str, default='GoMatch.pth')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on {device}')

    train_set, test_set = GoDataset.go_match_loader(
        args.path,
        args.split, 
        args.unlabeled_size,
        args.batch_size
    )
    go_match = GoMatch(args.model, device)
    
    optimizer = optim.SGD(
        go_match.net.parameters(), 
        lr=args.lr,
        momentum=args.momentum,
        nesterov=args.nesterov,
        weight_decay=args.weight_decay
    )

    go_match.fit(
        train_set,
        test_set,
        optimizer,
        args.epoch,
        args.tau,
        args.unsupervised_coef,
        args.save_dir,
        args.file_name
    )