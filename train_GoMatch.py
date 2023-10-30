from argparse import ArgumentParser
from GoMatch import GoMatch
import GoDataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch

def get_optim(optim_type, net, lr, momentum, nesterov, weight_decay):
    if optim_type == 'sgd':
        return optim.SGD(
            net.parameters(), 
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    elif optim_type == 'adam':
        return optim.Adam(
            net.parameters(),
            lr=lr
        )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--unlabeled_size', type=int, default=12)
    parser.add_argument('--path', '-p', type=str, default='./dataset/training/play_style_train.csv')
    parser.add_argument('--split', '-s', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=2e-2)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--tau', type=float, default=0.95)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_false')
    parser.add_argument('--unsupervised_coef', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='model_params')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--rand_move', action='store_true')
    parser.add_argument('--label_smoothing', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--optim_type', type=str, default='sgd')
    parser.add_argument('--pretrained', type=str)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on {device}')
    file_name = f'momentum_{args.momentum}_crop_{args.crop}_rand_move_{args.rand_move}_coef_{args.unsupervised_coef}_{args.model}.pth'
    print(f'file name: {file_name}')

    train_set, test_set = GoDataset.go_match_loader(
        args.path,
        args.split, 
        args.unlabeled_size,
        args.batch_size,
        args.crop,
        args.rand_move,
    )
    go_match = GoMatch(
        args.model,
        args.dropout, # to initialize resnet 
        args.label_smoothing, # to initialize loss function
        device,
        args.pretrained,
    )
    
    optimizer = get_optim(
        optim_type=args.optim_type,
        net=go_match.net,
        lr=args.lr,
        momentum=args.momentum,
        nesterov=args.nesterov,
        weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=args.epoch,
    )


    go_match.fit(
        train_set,
        test_set,
        optimizer,
        scheduler,
        args.epoch,
        args.tau,
        args.unsupervised_coef,
        args.save_dir,
        file_name=file_name
    )