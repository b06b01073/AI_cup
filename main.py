from argparse import ArgumentParser
import GoDataset
import os
import torch.nn as nn
import torch.optim as optim
import torch
import govars
from torchvision.models.vision_transformer import VisionTransformer as ViT
from trainer import Trainer



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='./dataset/training/dan_train.csv')
    parser.add_argument('--model', type=str, default='ViT')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', '-e', type=int, default=200)
    parser.add_argument('--patch_size', '-p', default=7, type=int)
    parser.add_argument('--embedded_dim', '-d', default=768, type=int)
    parser.add_argument('--encoder_layer', '-l', default=6, type=int)
    parser.add_argument('--num_class', '-c', default=362, type=int)
    parser.add_argument('--num_head', '-nh', default=8, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--weight_decay', '--wd', default=0, type=float)
    parser.add_argument('--pretrained', '--pt', type=str)
    parser.add_argument('--split', '-s', type=float, default=0.9)
    parser.add_argument('--save_dir', '--sd', type=str, default='./model_params')
    parser.add_argument('--task', '-t', type=str, default='dan')
    parser.add_argument('--patience', type=int, default=-1)


    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    

    path = args.path
    train_set, test_set = GoDataset.get_loader(path, args.split)
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
            in_channels=govars.FEAT_CHNLS,
            dropout=args.dropout
        )


    optimizer = optim.Adam(net.parameters(), lr=args.lr) 
    loss_func = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        optimizer,
        loss_func,
        args.save_dir,
        args.task,
        device,
    )

    trainer.fit(
        net, 
        train_set, 
        test_set, 
        args.epoch, 
        args.patience,
        args.encoder_layer,
    )
