import torchvision.models as models
import govars
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
import os
from torchvision.models.vision_transformer import VisionTransformer as ViT
from ema_pytorch import EMA
import My_ResNet

class GoMatch:
    def __init__(self, model, label_smoothing, decay, mean_pooling, device, pretrained):
        self.net = self.init_model(model, mean_pooling, pretrained).to(device)

        print(self.net)

        self.ema_net = EMA(
            self.net,
            beta = decay,              
            update_after_step = 100,    
            update_every = 10,          
        ).to(device)

        self.device = device
        self.loss_fn_avg = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.loss_fn_none = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def resnet_post_process(self, net, mean_pooling):

        # adjust i/o layer
        net.conv1 = nn.Conv2d(govars.FEAT_CHNLS, 64, kernel_size=7, stride=1, padding=3)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, govars.STYLE_CAT)

        if mean_pooling:
            net.maxpool = nn.AvgPool2d(net.maxpool.kernel_size, net.maxpool.stride, net.maxpool.padding)
            def max_2_mean(layer):
                if isinstance(layer, nn.MaxPool2d):
                    return nn.AvgPool2d(layer.kernel_size, layer.stride, layer.padding)
        
            net.apply(max_2_mean)


    def init_model(self, model, mean_pooling, pretrained=None):
        if pretrained is not None:
            print(f'Using pretrained model: {pretrained}')
            return torch.load(pretrained)
            
        if model == 'resnet18':
            net = models.resnet18(weights=None)
        if model == 'resnet50':
            net = models.resnet50(weights=None)
        elif model == 'resnet101':
            net = models.resnet101(weights=None)
        elif model == 'wresnet':
            net = models.wide_resnet50_2(weights=None)
        elif model == 'no_downsample':
            net = My_ResNet.ResNet()
        elif model == 'vit':
            net = ViT(
                image_size=govars.PADDED_SIZE,
                patch_size=7,
                num_classes=govars.STYLE_CAT,
                num_heads=8,
                num_layers=6,
                hidden_dim=768,
                mlp_dim=768,
                in_channels=govars.STYLE_CHNLS
            )

        
        if 'resnet' in model:
            # adjust i/o layers and avg pooling
            self.resnet_post_process(net, mean_pooling)



        return net

    def save_model(self, save_dir, file_name):
        torch.save(self.net, os.path.join(save_dir, file_name))


    def fit(self, train_set, test_set, optimizer, scheduler, epoch, tau, unsupervised_coef,  save_dir, file_name):
        best_acc = 0
        for i in range(epoch):
            print(f'Epoch [{i + 1}/{epoch}]')

            train_acc, supervised_loss, unsupervised_loss, mask_rate = self.train(train_set, optimizer, tau, unsupervised_coef)
            test_acc = self.test(test_set)
            scheduler.step()

            print(f'train acc: {train_acc}, test acc: {test_acc}, lr: {scheduler.get_last_lr()}')
            print(f'supervised_loss: {supervised_loss}, unsupervised_loss: {unsupervised_loss}, mask_rate: {mask_rate}')
            if test_acc > best_acc:
                best_acc = test_acc
                print('saving model...')
                self.save_model(save_dir, file_name)

    def train(self, train_set, optimizer, tau, unsupervised_coef):
        self.net.train()
        correct = 0
        total = 0
        masked_data = torch.zeros((1,)).to(self.device)
        total_unlabeled = torch.ones((1,)).to(self.device) # prevent devide by 0 when unsupervised_coef == 0
        total_unsupervised_loss = torch.zeros((1,)).to(self.device)
        total_supervised_loss = torch.zeros((1,)).to(self.device)

        with tqdm(train_set, leave=False, dynamic_ncols=True) as pbar:
            for labeled_states, labels, weak_augmented_states, strong_augmented_states in pbar:
                labeled_states, labels, weak_augmented_states, strong_augmented_states = labeled_states.to(self.device), labels.to(self.device), weak_augmented_states.to(self.device), strong_augmented_states.to(self.device)

                # supervised loss
                preds = self.net(labeled_states)
                supervised_loss = self.loss_fn_avg(preds, labels)


                # unsupervised loss
                if unsupervised_coef == 0:
                    # pure supervised learning
                    unsupervised_loss = 0
                else:
                    weak_augmented_states = rearrange(weak_augmented_states, 'b u c h w -> (b u) c h w')
                    strong_augmented_states = rearrange(strong_augmented_states, 'b u c h w -> (b u) c h w')


                    weak_preds = F.softmax(self.net(weak_augmented_states), dim=1)
                    strong_preds = self.net(strong_augmented_states)
                    confidence, _ = torch.max(weak_preds, dim=1) 

                    confidence_mask = (confidence > tau).float().squeeze()


                    pseudo_labels = torch.argmax(weak_preds, dim=1)

                    unsupervised_loss = (confidence_mask * self.loss_fn_none(strong_preds, pseudo_labels)).mean()


                # update model
                optimizer.zero_grad()
                loss = supervised_loss + unsupervised_coef * unsupervised_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                optimizer.step()

                # update ema network
                self.ema_net.update() 
        

                # stats
                with torch.no_grad():
                    # acc
                    predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)
                    correct += torch.sum(predicted_classes == labels).item()
                    total += labels.size(0)

                    # loss
                    total_unsupervised_loss += unsupervised_loss
                    total_supervised_loss += supervised_loss

                    # mask
                    if unsupervised_coef != 0:
                        masked_data += torch.count_nonzero(confidence_mask)
                        total_unlabeled += len(confidence_mask)

                pbar.set_description(f'Train Accuracy: {100 * correct / total:.2f}%')

        return 100 * correct / total, total_supervised_loss.item(), total_unsupervised_loss.item(), (masked_data / total_unlabeled).item()



    
    def test(self, test_set):
        correct = 0
        total = 0
        self.ema_net.eval()

        with torch.no_grad(), tqdm(test_set, leave=False, dynamic_ncols=True) as pbar:
            for states, labels in pbar:
                states, labels = states.to(self.device), labels.to(self.device)
                preds = self.ema_net(states)

                predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)


                correct += torch.sum(predicted_classes == labels).item()
                total += labels.shape[0]

                pbar.set_description(f'Test Accuracy: {100 * correct / total:.2f}%')

        return 100 * correct / total

