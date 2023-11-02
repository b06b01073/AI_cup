import torchvision.models as models
import govars
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
import os
from torchvision.models.vision_transformer import VisionTransformer as ViT
import copy

class GoMatch:
    def __init__(self, model, dropout, label_smoothing, decay, device, pretrained):
        self.net = self.init_model(model, dropout, pretrained).to(device)
        self.ema_net = EMA(decay=decay, device=device)
        self.ema_net.copy(self.net)

        self.device = device
        self.loss_fn_avg = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.loss_fn_none = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def resnet_post_process(self, net):
        net.conv1 = torch.nn.Conv2d(govars.FEAT_CHNLS, 64, kernel_size=7, stride=1, padding=3)
        in_features = net.fc.in_features
        net.fc = torch.nn.Linear(in_features, govars.STYLE_CAT)

        return net


    def init_model(self, model, dropout, pretrained=None):
        if pretrained is not None:
            print(f'Using pretrained model: {pretrained}')
            return torch.load(pretrained)
            
        if model == 'resnet18':
            net = models.resnet18(weights=None, dropout=dropout)
            net = self.resnet_post_process(net)
        if model == 'resnet50':
            net = models.resnet50(weights=None, dropout=dropout)
            net = self.resnet_post_process(net)
        elif model == 'resnet101':
            net = models.resnet101(weights=None, dropout=dropout)
            net = self.resnet_post_process(net)
        elif model == 'wresnet':
            net = models.wide_resnet50_2(weights=None, dropout=dropout)
            net = self.resnet_post_process(net)
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

        return net

    def save_model(self, save_dir, file_name):
        torch.save(self.net, os.path.join(save_dir, file_name))


    def fit(self, train_set, test_set, optimizer, scheduler, epoch, tau, unsupervised_coef,  save_dir, file_name):
        best_acc = 0
        for i in range(epoch):
            print(f'Epoch [{i + 1}/{epoch}]')

            train_acc, supervised_loss, unsupervised_loss, mask_rate = self.train(train_set, optimizer, tau, unsupervised_coef)

            self.ema_net.update(self.net) # update ema network

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
        masked_data = 0
        total_unlabeled = 0
        total_unsupervised_loss = 0
        total_supervised_loss = 0

        with tqdm(train_set, leave=False, dynamic_ncols=True) as pbar:
            for labeled_states, labels, weak_augmented_states, strong_augmented_states in pbar:
                labeled_states, labels, weak_augmented_states, strong_augmented_states = labeled_states.to(self.device), labels.to(self.device), weak_augmented_states.to(self.device), strong_augmented_states.to(self.device)

                # supervised loss
                preds = self.net(labeled_states)
                supervised_loss = self.loss_fn_avg(preds, labels)


                # unsupervised loss
                # unsupervised_loss = 0
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
        
                with torch.no_grad():
                    # acc
                    predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)
                    correct += torch.sum(predicted_classes == labels).item()
                    total += labels.size(0)

                    # loss
                    total_unsupervised_loss += unsupervised_loss
                    total_supervised_loss += supervised_loss

                    # mask
                    masked_data += torch.count_nonzero(confidence_mask)
                    total_unlabeled += len(confidence_mask)

                pbar.set_description(f'Train Accuracy: {100 * correct / total:.2f}%')

        return 100 * correct / total, total_supervised_loss.item(), total_unsupervised_loss.item(), (masked_data / total_unlabeled).item()



    
    def test(self, test_set):
        self.ema_net.eval()
        correct = 0
        total = 0

        with torch.no_grad(), tqdm(test_set, leave=False, dynamic_ncols=True) as pbar:
            for states, labels in pbar:
                states, labels = states.to(self.device), labels.to(self.device)
                preds = self.ema_net(states)

                predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)


                correct += torch.sum(predicted_classes == labels).item()
                total += labels.shape[0]

                pbar.set_description(f'Test Accuracy: {100 * correct / total:.2f}%')

        return 100 * correct / total


class EMA:
    def __init__(self, decay, device):
        self.decay = decay
        self.device = device
        self.net = None

    def copy(self, source_net):
        self.net = copy.deepcopy(source_net).to(self.device)


    def update(self, source_net):
        with torch.no_grad():
            for target_param, source_param in zip(self.net.parameters(), source_net.parameters()):
                target_param.data.copy_(self.decay * target_param.data + (1 - self.decay) * source_param.data)