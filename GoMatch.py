import torchvision.models as models
import govars
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
import os

class GoMatch:
    def __init__(self, model, device):
        self.net = self.init_model(model).to(device)
        self.device = device
        self.loss_fn_avg = nn.CrossEntropyLoss()
        self.loss_fn_sum = nn.CrossEntropyLoss(reduction='sum')

    def init_model(self, model):
        if model == 'resnet':
            net = models.resnet18(weights=None)

            # adjust the input and output layer
            net.conv1 = torch.nn.Conv2d(govars.FEAT_CHNLS, 64, kernel_size=7, stride=1, padding=3)
            in_features = net.fc.in_features
            net.fc = torch.nn.Linear(in_features, govars.STYLE_CAT)
            return net


    def save_model(self, save_dir, file_name):
        if not os.path.exists(save_dir):
            torch.save(self.net, os.path.join(save_dir, file_name))


    def fit(self, train_set, test_set, optimizer, scheduler, epoch, tau, unsupervised_coef, save_dir, file_name):
        best_acc = 0
        for i in range(epoch):
            print(f'Epoch [{i + 1}/{epoch}]')

            train_acc = self.train(train_set, optimizer, tau, unsupervised_coef)
            test_acc = self.test(test_set)
            scheduler.step()

            print(f'train acc: {train_acc}, test acc: {test_acc}, lr: {scheduler.get_last_lr()}')
            if test_acc > best_acc:
                best_acc = test_acc
                print('saving model...')
                self.save_model(save_dir, file_name)

    def train(self, train_set, optimizer, tau, unsupervised_coef):
        self.net.train()
        correct = 0
        total = 0
        with tqdm(train_set, leave=False, dynamic_ncols=True) as pbar:
            for labeled_states, label_onehots, weak_augmented_states, strong_augmented_states in pbar:
                labeled_states, label_onehots, weak_augmented_states, strong_augmented_states = labeled_states.to(self.device), label_onehots.to(self.device), weak_augmented_states.to(self.device), strong_augmented_states.to(self.device)

                # supervised loss
                preds = self.net(labeled_states)
                supervised_loss = self.loss_fn_avg(preds, label_onehots)


                # unsupervised loss
                # unsupervised_loss = 0
                weak_augmented_states = rearrange(weak_augmented_states, 'b u c h w -> (b u) c h w')
                strong_augmented_states = rearrange(strong_augmented_states, 'b u c h w -> (b u) c h w')


                unlabeled_size = weak_augmented_states.shape[0]
                weak_preds = F.softmax(self.net(weak_augmented_states), dim=1)
                strong_preds = self.net(strong_augmented_states)
                confidence, _ = torch.max(weak_preds, dim=1) 
                confidence_mask = torch.nonzero((confidence > tau).type(torch.int32)).squeeze()


                pseudo_labels = F.one_hot(torch.argmax(weak_preds, dim=1)[confidence_mask], govars.STYLE_CAT).type(torch.float32)

                unsupervised_loss = self.loss_fn_sum(strong_preds[confidence_mask], pseudo_labels.detach()) / unlabeled_size

                # update model
                optimizer.zero_grad()
                loss = supervised_loss + unsupervised_coef * unsupervised_loss
                loss.backward()
                optimizer.step()
        
                with torch.no_grad():
                    predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)
                    target_index = torch.argmax(label_onehots, dim=1)
                    correct += torch.sum(predicted_classes == target_index).item()
                    total += label_onehots.size(0)

                pbar.set_description(f'Train Accuracy: {100 * correct / total:.2f}%')

        return 100 * correct / total



    
    def test(self, test_set):
        self.net.eval()
        correct = 0
        total = 0

        with torch.no_grad(), tqdm(test_set, leave=False, dynamic_ncols=True) as pbar:
            for states, labels in pbar:
                states, labels = states.to(self.device), labels.to(self.device)
                preds = self.net(states)

                predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)


                correct += torch.sum(predicted_classes == labels).item()
                total += labels.shape[0]

                pbar.set_description(f'Test Accuracy: {100 * correct / total:.2f}%')

        return 100 * correct / total

