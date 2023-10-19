from tqdm import tqdm
import torch
import os

class Trainer:
    def __init__(self, optimizer=None, loss_func=None, save_dir=None, task=None, device='cuda'):
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.save_dir = save_dir
        self.device = device
        self.task = task

        print(f'Training on {self.device}')


    def fit(self, net, train_set, test_set, epoch, patience, encoder_layer, split_index=0):
        # encoder_layer is passed to distinguish model parameter file, split_index is passed to specify the boostrap dataset index
        net = net.to(self.device)
        patience_count = 0
        best_acc = 0
        best_ai_cup_score = 0

        for e in range(epoch):
            print(f'Epoch {e} just started...')

            patience_count += 1
            train_acc = self.train(train_set, net, self.optimizer, self.loss_func)
            test_acc, top5_acc = self.test(test_set, net, e)

            print(f'training acc: {train_acc:.4f}, testing acc: {test_acc:.4f}, test top5: {top5_acc:.4f}')

            if test_acc >= best_acc:
                best_acc = test_acc
                torch.save(net, os.path.join(self.save_dir, f'{split_index}_{encoder_layer}_{self.task}.pth'))
                print(f'saving new model with test_acc: {test_acc:.6f}')
                patience_count = 0
            
            test_ai_cup_score = 0.25 * test_acc + 0.1 * top5_acc
            if test_ai_cup_score >= best_ai_cup_score:
                best_ai_cup_score = test_ai_cup_score
                torch.save(net, os.path.join(self.save_dir, f'ai_{split_index}_{encoder_layer}_{self.task}.pth'))
                print(f'saving new model with best score: {test_ai_cup_score:.6f}')
                patience_count = 0

            if patience_count > patience:
                break

        with open(f'{self.task}_result.txt', 'a') as f:
            f.write(f'l: {encoder_layer}, acc: {best_acc}, ai_cup_score: {best_ai_cup_score}\n')

    
    def train(self, dataset, net, optimizer, loss_func):
        net.train()

        correct_preds = 0
        total_preds = 0
        top5_hit = 0 
            
        acc_interval = int(len(dataset) * 0.1)

        for iter, (states, target) in enumerate(dataset):
            states = states.squeeze(dim=0)
            target = target.squeeze(dim=0)

            states = states.to(self.device)
            target = target.to(self.device)

            preds = net(states) 

            optimizer.zero_grad()

            loss = loss_func(preds, target)
            loss.backward()
            optimizer.step()

            predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)
            target_index = torch.argmax(target, dim=1)
            # Compare the predicted classes to the target labels
            correct_preds += torch.sum(predicted_classes == target_index).item()
            top5_hit += self.batch_topk_hit(preds, target_index)

            total_preds += target.shape[0]

            if iter % acc_interval == 0 and iter != 0:
                print(f'Accumulated training accuracy [{100 * iter / len(dataset):.2f}%]: top1: {correct_preds / total_preds:.4f}, top5: {top5_hit / total_preds:.4f}')

        return correct_preds / total_preds
    

    def test(self, dataset, net, e):
        net.eval()
        correct_preds = 0
        total_preds = 0
        top5_hit = 0
        with torch.no_grad():
            for states, target in tqdm(dataset, desc=f'epoch {e}'):

                states = states.squeeze(dim=0)
                target = target.squeeze(dim=0)

                states = states.to(self.device)
                target = target.to(self.device)

                preds = net(states) 

                predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)
                target_index = torch.argmax(target, dim=1)
                # Compare the predicted classes to the target labels
                correct_preds += torch.sum(predicted_classes == target_index).item()
                top5_hit += self.batch_topk_hit(preds, target_index)

                total_preds += target.shape[0]


        return correct_preds / total_preds, top5_hit / total_preds
    

    def batch_topk_hit(self, preds, label_index, k=5):
        preds = torch.softmax(preds, dim=1)
        _, topk_indices = preds.topk(k, dim=-1) # output (batch, k)

        # Check if the true label_index is in the top-k predicted labels for each example
        batch_size, pred_size = preds.shape

        correct = 0

        for i in range(batch_size):
            if label_index[i] in topk_indices[i]:
                correct += 1

        return correct

