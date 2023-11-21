import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class MetaLearner(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

class BlendingClassifier(nn.Module):
    def __init__(self, estimators, train_epochs=4, meta_epochs=15, device='cuda'):
        '''
            estimators: a list of class of model 
            meta_estimator: the class of meta learner
        '''
        super().__init__()

        self.estimators = estimators
        self.meta_estimator = MetaLearner(in_features=len(estimators) * 3, out_features=3)

        self.train_epochs = train_epochs
        self.device = device

        self.meta_epochs = meta_epochs

        self.meta_train_X = None
        self.meta_train_y = None

        self.meta_test_X = None
        self.meta_test_y = None


    def fit(self, train_X, train_y, val_X, val_y, test_X, test_y, bagging=True, save_path='model_params/blender.pth', plot=True):
        self.meta_train_X = [] # the inputs of meta learner are the output of estimator from the val set
        self.meta_train_y = val_y # the labels of val set are the labels for meta learner

        self.meta_test_X = [] # the inputs of meta leaner are the output of estimator from the test set 
        self.meta_test_y = test_y # the labels (evaluation only) of test set are the labels for meta learner
        
        train_accs = []
        for estimator in tqdm(self.estimators, dynamic_ncols=True, desc='fitting estimators'):
            estimator.to(self.device)

            pred_val, pred_test = self.train_estimator(estimator, train_X.copy(), train_y.copy(), val_X, test_X, bagging)
            train_accs.append(self.eval_estimator(estimator, train_X, train_y))
    
            self.meta_train_X.append(pred_val)
            self.meta_test_X.append(pred_test)


            estimator.cpu()

        
        self.meta_train_X = np.array(self.meta_train_X)
        self.meta_test_X = np.array(self.meta_test_X)


        self.meta_train_X = np.transpose(self.meta_train_X, (1, 0, 2))
        self.meta_test_X = np.transpose(self.meta_test_X, (1, 0, 2))

        self.meta_train_X = np.reshape(self.meta_train_X, (self.meta_train_X.shape[0], -1))
        self.meta_test_X = np.reshape(self.meta_test_X, (self.meta_test_X.shape[0], -1))


        train_accs, test_accs = self.train_meta_estimator(save_path)


        if plot:
            if not os.path.exists('fig'):
                print('mkdir fig')
                os.mkdir('fig')
            plt.plot(train_accs, label='train acc')
            plt.plot(test_accs, label='test acc')
            plt.legend()
            plt.savefig('fig/meta_curve.png')


        print(f'train accs of meta learner: {train_accs}')
        print(f'test_accs of meta learner: {test_accs}')



    def train_estimator(self, estimator, train_X, train_y, val_X, test_X, bagging):
        estimator.train()
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            estimator.parameters(), 
            lr=1e-3, 
        ) 
        
        if bagging:
            bagging_indices = np.random.choice(range(len(train_X)), int(len(train_X)))
            train_X = train_X[bagging_indices]
            train_y = train_y[bagging_indices]


        train_set = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
        train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=6)

        # train the estimator
        for _ in range(self.train_epochs):
            for X, y in train_loader:
                X, y = X.to(self.device), y.type(torch.LongTensor).to(self.device)

                y_pred = estimator(X)

                optimizer.zero_grad()
                loss = loss_func(y_pred, y)
                loss.backward()
                optimizer.step()

        
        # get pred on val set and test set
        with torch.no_grad():
            estimator.eval()

            # get pred_val
            pred_val = torch.empty(0)
            val_set = TensorDataset(torch.from_numpy(val_X), torch.zeros(np.shape(val_X))) # dummy label 
            val_loader = DataLoader(val_set, batch_size=256, num_workers=6)
            with torch.no_grad():
                for X, _ in val_loader:
                    X = X.to(self.device)
                    y_pred = estimator(X)

                    y_pred = torch.softmax(y_pred, dim=1).cpu()
                    pred_val = torch.concat((pred_val, y_pred), dim=0)


            # get pred_test 
            pred_test = torch.empty(0)
            test_set = TensorDataset(torch.from_numpy(test_X), torch.zeros(np.shape(test_X))) # dummy label
            test_loader = DataLoader(test_set, batch_size=256, num_workers=6)
            with torch.no_grad():
                for X, _ in test_loader:
                    X = X.to(self.device)
                    y_pred = estimator(X)
                    
                    y_pred = torch.softmax(y_pred, dim=1).cpu()
                    pred_test = torch.concat((pred_test, y_pred), dim=0)


        return np.array(pred_val), np.array(pred_test)


    def train_meta_estimator(self, save_path):
        self.meta_estimator.to(self.device)

        # fit the final estimator
        train_set = TensorDataset(torch.from_numpy(self.meta_train_X), torch.Tensor(self.meta_train_y))
        train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=6)
        test_set = TensorDataset(torch.from_numpy(self.meta_test_X), torch.Tensor(self.meta_test_y))
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=6)

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.meta_estimator.parameters(), 
            lr=1e-3, 
        ) 

        train_accs = []
        test_accs = []
        for _ in tqdm(range(self.meta_epochs), dynamic_ncols=True, desc='fitting meta learner'):
            # training
            self.meta_estimator.train()
            correct_preds = 0
            total_preds = 0
            for X, y in train_loader:
                X, y = X.to(self.device), y.type(torch.LongTensor).to(self.device)

                y_pred = self.meta_estimator(X)

                optimizer.zero_grad()
                loss = loss_func(y_pred, y)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    y_pred = torch.softmax(y_pred, dim=1)
                    predicted_classes = torch.argmax(y_pred, dim=1)

                    correct_preds += torch.sum(predicted_classes == y).item()
                    total_preds += y.shape[0]

            train_accs.append(correct_preds / total_preds)

            
            # testing
            self.meta_estimator.eval()            
            correct_preds = 0
            total_preds = 0
            best_acc = 0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    y_pred = self.meta_estimator(X)

                    y_pred = torch.softmax(y_pred, dim=1)
                    predicted_classes = torch.argmax(y_pred, dim=1)

                    correct_preds += torch.sum(predicted_classes == y).item()
                    total_preds += y.shape[0]

            acc = correct_preds / total_preds
            if acc > best_acc:
                best_acc = acc
                self.save(save_path)

            test_accs.append(acc)




        self.meta_estimator.cpu()

        return train_accs, test_accs
    
    def eval_estimators(self, X, y):
        estimator_acc = []
        for estimator in self.estimators:
            estimator_acc.append(self.eval_estimator(estimator, X, y))

        return estimator_acc

    def eval_estimator(self, estimator, X, y):
        estimator.to(self.device)
        estimator.eval()
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=256, num_workers=6)

        with torch.no_grad():
            correct_preds = 0
            total_preds = 0
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                
                y_pred = estimator(X)
                y_pred = torch.softmax(y_pred, dim=1)
                predicted_classes = torch.argmax(y_pred, dim=1)

                correct_preds += torch.sum(predicted_classes == y).item()
                total_preds += y.shape[0]
                
        estimator.cpu()
        return correct_preds / total_preds


   

    def pred_proba(self, X):
        dataset = TensorDataset(torch.from_numpy(X), torch.zeros(X.shape)) # dummy labels
        loader = DataLoader(dataset, batch_size=256, num_workers=6)
        
        with torch.no_grad():

            # processing input of meta learner
            total_preds = []
            for estimator in self.estimators:
                estimator.eval()
                estimator.to(self.device)
                estimator_pred = torch.empty(0)
                for X, _ in loader:
                    X = X.to(self.device)
                    y_pred = estimator(X)
                    y_pred = torch.softmax(y_pred, dim=1).cpu()

                    estimator_pred = torch.concat((estimator_pred, y_pred), dim=0)
                
                total_preds.append(np.array(estimator_pred))
                estimator.cpu()
            
            total_preds = np.array(total_preds)
            total_preds = np.transpose(total_preds, (1, 0, 2))
            total_preds = np.reshape(total_preds, (total_preds.shape[0], -1))

            
            # meta learner prediction
            self.meta_estimator.eval()
            self.meta_estimator.to(self.device)
            meta_dataset = TensorDataset(torch.from_numpy(total_preds), torch.zeros(total_preds.shape)) # dummy labels
            meta_loader = DataLoader(meta_dataset, batch_size=256, num_workers=6)
            meta_preds = torch.empty(0)
            for X, _ in meta_loader:
                X = X.to(self.device)
                y_pred = self.meta_estimator(X)
                y_pred = torch.softmax(y_pred, dim=1).cpu()
                meta_preds = torch.concat((meta_preds, y_pred), dim=0)

            self.meta_estimator.cpu()

        return meta_preds


    def pred_proba_tta(self, X):
        '''
            make prediction with tta input: (dataset size, 8, c, h, w)
            each estimator makes prediction base on the 8 given inputs and comes up with the mean prob of 8 outputs, which then serves as the input of meta learner
        '''

        with torch.no_grad():
            dataset = torch.from_numpy(X) # dummy labels
            total_preds = []
            for estimator in self.estimators:
                estimator.eval()
                estimator.to(self.device)
                estimator_pred = torch.empty(0)
                for X in dataset:
                    # X: (8, c, h, w)
                    X = X.to(self.device)
                    pred = estimator(X)


                    pred = torch.softmax(pred, dim=1).mean(dim=0, keepdim=True).cpu()

                    estimator_pred = torch.concat((estimator_pred, pred), dim=0)
                
                total_preds.append(np.array(estimator_pred))
                estimator.cpu()

            total_preds = np.array(total_preds)
            total_preds = np.transpose(total_preds, (1, 0, 2))
            total_preds = np.reshape(total_preds, (total_preds.shape[0], -1))

            # meta learner prediction
            self.meta_estimator.eval()
            self.meta_estimator.to(self.device)
            meta_dataset = TensorDataset(torch.from_numpy(total_preds), torch.zeros(total_preds.shape)) # dummy labels
            meta_loader = DataLoader(meta_dataset, batch_size=256, num_workers=6)
            meta_preds = torch.empty(0)
            for X, _ in meta_loader:
                X = X.to(self.device)
                y_pred = self.meta_estimator(X)
                y_pred = torch.softmax(y_pred, dim=1).cpu()
                meta_preds = torch.concat((meta_preds, y_pred), dim=0)

            self.meta_estimator.cpu()


            return meta_preds
            


    def acc_score(self, X, y, tta=False):
        class_preds = torch.argmax(self.pred_proba(X), dim=1) if not tta else torch.argmax(self.pred_proba_tta(X), dim=1)
        return (torch.sum(class_preds == torch.from_numpy(y)) / y.size).item()
    

    def save(self, save_path):
        torch.save(self, save_path)