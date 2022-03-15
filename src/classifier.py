from transformers import get_scheduler
from torch.optim import AdamW
import torch.nn as nn
import torch
import numpy as np

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from model import CustomBERTModel
from create_dataset import create_data_loader

from config import device


class Classifier:
    """The Classifier"""

    #############################################
    def train(self, trainfile, devfile=None, lr = 1e-5, num_epochs = 30):
        """
        Trains the classifier model on the training set stored in file trainfile
        WARNING: DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        self.model = CustomBERTModel()
        model = CustomBERTModel()
        optimizer = AdamW(model.parameters(), lr=lr)

        data_loader_train = create_data_loader(trainfile, train=True)
        data_loader_test = create_data_loader(devfile, train=False)

        num_training_steps = num_epochs * len(data_loader_train)
        lr_scheduler = get_scheduler(name="linear", 
                                    optimizer=optimizer, 
                                    num_warmup_steps=0, 
                                    num_training_steps=num_training_steps * 2
        )

        
        model = model.to(device)

        TEST_SIZE = 388
        loss_f = nn.CrossEntropyLoss()
        loss_hist = []
        acc_hist = []
        loss_hist_test = []
        acc_hist_test = []
        best_test_acc = 0
        progress_bar = tqdm(range(num_epochs*len(data_loader_train)))
        
        for epoch in range(num_epochs):
            print(f"Running epoch : {epoch}/{num_epochs}")
            for batch in data_loader_train:
                labels = batch['label'].to(torch.int64)
                labels = labels.to(device)
                asp_id = batch['aspect_id']
                batch = {k: v.to(device) for k, v in batch.items() if k != 'label'}

                outputs = model(**batch)
                loss = loss_f(outputs, labels)
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                loss_hist.append(loss.detach().cpu().numpy())
                acc_hist.append(torch.mean((outputs.argmax(dim=1)== labels).to(torch.float32)).detach().cpu().numpy())

            fig,axes = plt.subplots(1,2, figsize = (10,10))
            N=100
            axes[0].set_title("Loss")
            axes[0].set_xlabel("Iteration")
            axes[0].set_ylabel("Loss")
            axes[0].plot(loss_hist, alpha = 0.2, color = 'b')
            axes[0].plot(np.convolve(loss_hist, np.ones(N)/N, mode='valid'), color='b')

            axes[1].set_title("Accuracy")
            axes[1].set_xlabel("Iteration")
            axes[1].set_ylabel("Accuracy")
            axes[1].plot(acc_hist, alpha = 0.2, color='b')
            axes[1].plot(np.convolve(acc_hist, np.ones(N)/N, mode='valid'), color='b')

            acc_test = 0
            loss_test = 0    
            for batch in data_loader_test:
                labels = batch['label'].to(torch.int64)
                labels = labels.to(device)
                asp_id = batch['aspect_id']
                batch = {k: v.to(device) for k, v in batch.items() if k != 'label'}

                outputs = model(**batch)
                loss = loss_f(outputs, labels)   
                loss_test += loss.detach().cpu().numpy() * outputs.shape[0]
                acc_test += torch.sum((outputs.argmax(dim=1) == labels).to(torch.float32)).detach().cpu().numpy()
            loss_hist_test.append(loss_test / TEST_SIZE)
            acc_hist_test.append(acc_test / TEST_SIZE)
            
            
            if acc_hist_test[-1] >best_test_acc:
                best_test_acc = acc_hist_test[-1]
                print(f"Best accuracy : {best_test_acc}")
                self.model = model
            X = [i * len(data_loader_train) for i in range(epoch+1)]    

            axes[0].plot(X,loss_hist_test, color = 'r')
            axes[1].plot(X,acc_hist_test, color = 'r')


            plt.show()

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        y_pred = []

        data_loader_test = create_data_loader(datafile, train=False)
        for batch in data_loader_test:
            batch = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            outputs = list(self.model(**batch).argmax(dim=1).cpu().numpy())
            y_pred = y_pred + outputs
        
        polarity = ["positive", "negative", "neutral"]
        polarity_out = [polarity[i] for i in y_pred]
        return polarity_out
