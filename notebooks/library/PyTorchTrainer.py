import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import time
import sys
import os
import numpy as np
from library.LSTM import NeuralNet
import math


class PyTorchTrainer():
    """
    Class responsible for abstracting the Neural Network initialisation,
    PyTorch data initialisation, forward and backward propagation and loss 
    calculation
    """
    output_dim = 1
    epoch_log_summary = []
    

    def __init__(self, X_train, y_train, embedding_matrix, max_features, cudaEnabled = False):
        super().__init__()

        self.cudaEnabled = cudaEnabled

        print('Creating tensors')
        # Setting up the various Pytorch tensors
        self.X_train = torch.tensor(X_train, dtype=torch.long)
        # self.y_train = torch.tensor(np.vstack(y_train.iloc[:, np.newaxis]), dtype=torch.float32)
        self.y_train = torch.tensor(np.hstack([y_train]), dtype=torch.float32)

        print('Creating model')
        # Initialising the Neural Network
        self.model = NeuralNet(embedding_matrix, (self.y_train.shape[-1] - 1), max_features)

        # Converting the tensor into cuda tensors if GPU is available
        if self.cudaEnabled:
            self.model.cuda()
            self.X_train = self.X_train.cuda()
            self.y_train = self.y_train.cuda()

        print('Creating datasets')
        # Packaging multiple tensors into PyTorch TensorDataset
        self.train_dataset = data.TensorDataset(self.X_train, self.y_train)

    def _sigmoid(self, x):
        """
        Sigmoid function, a differentiable function that converts a real number to one of value between 0-1
        """
        return 1 / (1 + np.exp(-x))

    def train_and_test(self, X_test, loss_fn = nn.BCEWithLogitsLoss(reduction='mean'), learning_rate = 0.001, batch_size = 512,
                    n_epochs = 1, enable_checkpoint_ensemble = True):

        """
        Function wrapping the train() function (responsible for the neural network forward propagation, 
        loss calculating, back propagation) adding test set prediction to the train 


        Parameters:
        ---------------
        
        loss_fn: a loss function like BCEWithLogitsLoss, CrossEntropyLoss.
            For soft-labels use BCEWithLogitsLoss 
            ref: https://discuss.pytorch.org/t/loss-function-crossentropyloss-vs-bcewithlogitsloss/16089
        
        learning_rate: learning rate which will be adjusted by the scheduler 
            ref: https://discuss.pytorch.org/t/using-the-new-learning-rate-scheduler/6726
        
        batch_size: defines the batch size that is fed to the NN
        
        n_epochs: number of epochs used for the training
        
        epoch_fn: optional closure function that is executed before the end of an epoch.
            This allows for additional functionality (e.g. test set prediction) to run at
            the end of each epoch.
        
        enable_checkpoint_ensemble: boolean flag that determines whether the outputted 
            prediction is the result of the weighted average of different predictions
            performed at each epoch or just the last model prediction

        """
        #  Packaging test dataset into PyTorch TensorDataset 
        test_dataset = self._convert_to_tensor_dataset(X_test)

        all_test_preds = []
        # calculating a different weight to be used to calculate a weighed average prediction 
        checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]

        # closure passed to to the train function to be executed at the end of each batch
        # it predicts on the test set and stores the predictions
        predict_closure = lambda: all_test_preds.append(self.predict( test_dataset, batch_size = batch_size ))

        # Calling the train function
        self.train(loss_fn, learning_rate, batch_size, n_epochs, predict_closure)
 

        if enable_checkpoint_ensemble:
            # test set prediction is the result of the weighted average of different predictions
            test_preds = [np.average(all_test_preds, weights=checkpoint_weights, axis=0)]
        else:
            # test set prediction is the result of the last epoch's prediction
            test_preds = all_test_preds

        return test_preds

    def train(self, loss_fn = nn.BCEWithLogitsLoss(reduction='mean'), learning_rate = 0.001, batch_size = 512,
                    n_epochs = 1, epoch_fn = None):
        """
        Function responsible for the neural network forward propagation, 
        loss calculating and back propagation

        Parameters:
        ---------------
        
        loss_fn: a loss function like BCEWithLogitsLoss, CrossEntropyLoss.
            For soft-labels use BCEWithLogitsLoss 
            ref: https://discuss.pytorch.org/t/loss-function-crossentropyloss-vs-bcewithlogitsloss/16089
        
        learning_rate: learning rate which will be adjusted by the scheduler 
            ref: https://discuss.pytorch.org/t/using-the-new-learning-rate-scheduler/6726
        
        batch_size: defines the batch size that is fed to the NN
        
        n_epochs: number of epochs used for the training
        
        epoch_fn: optional closure function that is executed before the end of an epoch.
            This allows for additional functionality (e.g. test set prediction) to run at
            the end of each epoch.

        """
        print('Training model')

        self.output_dim= self.y_train.shape[-1]

        param_lrs = [{'params': param, 'lr': learning_rate} for param in self.model.parameters()]
        optimizer = torch.optim.Adam(param_lrs, lr=learning_rate)

        # Sets the learning rate of each parameter group to the initial lr
        # times a given function. When last_epoch=-1, sets initial lr as lr.
        # ref: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

        # Data loader is to combines a dataset and a sampler, and provides 
        # single- or multi-process iterators over the dataset.
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(n_epochs):

            # Execution metrics gathering
            start_time = time.time()

            print('\nStarting Epoch {}'.format(epoch + 1))

            # Decays the learning rate
            scheduler.step()

            # Setting the model in "train" mode
            # ref: https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
            self.model.train()
            # Initialising/resetting avg_loss and counter variables
            avg_loss = 0.
            counter = 0

            # Looping through the data in batches
            for data in train_loader:

                counter += 1

                x_batch = data[:-1]
                y_batch = data[-1]

                # Forward propagation is called
                y_pred = self.model(*x_batch)

                # Calculating loss between prediction and 
                # true labels
                loss = loss_fn(y_pred, y_batch)

                # Manually zeroing the gradients before the backward 
                # as Pytorch retains gradients
                # ref: https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/12
                optimizer.zero_grad()

                # backpropagating and calculating the gradient
                loss.backward()
                # updating the parameters based on the calculated gradient
                optimizer.step()

                # Averaging the loss
                # Note: here the len(train_loader) is the the number 
                # of iteration will be performed given the batch_size and training data
                # as in len = math.ceil(n_records/batch_size)
                avg_loss += loss.item() / len(train_loader)

                # Logging to console percent of epoch traning done
                self._log_epoch_progress(counter, batch_size, max_logs = 10)

            # Logging current epoch summary
            elapsed_time = time.time() - start_time
            print('\n' + self._log_epoch_summary(epoch, n_epochs, avg_loss, elapsed_time))

            # epoch_fn is a closure that the callee can pass and is executed
            # at the end of the epoch
            if epoch_fn is not None:
                epoch_fn()

        # At the end of the training we are printing a summary of the
        # training (this is glance at once at the loss)
        [print(es) for es in self.epoch_log_summary]


    def predict(self, test_dataset, batch_size = 512):
        """
        Runs a prediction on the submitted dataset

        Parameters
        -----------

        test_dataset
        batch_size

        """
        # Sets the module in evaluation mode
        # ref https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615        
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialising the output numpy array
        test_preds = np.zeros((len(test_dataset), self.output_dim))

        # Looping through the data in batches
        for i, x_batch in enumerate(test_loader):
            # Performing a prediction (model.forward) on the batch 
            # calling the sigmoid to map the result to a value between  0-1 
            y_pred = self._sigmoid(self.model(*x_batch).detach().cpu().numpy())
            # Appending to the output variable
            batch_lower_bound = i * batch_size
            batch_upper_bound = (i+1) * batch_size
            test_preds[batch_lower_bound:batch_upper_bound, :] = y_pred

        # Resetting the model to train mode
        self.model.train()

        return test_preds[:, 0]

    def _log_epoch_summary(self, epoch, n_epochs, avg_loss, elapsed_time):
       
        self.epoch_log_summary\
            .append('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'\
                .format(epoch + 1, n_epochs, avg_loss, elapsed_time))
        
        return self.epoch_log_summary[-1]

    def _log_epoch_progress(self, counter, batch_size, max_logs = 20):
        
        # Evaluates to true/false: based on the max_logs determines if it is time to log 
        is_time_to_log = (counter * batch_size) % math.floor(len(self.train_dataset)/max_logs) < batch_size
        
        if is_time_to_log:
            # Calculates the percent of progress so far
            progress = counter * batch_size / len(self.train_dataset) * 100
            print('{0:.2f}'.format(progress))

    
    def convert_to_tensor_dataset(self, X_data):
        
        pdata = torch.tensor(X_data, dtype=torch.long)

        if self.cudaEnabled:
            pdata = pdata.cuda()

        return data.TensorDataset(pdata)


