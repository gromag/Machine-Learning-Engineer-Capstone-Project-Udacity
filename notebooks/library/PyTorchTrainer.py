import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import time
# TODO remove is not used
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
import sys
import os
import numpy as np
from library.LSTM import NeuralNet
import math


# disable progress bars when submitting
def is_interactive():
    return 'SHLVL' not in os.environ

if not is_interactive():
    print('Not interactive mode')
    def nop(it, *a, **k):
        return it

    tqdm = nop

class PyTorchTrainer():

    def __init__(self, X_train, X_test, y_train, embedding_matrix, max_features, cudaEnabled = False):
        super().__init__()

        print('Creating tensors')
        self.X_train = torch.tensor(X_train, dtype=torch.long)
        self.y_train = torch.tensor(np.vstack(y_train[:, np.newaxis]), dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.long)

        print('Creating model')
        self.model = NeuralNet(embedding_matrix, self.y_train.shape[-1], max_features)

        if cudaEnabled:
            self.model.cuda()
            self.X_train = self.X_train.cuda()
            self.y_train = self.y_train.cuda()
            self.X_test = self.X_test.cuda()

        print('Creating datasets')
        self.train_dataset = data.TensorDataset(self.X_train, self.y_train)
        self.test_dataset = data.TensorDataset(self.X_test)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train_model(self, loss_fn = nn.BCEWithLogitsLoss(reduction='mean'), lr = 0.001, batch_size = 512,
                    n_epochs = 1, enable_checkpoint_ensemble = True):

        print('Training model')

        output_dim= self.y_train.shape[-1]

        param_lrs = [{'params': param, 'lr': lr} for param in self.model.parameters()]
        optimizer = torch.optim.Adam(param_lrs, lr=lr)

        # provides several methods to adjust the learning rate based on the number of epochs.
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        all_test_preds = []

        checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]

        print(checkpoint_weights)

        for epoch in range(n_epochs):

            print('Epoch {}'.format(n_epochs))

            start_time = time.time()

            scheduler.step()

            self.model.train()
            avg_loss = 0.


            counter = 0


            for data in tqdm(train_loader, disable=False):

                counter += 1

                x_batch = data[:-1]
                y_batch = data[-1]

                y_pred = self.model(*x_batch)

                # Output max 20 times per epoch
                if (counter * batch_size) % math.floor(len(self.train_dataset)/20) < batch_size:
                    output_step = '\r {0:.2f}%'.format(counter * batch_size / len(self.train_dataset) * 100)
                    print(output_step, end="")
                    sys.stdout.flush()

                loss = loss_fn(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss += loss.item() / len(train_loader)

            #sets the module in evaluation mode
            self.model.eval()
            test_preds = np.zeros((len(self.test_dataset), output_dim))

            for i, x_batch in enumerate(test_loader):
                y_pred = self._sigmoid(self.model(*x_batch).detach().cpu().numpy())
                batch_lower_bound = i * batch_size
                batch_upper_bound = (i+1) * batch_size
                test_preds[batch_lower_bound:batch_upper_bound, :] = y_pred

            all_test_preds.append(test_preds)

            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
                  epoch + 1, n_epochs, avg_loss, elapsed_time))

        if enable_checkpoint_ensemble:
            test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)
        else:
            test_preds = all_test_preds[-1]

        return test_preds
