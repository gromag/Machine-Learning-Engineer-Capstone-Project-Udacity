import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import time

class NeuralNet(nn.Module):
    """
     Neural Network comprised of 3 main components:

     An embeddings layer, non trainable which holds the embeddings vectors as parameters, 
     it provides a look up for each words so to be converted from a dictionary index to a 
     pre-trained embedding vector.

     Two bidirectional LSTM layers. The bidirectional flag allows both earlier and later 
     words to provide cell state to the cell at time step t.

     Global average pooling followed by a max pooling.

     The result is 

     Two fully connected linear layers with identical input and output dimensions.


     Credits go to Benjamin Minixhofer, as this class is based on Benjamin's Kernel:
     https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version
    """

    def __init__(self, embedding_matrix, num_aux_targets, dict_length, lstm_output_dim = 90, embeddings_spacial_dropout=0.3):
        """
        
        Parameters:
        ---------------
        embedding_matrix:  a matrix containing the word embeddings with dimension (h, w) where 
            `h` is the length of the dictionary,
            `w` is each single word dimension in a vectorial space
        dict_length: dictionary size

        """

        super(NeuralNet, self).__init__()

        print('Neural Net initialising')


        # retrieving the length of the vector representing each word
        # this is going to be the first LSTM layer input dimension
        word_vect_dimension = embedding_matrix.shape[1]

        print('Creating embeddings')
        self.embedding = nn.Embedding(dict_length, word_vect_dimension)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        # We are using a pre-trained embedding layer so we don't 
        # want to update weights during  backpropagation
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(embeddings_spacial_dropout)


        print('Creatingg LSTMs')
        is_bidirectional = True
        direction_count = 2 if is_bidirectional else 1
        self.lstm1 = nn.LSTM(word_vect_dimension, lstm_output_dim, bidirectional= is_bidirectional, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_output_dim * direction_count, lstm_output_dim, bidirectional = is_bidirectional , batch_first=True)
        self.lstm3 = nn.LSTM(lstm_output_dim * direction_count, lstm_output_dim, bidirectional = is_bidirectional , batch_first=True)
        
        print('Creating linear')
        num_of_pooling_layers = 2
        linear_in_out_dim = num_of_pooling_layers * lstm_output_dim  * direction_count

        self.linear1 = nn.Linear(linear_in_out_dim, linear_in_out_dim)
        self.linear2 = nn.Linear(linear_in_out_dim, linear_in_out_dim)

        print('Creating output')
        self.linear_out = nn.Linear(linear_in_out_dim, 1)
        self.linear_aux_out = nn.Linear(linear_in_out_dim, num_aux_targets)

    def forward(self, x):

        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        h_lstm3, _ = self.lstm3(h_lstm2)

        # global average pooling
        avg_pool = torch.mean(h_lstm3, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm3, 1)

        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
       
        out = torch.cat([result, aux_result], 1)

        return out

class SpatialDropout(nn.Dropout2d):

    def forward(self, x):
        """
        N: sample dimension (equal to the batch size)
        T: time dimension (equal to MAX_LEN)
        K: feature dimension (equal to 300 because of the 300d embeddings)
        
        """
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
