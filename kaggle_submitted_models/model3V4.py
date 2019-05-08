# SETTINGS SECTION
# ---------------------------------------------------------------
# KAGGLE SETTINGS
import os
import time
import gc

# SETTINGS
TOXICITY_COLUMN = 'toxic'
IS_ON_KAGGLE =  os.path.exists('../input')
PATH_PREFIX = '../' if IS_ON_KAGGLE  else ''
FASTTEXT_PATH = PATH_PREFIX + 'input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
FASTTEXT_SAMPLES = -1
TRAIN_PATH = PATH_PREFIX + 'input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
TEST_PATH = PATH_PREFIX + 'input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
STOP_WORDS_PATH = PATH_PREFIX + 'input/stopwords/english'
SAMPLE_PERCENT = 1
TEST_SAMPLE_PERCENT = 1
EPOCHS = 10
TEXT_TOKEN_LENGTH = 220
TRAIN_TEST_SPLIT_PERCENT = 0.01


# AUTO ASSEMBLED CLASS IMPORT SECTION 
# ---------------------------------------------------------------


# ObsceneTextPreprocessor.py
# ---------------------------------------------------------------

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import string



class ObsceneTextPreprocessor():

    # Words we want to retain in our set
    leave_words = ['i', 'you', 'myself', 'yourself', 'no', 'she', 'themselves', 'he', 'we', 'only',\
                        'up', 'your', 'my', 'mine', 'yours', 'hers', 'her', 'his', 'from', 'to', 'for']
    #Regex to leave special characters for obscene words like f##k
    tokenizer = RegexpTokenizer(r'[\%\@\&\$\#\w\!\*]+')


    def __init__(self, lower=False, stop_words_path = './input/stopwords/english'):
        super().__init__()

        self.lower = lower
        #Use the English stopword list
        self.__load_stopwords(stop_words_path)
        # stop_words = set(stopwords.words('english'))
        self.stop_words = [w for w in self.stop_words if (not w in self.leave_words)]


    def __lower_fn(self, w):
        if self.lower:
            return w.lower()
        return w

    def __load_stopwords(self, stop_words_path):
        # TODO: FIX for both local and Kaggle environment
        f = open(stop_words_path, 'r')
        self.stop_words = f.readlines()
        f.close()

    def clean_doc(self, doc):
        out = [self.__lower_fn(w) for w in self.tokenizer.tokenize(doc) if (not w.lower() in self.stop_words)]
        return out

    def clean(self, data):

        return [' '.join(self.clean_doc(doc)) for doc in data]


# Reproducibility.py
# ---------------------------------------------------------------

import random
import torch
import os
import numpy as np


class Reproducibility():

    RANDOM_SEED = 1234


    def seed_everything(seed=None):
        """
        Thanks to
        https://www.kaggle.com/kunwar31/simple-lstm-with-identity-parameters-fastai/log
        https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version

        """
        if seed is None:
            seed = Reproducibility.RANDOM_SEED

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True



# KaggleSubmitter.py
# ---------------------------------------------------------------

import pandas as pd
import numpy as np

class KaggleSubmitter():

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def save_submission(data, predictions, filename = 'submission.csv'):

        assert (len(data) == len(predictions)), "Predictions and test data have different lengths"

        submission = pd.DataFrame.from_dict({
            'id': data['id'],
            'prediction': predictions.flatten()
        })

        submission.to_csv(filename, index=False)

        return submission


# Embeddings.py
# ---------------------------------------------------------------

# Embeddings
import io
import os
import numpy as np
import gc

class AbstractEmbeddings():

    _words = None
    _average_embedding = None
    _dim = (0,0)

    def __init__(self, sub_instance, path = '', max_embeddings_count = -1):

        assert(os.path.exists(path))

        self.path = path
        self.MAX_EMBEDDINGS_COUNT = max_embeddings_count

        self._load_embeddings()

    def _load_embeddings(self):
        raise NotImplementedError()


    def get_embedding(self, word):
        """
        Get the embedding corresponding to a word if found
        otherwise it returns the average embedding

        Input:
        ------
        word: string

        Returns:
        ------
        (embedding, is_found)

        """
        if word not in self.embeddings:
            return (self._average_embedding, False)

        return (self.embeddings[word], True)

    # def get_all_words(self):

    #     return self._words

    def shape(self):
        return self._dim


class FastTextEmbeddings(AbstractEmbeddings):

    def __init__(self, path = '', max_embeddings_count = -1):
        super().__init__(self, path, max_embeddings_count)

    def _load_embeddings(self):
        """
        Loads FastText embeddings.
        https://fasttext.cc/docs/en/english-vectors.html
        """
        fin = io.open(self.path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())

        length = n

        if(self.MAX_EMBEDDINGS_COUNT != -1):
            length = self.MAX_EMBEDDINGS_COUNT
            
        print('Fasttext dim: {}'.format((length, d)))

        data = {}

        for index in range(length):
            line = fin.readline()

            tokens = line.rstrip().split(' ')
            # data[tokens[0]] = list(map(float, tokens[1:])) <- MEMORY HUNGRY
            # MEMORY EFFICIENT below
            data[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            
            if index % (length/20) == 0:
                print('Fasttext loaded {} words'.format(index))
        
        fin.close()
        
        self.embeddings  = data
        print('Fasttext embedding loaded')

        # Gathering further datapoints
        arr = np.array([self.embeddings[a] for a in self.embeddings])
        self._average_embedding = arr.mean(axis=0)
        # self._words = np.array([a for a in self.embeddings])
        self._dim = (length, d)


# TextTokenizer.py
# ---------------------------------------------------------------

import numpy as np
from keras.preprocessing import text, sequence


class TextTokenizer():

    UNKNOWN = '<UNKNOWN>'
    PADDING = ''
    MAXLEN = 100
    def __init__(self, train, test, num_words = None, maxlen=100, filters = '"()+-./:;<=>[\\]^_`{|}~\t\n', lower=False):
        super().__init__()

        self.filters = filters
        self.MAXLEN = maxlen

        # It expects to be passed both training and testing data
        corpus = train + test

        self.tokenizer = text.Tokenizer(num_words = num_words, filters = self.filters, lower=lower, oov_token= self.UNKNOWN)
        self.tokenizer.fit_on_texts(corpus)

        self.document_count = self.tokenizer.document_count
        self.all_words_count =  sum([self.tokenizer.word_counts[w] for w in self.tokenizer.word_counts])
        self.reverse_word_map = dict(map(reversed, self.tokenizer.word_index.items()))



    def transform(self, data):
        """
        Converts each document into an integer sequence representing
        the index of each word in the vocabular.

        The vocabulary was generate in the __init__() of this class
        based on the corpora passed in.
        """
        data = self.tokenizer.texts_to_sequences(data)
        data = sequence.pad_sequences(data, maxlen=self.MAXLEN)
        return data

    def get_word_index_mapping(self):
        return self.tokenizer.word_index

    def get_index_by_word(self, word):

        if word not in self.tokenizer.word_index:
            return len(self.tokenizer.word_index)

        return self.tokenizer.word_index[word]

    def get_word_by_index(self, index):
        """
        Further remarks:
        (see https://github.com/keras-team/keras/issues/9637#issuecomment-402406020)
            - index[0] is reserved for padding
            - index[num_words+1] is reserved for unknown

        """
        if index == 0:
            return self.PADDING

        if index not in self.reverse_word_map:
            return self.UNKNOWN

        return self.reverse_word_map[index]

    def get_stats(self):
        """
        Gets stats of the corpus tokenised.

        Returns
        -----
        Tuple: (document_count, all_words_count, unique_words_count)
        """
        return (self.document_count, self.all_words_count, len(self.tokenizer.word_index))

    def build_embedding_matrix(self, embeddings):
        print('Building matrix')

        emb_width = embeddings.shape()[-1]

        embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1, emb_width))
        unknown_words = []

        for word, i in self.tokenizer.word_index.items():

            embedding_matrix[i], was_it_found = embeddings.get_embedding(word)

            if not was_it_found:
                unknown_words.append(word)

        return embedding_matrix, unknown_words




# PyTorchTrainer.py
# ---------------------------------------------------------------

import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import time
import sys
import os
import numpy as np

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
        self.y_train = torch.tensor(np.vstack(y_train[:, np.newaxis]), dtype=torch.float32)

        print('Creating model')
        # Initialising the Neural Network
        self.model = NeuralNet(embedding_matrix, self.y_train.shape[-1], max_features)

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
        Sigmoid function that converts a real number to one of value between 0-1
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
            test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)
        else:
            # test set prediction is the result of the last epoch's prediction
            test_preds = all_test_preds[-1]

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
                self._log_epoch_progress(counter, batch_size, max_logs = 20)

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

        return test_preds

    def _log_epoch_summary(self, epoch, n_epochs, avg_loss, elapsed_time):
       
        self.epoch_log_summary\
            .append('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'\
                .format(epoch + 1, n_epochs, avg_loss, elapsed_time))
        
        return self.epoch_log_summary[-1]

    def _log_epoch_progress(self, counter, batch_size, max_logs = 20):
        
        # Evaluates to true maximum the passed in max_logs number of times 
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





# ExploratoryAnalysis.py
# ---------------------------------------------------------------

import numpy as np
import pandas as pd

class ExploratoryAnalysis():

    def __init__(self, train_path=None, test_path=None, train=None, test=None):
        self.train = train if train is not None else pd.read_csv(train_path)
        self.test = test if test is not None else pd.read_csv(test_path)

    def describe_train(self):
        return self.train.describe()

    def describe_test(self):
        return self.test.describe()


class JigsawExploratoryAnalysis(ExploratoryAnalysis):
    """

    """
    subgroup_columns = ['severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']

    identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black',
                    'white', 'psychiatric_or_mental_illness']

    TOXIC_COLUMN = 'toxic'
    COMMENT_COLUMN = 'comment_text'

    def __init__(self, train_path = None, test_path = None, train = None, test = None):
        super().__init__(train_path, test_path, train, test)

        self.train[self.identity_columns] = self.train[self.identity_columns].fillna(0)
        self._add_toxic_column()

    def _add_toxic_column(self):
        train_toxic =(self.train['target'].values > 0.5).astype(bool).astype(np.int)
        self.train[self.TOXIC_COLUMN] = train_toxic

    def calculate_stats(self):

        assert self.TOXIC_COLUMN in self.train.columns

        test_percent = len(self.test)/len(self.train)
        toxic_comment_distribution = self.train['toxic'].sum()/len(self.train['toxic'])

        return (test_percent, toxic_comment_distribution)

    def __calculate_toxic_stats_for_column(self, col, threshold = 0.0):

        is_relevant = self.train[col] > threshold
        na_count = self.train[col].isnull().sum()
        rel_records = self.train[is_relevant]
        toxic_count = rel_records['toxic'].sum()
        toxic_percent = toxic_count/len(rel_records)

        return (col, toxic_count, len(rel_records), toxic_percent, na_count)

    def calculate_toxic_stats_for_identities(self, threshold = 0.0):

        toxic_stats = pd.DataFrame(columns=["identity", "toxic_count", "count", "toxic_percent", "na_count"])

        for index, col in enumerate(JigsawExploratoryAnalysis.identity_columns):
            toxic_stats.loc[index] = self.__calculate_toxic_stats_for_column(col, threshold)

        return toxic_stats

    def calculate_toxic_stats_for_subgroups(self, threshold = 0.0):

        toxic_subgroup_stats = pd.DataFrame(columns=["subgroup", "toxic_count", "count", "toxic_percent", "na_count"])

        for index, col in enumerate(JigsawExploratoryAnalysis.subgroup_columns):
            toxic_subgroup_stats.loc[index] = self.__calculate_toxic_stats_for_column(col, threshold)

        return toxic_subgroup_stats


    def __build_no_identities_set_expression(self):
        expr = None
        for col in JigsawExploratoryAnalysis.identity_columns:

            tmp = (self.train[col] == 0)

            if expr is None:
                expr = tmp
            else:
                expr = expr & tmp
        return expr


    def __build_at_least_one_identity_set_expression(self):

        expr = None
        for col in JigsawExploratoryAnalysis.identity_columns:

            tmp = (self.train[col] > 0)

            if expr is None:
                expr = tmp
            else:
                expr = expr | tmp
        return expr

    def calculate_no_identity_stats(self):

        no_identity_train = self.train[self.__build_no_identities_set_expression()]
        identity_train = self.train[self.__build_at_least_one_identity_set_expression()]

        return (no_identity_train, identity_train)




# EvaluationMetrics.py
# ---------------------------------------------------------------

from sklearn import metrics
from sklearn import model_selection
import numpy as np
import pandas as pd

class EvaluationMetrics:
    """
    Benchmark according to Kaggle competition evaluation

    Thanks to https://www.kaggle.com/dborkan/benchmark-kernel
    """
    SUBGROUP_AUC = 'subgroup_auc'
    BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
    BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

    def compute_auc(y_true, y_pred):
        """
        Computes the AUC.

        Note: this implementation is restricted to the binary classification task
        or multilabel classification task in label indicator format.


        Parameters
        ----------
        y_true : array, shape = [n_samples] or [n_samples, n_classes]
            True binary labels or binary label indicators.

        y_pred : array, shape = [n_samples] or [n_samples, n_classes]
            Target scores, can either be probability estimates of the positive
            class, confidence values, or non-thresholded measure of decisions
            (as returned by "decision_function" on some classifiers). For binary
            y_true, y_score is supposed to be the score of the class with greater
            label.
        """
        try:
            return metrics.roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def compute_subgroup_auc(df, subgroup, label, model_name):
        """Computes the AUC for the within-subgroup positive and negative examples."""
        subgroup_examples = df[(df[subgroup] > .5)]
        return EvaluationMetrics.compute_auc(subgroup_examples[label], subgroup_examples[model_name])

    def compute_bpsn_auc(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
        subgroup_negative_examples = df[(df[subgroup] > 0.5) & ~(df[label] > 0.5)]
        non_subgroup_positive_examples = df[~(df[subgroup] > 0.5) & (df[label] > 0.5)]
        examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
        return EvaluationMetrics.compute_auc(examples[label], examples[model_name])

    def compute_bnsp_auc(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
        subgroup_positive_examples = df[(df[subgroup] > 0.5) & (df[label] > 0.5)]
        non_subgroup_negative_examples = df[~(df[subgroup] > 0.5) & ~(df[label] > 0.5)]
        examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
        return EvaluationMetrics.compute_auc(examples[label], examples[model_name])

    def compute_bias_metrics_for_model(dataset,
                                       subgroups,
                                       model,
                                       label_col,
                                       include_asegs=False):
        """Computes per-subgroup metrics for all subgroups and one model."""

        records = []
        for subgroup in subgroups:
            record = {
                'subgroup': subgroup,
                'subgroup_size': len(dataset[(dataset[subgroup] > 0.5)])
            }

            record[EvaluationMetrics.SUBGROUP_AUC] = EvaluationMetrics.compute_subgroup_auc(dataset, subgroup, label_col, model)
            record[EvaluationMetrics.BPSN_AUC] = EvaluationMetrics.compute_bpsn_auc(dataset, subgroup, label_col, model)
            record[EvaluationMetrics.BNSP_AUC] = EvaluationMetrics.compute_bnsp_auc(dataset, subgroup, label_col, model)
            records.append(record)

        return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)

    def calculate_overall_auc(df, model_name, label_column):
        true_labels = df[label_column]
        predicted_labels = df[model_name]
        return metrics.roc_auc_score(true_labels, predicted_labels)

    def power_mean(series, p):
        total = sum(np.power(series, p))
        return np.power(total / len(series), 1 / p)

    def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
        bias_score = np.average([
            EvaluationMetrics.power_mean(bias_df[EvaluationMetrics.SUBGROUP_AUC], POWER),
            EvaluationMetrics.power_mean(bias_df[EvaluationMetrics.BPSN_AUC], POWER),
            EvaluationMetrics.power_mean(bias_df[EvaluationMetrics.BNSP_AUC], POWER)
        ])
        return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)



# LSTM.py
# ---------------------------------------------------------------

import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import time
from tqdm._tqdm_notebook import tqdm_notebook as tqdm

# TODO: add credits
class NeuralNet(nn.Module):

    def __init__(self, embedding_matrix, num_aux_targets, max_features, LSTM_UNITS = 128, DENSE_HIDDEN_LAYERS = 4):
        super(NeuralNet, self).__init__()

        print('Neural Net initialising')

        DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

        embed_size = embedding_matrix.shape[1]

        print('Creating embeddings')
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)


        print('Creatingg LSTMs')
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        print('Creating linear')
        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        print('Creating output')
        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)

    def forward(self, x):

        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)

        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return result

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x



# DataSampler.py
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
from random import sample, seed


class DataSampler():
    """Class responsible for holding, sampling and splitting the training in train and validation set"""

    def __init__(self, data=None, sample_percent = 1):
        """

        """
        super().__init__()

        assert (data is not None), "No data passed"

        try:
            self.RANDOM_SEED = RANDOM_SEED
        except:
            self.RANDOM_SEED = 1234

        assert(sample_percent > 0 and sample_percent <=1)

        self._data = data

        if sample_percent < 1:
            sample_size = math.ceil(len(data) * sample_percent)
            self._data = data.sample(sample_size)

    def data(self):
        return self._data


    def train_test_split(self, X, y, test_size = 0.2):
        """
        Split train data into `train` `test` set

        Input:
        ------
        X:
        y:
        test_size:

        Returns:
        ------
        (X_train, X_test, y_train, y_test)

        """
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=1)

        return (X_train, X_valid, y_train, y_valid)

    def train_test_split_by_columns(self, X_columns, y_column, rows_filter = None, sample_percent = 1, test_size = 0.2):

        """
        Split train data into `train` `test` set

        Input:
        ------
        X_columns:
        y_column:
        rows_filter:
        test_size:
        sample_percent: float 0.0 < n <= 1.0 percentage of samples


        Returns:
        ------
        (X_train, X_test, y_train, y_test)

        """
        X = self.data[X_columns]
        y = self.data[y_column]
        
        assert(len(X) == len(y))
        assert(sample_percent > 0 and sample_percent <=1)


        assert(rows_filter is None or len(rows_filter) == len(X))

        if rows_filter is not None:
            X = X[rows_filter]
            y = y[rows_filter]

        if sample_percent < 1:
            sample_size = math.ceil(len(X) * sample_percent)
            X = X.sample(sample_size, random_state=self.RANDOM_SEED)
            y = y.loc[X.index]

        return self.train_test_split(X, y, test_size)



# WORKFLOW
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------

global_start = time.time()

# Seeding
# -----------------------------------------------------------------
print('Seeding')
Reproducibility.RANDOM_SEED = 1234
Reproducibility.seed_everything()



# Loading traning and test data
# -----------------------------------------------------------------
print('Loading traning and test data')
s = time.time()
explore = JigsawExploratoryAnalysis(train_path = TRAIN_PATH, test_path = TEST_PATH)
print('Finished in {} sec'.format(time.time() - s))



# Collecting sample of data for training
# -----------------------------------------------------------------
print('Collecting sample of data for training')
s = time.time()
train_data_sampler = DataSampler(explore.train, sample_percent = SAMPLE_PERCENT)
test_data_sampler = DataSampler(explore.test,  sample_percent = TEST_SAMPLE_PERCENT)
print('Finished in {} sec'.format(time.time() - s))



# Preprocessing text 
# -----------------------------------------------------------------
print('Preprocessing text ')
s = time.time()
text_processor = ObsceneTextPreprocessor(stop_words_path = STOP_WORDS_PATH)
clean_train_comments = text_processor.clean(train_data_sampler.data()['comment_text'])
clean_test_comments = text_processor.clean(test_data_sampler.data()['comment_text'])
print('Finished in {} sec'.format(time.time() - s))



# Tokenisation
# ----------------------------------------------------------------- 
print('Tokeniser building')
s = time.time()
tokenizer = TextTokenizer(clean_train_comments, clean_test_comments,  maxlen = TEXT_TOKEN_LENGTH)
print('Finished in {} sec'.format(time.time() - s))

print('Tokenisation  of train')
s = time.time()
tokenised_train_comments = tokenizer.transform(clean_train_comments)
print('Finished in {} sec'.format(time.time() - s))

print('Tokenisation  of test')
s = time.time()
tokenised_test_comments = tokenizer.transform(clean_test_comments)
print('Finished in {} sec'.format(time.time() - s))



# Splitting data for Train / Test / Validation
# -----------------------------------------------------------------
print('Splitting data for Train / Test')
s = time.time()
X_train, X_valid, y_train, y_valid =  train_data_sampler.train_test_split(tokenised_train_comments, train_data_sampler.data()['toxic'], test_size = TRAIN_TEST_SPLIT_PERCENT  )
X_test = tokenised_test_comments
print('Finished in {} sec'.format(time.time() - s))



# Memory freeing
# -----------------------------------------------------------------
# del explore
# gc.collect()



# Loading embeddings
# -----------------------------------------------------------------
print('Loading embeddings')
s = time.time()
fast_text = FastTextEmbeddings(FASTTEXT_PATH, FASTTEXT_SAMPLES)
print('Finished in {} sec'.format(time.time() - s))
print('Embeddings shape: {}'.format(fast_text.shape()))



# Embeddings matrix creation
# -----------------------------------------------------------------
print('Embeddings matrix creation')
s = time.time()
embeddings, unknown = tokenizer.build_embedding_matrix(fast_text)
print('Finished in {} sec'.format(time.time() - s))



# Memory freeing
# -----------------------------------------------------------------
del fast_text
gc.collect()



# Loading model trainer
# -----------------------------------------------------------------
print('Loading model trainer')
s = time.time()
max_features = tokenizer.get_stats()[2] + 1
trainer = PyTorchTrainer(X_train, y_train, embeddings, max_features, cudaEnabled = IS_ON_KAGGLE )
print('Finished in {} sec'.format(time.time() - s))



# Training
# -----------------------------------------------------------------
print('Training')
s = time.time()
# output = trainer.train_and_test(X_test, n_epochs = EPOCHS)
trainer.train(n_epochs = EPOCHS)
print('Finished in {} sec'.format(time.time() - s))



# Kaggle prediction and submission
# -----------------------------------------------------------------
if IS_ON_KAGGLE:

    print('Kaggle prediction')
    s = time.time()
    output = trainer.predict(trainer.convert_to_tensor_dataset(X_test), batch_size = 512)
    print('Finished in {} sec'.format(time.time() - s))

    print('Saving submission')
    s = time.time()
    KaggleSubmitter.save_submission(test_data_sampler.data(), output, 'submission.csv')
    print('Finished in {} sec'.format(time.time() - s))



# Validation prediction
# -----------------------------------------------------------------
print('Validation prediction')
s = time.time()
output = trainer.predict(trainer.convert_to_tensor_dataset(X_valid), batch_size = 512)
print('Finished in {} sec'.format(time.time() - s))



# Coputing AUC on validation set
# -----------------------------------------------------------------
print('Coputing AUC on validation set')
auc = EvaluationMetrics.compute_auc(y_valid, output)
print(auc)



# Calculating and logging Bias metrics
# -----------------------------------------------------------------
print('Calculating Bias metrics')
s = time.time()

def save_score(model_name, data, y_valid, output):
    validate_df = data.loc[y_valid.index]
    validate_df[model_name] = output    
    return validate_df

validate_df = save_score('M1', train_data_sampler.data(), y_valid, output)
bias_metrics_df = EvaluationMetrics.compute_bias_metrics_for_model(validate_df, explore.identity_columns, 'M1', TOXICITY_COLUMN)
print(bias_metrics_df)

try:
    final_metrics = EvaluationMetrics.get_final_metric(bias_metrics_df, EvaluationMetrics.calculate_overall_auc(validate_df, 'M1', TOXICITY_COLUMN))
    print('Final metrics {}'.format(final_metrics))
except:
    print('Final metrics could not be calculated')

print('Finished in {} sec'.format(time.time() - s))



# Time Summary
# -----------------------------------------------------------------
print("Entire execution took {}s".format(time.time() - global_start))

