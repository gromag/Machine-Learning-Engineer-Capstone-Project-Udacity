from library.DataSampler import *
from library.PyTorchTrainer import *
from library.ObsceneTextPreprocessor import *
from library.TextTokenizer import *
from library.EvaluationMetrics import *
from library.Reproducibility import *
from library.LSTM import NeuralNet
from library.ExploratoryAnalysis import ExploratoryAnalysis, JigsawExploratoryAnalysis
from library.Embeddings import FastTextEmbeddings
from library.KaggleSubmitter import KaggleSubmitter



import os
import time
import gc

# SETTINGS
TOXICITY_COLUMN = 'toxic'
IS_ON_KAGGLE =  os.path.exists('../input')
PATH_PREFIX = '../' if IS_ON_KAGGLE  else ''
FASTTEXT_PATH = PATH_PREFIX + 'input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
FASTTEXT_SAMPLES = 100
TRAIN_PATH = PATH_PREFIX + 'input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
TEST_PATH = PATH_PREFIX + 'input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
SAMPLE_PERCENT = 0.001
EPOCHS = 1

# AUTO-REMOVE-ABOVE


# RAM 481.6 MB
# RAM 544.1



print('Seeding')
Reproducibility.RANDOM_SEED = 1234
Reproducibility.seed_everything()

# --- Initial RAM Usage
# RAM 481.8 MB

# #Loading embeddings
# print('Loading embeddings')
# s = time.time()
# fast_text = FastTextEmbeddings(FASTTEXT_PATH, FASTTEXT_SAMPLES)
# print('Finished in {} sec'.format(time.time() - s))
# print('Embeddings shape: {}'.format(fast_text.shape()))



# --- OLD Algorithm
# RAM 7.2 GB - 1/2 M Words - OLD Algorithm
# RAM 10 GB - 1 M Words - OLD Algoritm
# RAM 14 GB - 1.3 M words - OLD Algoritm

# --- REVISED Algorithm
# RAM 2.2 GB 1/2 M Words - REVISED Algorithm
# RAM 4.7 GB 1 M Words - REVISED Algorithm
# RAM 7 GB   1.6 M W Words - REVISED Algorithm
# RAM 8.2 GB 2 M Words - REVISED Algorithm
# RAM 3.7 GB 2 M Words (Auto GC/Memory release) - REVISED Algorithm

#Loading traning and test data
print('Loading traning and test data')
s = time.time()
explore = JigsawExploratoryAnalysis(train_path = TRAIN_PATH, test_path = TEST_PATH)
print('Finished in {} sec'.format(time.time() - s))

#Collecting sample of data for training
print('Collecting sample of data for training')
s = time.time()
train_data_sampler = DataSampler(explore.train, sample_percent = SAMPLE_PERCENT)
test_data_sampler = DataSampler(explore.test)
print('Finished in {} sec'.format(time.time() - s))

# RAM 6.7 GB - 2 M Words

#Preprocessing text 
print('Preprocessing text ')
s = time.time()
text_processor = ObsceneTextPreprocessor()
clean_train_comments = text_processor.clean(train_data_sampler.data()['comment_text'])
clean_test_comments = text_processor.clean(test_data_sampler.data()['comment_text'])
print('Finished in {} sec'.format(time.time() - s))
# RAM 6.9 GB - 2 M Words
# RAM 7 GB - 2 M Words


#Tokenisation 
print('Tokenisation ')
s = time.time()
tokenizer = TextTokenizer(clean_train_comments, clean_test_comments)
print('Finished in {} sec'.format(time.time() - s))

# RAM 7.7 GB - 2 M Words
# RAM 7.9 GB - 2 M Words

print('Tokenisation  of train')
s = time.time()
tokenised_train_comments = tokenizer.transform(clean_train_comments)
print('Finished in {} sec'.format(time.time() - s))

# RAM 9.2 GB - 2 M Words

print('Tokenisation  of test')
s = time.time()
tokenised_test_comments = tokenizer.transform(clean_test_comments)
print('Finished in {} sec'.format(time.time() - s))

# RAM 9.2 GB - 2 M Words

#Splitting data for Train / Test
print('Splitting data for Train / Test')
s = time.time()
# X_train, X_test, y_train, y_test =  train_data_sampler.train_test_split(tokenised_train_comments, train_data_sampler.data()['toxic'])
X_train, X_test, y_train = (tokenised_train_comments, tokenised_test_comments, train_data_sampler.data()['toxic'])
print('Finished in {} sec'.format(time.time() - s))

# RAM 9.2 GB - 2 M Words

del train_data_sampler, explore
gc.collect()


#Loading embeddings
print('Loading embeddings')
s = time.time()
fast_text = FastTextEmbeddings(FASTTEXT_PATH, FASTTEXT_SAMPLES)
print('Finished in {} sec'.format(time.time() - s))
print('Embeddings shape: {}'.format(fast_text.shape()))

# RAM 9.6 1/2 M
# RAM 11.4 1 M
# RAM 12 1.3 M
# RAM 12.8 1.7 M
# RAM 13.2 1.7 M
# RAM 11.6 2 M


#Embeddings creating
print('Embeddings creating')
s = time.time()
embeddings, unknown = tokenizer.build_embedding_matrix(fast_text)
print('Finished in {} sec'.format(time.time() - s))

# RAM 12.7 GB 2 M

del fast_text
gc.collect()

# RAM 12.5 GB 2 M


#Loading model trainer
print('Loading model trainer')
s = time.time()
max_features = tokenizer.get_stats()[2] + 1
trainer = PyTorchTrainer(X_train, X_test, y_train, embeddings, max_features, cudaEnabled = IS_ON_KAGGLE )
print('Finished in {} sec'.format(time.time() - s))

# RAM 14

#Training and predicting
print('Training')
s = time.time()
output = trainer.train_model(n_epochs = EPOCHS)
print('Finished in {} sec'.format(time.time() - s))

print('Saving submission')
s = time.time()
KaggleSubmitter.save_submission(test_data_sampler.data(), output, 'submission.csv')
print('Finished in {} sec'.format(time.time() - s))


# #Coputing AUC
# print('Coputing AUC')
# auc = EvaluationMetrics.compute_auc(y_test, output)
# print(auc)



# def save_score(model_name, data, y_test, output):
#     validate_df = data.loc[y_test.index]
#     validate_df[model_name] = output
    
#     return validate_df

# validate_df = save_score('M1', train_data_sampler.data(), y_test, output)

# bias_metrics_df = EvaluationMetrics.compute_bias_metrics_for_model(validate_df, explore.identity_columns, 'M1', TOXICITY_COLUMN)
# print(bias_metrics_df)

# try:
#     final_metrics = EvaluationMetrics.get_final_metric(bias_metrics_df, EvaluationMetrics.calculate_overall_auc(validate_df, 'M1', TOXICITY_COLUMN))
#     print('Final metrics {}'.format(final_metrics))
# except:
#     print('Final metrics could not be calculated')


