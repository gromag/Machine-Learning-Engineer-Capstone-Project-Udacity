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

print('Seeding')
Reproducibility.RANDOM_SEED = 1234
Reproducibility.seed_everything()

#Loading embeddings
print('Loading embeddings')
fast_text = FastTextEmbeddings(FASTTEXT_PATH, FASTTEXT_SAMPLES)
print('Embeddings shape: {}'.format(fast_text.shape()))

#Loading traning and test data
print('Loading traning and test data')
explore = JigsawExploratoryAnalysis(train_path = TRAIN_PATH, test_path = TEST_PATH)

#Collecting sample of data for training
print('Collecting sample of data for training')
train_data_sampler = DataSampler(explore.train, sample_percent = SAMPLE_PERCENT)
test_data_sampler = DataSampler(explore.test)

#Preprocessing text 
print('Preprocessing text ')
text_processor = ObsceneTextPreprocessor()
clean_train_comments = text_processor.clean(train_data_sampler.data()['comment_text'])
clean_test_comments = text_processor.clean(test_data_sampler.data()['comment_text'])

#Tokenisation 
print('Tokenisation ')
tokenizer = TextTokenizer(clean_train_comments, clean_test_comments)
tokenised_train_comments = tokenizer.transform(clean_train_comments)
tokenised_test_comments = tokenizer.transform(clean_test_comments)

#Embeddings creating
print('Embeddings creating')
embeddings, unknown = tokenizer.build_embedding_matrix(fast_text)

#Splitting data for Train / Test
print('Splitting data for Train / Test')
# X_train, X_test, y_train, y_test =  train_data_sampler.train_test_split(tokenised_train_comments, train_data_sampler.data()['toxic'])
X_train, X_test, y_train = (tokenised_train_comments, tokenised_test_comments, train_data_sampler.data()['toxic'])

#Loading model trainer
print('Loading model trainer')
max_features = tokenizer.get_stats()[2] + 1
trainer = PyTorchTrainer(X_train, X_test, y_train, embeddings, max_features, cudaEnabled = IS_ON_KAGGLE )

#Training and predicting
print('Training')
output = trainer.train_model(n_epochs = EPOCHS)

KaggleSubmitter.save_submission(test_data_sampler.data(), output, 'submission2.csv')


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


