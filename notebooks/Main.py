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
STOP_WORDS_PATH = PATH_PREFIX + 'input/stopwords/english'
SAMPLE_PERCENT = 0.001
TEST_SAMPLE_PERCENT = 0.001
EPOCHS = 1
TEXT_TOKEN_LENGTH = 100
TRAIN_TEST_SPLIT_PERCENT = 0.2

# Imports

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

# AUTO-REMOVE-ABOVE


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
max_features = tokenizer.get_dictionary_size()
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

