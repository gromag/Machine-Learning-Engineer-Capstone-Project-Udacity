
# LOCAL SETTINGS

import os
import time
import gc

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