from library.DataSampler import *
from library.PyTorchTrainer import *
from library.ObsceneTextPreprocessor import *
from library.TextTokenizer import *
from library.EvaluationMetrics import *
from library.Reproducibility import *
from library.LSTM import NeuralNet
from library.ExploratoryAnalysis import ExploratoryAnalysis, JigsawExploratoryAnalysis
from library.Embeddings import FastTextEmbeddings
import os

# SETTINGS
kaggle_path_prefix = '../' if os.path.exists('../input') else ''
fast_text_path = kaggle_path_prefix + 'input/crawl-300d-2M.vec'
train_path = kaggle_path_prefix + 'input/train.csv'
test_path = kaggle_path_prefix + 'input/test.csv'



print('Seeding')
Reproducibility.RANDOM_SEED = 1234
Reproducibility.seed_everything()

#Loading embeddings
print('Loading embeddings')
fast_text = FastTextEmbeddings(fast_text_path, 100)
print('Embeddings shape: {}'.format(fast_text.shape()))

#Loading traning and test data
print('Loading traning and test data')
explore = JigsawExploratoryAnalysis(train_path= train_path, test_path= test_path)

#Collecting sample of data for training
print('Collecting sample of data for training')
data_sampler = DataSampler(explore.train, sample_percent=0.01)

#Preprocessing text 
print('Preprocessing text ')
text_processor = ObsceneTextPreprocessor()
clean_comments = text_processor.clean(data_sampler.data()['comment_text'])

#Tokenisation 
print('Tokenisation ')
tokenizer = TextTokenizer(clean_comments)
tokenised_comments = tokenizer.transform(clean_comments)

#Embeddings creating
print('Embeddings creating')
embeddings, unknown = tokenizer.build_embedding_matrix(fast_text)

#Splitting data for Train / Validation
print('Splitting data for Train / Validation')
X_train, X_test, y_train, y_test =  data_sampler.train_test_split(tokenised_comments, data_sampler.data()['toxic'])

#Loading model trainer
print('Loading model trainer')
max_features = tokenizer.get_stats()[2] + 1
trainer = PyTorchTrainer(X_train, X_test, y_train, y_test, embeddings, max_features )

#Training
print('Training')
output = trainer.train_model(n_epochs=1)

#Coputing AUC
print('Coputing AUC')
EvaluationMetrics.compute_auc(y_test, output)


TOXICITY_COLUMN = 'toxic'

def save_score(model_name, data, y_test, output):
    validate_df = data.loc[y_test.index]
    validate_df[model_name] = output
    
    return validate_df

validate_df = save_score('M1', data_sampler.data(), y_test, output)

bias_metrics_df = EvaluationMetrics.compute_bias_metrics_for_model(validate_df, explore.identity_columns, 'M1', TOXICITY_COLUMN)
print(bias_metrics_df)

try:
    final_metrics = EvaluationMetrics.get_final_metric(bias_metrics_df, EvaluationMetrics.calculate_overall_auc(validate_df, 'M1', TOXICITY_COLUMN))
    print('Final metrics {}'.format(final_metrics))
except:
    print('Final metrics could not be calculated')


