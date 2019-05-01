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