# ObsceneTextPreprocessor
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import string
#Download stopwords
nltk.download('stopwords')


class ObsceneTextPreprocessor():

    # Words we want to retain in our set
    leave_words = ['i', 'you', 'myself', 'yourself', 'no', 'she', 'themselves', 'he', 'we', 'only',\
                        'up', 'your', 'my', 'mine', 'yours', 'hers', 'her', 'his', 'from', 'to', 'for']
    #Regex to leave special characters for obscene words like f##k
    tokenizer = RegexpTokenizer(r'[\%\@\&\$\#\w\!\*]+')


    def __init__(self, lower=False):
        super().__init__()

        self.lower = lower
        #Use the English stopword list
        stop_words = set(stopwords.words('english'))
        self.stop_words = [w for w in stop_words if (not w in self.leave_words)]


    def __lower_fn(self, w):
        if self.lower:
            return w.lower()
        return w


    def clean(self, data):

        def p(doc):
            out = [self.__lower_fn(w) for w in self.tokenizer.tokenize(doc) if (not w.lower() in self.stop_words)]
            return out

        return [' '.join(p(doc)) for doc in data]


