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

    def get_dictionary_size(self):
        """
        Returns the size of the dictionary, that is the length of 
        word_index plus 1 to account for the unknown word
        
        """
        return len(self.tokenizer.word_index) + 1

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

