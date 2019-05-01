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