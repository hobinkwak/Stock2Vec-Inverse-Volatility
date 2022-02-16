import os
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans


class Stock2Vec:
    def __init__(self, rtn_df, num_clusters=4):
        self.rtn_df = rtn_df
        self.num_clusters = num_clusters
        self.train_df = None
        self.skipgram = None
        self.vectors = None
        self.clusters = None
        self.company2cluster = None
        np.random.seed(42)

    def make_rtn_data(self):
        for col in self.rtn_df.columns:
            self.rtn_df[col] = self.rtn_df.loc[:, col].map(lambda x: [(col, x)])
        train_df = self.rtn_df.sum(axis=1).to_frame(name='rtn')
        self.train_df = train_df
        return train_df

    def sort_by_rtn(self, train_df):
        train_df.rtn = train_df.rtn.map(lambda ls: sorted(ls, key=lambda ls: ls[-1], reverse=True))
        train_df.rtn = train_df.rtn.map(lambda ls: [tup[0] for tup in ls])
        train_df = train_df.values.flatten().tolist()
        self.train_df = train_df
        return train_df

    def train_n_save_word2vec(self, train_df, size=100, window=5, min_count=5, workers=4, skipgram=True,
                              path='result/'):
        os.makedirs(path, exist_ok=True)
        skipgram = Word2Vec(sentences=train_df, vector_size=size, window=window, min_count=min_count, workers=workers,
                            sg=skipgram)
        skipgram.wv.save_word2vec_format(path + 'result.pt')
        self.skipgram = skipgram
        return skipgram

    def load_word2vec(self, path):
        self.skipgram = KeyedVectors.load_word2vec_format(path)
        return self.skipgram

    def get_sg_vectors(self):
        if isinstance(self.skipgram, KeyedVectors):
            self.vectors = self.skipgram.vectors
            self.index2key = self.skipgram.index_to_key
        else:
            self.vectors = self.skipgram.wv.vectors
            self.index2key = self.skipgram.wv.index_to_key
        return self.vectors

    def kmeans_clustering(self, vectors):
        kc = KMeans(n_clusters=self.num_clusters)
        clusters = kc.fit_predict(vectors)

        clusters = list(zip(self.index2key, clusters))
        clusters = sorted(clusters, key=lambda x: x[1])
        self.clusters = clusters
        return self.clusters

    def extract_ticker(self, clusters):
        result = []
        for i in range(self.num_clusters):
            ls = []
            for tup in clusters:
                if tup[-1] == i:
                    ls.append(tup[0])
            result.append(ls)
        return result
