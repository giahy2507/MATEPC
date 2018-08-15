import nltk
from nltk.corpus import stopwords
from collections import OrderedDict, defaultdict
import pickle
import numpy as np
import os
import re
from sklearn.model_selection import KFold
import copy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
import itertools

from utils import dense_to_one_hot, collect_data_infor_from_tsv, pad_sequences, load_word_embeddings


UNK = '<UNK>'
PAD = '<PAD>'


# PreProcessor
# WC
class WordPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, lowercase=True, num_norm=True):
        self.lowercase = lowercase
        self.num_norm = num_norm
        self.vocab_word = None
        self.vocab_tag = None
        self.stopwords = OrderedDict({PAD: 0, UNK: 1})

    def fit(self, X1, Y):
        vocab_word = OrderedDict({PAD: 0, UNK: 1})
        vocab_tag = OrderedDict({PAD: 0})

        stop_words = set()
        stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '@', '#', '$', '%', '^', '&', '*', '+', '=', '\\', '|', '`', '~'])

        self.max_length = 0
        for words in X1:
            self.max_length = max(self.max_length, len(words))
            for w in words:
                w = self._lower(w)
                w = self._normalize_num(w)
                if w not in vocab_word:
                    vocab_word[w] = len(vocab_word)
                    if w in stop_words:
                        self.stopwords[w] = vocab_word[w]

        for t in itertools.chain(*Y):
            if t not in vocab_tag:
                vocab_tag[t] = len(vocab_tag)

        self.vocab_word = vocab_word
        self.vocab_tag = vocab_tag
        self.reverse_vocab_word = {i: t for t, i in self.vocab_word.items()}
        self.reverse_vocab_tag = {i: t for t, i in self.vocab_tag.items()}

        self.number_of_classes = len(self.vocab_tag)
        self.word_vocab_size = len(self.vocab_word)
        self.max_length = self.max_length

        return self

    def transform(self, X1, Y=None):
        sents = []
        lengths = []

        # transform label X
        for sent in X1:
            word_ids = []
            for w in sent:
                w = self._lower(w)
                w = self._normalize_num(w)
                if w in self.vocab_word:
                    word_id = self.vocab_word[w]
                else:
                    word_id = self.vocab_word[UNK]
                word_ids.append(word_id)
            lengths.append(len(word_ids))
            sents.append(word_ids)

        # transform label Y
        if Y is not None:
            sent_labels = [[self.vocab_tag[l] for l in labels] for labels in Y]
        else:
            sent_labels = None

        # sequence_length
        sequence_length = np.asarray(lengths)

        # padding
        X_result = pad_sequences(sents, 0, max_length=self.max_length)
        Y_result = pad_sequences(sent_labels, 0, max_length=self.max_length)
        intput_mask = np.array( (Y_result > 0), dtype=np.float32)
        X_result = [X_result, intput_mask ,sequence_length]
        return X_result, Y_result

    def inverse_transform(self, y):
        indice_tag = {i: t for t, i in self.vocab_tag.items()}
        return [indice_tag[y_] for y_ in y]

    def _lower(self, word):
        return word.lower() if self.lowercase else word

    def _normalize_num(self, word):
        if self.num_norm:
            tmp = re.sub(r"\d{1,10}[\.]\d{1,10}", "0", word)
            return re.sub(r"\d{1,10}", "0", tmp)
        else:
            return word

    def save(self, file_path):
        with open(file_path+".p.pickle",mode="wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path):
        with open(file_path+".p.pickle",mode="rb") as f:
            p = pickle.load(f)
            return p



import time
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-kr_name', type=str, default="WCH1", help='keras_model_name')
    parser.add_argument('-data_name', type=str, default="laptops", help='data_name')
    parser.add_argument('-hand_features', type=str, default=None, help='hand_features')
    parser.add_argument('-params_str', type=str, default="100,100,100,100,100,0.5,20,0.0010,1,1,0.45",
                        help='parameters')
    args = parser.parse_args()

    kr_names = ["WCH1", "WCH2", "WH1", "WH2", "WH", "WCH", "WC", "W", "WCP", "WP", "WPH", "WCPH", "WPD", "WPHD"]
    data_names = ["laptops", "restaurants"]

    if args.hand_features is None:
        hand_features = None
    else:
        hand_features = args.hand_features.split(",")

    params_str = args.params_str.strip()

    data_name = "laptops"
    task_name = "ATEPC2"
    DATA_ROOT = 'data'
    SAVE_ROOT = './models'  # trained models
    LOG_ROOT = './logs'  # checkpoint, tensorboard
    w_embedding_path = '/home/s1610434/Documents/Data/Vector/w2v/w2v.word.150.txt'
    pos_embedding_path = '/home/s1610434/Documents/Data/Vector/w2v/w2v.pos.50.txt'
    pos_embedding_path = 'models/w2v.pos.50.txt'


    keras_model_name = "WPH"
    hand_features = ['NEGAT', 'BING', 'SWN', 'NAMEL', 'DEPENCY', "HEADVOTE"]

    hand_features_dict = {"POS": 0, "UNIPOS": 0, "NEGAT": 0, "BING": 0, "BINGBIN": 0, "SWN": 0, "NAMEL": 0,
                          "DEPENCY": 0}
    print("-----{0}-----{1}-----{2}-----{3}-----".format(task_name, data_name, keras_model_name, hand_features))
    save_path = SAVE_ROOT + "/{0}/{1}".format(data_name, task_name)
    train_path = os.path.join(DATA_ROOT, '{0}.{1}.train.tsv'.format(data_name, task_name))
    test_path = os.path.join(DATA_ROOT, '{0}.{1}.test.tsv'.format(data_name, task_name))

    # train set
    sents1, poses1, dep_idxs1, dep_relations1, labels1, preds1 = collect_data_infor_from_tsv(train_path,keep_conflict=False)
    X1_train_valid = sents1
    X2_train_valid = np.asarray(list(zip(poses1, dep_idxs1, dep_relations1)))
    Y_train_valid = labels1

    # test set
    sents2, poses2, dep_idxs2, dep_relations2, labels2, preds2 = collect_data_infor_from_tsv(test_path,keep_conflict=True)
    X1_test = sents2
    X2_test = np.asarray(list(zip(poses2, dep_idxs2, dep_relations2)))
    Y_test = labels2

    # train + test
    X1_train_test = np.concatenate((X1_train_valid, X1_test), axis=0)
    X2_train_test = np.concatenate((X2_train_valid, X2_test), axis=0)
    Y_train_test = np.concatenate((Y_train_valid, Y_test), axis=0)

    p = WordPreprocessor()
    p.fit(X1=X1_train_test, X2=X2_train_test, Y=Y_train_test)
    A, B = p.transform(X1_train_test, Y=Y_train_test)
    # # preprocessor
    # print(p.max_length)
    # print(A[0].shape)
    # print(A[1].shape)
    # print(A[2].shape)
    # print(A[3].shape)
    # print(B.shape)

    POS_embeddings = load_word_embeddings(p.pos_extractor.features_dict, pos_embedding_path, 50)
    print(POS_embeddings)
    p.save("logs/p")


