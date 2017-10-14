#!/usr/bin/env python
# encoding: UTF-8

"""
This module provides functionalities for parsing the data
"""

import os
import csv
import numpy as np
import datetime
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import itertools
import sys
import tensorflow as tf
from prettytable import  PrettyTable
from utils import _int64_feature
from utils import _bytes_feature
from utils import _parse_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

'''
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

'''

csv.field_size_limit(sys.maxsize)


class DataParser(object):
    def __init__(self,base_folder, dataset, papers_presentation, use_embeddings=False,**kwargs):
        """
        Initializes the data parser given a dataset name
        """
        self.base_folder = base_folder
        self.dataset = dataset
        self.papers_presentation = papers_presentation

        if dataset == 'citeulike-a':
            self.dataset_folder = self.base_folder + '/citeulike_a_extended'
        elif dataset == 'citeulike-t':
            self.dataset_folder = self.base_folder + '/citeulike_t_extended'
        elif dataset == 'dummy':
            self.dataset_folder = self.base_folder + '/dummy'
        else:
            print("Warning: Given dataset not known, setting to citeulike-a")
            self.dataset_folder = self.base_folder + '/citeulike_a_extended'


        if use_embeddings:
            self.use_pre_trained_embed = True
            if 'embed_dir' in kwargs.keys():
                self.embed_dir = kwargs['embed_dir']
            if 'embed_dim' in kwargs.keys():
                self.embed_dim = kwargs['embed_dim']
            self.embed_vocab = []
            self.embeddings = None
            self.embed_word_to_id = {}


        self.raw_labels = None
        #Papers are represented as vectors of strings
        self.raw_data = None
        #Papers are represented as vectors of word ids instead of strings
        self.all_documents = {}

        self.paper_count = None
        self.user_count = None
        self.unkown_words = {}
        self.unkown_words_count = 0
        self.numbers_count = 0
        self.words = {}
        self.words_count = 0
        self.id_map = {}
        self.paper_count_threshold = 2
        self.authors_map = {}

        self.train_ratings = None
        self.test_ratings = None
        self.confidence_matrix = None
        self.process()



    def process(self):
        """
        Starts parsing the data and gets matrices ready for training
        """
        # self.raw_labels, self.raw_data = self.parse_paper_raw_data()

        self.ratings = self.generate_ratings_matrix()
        self.raw_ratings= self.generate_raw_ratings_matrix()
        # self.build_document_word_matrix()
        # print("shape")
        # print(self.document_words.shape)
        if self.papers_presentation == 'attributes':
            self.feature_labels, self.feature_matrix = self.parse_paper_features()
            self.get_features_distribution()
        self.parse_authors()

    def build_document_word_matrix(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dataset_folder, 'mult.dat')
        documents_hash = {}
        docs_count = 0
        words_count = 1
        with open(path, "r") as f:
            for doc_id, line in enumerate(f):
                splitted = line.split(" ")
                count = int(splitted[0])
                words = []
                for i in range(count):
                    word_id_count = splitted[i + 1].split(":")
                    word_id = int(word_id_count[0])
                    count = int(word_id_count[1])
                    words.append((word_id, count))
                    words_count = max(word_id + 1, words_count)
                documents_hash[doc_id] = words
                docs_count += 1
        document_words = np.zeros((docs_count, words_count))
        for key, entry in documents_hash.items():
            for value in entry:
                word_id = value[0]
                word_count = value[1]
                document_words[key][word_id] = word_count
        del documents_hash

        self.document_words = document_words

    def get_document_word_distribution(self):
        return self.document_words

    def parse_paper_features(self):
        """
        Parses paper features
        """
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dataset_folder, 'paper_info_processed.csv')
        with open(path, "r",encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter='\t')
            first_line = True
            feature_vec = []
            i = 0
            row_length = 0
            labels_ids = []
            for line in reader:
                if first_line:
                    labels = []
                    for j, entry in enumerate(line):
                            if entry != 'citeulike_id':
                                labels.append(entry)
                                labels_ids.append(j)
                    row_length = len(labels_ids)
                    first_line = False
                    i += 1
                    continue
                paper_id = line[0]
                if line[1] != 'NaN':
                    self.id_map[int(line[1])] = paper_id
                if int(paper_id) != i:
                    for _ in range(int(paper_id) - i):
                        feature_vec.append(['\\N'] * row_length)
                        i += 1
                current_entry = []
                for label_id in labels_ids:
                    current_entry.append(line[label_id])
                feature_vec.append(current_entry)
                i += 1

        if self.paper_count is None:
            self.paper_count = len(feature_vec)
        return labels, np.asarray(feature_vec)


    def feature_index(self,feature):
        for index, label in enumerate(self.feature_labels):
            if label == feature:
                return index
        return -1

    def get_features_distribution(self):
        # if self.papers_presentation == 'attributes':
        #     return self.feature_matrix
        if self.feature_labels is None:
            self.feature_labels, self.feature_matrix = self.parse_paper_features()

        # A dict of tuples. First element of the tuple is the unique values of a the chosen attribute, the second is
        # their frequencies
        uniqe_freq = {}
        # A dict that has the id of the missing token '\\N' for each attribute
        missing_value_id = {}

        #Number of papers that have a value for each attribute
        att_count = {}
        t = PrettyTable(['feature','# of unique values', '# of papers that have value','# of papers that have missing values'])
        for feature in self.feature_labels:
            uniqe_freq[feature] = np.unique(self.feature_matrix[:, self.feature_index(feature)], return_counts=True)
            missing_value_id[feature] = np.where(uniqe_freq[feature][0] == '\\N')

            att_count[feature] = np.sum(uniqe_freq[feature][1]) \
                                 - (uniqe_freq[feature][1][missing_value_id[feature]]
                                    if len(missing_value_id[feature][0]) != 0 else 0)

            t.add_row([feature,len(uniqe_freq[feature][0]),att_count[feature],(uniqe_freq[feature][1][missing_value_id[feature]]
                                    if missing_value_id[feature][0] is not None else 0)])
        print (t)

        att_per_paper = {}
        for paper in self.feature_matrix:
            att_per_paper[paper[0]] = len(np.where(paper != '\\N')[0])

        plt.hist(list(att_per_paper.values()))
        plt.title("Attribute per paper")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        #fig = plt.gcf()
        plt.savefig('{0}attribute-per-paper-histogram_{1}'.format(self.dataset_folder+'/',self.dataset))
        print('')
        # for feature in self.feature_labels:
        #     print('Number of unique {0} {1}, frequencies {2}'.format(feature,len(uniqe_freq[feature][0]), uniqe_freq[feature][1]))
        #     print('', end="")

    def _embedding_parse_function(self, example_proto):
        feature = {'word': tf.FixedLenFeature([], tf.string),
                   'glove_id': tf.FixedLenFeature([], tf.int64),
                   'embed': tf.FixedLenFeature([self.embed_dim], tf.float32)}
        # Decode the record read by the reader
        features = tf.parse_single_example(example_proto, feature)
        word = tf.cast(features['word'], tf.string)
        glove_id = tf.cast(features['glove_id'], tf.int32)
        embed = tf.cast(features['embed'], tf.float32)

        return word, glove_id, embed

    def load_embeddings(self):
        filename = os.path.join(self.dataset_folder,'{0}-embeddings-{1}-w2v.tfrecord'.format(self.dataset, self.embed_dim))
        print('Loading embeddings {0} ...'.format(filename))
        unsorted_embeddings={}
        word_id =0
        for record in tf.python_io.tf_record_iterator(filename):
            example = tf.train.Example()
            example.ParseFromString(record)
            glove_id = example.features.feature['glove_id'].int64_list.value[0]
            word = (example.features.feature['word'].bytes_list.value[0]).decode("utf-8")
            embed = example.features.feature['embed'].float_list.value[:]
            self.embed_vocab.append(word)
            unsorted_embeddings[word_id] = embed
            self.embed_word_to_id[word] = word_id
            word_id +=1
        self.embeddings =np.zeros([word_id,self.embed_dim], dtype=np.float32)
        for word_id in unsorted_embeddings:
            self.embeddings[word_id] = unsorted_embeddings[word_id]
        del unsorted_embeddings


    def load_glove_embeddings(self):

        # Load pre-trained embeddings
        f = open(os.path.join(self.embed_dir, 'glove.6B.{0}d.txt'.format(self.embed_dim)))
        print("Loading GloVe embedding data  " + str(f.name))
        self.embeddings = []
        for line in f:
            row = line.split()
            word = row[0]
            # Add the word to the vocabulary list of the pre-trained word embeddings dataset
            self.embed_vocab.append(word)
            # Add the embedding of th word
            self.embeddings.append([float(val) for val in row[1:]])
        f.close()
        # Create a word to id index
        self.embed_word_to_id = {word: i for i, word in enumerate(self.embed_vocab)}
        return self.embed_vocab, self.embeddings

    def get_word_id(self, word):
        """
        Get the word id from the vocabolary of the pre-trianed embeddings .
            :returns: Word id
            :rtype int
        """

        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        if word in self.embed_vocab:
            return self.embed_word_to_id[word]
        elif is_number(word):
            self.numbers_count += 1
            # print('is_number {0}'.format(word))
            return self.embed_word_to_id['<num>']
        else:
            # print('Unknow word: {0}'.format(word))
            self.unkown_words[word] = 1
            return self.embed_word_to_id['unk']

    def insert_word(self, word):

        if word in self.words:
            self.words[word] += 1
            return
        self.words[word] = 1
        self.words_count += 1

    def parse_paper_raw_data(self):
        """
        Parses paper raw data
        :return: A tuple of Papers' labels and abstracts, where abstracts are returned as strings
        """
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dataset_folder, 'processed-raw-data.csv')
        delimiter = ','
        total = 0
        if self.dataset == 'citeulike-t':
            delimiter = '\t'
        documents = {}
        with open(path, "r",encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter=delimiter)
            first_line = True
            row_length = 0
            for line in reader:
                if first_line:
                    labels = line[1:]
                    row_length = len(line)
                    first_line = False
                    continue
                doc_id = int(line[0])
                # for citeulike-a, it starts from 1, we need it to start from 0
                if self.dataset == 'citeulike-a':
                    doc_id -=1
                documents[doc_id]=line[1]
                # if self.dataset == 'citeulike-t':
                #     for word in line[1].split(" "):
                #         self.insert_word(word)
        if self.paper_count is None:
            self.paper_count = len(documents)

        # print "Total is "
        # print(self.words_count)
        return labels, documents

    def get_papar_as_word_ids(self):
        """ Convert the papers raw data to vectors of word ids of the pre-trained embeddings' vocabulary
        :return: Papers' abstracts as ids
        """
        if self.paper_count is None:
            self.raw_labels, self.raw_data = self.parse_paper_raw_data()
        if self.use_pre_trained_embed:
            for doc_id in self.raw_data:
                # tokens = self.raw_data[doc_id].split()
                word_idx= [self.get_word_id(word) for word in self.raw_data[doc_id].split()]
                self.all_documents[doc_id] = word_idx
        # TODO: Get word ids when not using pre-trained word embeddings
        return self.all_documents

    def get_paper_conference(self, paper):
        journal_index = -1
        booktitle_index = -1
        series_index = -1
        for index, label in enumerate(self.feature_labels):
            if label == 'journal':
                journal_index = index
            if label == 'booktitle':
                booktitle_index = index
            if label == 'series':
                series_index = index
        if self.feature_matrix[paper][journal_index] != '\\N':
            return self.feature_matrix[paper][journal_index]
        if self.feature_matrix[paper][booktitle_index] != '\\N':
            return self.feature_matrix[paper][booktitle_index]
        if self.feature_matrix[paper][series_index] != '\\N':
            return self.feature_matrix[paper][series_index]
        return None

    def parse_authors(self):
        if bool(self.id_map) == False:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dataset_folder, 'paper_info.csv')
            with open(path, "r",encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f, delimiter='\t')
                first_line = True
                for line in reader:
                    if first_line:
                        first_line = False
                        continue
                    paper_id = line[0]
                    self.id_map[int(line[1])] = paper_id
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dataset_folder, 'authors.csv')
        with open(path, "r",encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter='\t')
            first_line = True
            for line in reader:
                if first_line:
                    first_line = False
                    continue
                key = self.id_map[int(line[0])]
                if key in self.authors_map:
                    if self.authors_map[key] is not None:
                        self.authors_map[key] = self.authors_map[key].append(line[1])
                    else:
                        self.authors_map[key] = [line[1]]
                else:
                    self.authors_map[key] = [line[1]]

    def get_author_similarity(self, paper, user_papers):
        print(type(self.authors_map))
        sim = 0
        paper_author = self.authors_map[paper]
        user_authors = []
        for user_paper in user_papers:
            if self.authors_map[user_paper] is not None:
                user_authors.append(self.authors_map[user_paper])
        intersection = [filter(lambda x: x in paper_author, sublist) for sublist in user_authors]
        sim = sum(len(x) for x in intersection)
        return sim / sum(len(x) for x in user_authors) * 1.0

    def get_conference_similarity(self, paper, user_papers):
        paper_conference = self.get_paper_conference(paper)
        denom = len(user_papers) * 1.0
        numer = 0.0
        for user_paper in user_papers:
            user_conference = self.get_paper_conference(user_paper)
            if user_conference == paper_conference:
                numer += 1.0
        return numer / denom

    def generate_ratings_matrix(self):
        """
        Generates a rating matrix of user x paper
        """
        if self.paper_count is None:
            self.raw_labels, self.raw_data = self.parse_paper_raw_data()

        # print self.paper_count
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dataset_folder, 'users.dat')
        self.user_count = sum(1 for line in open(path))
        ratings = np.zeros((self.user_count, self.paper_count))
        i = 0
        with open(path, "r") as f:
            for line in f:
                splitted = line.replace("\n", "").split(" ")
                for paper_id in splitted:
                    ratings[i][int(paper_id)] = 1
                i += 1
        return ratings

    def generate_raw_ratings_matrix(self):
        """
        Generates a rating matrix of user x paper
        """
        if self.paper_count is None:
            self.raw_labels, self.raw_data = self.parse_paper_raw_data()

        # print self.paper_count
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dataset_folder, 'users.dat')
        self.user_count = sum(1 for line in open(path))
        ratings = np.zeros((self.user_count, self.paper_count))
        i = 0
        with open(path, "r") as f:
            for line in f:
                splitted = line.replace("\n", "").split(" ")
                for paper_id in splitted:
                    ratings[i][int(paper_id)] = int(paper_id)
                i += 1
        return ratings

    def get_raw_data(self):
        return self.raw_labels, self.raw_data

    def del_raw_data(self):
        # empty space in the ram
        del self.raw_data

    def get_feature_vector(self):
        return self.feature_labels, self.feature_matrix

    def get_ratings_matrix(self):
        return self.ratings

    def get_raw_ratings_matrix(self):
        return self.raw_ratings

    def get_vocab_size(self):
        if self.use_pre_trained_embed:
            return len(self.embed_vocab)
        else:
            return self.words_count

    def split_warm_start_user(self,folds):
        idx = np.arange(0,self.paper_count)
        self.test_ratings = [[[] for u in range(self.user_count)]for x in range(folds)]
        self.train_ratings = [[[] for u in range(self.user_count)] for x in range(folds)]
        items_frequency = np.sum(self.ratings, axis=0,dtype=int)
        always_in_train=[]
        for i, count in enumerate(items_frequency):
            if count < self.paper_count_threshold:
                always_in_train.append(i)
        for user in range(self.user_count):
            rated_items_indices = self.ratings[user].nonzero()[0]
            # Shuffle all rated items indices
            np.random.seed(0)
            np.random.shuffle(rated_items_indices)
            item_per_fold = len(rated_items_indices)/ folds
            for fold in range(folds):
                start = int( fold * item_per_fold)
                end = int((fold + 1) * item_per_fold)
                if fold == folds - 1:
                    end = len(idx)
                u_test_indices = rated_items_indices[start:end]
                u_test_indices = [i for i in u_test_indices if i not in always_in_train]
                # mask = np.ones(len(rated_items_indices), dtype=bool)
                # mask[[rated_items_indices]] = False
                # train_idx = rated_items_indices[mask]
                u_train_indices = [i for i in rated_items_indices if i not in u_test_indices]
                assert (len(u_test_indices) + len(u_train_indices) == len(rated_items_indices))
                u_train_indices.sort()
                u_test_indices.sort()
                self.train_ratings[fold][user] = u_train_indices
                self.test_ratings[fold][user] = u_test_indices
        return self.train_ratings,self.test_ratings

    def split_warm_start_item(self, folds):
        #List of lists to store test ratings.
        #A list of size fold, where each item is a list that contains a list of size paper_count
        #It has the shape [folds,items,users]
        test_ratings_item = [[[] for i in range(self.paper_count)] for x in range(folds)]
        # List of lists to store train ratings.
        # A list of size fold, where each  item is a list that contains a list of size paper_count
        train_ratings_item = [[[] for i in range(self.paper_count)] for x in range(folds)]
        items_with_users_bigger_folds = 0
        for item in range(self.paper_count):
            # List of users ids that rated the item
            rated_items_indices = self.ratings[:,item].nonzero()[0]
            #Check if the number of users who rated this item is less than the number of folds
            if len(rated_items_indices) < folds:
                for fold in range(folds):
                    #Always add to train
                    train_ratings_item[fold][item] = rated_items_indices
            else:
                items_with_users_bigger_folds += 1
                # Shuffle all rated items indices
                np.random.seed(0)
                np.random.shuffle(rated_items_indices)
                item_per_fold = len(rated_items_indices) / folds
                for fold in range(folds):
                    #calculate the start and the end of the split for each fold
                    start = int(fold * item_per_fold)
                    end = int((fold + 1) * item_per_fold)
                    if fold == folds - 1:
                        end = len(rated_items_indices)
                    #indices of the users in test
                    item_test_indices = rated_items_indices[start:end]

                    # indices of the users in train
                    item_train_indices = [i for i in rated_items_indices if i not in item_test_indices]

                    # mask = np.ones(len(rated_items_indices), dtype=bool)
                    # mask[[item_test_indices]] = False
                    # item_train_indices = rated_items_indices[mask]

                    train_ratings_item[fold][item] = item_train_indices
                    test_ratings_item[fold][item] = item_test_indices

        print('Items w/ users bigger than {0}: {1}  '.format(folds,items_with_users_bigger_folds))
        # A list of size fold, where each item is a list that contains a list of size user_count
        # It has the shape [folds,users,items]
        test_ratings_user = [[[] for i in range(self.user_count)] for x in range(folds)]
        # List of lists to store train ratings.
        # A list of size fold, where each  item is a list that contains a list of size user_count
        train_ratings_user = [[[] for i in range(self.user_count)] for x in range(folds)]

        #convert the train_ratings and test ratings from [fold,item,user] to [fold,user,item]
        for fold in range(folds):
            ratings = np.zeros((self.user_count, self.paper_count),dtype=np.int32)
            for i, item in enumerate(train_ratings_item[fold]):
                for user in item:
                    train_ratings_user[fold][user].append(i)
            for i, item in enumerate(test_ratings_item[fold]):
                for user in item:
                    test_ratings_user[fold][user].append(i)

            #sort the items indices
            for user in range(self.user_count):
                #Randomly sample a negative data example for each positive one
                negative_idx =self.get_negative_samples(train_ratings_user[fold][user],test_ratings_user[fold][user],self.paper_count)
                train_ratings_user[fold][user].extend(negative_idx)
                train_ratings_user[fold][user].sort()
                test_ratings_user[fold][user].sort()

        self.train_ratings = train_ratings_user
        self.test_ratings = test_ratings_user
        return self.train_ratings, self.test_ratings

    def split_cold_start(self, folds):
        item_per_fold = self.paper_count / folds
        idx = np.arange(0,self.paper_count)
        np.random.seed(0)
        np.random.shuffle(idx)
        test_ratings = [[[] for u in range(self.user_count)]for x in range(folds)]
        train_ratings = [[[] for u in range(self.user_count)] for x in range(folds)]
        items_count = np.sum(self.ratings, axis=0,dtype=int)
        always_in_train=[]
        for i, count in enumerate(items_count):
            if count < self.paper_count_threshold:
                always_in_train.append(i)
        for fold in range(folds):
            start = int(fold *item_per_fold)
            end = int ((fold+1) *item_per_fold)
            if fold == folds -1:
                end = len(idx)
            test_idx = idx[start:end]
            test_idx = [i for i in test_idx if i not in always_in_train]
            mask = np.ones(self.paper_count, dtype=bool)
            mask[[test_idx]] = False
            train_idx = idx[mask]
            assert (len(test_idx)+len(train_idx) == self.paper_count)
            for user in range(self.user_count):
                rated_items_indices = self.ratings[user].nonzero()[0]
                u_train_indices = np.intersect1d(train_idx,rated_items_indices)
                u_test_indices = np.intersect1d(test_idx,rated_items_indices)
                #Randomly sample a negative data example for each positive one
                negative_idx = self.get_negative_samples(u_train_indices, u_test_indices, self.paper_count)

                u_train_indices = np.concatenate((u_train_indices,negative_idx))
                u_train_indices.sort()
                u_test_indices.sort()
                train_ratings[fold][user] =u_train_indices
                test_ratings[fold][ user] =u_test_indices
        self.train_ratings = train_ratings
        self.test_ratings = test_ratings
        return self.train_ratings,self.test_ratings

    def get_negative_samples(self,train_idx,test_idx,paper_count):
        idx= np.arange(self.paper_count,dtype=np.int32)
        mask = np.ones(self.paper_count, dtype=bool)
        mask[[train_idx]] = False
        mask[[test_idx]] = False
        negative_idx = idx[mask]
        np.random.seed(0)
        np.random.shuffle(negative_idx)
        return negative_idx[:len(train_idx)]



    def generate_batch_samples(self, batch_size,fold, train=True, validation=False, test=False):
        ratings = np.zeros((self.user_count, self.paper_count),dtype=np.int32)

        if test:
            ratings_ = self.test_ratings
        elif train:
            ratings_ = self.train_ratings

        for i, u in enumerate(ratings_[fold]):
                for v in u:
                    ratings[i][v]=self.ratings[i,v]

        self.nonzero_u_idx = ratings.nonzero()[0]
        self.nonzero_v_idx = ratings.nonzero()[1]
        num_rating = np.count_nonzero(ratings)
        idx = np.arange(num_rating)
        np.random.seed(0)
        np.random.shuffle(idx)

        nb_batches = num_rating // batch_size
        assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
        for batch in range(nb_batches):
            batch_idx = np.random.randint(num_rating, size=batch_size)
            u_idx = self.nonzero_u_idx[idx[batch_idx]]
            v_idx = self.nonzero_v_idx[idx[batch_idx]]
            r = ratings[u_idx, v_idx]
            docs = []
            docs = [self.all_documents[x] for x in v_idx]
            docs = np.array(docs)
            if batch_size > 1:
                yield u_idx, v_idx, r, docs
            else:
                yield u_idx[0], v_idx[0], r[0], docs[0]
        # return True

    def generate_samples(self,fold, train=True, validation=False, test=False):
        # ratings = np.zeros((self.user_count, self.paper_count),dtype=np.int32)
        #
        if test:
            ratings_ = self.test_ratings
        elif train:
            ratings_ = self.train_ratings

        for i, u in enumerate(ratings_[fold]):
                for v in u:
                    # ratings[i][v]=self.ratings[i,v]
                    u_id = i
                    v_id = v
                    r = self.ratings[i,v]
                    doc= np.array(self.all_documents[v_id])
                    yield u_id, v_id, int(r), doc

    def get_confidence_matrix(self,mode='default',**kwargs):
        '''

        :param mode: { 'constant','only-positive', 'user-dependant' }
        :return:
        '''
        self.confidence_matrix = np.ones((self.ratings.shape))
        if mode == 'constant':
            if 'alpha' in kwargs and 'beta' in kwargs:
                self.confidence_matrix[self.ratings == 1] = kwargs['alpha']
                self.confidence_matrix[self.ratings != 1] = kwargs['beta']
            else:
                raise Exception('alpha and beta values are required, where alpha >> beta ')
        elif mode == 'only-positive':
            self.confidence_matrix[self.ratings == 1] = 1
            self.confidence_matrix[self.ratings != 1] = 0
        elif mode == 'user-dependant':
            if 'alpha' in kwargs:
                alpha = kwargs['alpha']
            else:
                alpha = 40
                print('alpha value is not provided, using default value %d ' % alpha)
            epsilon = 1e-8
            count_nonzero = np.count_nonzero(self.ratings, axis=1)
            self.confidence_matrix = self.confidence_matrix .T *( 1 +  alpha * np.log(1 + count_nonzero/(epsilon)))
        else:
            print('Using default confidence mode, all negative and positive samples have confidence value 1  ')

        return self.confidence_matrix


