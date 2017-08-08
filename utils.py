# encoding: UTF-8
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
# import nltk
import itertools
import numpy as np
import pickle
import os
import sys
import time
from functools import wraps
# from pympler import asizeof
# import gc
from keras.preprocessing.sequence import pad_sequences


embedding_size=200
sentence_start_token = "SENTENCE_START"
sentence_start_token_embedding=[0]*embedding_size
sentence_end_token = "SENTENCE_END"
sentence_end_token_embedding=[1]*embedding_size
unknown_token = "UNKNOWN_TOKEN"
unknown_token_embedding= np.random.uniform(-2.3,2.3,embedding_size)

'''
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)


'''
class textloader():
    def __init__(self,embeddings_dir,batch_size):
        self.batch_size = batch_size
        def load_embeddings(GLOVE_DIR,word_index=None,EMBEDDING_DIM=None):
            f = open(os.path.join(GLOVE_DIR, 'glove.6B.200d.txt'))

            embeddings_index = {}
            vocab=[]
            embed=[]
            for line in f:
                row = line.split()
                word = row[0]#.decode('utf-8')
                vocab.append(word)
                embed.append([float(val) for val in row[1:]])
                # embeddings_index[word] = row[1:]
            print("Loading embeddings  " + str(f.name))
            f.close()
            return vocab,embed

        self.vocab, self.embed = load_embeddings('/home/wanli/data/glove.6B/')
        self.vocab_size = len(self.vocab)
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.M = np.array(0)
        self.train_idx = None
        self.valid_idx = None
        self.test_idx = None
        self.nonzero_u_idx = None
        self.nonzero_v_idx = None
        self.seq_length = 20
        self.max_length =0
        self.min_length = 15000

    def getWordEmbedding(self,word):
        if word in self.vocab:
            return self.word_to_id[word]
        else:
            return self.word_to_id['unk']

    def read_abstracts(self, path):
        df = pd.read_csv(path, usecols=[0, 1], index_col=0, header=0,
                         delimiter=',',encoding='utf-8')
        self.all_documents = {}
        for index, row in df.iterrows():
            sentences = itertools.chain(*[sent_tokenize(x.lower()) for x in row])
            print("Loading paper  " + str(index))
            sentences = [word_tokenize(x) for x in sentences]
            doc = ([self.getWordEmbedding(word) for sentence in sentences for word in sentence])

            self.all_documents[index]=doc

    def get_min_max_length(self):
        for doc in self.all_documents.values():
            self.max_length = max(self.max_length, len(doc))
            self.min_length = min(self.min_length, len(doc))
        print(" Max sequence length: " + str(self.max_length))
        print(" Min sequence length: " + str(self.min_length))
        return self.min_length

    def read_split_abstrcts(self,file,validation=True):
        df = pd.read_csv(file, usecols=[0, 4], index_col=0, header=1,
                         delimiter=',')
        all_documents = []
        docs_ranges =[]
        for index, row in df.iterrows():
            # sentences = itertools.chain(*[nltk.word_tokenize(x.lower()) for x in row])
            sentences = itertools.chain(*[sent_tokenize(x.lower()) for x in row])
            # words.extend(sentences)
            print("Loading paper  " + str(index))
            sentences = [word_tokenize(x) for x in sentences]
            start = len(all_documents)
            all_documents.extend([self.getWordEmbedding(word) for sentence in sentences for word in sentence])
            end = len(all_documents)
            docs_ranges.append({"start": start, "end": end, "id": index})

        if len(docs_ranges) == 0:
            sys.exit("No training data has been found. Aborting.")

        # For validation, use roughly 90K of text,
        # but no more than 10% of the entire text
        # and no more than 1 book in 5 => no validation at all for 5 files or fewer.

        # 10% of the text is how many files ?
        total_len = len(all_documents)
        validation_len = 0
        nb_books1 = 0
        for book in reversed(docs_ranges):
            validation_len += book["end"] - book["start"]
            nb_books1 += 1
            if validation_len > total_len // 10:
                break

        # 90K of text is how many books ?
        validation_len = 0
        nb_books2 = 0
        for book in reversed(docs_ranges):
            validation_len += book["end"] - book["start"]
            nb_books2 += 1
            if validation_len > 90 * 1024:
                break

        # 20% of the books is how many books ?
        nb_books3 = len(docs_ranges) // 5

        # pick the smallest
        nb_books = min(nb_books1, nb_books2, nb_books3)

        if nb_books == 0 or not validation:
            cutoff = len(all_documents)
        else:
            cutoff = docs_ranges[-nb_books]["start"]
        valitext = all_documents[cutoff:]
        codetext = all_documents[:cutoff]
        return codetext, valitext, docs_ranges
        # print(len(words))
        # words = [item for sublist in words for item in sublist]
        # # Count the word frequencies
        # word_freq = nltk.FreqDist(itertools.chain(*words))
        # print ("Found %d unique words tokens." % len(word_freq.items()))
        # vocabulary_size = len(word_freq.items())
        # # Get the most common words and build index_to_word and word_to_index vectors
        # # vocab = word_freq.most_common(vocabulary_size - 1)
        # uniq_words = set(itertools.chain(*words))
        # return words,uniq_words
    # # Embedings
    # glove_dir = '/home/wanli/data/glove.6B/'
    # vocab,embd = load_embeddings(glove_dir)
    # vocab_size = len(vocab)
    # embedding_dim = len(embd['the'])
    # embedding = np.asarray(embd)

    def convert_to_word(self,c):
        """Decode a code point
        :param c: code point
        :param avoid_tab_and_lf: if True, tab and line feed characters are replaced by '\'
        :return: decoded character
        """
        if c < len(self.vocab):
            return self.vocab[c]
        else:
            return '<UKN>'  # unknown

    def decode_to_text(self,c, avoid_tab_and_lf=False):
        """Decode an encoded string.
        :param c: encoded list of code points
        :param avoid_tab_and_lf: if True, tab and line feed characters are replaced by '\'
        :return:
        """
        return " ".join(map(lambda a: str(self.convert_to_word(a)), c))


    def rnn_minibatch_sequencer(self,raw_data, batch_size, sequence_size, nb_epochs):
        """
        Divides the data into batches of sequences so that all the sequences in one batch
        continue in the next batch. This is a generator that will keep returning batches
        until the input data has been seen nb_epochs times. Sequences are continued even
        between epochs, apart from one, the one corresponding to the end of raw_data.
        The remainder at the end of raw_data that does not fit in an full batch is ignored.
        :param raw_data: the training text
        :param batch_size: the size of a training minibatch
        :param sequence_size: the unroll size of the RNN
        :param nb_epochs: number of epochs to train on
        :return:
            x: one batch of training sequences
            y: on batch of target sequences, i.e. training sequences shifted by 1
            epoch: the current epoch number (starting at 0)
        """
        data = np.array(raw_data)
        data_len = data.shape[0]
        # using (data_len-1) because we must provide for the sequence shifted by 1 too
        nb_batches = (data_len - 1) // (batch_size * sequence_size)
        assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
        rounded_data_len = nb_batches * batch_size * sequence_size
        xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
        ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size])

        for epoch in range(nb_epochs):
            for batch in range(nb_batches):
                x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
                y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
                x = np.roll(x, -epoch, axis=0)  # to continue the text from epoch to epoch (do not reset rnn state!)
                y = np.roll(y, -epoch, axis=0)
                yield x, y, epoch

    def find_book(index, bookranges):
        return next(
            book["id"] for book in bookranges if (book["start"] <= index < book["end"]))


    def find_book_index(index, bookranges):
        return next(
            i for i, book in enumerate(bookranges) if (book["start"] <= index < book["end"]))

    def print_learning_learned_comparison(self,X, Y, losses, bookranges, batch_loss, batch_accuracy, epoch_size, index, epoch):
        """Display utility for printing learning statistics"""
        print()
        # epoch_size in number of batches
        batch_size = X.shape[0]  # batch_size in number of sequences
        sequence_len = X.shape[1]  # sequence_len in number of characters
        start_index_in_epoch = index % (epoch_size * batch_size * sequence_len)
        for k in range(batch_size):
            index_in_epoch = index % (epoch_size * batch_size * sequence_len)
            decx = self.decode_to_text(X[k], avoid_tab_and_lf=True)
            decy = self.decode_to_text(Y[k], avoid_tab_and_lf=True)
            bookname = self.find_book(index_in_epoch, bookranges)
            formatted_bookname = "{: <10.40}".format(str(bookname)) # min 10 and max 40 chars
            epoch_string = "{:4d}".format(index) + " (epoch {}) ".format(epoch)
            loss_string = "loss: {:.5f}".format(losses[k])
            print_string = epoch_string + formatted_bookname + " │ {} │ {} │ {}"
            print(print_string.format(decx, decy, loss_string))
            index += sequence_len
        # box formatting characters:
        # │ \u2502
        # ─ \u2500
        # └ \u2514
        # ┘ \u2518
        # ┴ \u2534
        # ┌ \u250C
        # ┐ \u2510
        format_string = "└{:─^" + str(len(epoch_string)) + "}"
        format_string += "{:─^" + str(len(formatted_bookname)) + "}"
        format_string += "┴{:─^" + str(len(decx) + 2) + "}"
        format_string += "┴{:─^" + str(len(decy) + 2) + "}"
        format_string += "┴{:─^" + str(len(loss_string)) + "}┘"
        footer = format_string.format('INDEX', 'BOOK NAME', 'TRAINING SEQUENCE', 'PREDICTED SEQUENCE', 'LOSS')
        print(footer)
        # print statistics
        batch_index = start_index_in_epoch // (batch_size * sequence_len)
        batch_string = "batch {}/{} in epoch {},".format(batch_index, epoch_size, epoch)
        stats = "{: <28} batch loss: {:.5f}, batch accuracy: {:.5f}".format(batch_string, batch_loss, batch_accuracy)
        print()
        print("TRAINING STATS: {}".format(stats))


    def print_data_stats(self,datalen, valilen, epoch_size):
        datalen_mb = datalen/1024.0/1024.0
        valilen_kb = valilen/1024.0
        print("Training text size is {:.2f}MB with {:.2f}KB set aside for validation.".format(datalen_mb, valilen_kb)
              + " There will be {} batches per epoch".format(epoch_size))


    def print_validation_header(self,validation_start, bookranges):
        bookindex = self.find_book_index(validation_start, bookranges)
        books = ''
        for i in range(bookindex, len(bookranges)):
            books += str(bookranges[i]["id"])
            if i < len(bookranges) - 1:
                books += ", "
        sys.stdout.flush()
        print("{: <60}".format("Validating on " + books))

    def print_validation_stats(self,loss, accuracy):
        print("VALIDATION STATS:                                  loss: {:.5f},       accuracy: {:.5f}".format(loss,accuracy))

    def reset_batch_pointer(self):
        self.pointer = 0

    def read_dataset(self, path,num_user,num_item):
        self.M = np.zeros([num_user, num_item], dtype=int)
        with open(path, 'r') as f:
            i =0
            for line in f.readlines():
                items_idx= line.split()
                items_idx = [int(x) - 1 for x in items_idx]
                user_id = i  # 0 base index
                rating = 1
                self.M[user_id, items_idx] = rating
                i +=1

    def split_data(self):
        num_rating = np.count_nonzero(self.M)
        idx = np.arange(num_rating)
        np.random.seed(0)
        np.random.shuffle(idx)

        train_prop = 0.9
        valid_prop = train_prop * 0.05
        self.train_idx = idx[:int((train_prop - valid_prop) * num_rating)]
        self.valid_idx = idx[int((train_prop - valid_prop) * num_rating):int(train_prop * num_rating)]
        self.test_idx = idx[int(0.9 * num_rating):]

        self.nonzero_u_idx = self.M.nonzero()[0]
        self.nonzero_v_idx = self.M.nonzero()[1]

        train_size = self.train_idx.size
        trainM = np.zeros(self.M.shape)
        trainM[self.nonzero_u_idx[self.train_idx], self.nonzero_v_idx[self.train_idx]] = self.M[
            self.nonzero_u_idx[self.train_idx], self.nonzero_v_idx[self.train_idx]]
        u_idx =self.nonzero_u_idx[self.train_idx]
        v_idx = self.nonzero_v_idx[self.train_idx]
        r = trainM[u_idx, v_idx]
        docs = [self.all_documents[x] for x in v_idx]
        return u_idx, v_idx, r, docs

    def generate_batches(self, nb_epochs,train=True, validation=False,test=False):
        num_rating = np.count_nonzero(self.M)
        idx = np.arange(num_rating)
        np.random.seed(0)
        np.random.shuffle(idx)

        train_prop = 0.9
        valid_prop = train_prop * 0.05
        self.train_idx = idx[:int((train_prop - valid_prop) * num_rating)]
        self.valid_idx = idx[int((train_prop - valid_prop) * num_rating):int(train_prop * num_rating)]
        self.test_idx = idx[int(0.9 * num_rating):]

        self.nonzero_u_idx = self.M.nonzero()[0]
        self.nonzero_v_idx = self.M.nonzero()[1]
        if train:
            idx = self.train_idx
        if validation:
            idx = self.valid_idx
        if test:
            idx = self.test_idx

        train_size =  idx.size
        trainM = np.zeros(self.M.shape)
        trainM[self.nonzero_u_idx[idx], self.nonzero_v_idx[idx]] = self.M[
            self.nonzero_u_idx[idx], self.nonzero_v_idx[idx]]
        nb_batches = train_size // self.batch_size
        assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
        rounded_data_len = nb_batches * self.batch_size
        udata = np.reshape(idx[0:rounded_data_len], [self.batch_size, nb_batches])
        i =0
        for epoch in range(nb_epochs):
            for batch in range(nb_batches):
                u_idx = self.nonzero_u_idx[udata[:, batch]]
                v_idx = self.nonzero_v_idx[udata[:, batch]]
                r = trainM[u_idx, v_idx]
                docs = []
                docs = [self.all_documents[x] for x in v_idx]
                # docs = [x[:self.min_length] for x in docs]
                docs = np.array(docs)
                i += 1
                yield u_idx, v_idx, r, docs, epoch
        print('Total number of batches: {0} of size {1}'.format(i,self.batch_size))

    def get_valid_idx(self):
        valid_u_idx = self.nonzero_u_idx[self.valid_idx]
        valid_v_idx = self.nonzero_v_idx[self.valid_idx]
        valid_m = self.M[valid_u_idx, valid_v_idx]
        docs = [self.all_documents[x] for x in valid_v_idx]
        # docs = [x[:self.min_length] for x in docs]
        docs = np.array(docs)
        return valid_u_idx, valid_v_idx, valid_m, docs
    def get_test_idx(self):
        test_u_idx = self.nonzero_u_idx[self.test_idx]
        test_v_idx = self.nonzero_v_idx[self.test_idx]
        test_m = self.M[test_u_idx, test_v_idx]
        docs = [self.all_documents[x] for x in test_v_idx]
        # docs = [x[:self.min_length] for x in docs]
        docs = np.array(docs)
        return test_u_idx, test_v_idx,test_m, docs

    def rounded_predictions(self, predictions):
        """
        The method rounds up the predictions and returns a prediction matrix containing only 0s and 1s.
        :returns: predictions rounded up matrix
        :rtype: int[][]
        """
        predictions = predictions.copy()
        n_users = self.M.shape[0]
        for user in range(n_users):
            avg = sum(predictions[user]) / predictions.shape[1]
            low_values_indices = predictions[user, :] < avg
            predictions[user, :] = 1
            predictions[user, low_values_indices] = 0
        return predictions

    def static_padding(self,docs):
        lengths = []
        long_docs = 0
        max_length = 300
        for d in docs:
            if len(d) <= max_length:
                lengths.append(len(d))
            else:
                lengths.append(max_length)
        padded_seq = pad_sequences(docs, maxlen=max_length, padding='post')
        return padded_seq, lengths

    ''' 
    https: // github.com / chiphuyen / stanford - tensorflow - tutorials / blob / master / examples / cgru / data_reader.py
    '''

    # def batch_examples(examples, batch_size, bucket_boundaries=None):
    #     """Given a queue of examples, create batches of examples with similar lengths.
    #     We assume that examples is a dictionary with string keys and tensor values,
    #     possibly coming from a queue, e.g., constructed by examples_queue above.
    #     Each tensor in examples is assumed to be 1D. We will put tensors of similar
    #     length into batches togeter. We return a dictionary with the same keys as
    #     examples, and with values being batches of size batch_size. If elements have
    #     different lengths, they are padded with 0s. This function is based on
    #     tf.contrib.training.bucket_by_sequence_length so see there for details.
    #     For example, if examples is a queue containing [1, 2, 3] and [4], then
    #     this function with batch_size=2 will return a batch [[1, 2, 3], [4, 0, 0]].
    #     Args:
    #       examples: a dictionary with string keys and 1D tensor values.
    #       batch_size: a python integer or a scalar int32 tensor.
    #       bucket_boundaries: a list of integers for the boundaries that will be
    #         used for bucketing; see tf.contrib.training.bucket_by_sequence_length
    #         for more details; if None, we create a default set of buckets.
    #     Returns:
    #       A dictionary with the same keys as examples and with values being batches
    #       of examples padded with 0s, i.e., [batch_size x length] tensors.
    #     """
    #     # Create default buckets if none were provided.
    #     if bucket_boundaries is None:
    #         # Small buckets -- go in steps of 8 until 64.
    #         small_buckets = [8 * (i + 1) for i in xrange(8)]
    #         # Medium buckets -- go in steps of 32 until 256.
    #         medium_buckets = [32 * (i + 3) for i in xrange(6)]
    #         # Large buckets -- go in steps of 128 until maximum of 1024.
    #         large_buckets = [128 * (i + 3) for i in xrange(6)]
    #         # By default use the above 20 bucket boundaries (21 queues in total).
    #         bucket_boundaries = small_buckets + medium_buckets + large_buckets
    #     with tf.name_scope("batch_examples"):
    #         # The queue to bucket on will be chosen based on maximum length.
    #         max_length = 0
    #         for v in examples.values():  # We assume 0-th dimension is the length.
    #             max_length = tf.maximum(max_length, tf.shape(v)[0])
    #         (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
    #             max_length, examples, batch_size, bucket_boundaries,
    #             capacity=2 * batch_size, dynamic_pad=True)
    #         return outputs


