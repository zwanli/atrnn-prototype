import time
import tensorflow as tf
from tensorflow.contrib import rnn
import threading
import sys
import numpy as np


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)
def bias_variable(shape, name):
    b_init = tf.constant_initializer(0.)
    return tf.get_variable(name, shape, initializer=b_init)
def get_lengths(docs):
    lengths = []
    for d in docs:
            lengths.append(len(d))
    # print(len(lengths))
    return lengths

class Model():
    def __init__(self, args, M, embed,bucket_boundaries,data_loader, reg_lambda=0.01,random_shuffl=False,name='tr'):
        self.data_loader = data_loader
        self.args = args
        self.dataset = args.dataset
        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))
        self.n, self.m = M.shape
        self.k = args.embedding_dim
        self.learning_rate = args.learning_rate
        # self.batch_size = args.batch_size
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)

        self.batch_size = args.batch_size

        # queue = tf.FIFOQueue()
        self.input_text = tf.placeholder(tf.int32, [None, None],name="Input_text")
        self.seq_lengths = tf.placeholder(tf.int32,[None],name="seq_lengths")
        self.u_idx = tf.placeholder(tf.int32, [None],name="U_matrix")
        self.v_idx = tf.placeholder(tf.int32, [None],name="V_matrix")
        self.r = tf.placeholder(tf.float32, [None],name="R_target")

        with tf.device("/cpu:0"):
            # The actual queue of data. The queue contains a vector for
            filename = os.path.join(self.args.data_dir, self.dataset + '_train.tfrecords')
            if (not os.path.exists(filename)):
                print('Dataset file does not exist {0}'.format(filename))
                raise SystemExit
            # Create a list of filenames and pass it to a queue
            filename_queue = tf.train.string_input_producer([filename], num_epochs=args.num_epochs)
            train_features = read_and_decode(filename_queue=filename_queue)

            # the mnist features, and a scalar label.
            if (random_shuffl):
                self.input_queue = tf.RandomShuffleQueue(
                    capacity=15 * self.batch_size, dtypes=(tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
                    shared_name='shared_name{0}'.format(name), min_after_dequeue=0)
            else:
                self.input_queue = tf.FIFOQueue(
                    capacity=15 * self.batch_size, dtypes=(tf.int32,tf.int32, tf.int32, tf.int32,tf.int32),
                    # shapes=[[self.batch_size,None], [self.batch_size], [self.batch_size],[self.batch_size],[self.batch_size]],
                    # shapes=[[1, None], [1], [1], [1], [1]],
                    shared_name='shared_name{0}'.format(name))#
                #       self.input_queue = tf.PaddingFIFOQueue(
                #     capacity=2 * batch_size, dtypes=(tf.int32,tf.int32, tf.int32, tf.int32,tf.float32),
                #     # shapes=[[None,None], [None], [None],[None],[None]],
                #     shared_name='shared_name{0}'.format(name))

            self.queue_size = self.input_queue.size()
            # The symbolic operation to add data to the queue
            self.input_enqueue_op = self.input_queue.enqueue((train_features))

            numberOfThreads = 1
            # now setup a queue runner to handle enqueue_op outside of the main thread asynchronously
            qr = tf.train.QueueRunner(self.input_queue, [self.input_enqueue_op] * numberOfThreads)
            # now we need to add qr to the TensorFlow queue runners collection
            tf.train.add_queue_runner(qr)

            #self.input_t, self.lengths_t, self.u_idx_t, self.v_idx_t, self.r_t = self.input_queue.dequeue_many(self.batch_size)
            self.input_t, self.lengths_t, self.u_idx_t, self.v_idx_t, self.r_t = self.input_queue.dequeue()
            self.r_t = tf.Print((self.r_t),data=[self.input_queue.size()], message='This is how many items are left in q: ')

            # self.dummy_output = tf.reduce_sum(tf.multiply(self.u_idx_t,self.v_idx_t), reduction_indices=1)
        # self.lengths_t = tf.reshape(self.lengths_t, [1])
        # self.input_t = tf.reshape(self.input_t, [1, -1])
        # self.u_idx_t = tf.reshape(self.u_idx_t, [1])
        # self.v_idx_t = tf.reshape(self.v_idx_t, [1])
        # self.r_t = tf.reshape(self.r_t, [1])
        # self.close_input_op = self.input_queue.close()
        #The queue output is the input of the model
        # self.seq_len, (self.doc_b, self.lengths_b, self.u_idx_b, self.v_idx_b, self.r_b) = tf.contrib.training.bucket_by_sequence_length(
        #     self.lengths_t, tensors=[self.input_t, self.lengths_t, self.u_idx_t, self.v_idx_t, self.r_t], allow_smaller_final_batch=True, \
        #     batch_size=batch_size, bucket_boundaries=bucket_boundaries, \
        #     capacity=2 * batch_size, dynamic_pad=True)
        #
        # # self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length], name="Input_text")
        # # self.seq_lengths = tf.placeholder(tf.int32, [args.batch_size], name="seq_lengths")
        # # self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length],name="target_text")
        # self.batch_size = tf.placeholder(tf.int32,[], name="batch_size")
        # self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        # self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)
        # self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)
        # self.batch_time = tf.Variable(0.0, name="batch_time", trainable=False)
        # tf.summary.scalar("time_batch", self.batch_time)
        #
        # def variable_summaries(var):
        #     """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        #     with tf.name_scope('summaries'):
        #         mean = tf.reduce_mean(var)
        #         tf.summary.scalar('mean', mean)
        #         #with tf.name_scope('stddev'):
        #         #   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        #         #tf.summary.scalar('stddev', stddev)
        #         tf.summary.scalar('max', tf.reduce_max(var))
        #         tf.summary.scalar('min', tf.reduce_min(var))
        #         #tf.summary.histogram('histogram', var)
        #
        # with tf.variable_scope('rnnlm'):
        #     # softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
        #     # variable_summaries(softmax_w)
        #     # softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
        #     # variable_summaries(softmax_b)
        #     with tf.device("/cpu:0"):
        #         vocab_size = args.vocab_size
        #         embedding_dim = len(embed[0])
        #         embeddings = np.asarray(embed)
        #         # embeddings = tf.get_variable("embeddings", shape=[dim1, dim2], initializer=tf.constant_initializer(np.array(embeddings_matrix))
        #         embedding = tf.get_variable(name="embedding", shape=[vocab_size, embedding_dim],
        #                                      initializer=tf.constant_initializer(embeddings), trainable=False)
        #         # embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        #         # inputs = tf.split(, args.seq_length, 1)
        #         # inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        #         inputs = tf.nn.embedding_lookup(embedding, self.doc_b)
        #
        # cells = []
        #
        # cell_fw = cell_fn(args.rnn_size)
        # cell_bw = cell_fn(args.rnn_size)
        # self.init_state_fw =cell_bw.zero_state(self.batch_size,tf.float32)
        # self.init_state_bw =cell_fw.zero_state(self.batch_size,tf.float32)
        #
        # bi_outputs, bi_output_state = \
        #     tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=self.seq_len,
        #                                     initial_state_bw=self.init_state_bw, initial_state_fw=self.init_state_fw)
        # bi_outputs = tf.concat(bi_outputs, 2)
        # self.bi_output_state_fw, self.bi_output_state_bw = bi_output_state
        #
        # # (bi_outputs_fw, bi_outputs_bw), self.bi_output_state_fw, self.bi_output_state_bw = \
        # #         tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,inputs,sequence_length=self.seq_lengths,
        # #                                  initial_state_bw=self.init_state_bw,initial_state_fw=self.init_state_fw)
        # # bi_outputs = tf.concat((bi_outputs_fw,bi_outputs_bw),2)
        #
        # self.bi_output_state_fw = tf.identity(self.bi_output_state_fw, name='bi_state_fw')  # just to give it a name
        # self.bi_output_state_bw = tf.identity(self.bi_output_state_bw, name='bi_state_bw')  # just to give it a name
        #
        # for _ in range(args.num_layers):
        #     cell = cell_fn(args.rnn_size)
        #     cells.append(cell)
        #
        # self.cell = cell = rnn.MultiRNNCell(cells)
        # self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        #
        # # bi_outputs = tf.stack(bi_outputs,1)
        # self.Yr, self.H = tf.nn.dynamic_rnn(cell,bi_outputs,sequence_length=self.seq_lengths,
        #                                     initial_state=self.initial_state,dtype=tf.float32)
        # # Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
        # # H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence
        # self.H = tf.identity(self.H, name='H')  # just to give it a name
        # self.Yr = tf.identity(self.Yr, name='Yr')
        #
        # self.G = tf.reduce_mean(self.Yr, 1)
        #
        #
        # self.U = weight_variable([self.n, self.k], 'U')
        # self.V = weight_variable([self.m, self.k], 'V')
        #
        # self.U_bias = weight_variable([self.n], 'U_bias')
        # self.V_bias = weight_variable([self.m], 'V_bias')
        #
        # self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx_b)
        # self.V_embed = tf.nn.embedding_lookup(self.V, self.v_idx_b)
        #
        # self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
        # self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.v_idx)
        #
        # self.F = tf.add(self.V_embed,self.G)
        # self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.F), reduction_indices=1)
        #
        # # self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
        # self.r_hat = tf.add(self.r_hat, self.U_bias_embed)
        # self.r_hat = tf.add(self.r_hat, self.V_bias_embed,name="R_predicted")
        #
        # self.RMSE = tf.sqrt(tf.losses.mean_squared_error(self.r, self.r_hat))
        # self.l2_loss = tf.nn.l2_loss(tf.subtract(self.r_b, self.r_hat))
        # self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.r, self.r_hat)))
        # self.reg = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U)),
        #                   tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V)))
        # self.reg_loss = tf.add(self.l2_loss, self.reg)
        #
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # # self.train_step = self.optimizer.minimize(self.reg_loss)
        # self.train_step_u = self.optimizer.minimize(self.reg_loss, var_list=[self.U, self.U_bias])
        # self.train_step_v = self.optimizer.minimize(self.reg_loss, var_list=[self.V, self.V_bias])
        #
        # t_vars=tf.trainable_variables()
        # gru_vars = [var for var in t_vars if 'gru_cell' in var.name]
        # self.train_step_rnn = self.optimizer.minimize(self.reg_loss, var_list=[gru_vars])
        #
        # tf.summary.scalar("RMSE", self.RMSE)
        # tf.summary.scalar("MAE", self.MAE)
        # tf.summary.scalar("L2-Loss", self.l2_loss)
        # tf.summary.scalar("Reg-Loss", self.reg_loss)
        #
        # # add op for merging summary
        # self.summary_op = tf.summary.merge_all()
        #
        # self.recall = tf.Variable(0, trainable=False,dtype=tf.float32)
        # self.recall_10 = tf.Variable(0, trainable=False, dtype=tf.float32)
        # self.recall_50 = tf.Variable (0,trainable=False, dtype=tf.float32)
        # self.recall_100 = tf.Variable(0, trainable=False, dtype=tf.float32)
        # self.recall_200 = tf.Variable(0, trainable=False, dtype=tf.float32)
        # recall_sum =tf.summary.scalar("Recall",self.recall)
        # recall_10_sum = tf.summary.scalar('recall@10',self.recall_10)
        # recall_50_sum = tf.summary.scalar('recall@50',self.recall_50)
        # recall_100_sum = tf.summary.scalar('recall@100',self.recall_100)
        # recall_200_sum = tf.summary.scalar('recall@200',self.recall_200)
        #
        # self.ndcg_5 = tf.Variable(0, trainable=False,dtype=tf.float32)
        # self.ndcg_10 = tf.Variable(0, trainable=False,dtype=tf.float32)
        # ndcg_5_sum = tf.summary.scalar('ndcg@5',self.ndcg_5)
        # ndcg_10_sum = tf.summary.scalar('ndcg@10',self.ndcg_10)
        #
        #
        # self.eval_metrics = tf.summary.merge((recall_sum,recall_10_sum,recall_50_sum,recall_100_sum,recall_200_sum,
        #                                      ndcg_5_sum, ndcg_10_sum))
        #
        #
        #
        # # add Saver ops
        # self.saver = tf.train.Saver()

import argparse
import os
from evaluator import Evaluator
# from model import Model
import pickle
from data_parser import DataParser
import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/wanli/data/Extended_ctr/',
                        help='data directory containing input.txt')
    parser.add_argument("--dataset", "-d", type=str, default='dummy',
                        help="Which dataset to use", choices=['dummy', 'citeulike-a', 'citeulike-t'])
    parser.add_argument('--embedding_dir', type=str, default='/home/wanli/data/glove.6B/',
                        help='GloVe embedding directory containing embeddings file')
    parser.add_argument('--embedding_dim', type=int, default=200,
                        help='dimension of the embeddings', choices=['50', '100', '200', '300'])
    parser.add_argument('--input_encoding', type=str, default=None,
                        help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')

    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=200,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='gru',
                        help='Choose the RNN cell type', choices=['rnn, gru, or lstm'])
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=300,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    parser.add_argument('--gpu_mem', type=float, default=0.666,
                        help='%% of gpu memory to be allocated to this process. Default is 66.6%%')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])



def get_lengths(docs):
    lengths = []
    for d in docs:
            lengths.append(len(d))
    # print(len(lengths))
    return lengths


def convert_to(dir, parser, name, validation=False, test=False):
    """Converts a dataset to tfrecords."""

    filename = os.path.join(dir, name + '{0}.tfrecords'.format("_test" if test else "_train" ))
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for u_id, v_id, rating, doc in parser.generate_samples(1, validation=validation, test=test):
        context = tf.train.Features(feature={
            'u': _int64_feature(u_id),
            'v': _int64_feature(v_id),
            'r': _int64_feature(rating),
            # 'abstract': _int64_feature_list(doc),
            'abs_length': _int64_feature(len(doc))
        })
        feature_lists = tf.train.FeatureLists(feature_list={
            "abstract": _int64_feature_list(doc) })
        sequence_example = tf.train.SequenceExample(
            context=context, feature_lists=feature_lists)
        writer.write(sequence_example.SerializeToString())
        # example = tf.train.Example(feature))1
        # writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()

def convert_to_temp(dir, parser, name, validation=False, test=False):
    """Converts a dataset to tfrecords."""

    filename = os.path.join(dir, name + '{0}_temp.tfrecords'.format("_test" if test else "_train" ))
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for u_id, v_id, rating, doc in parser.generate_samples(1, validation=validation, test=test):
        features = tf.train.Features(feature={
            'u': _int64_feature(u_id),
            'v': _int64_feature(v_id),
            'r': _int64_feature(rating),
            # 'abstract': tf.train.Feature(int64_list=tf.train.Int64List(value=doc)),
            'abslength': _int64_feature(len(doc))
        })
        example = tf.train.Example(features=features)

        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()

def read_and_decode(filename_queue):
    context_feature = {'train/u': tf.FixedLenFeature([], tf.int64),
               'train/v': tf.FixedLenFeature([], tf.int64),
               'train/r': tf.FixedLenFeature([], tf.int64),
               'train/abs_length': tf.FixedLenFeature([], tf.int64)}

    sequence_feature={'train/abstract': tf.FixedLenSequenceFeature([], tf.int64)}


    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    context_feature,sequence_feature = tf.parse_single_sequence_example(serialized_example, context_features=context_feature,
    sequence_features=sequence_feature)

    u = tf.cast(context_feature['train/u'], tf.int32)
    v = tf.cast(context_feature['train/v'], tf.int32)
    r = tf.cast(context_feature['train/r'], tf.int32)
    abs_length = tf.cast(context_feature['train/abs_length'], tf.int32)
    abstract = tf.cast(sequence_feature['train/abstract'], tf.int32)
    return u, v, r, abstract, abs_length


def train(args):
    #Read text input
    parser = DataParser(args.data_dir,args.dataset,None,
                        use_embeddings=True,embed_dir=args.embedding_dir,embed_dim=args.embedding_dim)
    parser.load_embeddings()
    args.vocab_size = parser.get_vocab_size()
    if os.path.exists('abstracts_word_embeddings_{}.pkl'.format(args.dataset)):
        print('Loading abstracts')
        with open('abstracts_word_embeddings_{}.pkl'.format(args.dataset), 'rb') as f:
            parser.all_documents =  pickle.load(f)
    else:
        parser.get_papar_as_word_ids()
        with open("abstracts_word_embeddings_{}.pkl".format(args.dataset),'wb') as f:
            pickle.dump(parser.all_documents,f,pickle.HIGHEST_PROTOCOL)
            print("Saved abstracts")

    parser.split_cold_start(5)
    parser.split_warm_start(5)
    path_train = os.path.join(args.data_dir, args.dataset+ '_train.tfrecords')
    if os.path.exists(path_train):
        print("File already exists {0}".format(path_train))
    else:
        convert_to(args.data_dir,parser,args.dataset)

    path_test = os.path.join(args.data_dir, args.dataset+ '_test.tfrecords')
    if os.path.exists(path_test):
        print("File already exists {0}".format(path_test))
    else:
        convert_to(args.data_dir, parser, args.dataset, test=True)

    # convert_to_temp(args.data_dir,parser,args.dataset)
    # convert_to_temp(args.data_dir, parser, args.dataset, test=True)
    # parser.generate_batches(128)
    bucket_boundaries = [x for x in range(50, 500, 50)]
    model = Model(args, parser.get_ratings_matrix(),parser.embeddings,bucket_boundaries,data_loader=parser)
    startTime = time.time()
    # numberOfThreads = 1
    # qr = tf.train.QueueRunner(model.input_queue, [model.input_enqueue_op] * numberOfThreads)
    # tf.train.add_queue_runner(qr)
    with tf.Session() as sess:
        # ... init our variables, ...
        sess.run(tf.global_variables_initializer())

        # ... add the coordinator, ...
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        # model.start_threads(sess)
        # ... check the accuracy before training (without feed_dict!), ...
        sess.run(model.r_t)

        # ... train ...
        for i in range(5000):
            #  ... without sampling from Python and without a feed_dict !
            loss = sess.run([model.lengths_t])

            # We regularly check the loss
            # if i % 500 == 0:
            print('iter:{0} - loss:{1}'.format(i,0))

        # Finally, we check our final accuracy
        sess.run(model.dummy_output)

        coord.request_stop()
        coord.join(threads)
##########################################################################################################
    #
    # print('Vocabolary size {0}'.format(parser.words_count))
    # print("Uknown words {0}".format(parser.unkows_words_count))
    # print("Uknown numbers {0}".format(parser.numbers_count))
    #
    #
    # def construct_feed(u_idx, v_idx, r, docs,seq_lengths,bi_hid_fw, bi_hid_bw, batch_size):
    #     return {model.u_idx: u_idx, model.v_idx: v_idx, model.r: r, model.input_data: docs, model.seq_lengths:seq_lengths,
    #             model.init_state_fw: bi_hid_fw, model.init_state_bw: bi_hid_bw,model.batch_size: batch_size}
    #             # model.initial_state: hid_state,
    #
    # # def get_batches(sess, coord, batch_size,bucket_boundaries, valid=False, test=False):
    # #     with tf.device("/cpu:0"):
    # #         custom_runner = CustomRunner(batch_size, bucket_boundaries, parser,name='v' if valid else 't')
    # #         seq_len, outputs = custom_runner.get_outputs()
    # #     try:
    # #         custom_runner.enque_input(sess,valid,test)
    # #         custom_runner.close(sess)
    # #         # threads = tf.train.start_queue_runners(sess, coord)
    # #         b = 0
    # #         batches={}
    # #         while True:
    # #             out_lengths, (input_t, lengths_t, u_idx_t, v_idx_t, r_t) = sess.run([seq_len, outputs])
    # #             print(len(input_t[0, 0]))
    # #             input_t = np.squeeze(input_t, [1])
    # #             lengths_t = np.squeeze(lengths_t, [1])
    # #             u_idx_t = np.squeeze(u_idx_t, [1])
    # #             v_idx_t = np.squeeze(v_idx_t, [1])
    # #             r_t = np.squeeze(r_t, [1])
    # #             batches[b] = (input_t, lengths_t, u_idx_t, v_idx_t, r_t)
    # #             b += 1
    # #     except Exception as e:
    # #         # Report exceptions to the coordinator.
    # #         coord.request_stop(e)
    # #         print("Total number of {0} samples: {1}".format('',b * batch_size))
    # #         print("Total number of {0} batches: {1}".format('',b))
    # #     return batches
    #
    # bucket_boundaries = [x for x in range(50, 500, 50)]
    # batch_size = args.batch_size
    #
    #
    # dir_prefix = time.strftime("%d:%m-%H:%M:")
    # train_writer = tf.summary.FileWriter(args.log_dir+ '/{0}-train'.format(dir_prefix))
    # valid_writer = tf.summary.FileWriter(args.log_dir + '/{0}-validation'.format(time.strftime(dir_prefix)))
    # test_writer = tf.summary.FileWriter(args.log_dir + '/{0}-test'.format(time.strftime(dir_prefix)))
    # best_val_rmse = np.inf
    # best_val_mae = np.inf
    # best_test_rmse = 0
    # best_test_mae = 0
    #
    # evaluator = Evaluator(parser.get_ratings_matrix(),verbose=True)
    #
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
    # n_steps = args.num_epochs
    # with tf.device("/cpu:0"):
    #     custom_runner = CustomRunner(batch_size, bucket_boundaries,parser)
    #     seq_len, outputs = custom_runner.get_outputs()
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #     print('Saving graph to disk...')
    #     train_writer.add_graph(sess.graph)
    #     # valid_writer.add_graph(sess.graph)
    #     # test_writer.add_graph(sess.graph)
    #     tf.global_variables_initializer().run()
    #     tf.local_variables_initializer().run()
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess, coord)
    #     train_batches = {}
    #     valid_batches = {}
    #     test_batches = {}
    #
    #     try:
    #         print('Bucketing batches...')
    #         custom_runner.enque_input(sess)
    #         custom_runner.close(sess)
    #         # threads = tf.train.start_queue_runners(sess, coord)
    #         b = 0
    #         while True:
    #             out_lengths, (input_t, lengths_t, u_idx_t, v_idx_t, r_t) = sess.run([seq_len, outputs])
    #             print(len(input_t[0, 0]))
    #             input_t = np.squeeze(input_t, [1])
    #             lengths_t = np.squeeze(lengths_t, [1])
    #             u_idx_t = np.squeeze(u_idx_t, [1])
    #             v_idx_t = np.squeeze(v_idx_t, [1])
    #             r_t = np.squeeze(r_t, [1])
    #             train_batches[b] = (input_t, lengths_t, u_idx_t, v_idx_t, r_t)
    #             b += 1
    #     except Exception as e:
    #         # Report exceptions to the coordinator.
    #         coord.request_stop(e)
    #         print("Total number of training samples: {0}".format(b * batch_size))
    #         print("Total number of training batches: {0}".format(b))
    #     finally:
    #         coord.request_stop()
    #         coord.join(threads)
    #     # valid_u_idx, valid_v_idx, valid_m, valid_docs = data_loader.get_valid_idx()
    #     # test_u_idx, test_v_idx, test_m, test_docs = data_loader.get_test_idx()
    #     # valid_batches = get_batches(sess,coord,valid_docs.shape[0],[0],valid=True)
    #     # test_batches = get_batches(sess, coord, test_docs.shape[0],[0], test=True)
    #
    #     print('Finished batching ')
    #     # model.saver = tf.train.Saver(tf.global_variables())
    #
    #     bi_state_fw = sess.run(model.init_state_bw,feed_dict={model.batch_size: args.batch_size})
    #     bi_state_bw = sess.run(model.init_state_fw,feed_dict={model.batch_size: args.batch_size})
    #     h_state = sess.run(model.initial_state,feed_dict={model.batch_size: args.batch_size})
    #
    #     for step in range(n_steps):
    #         for d, s, u, v, r in train_batches.values():
    #             feed = construct_feed(u, v, r, d,s, bi_state_fw, bi_state_bw,args.batch_size)
    #             sess.run(model.train_step_v, feed_dict=feed)
    #             sess.run(model.train_step_u, feed_dict=feed)
    #             _, U,V ,U_b ,V_b , bi_out_fw, bi_out_bw, final_state, rmse, mae, summary_str = sess.run([model.train_step_rnn,
    #                                                                                            model.U,model.V,model.U_bias,model.V_bias,
    #                                                                                            model.bi_output_state_fw, model.bi_output_state_bw, model.H,
    #                                                                                            model.RMSE,model.MAE, model.summary_op],
    #                                                         feed_dict=feed)
    #             train_writer.add_summary(summary_str,step)
    #
    #             cond = True
    #             if cond and step % int(n_steps / 50) == 0:
    #                 # valid_u_idx, valid_v_idx, valid_m, valid_docs = data_loader.get_valid_idx()
    #                 # valid_docs, valid_docs_len = data_loader.static_padding(valid_docs)
    #                 # valid_bi_fw = sess.run(model.init_state_fw,feed_dict={model.batch_size:valid_docs.shape[0] })
    #                 # valid_bi_bw = sess.run(model.init_state_bw,feed_dict={model.batch_size:valid_docs.shape[0] })
    #                 # init_state = sess.run(model.initial_state,feed_dict={model.batch_size:valid_docs.shape[0] })
    #                 # feed_dict = construct_feed(valid_u_idx, valid_v_idx, valid_m,
    #                 #                            valid_docs,valid_docs_len, valid_bi_fw, valid_bi_bw, valid_docs.shape[0])
    #                 # rmse_valid, mae_valid, summary_str = sess.run(
    #                 #     [model.RMSE, model.MAE, model.summary_op], feed_dict=feed_dict)
    #                 # valid_writer.add_summary(summary_str, step)
    #
    #                 test_u_idx, test_v_idx, test_m, test_docs, test_ratings = parser.get_test_idx()
    #                 test_docs, test_docs_len = utils.static_padding(test_docs,maxlen=300)
    #
    #                 test_bi_fw = sess.run(model.init_state_fw, feed_dict={model.batch_size: test_docs.shape[0]})
    #                 test_bi_bw = sess.run(model.init_state_bw, feed_dict={model.batch_size: test_docs.shape[0]})
    #                 init_state = sess.run(model.initial_state,feed_dict={model.batch_size:test_docs.shape[0] })
    #                 feed_dict= construct_feed(test_u_idx, test_v_idx, test_m,
    #                                           test_docs, test_docs_len, test_bi_fw,test_bi_bw, test_docs.shape[0])
    #                 rmse_test, mae_test, summary_str = sess.run(
    #                     [model.RMSE, model.MAE, model.summary_op], feed_dict=feed_dict)
    #
    #                 test_writer.add_summary(summary_str, step)
    #
    #                 prediction_matrix = np.matmul(U,V.T)
    #                 prediction_matrix = np.add(prediction_matrix,np.reshape(U_b,[-1,1]))
    #                 prediction_matrix = np.add(prediction_matrix,V_b)
    #                 rounded_predictions = utils.rounded_predictions(prediction_matrix)
    #                 # testM = np.zeros(parser.ratings.shape)
    #                 # testM[data_loader.nonzero_u_idx[data_loader.test_idx], data_loader.nonzero_v_idx[data_loader.test_idx]] = data_loader.M[
    #                 #     data_loader.nonzero_u_idx[data_loader.test_idx], data_loader.nonzero_v_idx[data_loader.test_idx]]
    #
    #                 evaluator.load_top_recommendations_2(200,prediction_matrix,test_ratings)
    #                 recall_10 = evaluator.recall_at_x(10, prediction_matrix, parser.ratings,rounded_predictions )
    #                 recall_50 = evaluator.recall_at_x(50, prediction_matrix, parser.ratings, rounded_predictions)
    #                 recall_100 = evaluator.recall_at_x(100, prediction_matrix, parser.ratings, rounded_predictions)
    #                 recall_200 = evaluator.recall_at_x(200, prediction_matrix, parser.ratings, rounded_predictions)
    #                 recall = evaluator.calculate_recall(ratings=parser.ratings,predictions=rounded_predictions)
    #                 ndcg_at_five = evaluator.calculate_ndcg(5, rounded_predictions)
    #                 ndcg_at_ten = evaluator.calculate_ndcg(10, rounded_predictions)
    #
    #
    #
    #                 feed ={model.recall:recall, model.recall_10:recall_10, model.recall_50:recall_50,
    #                        model.recall_100:recall_100, model.recall_200:recall_200,
    #                        model.ndcg_5:ndcg_at_five, model.ndcg_10:ndcg_at_ten}
    #                 eval_metrics = sess.run([model.eval_metrics], feed_dict=feed)
    #                 test_writer.add_summary(eval_metrics[0], step)
    #
    #                 print("Step {0} | Train RMSE: {1:3.4f}, MAE: {2:3.4f}".format(
    #                     step, rmse, mae))
    #                 # print("         | Valid  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
    #                 #     rmse_valid, mae_valid))
    #                 print("         | Test  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
    #                     rmse_test, mae_test))
    #                 print("         | Recall@10: {0:3.4f}".format(recall_10))
    #                 print("         | Recall@50: {0:3.4f}".format(recall_50))
    #                 print("         | Recall@100: {0:3.4f}".format(recall_100))
    #                 print("         | Recall@200: {0:3.4f}".format(recall_200))
    #                 print("         | Recall: {0:3.4f}".format(recall))
    #                 print("         | ndcg@5: {0:3.4f}".format(ndcg_at_five))
    #                 print("         | ndcg@10: {0:3.4f}".format(ndcg_at_ten))
    #
    #
    #
    #
    #                 if best_val_rmse > rmse_test:
    #                     # best_val_rmse = rmse_valid
    #                     best_test_rmse = rmse_test
    #
    #                 if best_val_mae > rmse_test:
    #                     # best_val_mae = mae_valid
    #                     best_test_mae = mae_test
    #
    #             # loop state around
    #             h_state = final_state
    #             bi_state_fw = bi_out_fw
    #             bi_state_bw = bi_out_bw
    #             # if step > 0 and (step % args.save_every == 0 or ( step == args.num_epochs - 1)):  # save for the last result
    #             #     checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
    #             #     saver.save(sess, checkpoint_path, global_step=step)
    #             #     print("model saved to {}".format(checkpoint_path))
    #     model.saver.save(sess, args.log_dir+ "/{0}model.ckpt".format(time.strftime(dir_prefix)))
    #     print('Best test rmse:',best_test_rmse,'Best test mae',best_test_mae,sep=' ')
    #     # restore model
    #     # if args.init_from is not None:
    #     #     saver.restore(sess, ckpt.model_checkpoint_path)
    #     # for e in range(model.epoch_pointer.eval(), args.num_epochs):
    #     #     sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
    #     #     data_loader.reset_batch_pointer()
    #     #     state = sess.run(model.initial_state)
    #     #     speed = 0
    #     #     for b in range(data_loader.pointer, data_loader.num_batches):
    #     #         start = time.time()
    #     #         x, y = data_loader.next_batch()
    #     #         feed = {model.input_data: x, model.targets: y, model.initial_state: state,
    #     #                 model.batch_time: speed}
    #     #         summary, train_loss, state, _, _ = sess.run([merged, model.cost, model.final_state,
    #     #                                                      model.train_op, model.inc_batch_pointer_op], feed)
    #     #         train_writer.add_summary(summary, e * data_loader.num_batches + b)
    #     #         speed = time.time() - start
    #     #         if (e * data_loader.num_batches + b) % args.batch_size == 0:
    #     #             print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
    #     #                 .format(e * data_loader.num_batches + b,
    #     #                         args.num_epochs * data_loader.num_batches,
    #     #                         e, train_loss, speed))
    #     #         if (e * data_loader.num_batches + b) % args.save_every == 0 \
    #     #                 or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
    #     #             checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
    #     #             saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
    #     #             print("model saved to {}".format(checkpoint_path))
    #     train_writer.close()
    #     valid_writer.close()
    #     test_writer.close()



if __name__ == '__main__':
    main()
# We simulate some raw input data
# (think about it as fetching some data from the file system)
# let's say: batches of 128 samples, each containing 1024 data points
# x_input_data = tf.random_normal([128, 1024], mean=0, stddev=1)
#
# # We build our small model: a basic two layers neural net with ReLU
# with tf.variable_scope("queue"):
#     q = tf.FIFOQueue(capacity=5, dtypes=tf.float32) # enqueue 5 batches
#     # We use the "enqueue" operation so 1 element of the queue is the full batch
#     enqueue_op = q.enqueue(x_input_data)
#     numberOfThreads = 1
#     qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
#     tf.train.add_queue_runner(qr)
#     input = q.dequeue() # It replaces our input placeholder
#     # We can also compute y_true right into the graph now
#     y_true = tf.cast(tf.reduce_sum(input, axis=1, keep_dims=True) > 0, tf.int32)
#
# with tf.variable_scope('FullyConnected'):
#     w = tf.get_variable('w', shape=[1024, 1024], initializer=tf.random_normal_initializer(stddev=1e-1))
#     b = tf.get_variable('b', shape=[1024], initializer=tf.constant_initializer(0.1))
#     z = tf.matmul(input, w) + b
#     y = tf.nn.relu(z)
#
#     w2 = tf.get_variable('w2', shape=[1024, 1], initializer=tf.random_normal_initializer(stddev=1e-1))
#     b2 = tf.get_variable('b2', shape=[1], initializer=tf.constant_initializer(0.1))
#     z = tf.matmul(y, w2) + b2
#
# with tf.variable_scope('Loss'):
#     losses = tf.nn.sigmoid_cross_entropy_with_logits(None, tf.cast(y_true, tf.float32), z)
#     loss_op = tf.reduce_mean(losses)
#
# with tf.variable_scope('Accuracy'):
#     y_pred = tf.cast(z > 0, tf.int32)
#     accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))
#     accuracy = tf.Print(accuracy, data=[accuracy], message="accuracy:")
#
# # We add the training op ...
# adam = tf.train.AdamOptimizer(1e-2)
# train_op = adam.minimize(loss_op, name="train_op")
#
# startTime = time.time()
# with tf.Session() as sess:
#     # ... init our variables, ...
#     sess.run(tf.global_variables_initializer())
#
#     # ... add the coordinator, ...
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     # ... check the accuracy before training (without feed_dict!), ...
#     sess.run(accuracy)
#
#     # ... train ...
#     for i in range(5000):
#         #  ... without sampling from Python and without a feed_dict !
#         _, loss = sess.run([train_op, loss_op])
#
#         # We regularly check the loss
#         if i % 500 == 0:
#             print('iter:%d - loss:%f' % (i, loss))
#
#     # Finally, we check our final accuracy
#     sess.run(accuracy)
#
#     coord.request_stop()
#     coord.join(threads)
#
# print("Time taken: %f" % (time.time() - startTime))

# with tf.Session() as sess:
#   # Define options for the `sess.run()` call.
#   options = tf.RunOptions()
#   options.output_partition_graphs = True
#   options.trace_level = tf.RunOptions.FULL_TRACE
#
#   # Define a container for the returned metadata.
#   metadata = tf.RunMetadata()
#
#   sess.run(y, options=options, run_metadata=metadata)
#
#   # Print the subgraphs that executed on each device.
#   print(metadata.partition_graphs)
#
#   # Print the timings of each operation that executed.
#   print(metadata.step_stats)