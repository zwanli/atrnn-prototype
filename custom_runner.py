import tensorflow as tf
import numpy as np
import os
from data_loader import textloader
import pickle
import threading


def get_lengths(docs):
    lengths = []
    for d in docs:
            lengths.append(len(d))
    # print(len(lengths))
    return lengths

class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self,batch_size,bucket_boundaries,data_loader,random_shuffl=False,name=''):
        self.input_text = tf.placeholder(tf.int32, [None, None], name="Input_text_queue")
        self.seq_lengths = tf.placeholder(tf.int32, [None])
        self.u_idx = tf.placeholder(tf.int32, [None], name="U_matrix_queue")
        self.v_idx = tf.placeholder(tf.int32, [None], name="V_matrix_queue")
        self.r = tf.placeholder(tf.float32, [None], name="R_target_queue")


        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.
        if (random_shuffl):
            self.input_queue = tf.RandomShuffleQueue(
                5000, dtypes=(tf.int32, tf.int32, tf.int32, tf.int32, tf.float32),
                shared_name='shared_name{0}'.format(name), min_after_dequeue=0)
        else:
            self.input_queue = tf.PaddingFIFOQueue(
                5000, dtypes=(tf.int32,tf.int32, tf.int32, tf.int32,tf.float32),
                shapes=[[None,None], [None], [None],[None],[None]],
                shared_name='shared_name{0}'.format(name))
        self.queue_size = self.input_queue.size()
        # The symbolic operation to add data to the queue
        self.input_enqueue_op = self.input_queue.enqueue((self.input_text, self.seq_lengths,
                                                          self.u_idx, self.v_idx, self.r))
        # self.input_enqueue_many = self.input_queue.enqueue_many((self.input_text, self.seq_lengths,
        #                                                          self.u_idx, self.v_idx, self.r))

        self.input_t, self.lengths_t, self.u_idx_t, self.v_idx_t, self.r_t = self.input_queue.dequeue()
        self.input_t = tf.reshape(self.input_t, [1, -1])
        self.lengths_t = tf.reshape(self.lengths_t, [1])
        self.u_idx_t = tf.reshape(self.u_idx_t, [1])
        self.v_idx_t = tf.reshape(self.v_idx_t, [1])
        self.r_t = tf.reshape(self.r_t, [1])

        self.close_input_op = self.input_queue.close()
        self.seq_len, self.outputs = tf.contrib.training.bucket_by_sequence_length(
            self.lengths_t, tensors=[self.input_t, self.lengths_t, self.u_idx_t, self.v_idx_t, self.r_t], allow_smaller_final_batch=True, \
            batch_size=batch_size, bucket_boundaries=bucket_boundaries, \
            capacity=2 * batch_size, dynamic_pad=True)

        self.data_loader = data_loader

    def get_outputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        return self.seq_len, self.outputs

    def enque_input(self, sess, validation=False, test=False):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        # u_idx, v_idx, ratings, docs = data_loader.split_data()
        # for i in range(data_loader.train_idx.size):
        #     sess.run(self.input_enqueue_op, feed_dict={
        #         self.input_text: docs[i], self.seq_lengths: len(docs[i]),
        #         self.u_idx: u_idx[i], self.v_idx: v_idx[i], self.r: ratings[i]})
        for u_idx, v_idx, ratings, docs in self.data_loader.generate_batches(1,validation=validation,test=test):
            sess.run(self.input_enqueue_op, feed_dict={
                self.input_text: docs, self.seq_lengths: get_lengths(docs),
                self.u_idx: u_idx, self.v_idx: v_idx, self.r: ratings})
            # print("QueueSize = %i" % (sess.run(self.queue_size)))


    def close(self, sess):
        sess.run(self.close_input_op)

def get_batches(sess, coord,custom_runner, valid=False, test=False):
    with tf.device("/cpu:0"):
        seq_len, outputs = custom_runner.get_outputs()
    try:

        custom_runner.enque_input(sess,valid,test)
        custom_runner.close(sess)
        b = 0
        batches={}
        while True:
            out_lengths, (input_t, lengths_t, u_idx_t, v_idx_t, r_t) = sess.run([seq_len, outputs])
            print(len(input_t[0, 0]))
            input_t = np.squeeze(input_t, [1])
            lengths_t = np.squeeze(lengths_t, [1])
            u_idx_t = np.squeeze(u_idx_t, [1])
            v_idx_t = np.squeeze(v_idx_t, [1])
            r_t = np.squeeze(r_t, [1])
            batches[b] = (input_t, lengths_t, u_idx_t, v_idx_t, r_t)
            b += 1
    except Exception as e:
        # Report exceptions to the coordinator.
        coord.request_stop(e)
        print("Total number of {0} samples: {1}".format('',b ))
        print("Total number of {0} batches: {1}".format('',b))
    finally:
        coord.request_stop()
        # coord.join(threads)
    return batches

test = False
if test:
    # these values specify the length of the sequence and this controls how
    # the data is bucketed. The value is not required to be the acutal length,
    # which is also problematic when using pairs of sequences that have diffrent
    # length. In that case just specify a value that gives the best performance,
    # for example "the max length".

    batch_size = 1
    data_loader = textloader('/home/wanli/data/glove.6B/',batch_size)
    vocab_size = data_loader.vocab_size

    if os.path.exists('abstracrs_word_embeddings_dummy.pkl'):
        print('Loading abstracts')
        with open('abstracrs_word_embeddings_dummy.pkl', 'rb') as f:
            data_loader.all_documents= pickle.load(f)

    print('Loading ratings')
    ratings_path = '/home/wanli/data/Extended_ctr/dummy/users.dat'
    data_loader.read_dataset(ratings_path,50,1928)#CHANGE ++++++++


    bucket_boundaries = [x for x in range (50,500,50)]

    batch_size = 128

    data_loader.split_data()

    with tf.Session(config=tf.ConfigProto(log_device_placement=False,intra_op_parallelism_threads=0)) as sess:
        with tf.device("/cpu:0"):
            custom_runner = CustomRunner(batch_size, bucket_boundaries, data_loader)
            seq_len, outputs = custom_runner.get_outputs()
            valid_u_idx, valid_v_idx, valid_m, valid_docs = data_loader.get_valid_idx()
            test_u_idx, test_v_idx, test_m, test_docs = data_loader.get_test_idx()
            v_queue = CustomRunner(valid_docs.shape[0], [0], data_loader, name='v')
            v_seq_len, v_outputs = v_queue.get_outputs()
            t_queue = CustomRunner(test_docs.shape[0], [0], data_loader, name='t')
            t_seq_len, t_outputs = t_queue.get_outputs()

        init = tf.global_variables_initializer()
        sess.run(tf.local_variables_initializer())
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        batches = {}
        try:
            custom_runner.enque_input(sess)
            custom_runner.close(sess)

            # threads = tf.train.start_queue_runners(sess, coord)
            b = 0
            while not coord.should_stop():
                out_lengths, (input_t, lengths_t, u_idx_t, v_idx_t, r_t) = sess.run([seq_len, outputs])
                print(len(input_t[0, 0]))

                input_t = np.squeeze(input_t,[1])
                lengths_t = np.squeeze(lengths_t,[1])
                u_idx_t= np.squeeze(u_idx_t,[1])
                v_idx_t= np.squeeze(v_idx_t,[1])
                r_t= np.squeeze(r_t,[1])

                batches[b]= (input_t, lengths_t, u_idx_t, v_idx_t, r_t)
                b += 1

        except Exception as e:
            # Report exceptions to the coordinator.
            coord.request_stop(e)
            print("Total number of training samples: {0}".format(b * batch_size))
            print("Total number of batches: {0}".format(b))
            print('Finished batching ')




        valid_batches = {}
        test_batches = {}
        # valid_batches = get_batches(sess, coord,v_queue,valid=True)
        # test_batches = get_batches(sess, coord, t_queue, test=True)

        coord.request_stop()
        coord.join(threads)

        for batch_num in range(len(batches)):
            print(len(batches[batch_num][0][0]))
#############################################
#
# n_steps=1
# training = []
# label = []
# for u, v, r, d, step in data_loader.generate_batches(n_steps):
#     training.append((u,v,d,len(d)))
# label=np.ones(len(training))
#
#
#
# v_idx = data_loader.nonzero_v_idx[data_loader.train_idx]
# train_doc = [data_loader.all_documents[doc] for doc in v_idx]


# print(len(lengths))

# text_table = SequenceTable(train_doc,tf.int32)
# u_table = SequenceTable(data_loader.nonzero_u_idx[data_loader.train_idx],tf.int32)
# v_table = SequenceTable(data_loader.nonzero_v_idx[data_loader.train_idx],tf.int32)
# r_table = SequenceTable(np.ones((len(lengths))),tf.int32)
#
# bucket_boundaries = [x for x in range (50,500,50)]
# print(bucket_boundaries)

# batch_size = 128
# # data_loader.batch_size = batch_size
# # ,shapes=[[None], [None], [None],[None],[None]]
# input_queue = tf.RandomShuffleQueue(
#     5000, dtypes=(tf.int32,tf.int32, tf.int32, tf.int32,tf.float32),
#     shared_name = 'shared_name',min_after_dequeue=1)

# input_queue = tf.PaddingFIFOQueue(
#     5000, dtypes=(tf.int32,tf.int32, tf.int32, tf.int32,tf.float32),shapes=[[None,None], [None], [None],[None],[None]],
#     shared_name = 'shared_name')
#
# queue_size = input_queue.size()
#
# input_text = tf.placeholder(tf.int32, [None,None],name="Input_text")
# seq_lengths = tf.placeholder(tf.int32, [None])
# u_idx = tf.placeholder(tf.int32, [None],name="U_matrix")
# v_idx = tf.placeholder(tf.int32, [None],name="V_matrix")
# r = tf.placeholder(tf.float32, [None],name="R_target")
#
# input_enqueue_op = input_queue.enqueue((input_text, seq_lengths, u_idx, v_idx, r))
# input_t, lengths_t, u_idx_t, v_idx_t, r_t = input_queue.dequeue()
# input_t = tf.reshape(input_t,[1,-1])
# lengths_t = tf.reshape(lengths_t,[1])
# u_idx_t= tf.reshape(u_idx_t,[1])
# v_idx_t= tf.reshape(v_idx_t,[1])
# r_t = tf.reshape(r_t,[1])
#
# close_input_op = input_queue.close()
# doc_lengths = get_lengths(train_doc)
# seq_len, outputs = tf.contrib.training.bucket_by_sequence_length(
#     lengths_t,tensors=[input_t, lengths_t, u_idx_t, v_idx_t, r_t],allow_smaller_final_batch=True,\
#                                                                      batch_size=batch_size,bucket_boundaries=bucket_boundaries, \
#                                                                      capacity= 2 * batch_size,dynamic_pad= True)
#
# #
# # source_batch, target_batch = shuffle_bucket_batch(
# #     lengths, [text_table,u_table, v_table, r_table],
# #     batch_size=batch_size,
# #     # devices buckets into [len < 3, 3 <= len < 5, 5 <= len]
# #     bucket_boundaries=bucket_boundaries,
# #     # this will bad the source_batch and target_batch independently
# #     dynamic_pad=True,
# #     capacity=2 * batch_size
# # )
# nb_batches = data_loader.train_idx.size // batch_size
# with tf.Session() as sess:
#     try:
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess, coord)
#         for u, v, r_t, d, step in data_loader.generate_batches(n_steps):
#             sess.run(input_enqueue_op, feed_dict={
#                 input_text: d, seq_lengths: get_lengths(d), u_idx:u,v_idx: v, r:r_t })
#             print("Step = %i, QueueSize = %i" % (step, sess.run(queue_size)))
#         sess.run(close_input_op)
#         # Start the queue runners
#         threads = tf.train.start_queue_runners(coord=coord)
#         # Read off the top of the bucket and ensure correctness of output
#         b =0
#         while True:
#             out_lengths, (input_t, lengths_t, u_idx_t, v_idx_t, r_t) = sess.run((seq_len,outputs))
#             print(len(input_t[0,0]))
#             b += 1
#         print("Total number of training samples: {0}".format(b*batch_size))
#         # while True:
#         #     i1, _ ,_ ,_ ,_ = input_queue.dequeue()
#         #     ii = sess.run(i1)
#         #     # print(len(ii[0, 0]))
#     except Exception as e:
#         # Report exceptions to the coordinator.
#         coord.request_stop(e)
#         print('Error: ',e)
#     finally:
#         coord.request_stop()
#         coord.join(threads)