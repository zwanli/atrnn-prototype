import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
from utils import read_and_decode


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    b_init = tf.constant_initializer(0.)
    return tf.get_variable(name, shape, initializer=b_init)

class Model():
    def __init__(self, args, M, embed,filename, reg_lambda=0.01,random_shuffl=False,name='tr'):
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


        # self.input_text = tf.placehoslder(tf.int32, [None, None],name="Input_text")
        # self.seq_lengths = tf.placeholder(tf.int32,[None],name="seq_lengths")
        # self.u_idx = tf.placeholder(tf.int32, [None],name="U_matrix")
        # self.v_idx = tf.placeholder(tf.int32, [None],name="V_matrix")
        # self.r = tf.placeholder(tf.float32, [None],name="R_target")

        with tf.device("/cpu:0"):  # with tf.device("/cpu:0"):
            self.u_idx,self.v_idx, self.r, self.input_text, self.seq_lengths\
                = get_inputs(filename,batch_size=self.batch_size)
            
        # self.batch_size = tf.placeholder(tf.int32,[], name="batch_size")
        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)
        self.batch_time = tf.Variable(0.0, name="batch_time", trainable=False)
        tf.summary.scalar("time_batch", self.batch_time)

        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                #with tf.name_scope('stddev'):
                #   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                #tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                #tf.summary.histogram('histogram', var)

        with tf.variable_scope('rnnlm'):
            with tf.device("/cpu:0"):
                vocab_size = args.vocab_size
                embedding_dim = len(embed[0])
                embeddings = np.asarray(embed)
                # embeddings = tf.get_variable("embeddings", shape=[dim1, dim2], initializer=tf.constant_initializer(np.array(embeddings_matrix))
                embedding = tf.get_variable(name="embedding", shape=[vocab_size, embedding_dim],
                                             initializer=tf.constant_initializer(embeddings), trainable=False)
                # embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                # inputs = tf.split(, args.seq_length, 1)
                # inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
                inputs = tf.nn.embedding_lookup(embedding, self.input_text)

        cells = []

        cell_fw = cell_fn(args.rnn_size)
        cell_bw = cell_fn(args.rnn_size)
        self.init_state_fw =cell_bw.zero_state(self.batch_size,tf.float32)
        self.init_state_bw =cell_fw.zero_state(self.batch_size,tf.float32)

        bi_outputs, bi_output_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=self.seq_lengths,
                                            initial_state_bw=self.init_state_bw, initial_state_fw=self.init_state_fw)
        bi_outputs = tf.concat(bi_outputs, 2)
        self.bi_output_state_fw, self.bi_output_state_bw = bi_output_state

        # (bi_outputs_fw, bi_outputs_bw), self.bi_output_state_fw, self.bi_output_state_bw = \
        #         tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,inputs,sequence_length=self.seq_lengths,
        #                                  initial_state_bw=self.init_state_bw,initial_state_fw=self.init_state_fw)
        # bi_outputs = tf.concat((bi_outputs_fw,bi_outputs_bw),2)

        self.bi_output_state_fw = tf.identity(self.bi_output_state_fw, name='bi_state_fw')  # just to give it a name
        self.bi_output_state_bw = tf.identity(self.bi_output_state_bw, name='bi_state_bw')  # just to give it a name

        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells)
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        # bi_outputs = tf.stack(bi_outputs,1)
        self.Yr, self.H = tf.nn.dynamic_rnn(cell,bi_outputs,sequence_length=self.seq_lengths,
                                            initial_state=self.initial_state,dtype=tf.float32)
        # Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
        # H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence
        self.H = tf.identity(self.H, name='H')  # just to give it a name
        self.Yr = tf.identity(self.Yr, name='Yr')

        self.G = tf.reduce_mean(self.Yr, 1)


        self.U = weight_variable([self.n, self.k], 'U')
        self.V = weight_variable([self.m, self.k], 'V')

        self.U_bias = weight_variable([self.n], 'U_bias')
        self.V_bias = weight_variable([self.m], 'V_bias')

        self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.V_embed = tf.nn.embedding_lookup(self.V, self.v_idx)

        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.v_idx)

        self.F = tf.add(self.V_embed,self.G)
        self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.F), reduction_indices=1)

        # self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
        self.r_hat = tf.add(self.r_hat, self.U_bias_embed)
        self.r_hat = tf.add(self.r_hat, self.V_bias_embed,name="R_predicted")

        self.RMSE = tf.sqrt(tf.losses.mean_squared_error(self.r, self.r_hat))
        self.l2_loss = tf.nn.l2_loss(tf.subtract(self.r, self.r_hat))
        self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.r, self.r_hat)))
        self.reg = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U)),
                          tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V)))
        self.reg_loss = tf.add(self.l2_loss, self.reg)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # self.train_step = self.optimizer.minimize(self.reg_loss)
        self.train_step_u = self.optimizer.minimize(self.reg_loss, var_list=[self.U, self.U_bias])
        self.train_step_v = self.optimizer.minimize(self.reg_loss, var_list=[self.V, self.V_bias])

        t_vars=tf.trainable_variables()
        gru_vars = [var for var in t_vars if 'gru_cell' in var.name]
        self.train_step_rnn = self.optimizer.minimize(self.reg_loss, var_list=[gru_vars])

        tf.summary.scalar("RMSE", self.RMSE)
        tf.summary.scalar("MAE", self.MAE)
        tf.summary.scalar("L2-Loss", self.l2_loss)
        tf.summary.scalar("Reg-Loss", self.reg_loss)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        self.recall = tf.Variable(0, trainable=False,dtype=tf.float32)
        self.recall_10 = tf.Variable(0, trainable=False, dtype=tf.float32)
        self.recall_50 = tf.Variable (0,trainable=False, dtype=tf.float32)
        self.recall_100 = tf.Variable(0, trainable=False, dtype=tf.float32)
        self.recall_200 = tf.Variable(0, trainable=False, dtype=tf.float32)
        recall_sum =tf.summary.scalar("Recall",self.recall)
        recall_10_sum = tf.summary.scalar('recall@10',self.recall_10)
        recall_50_sum = tf.summary.scalar('recall@50',self.recall_50)
        recall_100_sum = tf.summary.scalar('recall@100',self.recall_100)
        recall_200_sum = tf.summary.scalar('recall@200',self.recall_200)

        self.ndcg_5 = tf.Variable(0, trainable=False,dtype=tf.float32)
        self.ndcg_10 = tf.Variable(0, trainable=False,dtype=tf.float32)
        ndcg_5_sum = tf.summary.scalar('ndcg@5',self.ndcg_5)
        ndcg_10_sum = tf.summary.scalar('ndcg@10',self.ndcg_10)


        self.eval_metrics = tf.summary.merge((recall_sum,recall_10_sum,recall_50_sum,recall_100_sum,recall_200_sum,
                                             ndcg_5_sum, ndcg_10_sum))



        # add Saver ops
        self.saver = tf.train.Saver()

def get_inputs(filename,batch_size,test=False):

    # The actual queue of data. The queue contains a vector for
    if (not os.path.exists(filename)):
        print('Dataset file does not exist {0}'.format(filename))
        raise SystemExit
    capacity = 20 * batch_size

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

    # Output order: u, v, r, doc, length
    train_features = read_and_decode(filename_queue=filename_queue)
    input_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=(tf.int32, tf.int32, tf.float32, tf.int32, tf.int32),
        shared_name='shared_name{0}'.format('_train'))
    # The symbolic operation to add data to the queue
    input_enqueue_op = input_queue.enqueue(train_features)
    numberOfThreads = 1
    # now setup a queue runner to handle enqueue_op outside of the main thread asynchronously
    qr = tf.train.QueueRunner(input_queue, [input_enqueue_op] * numberOfThreads)
    # now we need to add qr to the TensorFlow queue runners collection
    tf.train.add_queue_runner(qr)
    # outputs(u, v, r, abstract, abs_length)
    outputs = input_queue.dequeue()

    u_idx_t = tf.reshape(outputs[0], [1])
    v_idx_t = tf.reshape(outputs[1], [1])
    r_t = tf.reshape(outputs[2], [1])
    input_t = tf.reshape(outputs[3], [1, -1])
    lengths_t = tf.reshape(outputs[4], [1])

    bucket_boundaries = [x for x in range(50, 500, 50)]
    seq_len, outputs_b = tf.contrib.training.bucket_by_sequence_length(
        lengths_t, tensors=[u_idx_t, v_idx_t, r_t, input_t, lengths_t],
        allow_smaller_final_batch=True, \
        batch_size=batch_size, bucket_boundaries=bucket_boundaries, \
        capacity=capacity, dynamic_pad=True)
    u_idx = tf.squeeze(outputs_b[0], [1], name="U_matrix")
    v_idx = tf.squeeze(outputs_b[1], [1], name="V_matrix")
    r = tf.squeeze(outputs_b[2], [1], name='R_target')
    input_text = tf.squeeze(outputs_b[3], [1], name="Input_text")
    seq_lengths = tf.squeeze(outputs_b[4], [1], name="seq_lengths")
    return u_idx,v_idx,r,input_text,seq_lengths