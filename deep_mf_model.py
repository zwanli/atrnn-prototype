import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
from utils import read_and_decode
from model import get_input_dataset

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.get_variable( name=name,initializer=initial)

def bias_variable(shape, name):
    b_init = tf.constant_initializer(0.)
    return tf.get_variable(name, shape, initializer=b_init)

class DMF_Model():
    def __init__(self, args, M, embed,confidence_matrix,train_filename,test_filename,enabel_dropout=False, reg_lambda=0.01):
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
        # n: number of users
        # m: number of items
        self.n, self.m = M.shape
        self.training_samples_count = args.training_samples_count
        self.k = args.embedding_dim
        self.learning_rate = args.learning_rate
        self.maxlen = args.max_length
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)
        self.batch_size = args.batch_size

        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)

        ##Get input
        outputs,init_ops = get_input_dataset(train_filename,test_filename, batch_size=self.batch_size)
        self.u_idx,self.v_idx, self.r, self.input_text, self.seq_lengths = outputs

        confidence = tf.constant(confidence_matrix, dtype=tf.float32, shape=confidence_matrix.shape,
                                 name='confidence')
        u_v_idx = tf.stack([self.u_idx, self.v_idx], axis=1)
        confidence = tf.gather_nd(confidence, u_v_idx)

        # # limit the input_text sequence length [batch_size, max_lenght]
        # # if self.input_text.shape[1] > self.maxlen:
        # #     self.input_text = tf.slice(self.input_text, [0, 0], [-1, ])
        # def f1():
        #     return tf.slice(self.input_text, [0, 0], [-1, self.maxlen])
        #
        # def f2():
        #     return self.input_text
        # self.input_text = tf.cond(tf.less(self.maxlen,self.input_text.shape[1]),f1,f2)

        self.train_init_op, self.validation_init_op = init_ops

        self.dropout_second_layer = tf.placeholder(tf.float32, name='dropout_second_layer')
        self.dropout_bidir_layer = tf.placeholder(tf.float32, name='dropout_bidir_layer')
        self.dropout_embed_layer = tf.placeholder(tf.float32, name='dropout_embed_layer')

        with tf.variable_scope('rnn'):

            with tf.device("/cpu:0"):
                vocab_size = args.vocab_size
                embedding_dim = args.embedding_dim
                embeddings = np.asarray(embed)
                embedding = tf.get_variable(name="embedding", shape=[vocab_size, embedding_dim],
                                             initializer=tf.constant_initializer(embeddings), trainable=False)
                inputs = tf.nn.embedding_lookup(embedding, self.input_text)
                if enabel_dropout:
                    inputs = tf.contrib.layers.dropout(inputs, keep_prob=1.0-self.dropout_embed_layer)

            #First layer, bidirectional layer
            '''
                        https://arxiv.org/pdf/1512.05287.pdf
                        "The article says that dropout should be applied to RNN inputs+output as well as states,
                        using the same dropout mask for all the steps of the unrolled sequence.
                        This approach is called "variational dropout" and the primitives for implementing it
                        have recently been added to Tensorflow."
            '''

            cell_fw = cell_fn(args.rnn_size)
            if enabel_dropout:
                cell_fw = rnn.DropoutWrapper(
                    cell_fw, input_keep_prob=1.0-self.dropout_bidir_layer, output_keep_prob=1.0-self.dropout_bidir_layer,
                    state_keep_prob = 1.0-self.dropout_bidir_layer,
                    dtype=tf.float32, variational_recurrent=True, input_size= embedding_dim)
            cell_bw = cell_fn(args.rnn_size)
            if enabel_dropout:
                cell_bw = tf.contrib.rnn.DropoutWrapper(
                    cell_bw, input_keep_prob=1.0-self.dropout_bidir_layer, output_keep_prob=1.0-self.dropout_bidir_layer,
                    state_keep_prob=1.0-self.dropout_bidir_layer,
                    dtype=tf.float32, variational_recurrent=True, input_size= embedding_dim)
            self.init_state_fw =cell_bw.zero_state(self.batch_size, tf.float32)
            self.init_state_bw =cell_fw.zero_state(self.batch_size, tf.float32)

            bi_outputs, bi_output_state = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=self.seq_lengths,
                                                initial_state_bw=self.init_state_bw, initial_state_fw=self.init_state_fw)
            bi_outputs = tf.concat(bi_outputs, 2)
            self.bi_output_state_fw, self.bi_output_state_bw = bi_output_state


            self.bi_output_state_fw = tf.identity(self.bi_output_state_fw, name='bi_state_fw')  # just to give it a name
            self.bi_output_state_bw = tf.identity(self.bi_output_state_bw, name='bi_state_bw')  # just to give it a name

            #Second layer
            cells = []
            for _ in range(args.num_layers):
                cell = cell_fn(args.rnn_size)
                if enabel_dropout:
                    cell = rnn.DropoutWrapper(
                        cell, input_keep_prob=1.0-self.dropout_second_layer, output_keep_prob=1.0-self.dropout_second_layer,
                        state_keep_prob=1.0-self.dropout_second_layer)
                        #dtype=tf.float32, variational_recurrent=True, input_size= embedding_dim*2)
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

            # RNN output layer:
            # avg pool layer [batch_size, embedding_dim]
            self.G = tf.reduce_mean(self.Yr, 1)

        #Update RNN output
        with tf.device("/cpu:0"):
            # RNN output [num_items, embedding_dim]
            self.RNN = tf.get_variable(shape=[self.m, self.k], name='RNN_output', trainable=False, dtype=tf.float32
                                       ,initializer=tf.constant_initializer(0.))
            self.update_rnn_output = tf.scatter_update(self.RNN, self.v_idx, self.G)


        ## Attributes component


        ## Deep matrix factorization model
        self.Y_matrix = tf.constant(M,dtype=tf.float32,shape=M.shape,name="Y_actual")

        # Network Parameters
        # calculate the number of hidden units for each hidden layer
        # N_h = N_s / (alpha * (N_i + N_o))
        # N_i  = number of input neurons.
        # N_o = number of output neurons.
        # N_s = number of samples in training data set.
        # alpha = an arbitrary scaling factor usually 2-10.
        alpha = 0.1
        n_hidden_1_U = int (self.training_samples_count / (alpha * (self.m + self.k))) # 1st layer number of neurons
        n_hidden_1_V = int (self.training_samples_count / (alpha * (self.n + self.k)))  # 1st layer number of neurons
        alpha *= 2
        n_hidden_2_U = int (self.training_samples_count / (alpha * (self.m + self.k)))
        n_hidden_2_V = int (self.training_samples_count / (alpha * (self.n + self.k)))  # 2nd layer number of neurons
        alpha *= 3
        n_hidden_3_U = int(self.training_samples_count / (alpha * (self.m + self.k)))  # 1st layer number of neurons
        n_hidden_3_V = int(self.training_samples_count / (alpha * (self.n + self.k)))  # 1st layer number of neurons
        n_output = self.k


        # U, V biases
        self.U_bias = weight_variable([self.n], 'U_bias')
        self.V_bias = weight_variable([self.m], 'V_bias')

        # Users' raws form U matrix considered for the current batch [batch_size, num_items]
        self.U_embed = tf.nn.embedding_lookup(self.Y_matrix, self.u_idx)
        self.U_embed = tf.Print(self.U_embed, [tf.shape(self.U_embed), tf.reduce_sum(self.U_embed), self.u_idx],
                                message='U', first_n=20, summarize=4)

        # Items' raws form V matrix considered for the current batch [batch_size, num_users]
        self.V_embed = tf.nn.embedding_lookup(tf.transpose(self.Y_matrix), self.v_idx)
        self.V_embed = tf.Print(self.V_embed,[tf.shape(self.V_embed),tf.reduce_sum(self.V_embed),self.v_idx],message='V',first_n=20,summarize=4)


        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.v_idx)


        with tf.variable_scope('DeepMF_%d' % (args.mf_num_layers)):

            # First layer, User side
            with tf.name_scope('U_input_layer'):
                w_U_1 = weight_variable([self.m,n_hidden_1_U], 'W_U_1')
                b_U_1 = bias_variable(n_hidden_1_U, 'B_U_1')
                h_U_1 = tf.nn.relu(tf.add(tf.matmul(self.U_embed, w_U_1), b_U_1))


            # First layer, item side
            with tf.name_scope('V_input_layer'):
                w_V_1 = weight_variable([self.n,n_hidden_1_V], 'W_V_1')
                b_V_1 = bias_variable(n_hidden_1_V, 'B_V_1')
                h_V_1 = tf.nn.relu(tf.add(tf.matmul(self.V_embed, w_V_1),b_V_1))

            #Hidden layers
            for n in range(2,args.mf_num_layers +1 ):
                if n == 2:
                    # Hidden layer, User side
                    with tf.name_scope('U_layer%d' % n):
                        w_U_h = weight_variable([n_hidden_1_U,n_hidden_2_U], 'W_U_%d' % n)
                        b_U_h = bias_variable(n_hidden_2_U, 'B_U_%d' % n)
                        h_U_h = tf.nn.relu(tf.add(tf.matmul(h_U_1,w_U_h),b_U_h),'h_U_%d' % n)
                    # Hidden layer, item side
                    with tf.name_scope('V_layer%d' % n):
                        w_V_h = weight_variable([n_hidden_1_V,n_hidden_2_V], 'W_V_%d' % n)
                        b_V_h = bias_variable(n_hidden_2_V, 'B_V_%d' % n)
                        h_V_h = tf.nn.relu(tf.add(tf.matmul(h_V_1,w_V_h),b_V_h),'h_V_%d' % n)
                else:
                    # Hidden layer, User side
                    with tf.name_scope('U_layer%d' % n):
                        w_U_h = weight_variable([n_hidden_2_U,n_hidden_3_U], 'W_U_%d' % n)
                        b_U_h = bias_variable(n_hidden_3_U, 'B_U_%d' % n)
                        h_U_h = tf.nn.relu(tf.add(tf.matmul(h_U_h,w_U_h),b_U_h), 'h_U_%d' % n)
                    # Hidden layer, item side
                    with tf.name_scope('V_layer%d' % n):
                        w_V_h = weight_variable([n_hidden_2_V,n_hidden_3_V], 'W_V_%d' % n)
                        b_V_h = bias_variable(n_hidden_3_V, 'B_V_%d' % n)
                        h_V_h = tf.nn.relu(tf.add(tf.matmul(h_V_h,w_V_h),b_V_h), 'h_V_%d' % n)

            with tf.name_scope('output_layer'):
                with tf.name_scope('U_out_layer'):
                    if args.mf_num_layers > 2:
                        n_hidden_prev = n_hidden_3_U
                    else:
                        n_hidden_prev = n_hidden_2_U
                    w_U_out = weight_variable([n_hidden_prev, n_output], 'W_U_out')
                    b_U_out = bias_variable(n_output, 'B_U_out')
                    p = tf.nn.relu(tf.add(tf.matmul(h_U_h, w_U_out), b_U_out), 'item_embedding')
                # Hidden layer, item side
                with tf.name_scope('V_out_layer'):
                    if args.mf_num_layers > 2:
                        n_hidden_prev = n_hidden_3_V
                    else:
                        n_hidden_prev = n_hidden_2_V
                    w_V_out = weight_variable([n_hidden_prev, n_output], 'W_V_out')
                    b_V_out = bias_variable(n_output, 'B_V_out')
                    q = tf.nn.relu(tf.add(tf.matmul(h_V_h, w_V_out), b_V_out), 'user_embedding')

        with tf.name_scope('joint_output'):
            #model output
            #combine the rnn output with the item embedding
            # self.F = tf.add(q, self.G)
            self.F = q
            self.F = tf.Print(self.F, [self.F,tf.shape(self.F)], message='Q=',first_n=20,summarize=4)
            p = tf.Print(p, [p,tf.shape(p)], message='P=',first_n=20,summarize=4)

            mu = 1e-6
            #cosine distance
            cos_dist = tf.losses.cosine_distance(tf.nn.l2_normalize(p,1),tf.nn.l2_normalize(self.F,1)
                                                            ,reduction=tf.losses.Reduction.NONE,dim=1)
            cos_dist = tf.Print(cos_dist,[cos_dist,tf.shape(cos_dist)],message='cos_dist',first_n=20,summarize=4)

            #cos_similarity
            cos_similarity = tf.subtract(1.0, cos_dist)
            cos_similarity = tf.Print(cos_similarity,[tf.reduce_max(cos_similarity),tf.reduce_min(cos_similarity)],
                                      message='cos_sim',first_n=20,summarize=4)
            # Make sure the outpust is positive
            logits = tf.maximum(cos_similarity,mu)

            logits = tf.Print(logits, [logits,tf.shape(logits),tf.reduce_min(logits),tf.reduce_max(logits),tf.count_nonzero(logits)],
                              message='Logits=',first_n=20,summarize=4)

        with tf.name_scope('regularize'):
            # t_vars = tf.trainable_variables()
            # weights_biases = [var for var in t_vars if 'DeepMF' in var.name]
            # reg = tf.add_n([tf.nn.l2_loss(v) for v in weights_biases]) * self.reg_lambda
            weights_biases = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'DeepMF')
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_lambda)
            reg_term = tf.contrib.layers.apply_regularization(regularizer, weights_biases)

        with tf.name_scope('loss'):
            # Define loss and optimizer
            #Create a 2D labels array [Batch_size, 2]
            #The first dimension is the actual label, the second is (10-label)
            # r_2D = tf.stack([self.r, tf.subtract(1.0, self.r)], axis=1)
            logits_2D = tf.concat([logits, tf.subtract(1.0, logits)], axis=1)
            self.cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
                logits=logits_2D, labels=tf.to_int32(self.r), weights=confidence))

            self.cross_entropy = tf.Print(self.cross_entropy, [self.cross_entropy], message='CE=',first_n=20,summarize=4)

            loss = tf.add(self.cross_entropy , tf.reduce_sum(reg_term))
            loss = tf.add(loss,self.U_bias_embed)
            loss = tf.add(loss,self.V_bias_embed)
            loss = tf.Print(loss, [loss], message='Loss=',first_n=20,summarize=4)


            # self.train_step_rnn = self.optimizer.minimize(self.reg_loss, var_list=[gru_vars])
        with tf.name_scope('learning_rate_decay'):
            # self.global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            learning_rate = tf.train.exponential_decay(starter_learning_rate, self.batch_pointer,
                                                       100000, 0.96, staircase=True)

        with tf.name_scope('adam_optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = optimizer.minimize(loss)

        with tf.name_scope('metrics'):
            correct_prediction = tf.greater(logits,0.5)
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            correct_prediction = tf.equal(correct_prediction, self.r)
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)
            self.MSE = tf.losses.mean_squared_error(self.r,tf.reshape(logits,[-1]))
            self.RMSE = tf.sqrt(self.MSE)
        # stats for display
        # loss_summary = tf.summary.scalar("batch_loss", batchloss)
        acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
        tf.summary.scalar("RMSE", self.RMSE)
        tf.summary.scalar('Cross-entropy',self.cross_entropy)
        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        #
        #
        # self.F = tf.add(self.V_embed,self.G)
        #
        # self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.F), reduction_indices=1)
        #
        # # self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
        # self.r_hat = tf.add(self.r_hat, self.U_bias_embed)
        # self.r_hat = tf.add(self.r_hat, self.V_bias_embed,name="R_predicted")
        #
        # self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.r, self.r_hat)))
        # self.l2_loss =tf.nn.l2_loss(tf.multiply(confidence,tf.subtract(self.r, self.r_hat)))
        #
        # self.MSE = tf.losses.mean_squared_error(self.r, self.r_hat,weights=confidence)
        # self.RMSE = tf.sqrt(self.MSE)
        #
        # self.reg = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U)),
        #                   tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V)))
        # self.reg_loss = tf.add(self.l2_loss, self.reg)
        #
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        #
        # self.joint_train_step = self.optimizer.minimize(self.reg_loss)
        #
        # self.train_step_u = self.optimizer.minimize(self.reg_loss, var_list=[self.U, self.U_bias])
        # self.train_step_v = self.optimizer.minimize(self.reg_loss, var_list=[self.V, self.V_bias])
        #
        # t_vars=tf.trainable_variables()
        # gru_vars = [var for var in t_vars if 'gru_cell' in var.name]
        # self.train_step_rnn = self.optimizer.minimize(self.reg_loss, var_list=[gru_vars])
        #
        # tf.summary.scalar("MSE", self.MSE)

        # tf.summary.scalar("MAE", self.MAE)
        # tf.summary.scalar("L2-Loss", self.l2_loss)
        # tf.summary.scalar("Reg-Loss", self.reg_loss)
        #




        # add Saver ops
        self.saver = tf.train.Saver()

    def loss(self, logits, labels):
        """Calculates the loss from the logits and the labels.

        Args:
          logits: Logits tensor, float - [batch_size, 2].
          labels: Labels tensor, int32 - [batch_size, 2].

        Returns:
          loss: Loss tensor of type float.
        """
        with tf.name_scope('loss'):
            logits = tf.reshape(logits, (-1, 2))
            shape = [logits.get_shape()[0], 2]
            epsilon = tf.constant(value=1e-8, shape=shape)
            logits = logits + epsilon
            labels = tf.to_float(tf.reshape(labels, (-1, 2)))

            softmax = tf.nn.softmax(logits)
            cross_entropy = -tf.reduce_sum(labels * tf.log(softmax),
                                           reduction_indices=[1])

            cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                                name='xentropy_mean')
            tf.add_to_collection('losses', cross_entropy_mean)

            loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        return loss

    def metrics(self):
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

        self.mrr_10 = tf.Variable(0, trainable=False,dtype=tf.float32)
        mrr_10_sum = tf.summary.scalar('mrr@10', self.mrr_10)

        self.eval_metrics = tf.summary.merge((recall_sum,recall_10_sum,recall_50_sum,recall_100_sum,recall_200_sum,
                                             ndcg_5_sum, ndcg_10_sum,mrr_10_sum))


