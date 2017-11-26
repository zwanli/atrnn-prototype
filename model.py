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
    def __init__(self, args, ratings, features_matrix,
                 tags_count, confidence_matrix, train_filename, test_filename, enabel_dropout=False,
                 reg_lambda_u=0.01, reg_lambda_v=100, reg_lambda_att=0.5):
        self.args = args
        self.dataset = args.dataset

        # n: number of users
        # m: number of items
        self.n, self.m = ratings.shape
        self.k = args.embedding_dim
        self.training_samples_count = args.training_samples_count
        self.learning_rate = args.learning_rate
        self.maxlen = args.max_length
        self.reg_lambda_u = tf.constant(reg_lambda_u, dtype=tf.float32)
        self.reg_lambda_v = tf.constant(reg_lambda_v, dtype=tf.float32)

        self.batch_size = args.batch_size
        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)

        self.train_filename = tf.placeholder_with_default(train_filename, shape=(), name='train_filename')
        self.example_structure = args.example_structure
        self.test_filename = tf.placeholder_with_default(test_filename, shape=(), name='test_filename')

        outputs, init_ops = get_input_dataset(self.train_filename, self.test_filename, batch_size=self.batch_size,
                                              example_structure=self.example_structure)

        self.train_init_op, self.validation_init_op = init_ops
        self.u_idx, self.pos_idx, self.neg_idx, self.pos_len, self.neg_len = outputs

        # A matrix that contains all the abstracts
        self.abstracts_matrix_init = tf.placeholder(tf.int64, shape=(self.m, self.maxlen))
        self.abstracts_matrix = tf.Variable(self.abstracts_matrix_init, trainable=False)

        # get positive samples' abstracts
        self.pos_abstracts = tf.nn.embedding_lookup(self.abstracts_matrix, self.pos_idx)
        # get negative samples' abstracts
        self.neg_abstracts = tf.nn.embedding_lookup(self.abstracts_matrix, self.neg_idx)

        # Confidence matrix
        confidence = tf.constant(confidence_matrix, dtype=tf.float32, shape=confidence_matrix.shape,
                                 name='confidence')
        # Free some ram
        del confidence_matrix
        u_v_idx = tf.stack([self.u_idx, self.pos_idx], axis=1)
        confidence = tf.gather_nd(confidence, u_v_idx)

        use_rnn = args.use_rnn
        if use_rnn:
            with tf.variable_scope('RNN') as scope:
                self.pos_rnn_output = self.rnn_module(self.pos_abstracts, self.pos_len, args.rnn_size, args.vocab_size,
                                                      args.embedding_dim, enabel_dropout, model=args.model)
                scope.reuse_variables()
                self.neg_rnn_output = self.rnn_module(self.neg_abstracts, self.neg_len, args.rnn_size, args.vocab_size,
                                                      args.embedding_dim, enabel_dropout, model=args.model)

        use_attribues = args.use_att
        sum_joint_output = args.summation
        fc_joint_output = args.fc_layer
        att_ouput_dim = 50
        if use_attribues:
            ## Attributes component
            with tf.variable_scope('Attributes_component_%d-layers' % (args.num_layers)) as scope:
                self.pos_att_output = self.attribute_module(self.pos_idx, features_matrix, args.num_layers,
                                                            (att_ouput_dim if fc_joint_output else self.k))
                scope.reuse_variables()

                self.neg_att_output = self.attribute_module(self.neg_idx, features_matrix, args.num_layers,
                                                            (att_ouput_dim if fc_joint_output else self.k))
        # Free some ram
        del features_matrix

        with tf.name_scope('Matrix_factorizatin'):
            # U matrix [num_users, embeddings_dim]
            self.U = weight_variable([self.n, self.k], 'U')
            # V matrix [num_items, embeddings_dim]
            self.V = weight_variable([self.m, self.k], 'V')

            # U, V biases
            self.U_bias = bias_variable([self.n], 'U_bias')
            self.V_bias = bias_variable([self.m], 'V_bias')

            # Users' raws form U matrix considered for the current batch [batch_size, embeddings_dim]
            self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)
            # Items' raws form V matrix considered for the current batch [batch_size, embeddings_dim]
            self.pos_V_embed = tf.nn.embedding_lookup(self.V, self.pos_idx)
            self.neg_V_embed = tf.nn.embedding_lookup(self.V, self.neg_idx)

            self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
            self.pos_V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.pos_idx)
            self.neg_V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.neg_idx)

        with tf.name_scope('joint_output'):
            with tf.variable_scope("foo"):
                if use_attribues and use_rnn:
                    if fc_joint_output:
                        # Positive samples
                        # concatonate the rnn output and the attributes output
                        self.Q = tf.concat([self.pos_rnn_output, self.pos_att_output], 1)
                        self.fc_layer = tf.layers.dense(inputs=self.Q, units=self.k, activation=tf.nn.relu,
                                                        name='Dense_layer')
                        self.pos_fc_layer = tf.identity(self.fc_layer, name='fc_pos')
                        self.pos_F = tf.add(self.pos_V_embed, self.pos_fc_layer)
                        # Negative samples
                        # concatonate the rnn output and the attributes output
                        self.Q = tf.concat([self.neg_rnn_output, self.neg_att_output], 1)
                        self.fc_layer = tf.layers.dense(inputs=self.Q, units=self.k, activation=tf.nn.relu,
                                                        name='Dense_layer', reuse=True)
                        self.neg_fc_layer = tf.identity(self.fc_layer, name='fc_neg')
                        self.neg_F = tf.add(self.neg_V_embed, self.neg_fc_layer)

                    elif sum_joint_output:
                        # Positive samples
                        self.pos_F = tf.add(self.pos_rnn_output, self.pos_att_output)
                        self.pos_F = tf.add(self.pos_F, self.pos_V_embed)
                        # Negative samples
                        self.neg_F = tf.add(self.neg_rnn_output, self.neg_att_output)
                        self.neg_F = tf.add(self.neg_F, self.neg_V_embed)


                elif use_rnn:
                    # Positive samples
                    self.pos_F = tf.add(self.pos_V_embed, self.pos_rnn_output)
                    # Negative samples
                    self.neg_F = tf.add(self.neg_V_embed, self.neg_rnn_output)

                else:
                    # Positive samples
                    self.pos_F = self.pos_V_embed
                    # Negative samples
                    self.neg_F = self.neg_V_embed

                # self.r = tf.reduce_sum(tf.multiply(self.U_embed, self.pos_F), reduction_indices=1)
                # In case of pure MF, don't add biases
                # if use_rnn or use_attribues:
                #     self.biases = tf.add(2 * self.U_bias_embed, self.pos_V_bias_embed)
                #     self.biases = tf.add(self.biases, self.neg_V_bias_embed)
                if use_rnn or use_attribues:
                    self.pos_r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.pos_F), reduction_indices=1)
                    self.pos_r_hat = tf.add(self.pos_r_hat,  self.pos_V_bias_embed)
                    # self.pos_r_hat = tf.Print(self.pos_r_hat, [tf.shape(self.pos_r_hat), self.pos_r_hat],
                    #                           message='pos_r_hat=', first_n=20,
                    #                           summarize=4)
                    self.neg_r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.neg_F), reduction_indices=1)
                    self.neg_r_hat = tf.add(self.neg_r_hat,  self.neg_V_bias_embed)

        multi_task = args.multi_task
        with tf.name_scope('Tag_prediction'):
            if multi_task:
                # Tag prediction task
                tags_loss = self.tag_module(tags_count, self.k)

        with tf.name_scope('loss'):

            # self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.r, self.r_hat)))
            # Regularizers
            # User vector regularizer
            self.U_reg = tf.multiply(self.reg_lambda_u, tf.nn.l2_loss(self.U_embed))

            # add attribute module regularizer term
            if args.use_att:
                # get weight and biase variables
                weights_biases = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                   'Attributes_component_%d-layers' % (args.num_layers))
                regularizer = tf.contrib.layers.l2_regularizer(scale=reg_lambda_att)
                # attribute module regularizer term
                att_reg_term = tf.reduce_sum(tf.contrib.layers.apply_regularization(regularizer, weights_biases))

                self.reg_term = tf.add(self.U_reg, att_reg_term)
            else:
                self.reg_term = self.U_reg

            # NO NEED FOR CONFIDENCE for BPR
            self.log_loss = tf.reduce_mean(tf.log(tf.nn.sigmoid(tf.subtract(self.pos_r_hat, self.neg_r_hat))))


            # add regularization terms to the loss function
            if multi_task:
                mt_lambda = args.mt_lambda
                self.reg_loss = tf.add(mt_lambda * tf.add(self.log_loss, self.reg_term), (1 - mt_lambda) * tags_loss)
            else:
                self.V_reg = tf.multiply(self.reg_lambda_v, tf.nn.l2_loss(self.pos_V_embed))
                self.V_reg = tf.add(self.V_reg, tf.multiply(self.reg_lambda_v, tf.nn.l2_loss(self.neg_V_embed)))
                self.reg_term = tf.add(self.reg_term, self.V_reg)
                self.reg_loss = tf.add(self.log_loss, self.reg_term)

        with tf.name_scope('learning_rate_decay'):
            # self.global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            learning_rate = tf.train.exponential_decay(starter_learning_rate, self.batch_pointer,
                                                       100000, 0.96, staircase=True)

        with tf.name_scope('adam_optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            # In case of joint optimization
            self.joint_train_step = self.optimizer.minimize(self.reg_loss)

            # In case of alternate learning method
            self.train_step_u = self.optimizer.minimize(self.reg_loss, var_list=[self.U, self.U_bias])
            self.train_step_v = self.optimizer.minimize(self.reg_loss, var_list=[self.V, self.V_bias])
            if use_rnn:
                t_vars = tf.trainable_variables()
                gru_vars = [var for var in t_vars if 'gru_cell' in var.name]
                self.train_step_rnn = self.optimizer.minimize(self.reg_loss, var_list=[gru_vars])

        with tf.name_scope('update_doc_att_embedding'):  # RNN output [num_items, embedding_dim]
            # Update RNN and/or attribute joint output
            with tf.device("/cpu:0"):
                if use_rnn:
                    # RNN and/or attribute joint output [num_items, embedding_dim]
                    self.doc_embed = tf.get_variable(shape=[self.m, self.k], name='doc_embed', trainable=False,
                                                     dtype=tf.float32
                                                     , initializer=tf.constant_initializer(0.))
                    self.update_doc_embed = tf.scatter_update(self.doc_embed,
                                                              tf.concat([self.pos_idx, self.neg_idx], 0),
                                                              tf.concat([self.pos_rnn_output, self.neg_rnn_output], 0))

                if use_attribues and args.use_rnn:
                    self.att_embed = tf.get_variable(shape=[self.m, (att_ouput_dim if fc_joint_output else self.k)],
                                                     name='att_embed', trainable=False, dtype=tf.float32
                                                     , initializer=tf.constant_initializer(0.))
                    self.update_att_embed = tf.scatter_update(self.att_embed, tf.concat([self.pos_idx, self.neg_idx], 0)
                                                              ,
                                                              tf.concat([self.pos_att_output, self.neg_att_output], 0))

                    self.joint_doc_att_embed = tf.get_variable(shape=[self.m, self.k],
                                                               name='joint_doc_att_embed', trainable=False,
                                                               dtype=tf.float32
                                                               , initializer=tf.constant_initializer(0.))
                    if fc_joint_output:
                        self.update_doc_att_embed = tf.scatter_update(self.joint_doc_att_embed,
                                                                      tf.concat([self.pos_idx, self.neg_idx], 0),
                                                                      tf.concat(
                                                                          [self.pos_fc_layer, self.neg_fc_layer], 0))
                    elif sum_joint_output:
                        self.summation_output = tf.add(tf.concat([self.pos_att_output, self.neg_att_output], 0),
                                                       tf.concat([self.pos_rnn_output, self.neg_rnn_output], 0))
                        self.update_doc_att_embed = tf.scatter_update(self.joint_doc_att_embed,
                                                                      tf.concat([self.pos_idx, self.neg_idx], 0),
                                                                      self.summation_output)

        with tf.name_scope('Calculate_prediction_matrix'):
            # Get predicted ratings matrix
            if use_attribues and use_rnn:
                self.R_hat = tf.add(self.V, self.joint_doc_att_embed)
            elif use_rnn:
                self.R_hat = tf.add(self.V, self.doc_embed)
            elif use_attribues:
                self.R_hat = tf.add(self.V, self.att_embed)
            else:
                self.R_hat = self.V

            self.R_hat = tf.matmul(self.U, self.R_hat, transpose_b=True)

            if use_rnn:
                # self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
                self.R_hat = tf.add(self.R_hat, tf.reshape(self.U_bias, shape=[-1, 1]))
                self.get_prediction_matrix = tf.add(self.R_hat, self.V_bias, name="Prediction_matrix")

            else:  # In case of pure MF, don't add biases
                self.get_prediction_matrix = self.R_hat

        with tf.name_scope('metrics'):
            # useless loss functions
            labels = tf.ones(shape=self.batch_size)

            self.MSE = tf.metrics.mean_squared_error(labels, self.pos_r_hat)
            self.RMSE = tf.sqrt(self.MSE)
            # accuracy, _ = tf.metrics.(labels, self.pos_r_hat, name='Accuracy')
            # tf.summary.scalar('Accuracy', accuracy)
            # tf.summary.scalar("MSE", self.MSE)
            tf.summary.scalar("RMSE", self.RMSE)
            tf.summary.scalar('Log-Loss', self.log_loss)
            tf.summary.scalar("Reg-Loss", self.reg_loss)

            # add op for merging summary
            self.summary_op = tf.summary.merge_all()

            self.recall = tf.Variable(0, trainable=False, dtype=tf.float32)
            self.recall_10 = tf.Variable(0, trainable=False, dtype=tf.float32)
            self.recall_50 = tf.Variable(0, trainable=False, dtype=tf.float32)
            self.recall_100 = tf.Variable(0, trainable=False, dtype=tf.float32)
            self.recall_200 = tf.Variable(0, trainable=False, dtype=tf.float32)
            recall_sum = tf.summary.scalar("Recall", self.recall)
            recall_10_sum = tf.summary.scalar('recall@10', self.recall_10)
            recall_50_sum = tf.summary.scalar('recall@50', self.recall_50)
            recall_100_sum = tf.summary.scalar('recall@100', self.recall_100)
            recall_200_sum = tf.summary.scalar('recall@200', self.recall_200)

            self.ndcg_5 = tf.Variable(0, trainable=False, dtype=tf.float32)
            self.ndcg_10 = tf.Variable(0, trainable=False, dtype=tf.float32)
            ndcg_5_sum = tf.summary.scalar('ndcg@5', self.ndcg_5)
            ndcg_10_sum = tf.summary.scalar('ndcg@10', self.ndcg_10)

            self.mrr_10 = tf.Variable(0, trainable=False, dtype=tf.float32)
            mrr_10_sum = tf.summary.scalar('mrr@10', self.mrr_10)

            self.eval_metrics = tf.summary.merge(
                (recall_sum, recall_10_sum, recall_50_sum, recall_100_sum, recall_200_sum,
                 ndcg_5_sum, ndcg_10_sum, mrr_10_sum))

        # add Saver ops
        self.saver = tf.train.Saver()

    def rnn_module(self, abs_idx, abs_len, rnn_size, vocab_size, embedding_dim, enabel_dropout, model='gru'):
        if model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif model == 'gru':
            cell_fn = rnn.GRUCell
        elif model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(model))
        self.dropout_second_layer = tf.placeholder(tf.float32, name='dropout_second_layer')
        self.dropout_bidir_layer = tf.placeholder(tf.float32, name='dropout_bidir_layer')
        self.dropout_embed_layer = tf.placeholder(tf.float32, name='dropout_embed_layer')
        with tf.device("/cpu:0"):
            # A matrix that contains all the abstracts
            self.embeddings_init = tf.placeholder(tf.float32, shape=(vocab_size, embedding_dim), name='embeddings_init')

            # shape=[vocab_size, embedding_dim]
            embeddings = tf.get_variable(name="word_embeddings",
                                         initializer=self.embeddings_init, trainable=False)
            abstracts = tf.nn.embedding_lookup(embeddings, abs_idx)
            if enabel_dropout:
                abstracts = tf.contrib.layers.dropout(abstracts, keep_prob=1.0 - self.dropout_embed_layer)
        # First layer, bidirectional layer
        '''
                    https://arxiv.org/pdf/1512.05287.pdf
                    "The article says that dropout should be applied to RNN inputs+output as well as states,
                    using the same dropout mask for all the steps of the unrolled sequence.
                    This approach is called "variational dropout" and the primitives for implementing it
                    have recently been added to Tensorflow."
        '''

        cell_fw = cell_fn(rnn_size)
        if enabel_dropout:
            cell_fw = rnn.DropoutWrapper(
                cell_fw, input_keep_prob=1.0 - self.dropout_bidir_layer,
                output_keep_prob=1.0 - self.dropout_bidir_layer,
                state_keep_prob=1.0 - self.dropout_bidir_layer,
                dtype=tf.float32, variational_recurrent=True, input_size=embedding_dim)
        cell_bw = cell_fn(rnn_size)
        if enabel_dropout:
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell_bw, input_keep_prob=1.0 - self.dropout_bidir_layer,
                output_keep_prob=1.0 - self.dropout_bidir_layer,
                state_keep_prob=1.0 - self.dropout_bidir_layer,
                dtype=tf.float32, variational_recurrent=True, input_size=embedding_dim)
        self.init_state_fw = cell_bw.zero_state(self.batch_size, tf.float32)
        self.init_state_bw = cell_fw.zero_state(self.batch_size, tf.float32)

        bi_outputs, bi_output_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, abstracts, sequence_length=abs_len,
                                            initial_state_bw=self.init_state_bw,
                                            initial_state_fw=self.init_state_fw)
        bi_outputs = tf.concat(bi_outputs, 2)
        self.bi_output_state_fw, self.bi_output_state_bw = bi_output_state

        self.bi_output_state_fw = tf.identity(self.bi_output_state_fw,
                                              name='bi_state_fw')  # just to give it a name
        self.bi_output_state_bw = tf.identity(self.bi_output_state_bw,
                                              name='bi_state_bw')  # just to give it a name

        # Second layer
        cells = []
        rnn_num_layers = 1
        for _ in range(rnn_num_layers):
            cell = cell_fn(rnn_size)
            if enabel_dropout:
                cell = rnn.DropoutWrapper(
                    cell, input_keep_prob=1.0 - self.dropout_second_layer,
                    output_keep_prob=1.0 - self.dropout_second_layer,
                    state_keep_prob=1.0 - self.dropout_second_layer)
                # dtype=tf.float32, variational_recurrent=True, input_size= embedding_dim*2)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells)
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        # bi_outputs = tf.stack(bi_outputs,1)
        self.Yr, self.H = tf.nn.dynamic_rnn(cell, bi_outputs, sequence_length=abs_len,
                                            initial_state=self.initial_state, dtype=tf.float32)
        # Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
        # H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence
        self.H = tf.identity(self.H, name='H')  # just to give it a name
        self.Yr = tf.identity(self.Yr, name='Yr')

        # RNN output layer:
        # avg pool layer [batch_size, embedding_dim]
        self.rnn_output = tf.reduce_mean(self.Yr, 1)
        return self.rnn_output

    def attribute_module(self, abs_idx, features_matrix, n_layers, output_dim):
        '''

        :param input:
        :param n_layers:
        :return:
        '''

        # Implementation of a simple MLP network with one hidden layer.
        x_size = features_matrix.shape[1]

        self.features_matrix = tf.constant(features_matrix, dtype=tf.float32, shape=features_matrix.shape,
                                           name="attributes_matrix")

        # Attribute features vector
        self.input_att = tf.nn.embedding_lookup(self.features_matrix, abs_idx)
        # self.input_att = tf.Print(self.input_att, [tf.shape(self.input_att), self.input_att],
        #                         message='Attributes', first_n=20, summarize=4)

        # Network Parameters
        # calculate the number of hidden units for each hidden layer
        # N_h = N_s / (alpha * (N_i + N_o))
        # N_i  = number of input neurons.
        # N_o = number of output neurons.
        # N_s = number of samples in training data set.
        # alpha = an arbitrary scaling factor usually 2-10.
        alpha = 3
        n_hidden_1 = int(
            self.training_samples_count / (alpha * (x_size + self.k)))  # Number of neurons in the 1st layer
        alpha = 5
        n_hidden_2 = int(self.training_samples_count / (alpha * (x_size + self.k)))  # 2nd  layer
        alpha = 7
        n_hidden_3 = int(self.training_samples_count / (alpha * (x_size + self.k)))  # 3rd layer on wards
        y_size = output_dim

        # Input layer, User side
        with tf.name_scope('U_input_layer'):
            w_input = weight_variable([x_size, n_hidden_1], 'W_input')
            b_input = bias_variable(n_hidden_1, 'B_input')
            h_1 = tf.nn.relu(tf.add(tf.matmul(self.input_att, w_input), b_input))

        # Hidden layers
        for n in range(1, n_layers + 1):
            if n == 1:
                # Hidden layer
                with tf.name_scope('U_layer%d' % n):
                    w_h = weight_variable([n_hidden_1, n_hidden_2], 'W_%d' % n)
                    b_h = bias_variable(n_hidden_2, 'B_%d' % n)
                    h_h = tf.nn.relu(tf.add(tf.matmul(h_1, w_h), b_h), 'h_%d' % n)
            else:
                # Hidden layer
                with tf.name_scope('U_layer%d' % n):
                    w_h = weight_variable([n_hidden_2, n_hidden_3], 'W_%d' % n)
                    b_h = bias_variable(n_hidden_3, 'B_%d' % n)
                    h_h = tf.nn.relu(tf.add(tf.matmul(h_h, w_h), b_h), 'h_%d' % n)
        with tf.name_scope('output_layer'):
            if n_layers > 1:
                n_hidden_prev = n_hidden_3
            else:
                n_hidden_prev = n_hidden_2
            w_U_out = weight_variable([n_hidden_prev, y_size], 'W_out')
            b_U_out = bias_variable(y_size, 'B_out')
            attribute_output = tf.nn.relu(tf.add(tf.matmul(h_h, w_U_out), b_U_out), 'Attributes_output')

        print('  --Attribute module:\n number of layers')
        print('    Number of layers %d' % n_layers)
        print('    Number of unites per layer:')
        print('     input layer: %d' % n_hidden_1)
        print('     1st layer: %d' % n_hidden_2)
        if n_layers > 1:
            print('     2nd layer onwards: %d' % n_hidden_3)
        return attribute_output

    def tag_module(self, tags_count, embedding_dim):
        '''

        :param tags: is a tuple (tag_count, tag_idx)
        :param embedding_dim:
        :return:
        '''
        with tf.device("/cpu:0"):
            self.tags_matrix_init = tf.placeholder(tf.float32, shape=(self.m, tags_count))
            tags_matrix = tf.Variable(self.tags_matrix_init, trainable=False)
            # tags_matrix = tf.constant(tags, dtype=tf.int32, shape=tags.shape,
            #                           name='confidence')
            # tags_sparse = tf.SparseTensor(indices=tags[1], values= tf.ones(len(tags[1])), dense_shape=(self.m, tags[0]))
            tags_actual = tf.nn.embedding_lookup(tags_matrix, self.pos_idx)  # [batch_size, max_tags]

            tags_embeddings = tf.get_variable(name="embedding", shape=[tags_count, embedding_dim])
            # tags_embeddings = tf.nn.embedding_lookup(embedding_var,tags_actual) # [batch_size, max_tags, embeding_dim]
            # todo: change F to rnn output only
            tags_probalities = tf.einsum('ai,bi->ab', self.pos_rnn_output, tags_embeddings)

            # todo: add down weights for predicting the unobserved tags

            # tags_loss = tf.losses.sigmoid_cross_entropy(tags_actual,tags_probalities,)

            tags_sigmoid = tf.nn.sigmoid(tags_probalities)
            unobserved_tags_weight = 0.01
            tags_loss = -tf.reduce_mean(
                ((tags_actual * tf.log(tags_sigmoid)) + (
                    unobserved_tags_weight * (1 - tags_actual) * tf.log(1 - tags_sigmoid))),
                name='tags_loss')

        return tags_loss


def get_inputs(filename, batch_size, test=False):
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
    return u_idx, v_idx, r, input_text, seq_lengths


def _parse_function(sequence_example_proto):
    context_feature = {'u': tf.FixedLenFeature([], tf.int64),
                       'v': tf.FixedLenFeature([], tf.int64),
                       'r': tf.FixedLenFeature([], tf.int64),
                       'abs_length': tf.FixedLenFeature([], tf.int64)}

    sequence_feature = {'abstract': tf.FixedLenSequenceFeature([], tf.int64)}

    # Decode the record read by the reader
    context_feature, sequence_feature = tf.parse_single_sequence_example(sequence_example_proto,
                                                                         context_features=context_feature,
                                                                         sequence_features=sequence_feature)
    u = tf.cast(context_feature['u'], tf.int32)
    v = tf.cast(context_feature['v'], tf.int32)
    r = tf.cast(context_feature['r'], tf.int32)
    abs_length = tf.cast(context_feature['abs_length'], tf.int32)
    abstract = tf.cast(sequence_feature['abstract'], tf.int32)
    return u, v, r, abstract, abs_length


def _parse_triplets_function(example_proto):
    feature = {'u': tf.FixedLenFeature([], tf.int64),
               'v': tf.FixedLenFeature([], tf.int64),
               'r': tf.FixedLenFeature([], tf.int64),
               'abs_length': tf.FixedLenFeature([], tf.int64)}

    sequence_feature = {'abstract': tf.FixedLenSequenceFeature([], tf.int64)}
    feature = {
        'u_id': tf.FixedLenFeature([], tf.int64),
        'pos_id': tf.FixedLenFeature([], tf.int64),
        'neg_id': tf.FixedLenFeature([], tf.int64),
        'pos_length': tf.FixedLenFeature([], tf.int64),
        'neg_length': tf.FixedLenFeature([], tf.int64),
    }
    # Decode the record read by the reader
    features = tf.parse_single_example(example_proto, features=feature)
    u_id = tf.cast(features['u_id'], tf.int32)
    pos_id = tf.cast(features['pos_id'], tf.int32)
    neg_id = tf.cast(features['neg_id'], tf.int32)
    pos_length = tf.cast(features['pos_length'], tf.int32)
    neg_length = tf.cast(features['neg_length'], tf.int32)
    return u_id, pos_id, neg_id, pos_length, neg_length


def get_input_test(filenames, batch_size):
    # Creates a dataset that reads all of the examples from filenames.
    dataset = tf.contrib.data.TFRecordDataset(filenames)

    # Repeat the input indefinitely.
    dataset = dataset.repeat()
    # Parse the record into tensors.
    dataset = dataset.map(_parse_function)
    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=10000)
    # Generate batches
    # dataset = dataset.batch(128)

    # iterator = dataset.make_initializable_iterator()
    # print(dataset.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
    # print(dataset.output_shapes)  # ==> "(10, ((), (100,)))"

    dataset = dataset.padded_batch(batch_size, padded_shapes=((), (), (), [None], ()))
    # Create a one-shot iterator
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    # with tf.Session() as sess:
    #     for i in range(100):
    #         record = sess.run(next_element)
    return next_element


def get_input_dataset(train_filename, test_filename, batch_size, example_structure):
    '''

    :param train_filename:
    :param test_filename:
    :param batch_size:
    :param example_structure: Either 'single' sample or 'triplets' (u_id, i_id, j_id)
    :return:
    '''
    with tf.device("/cpu:0"):
        with tf.variable_scope('input'):
            # Creates a dataset that reads all of the examples from filenames.
            test_dataset = tf.contrib.data.TFRecordDataset(test_filename)
            training_dataset = tf.contrib.data.TFRecordDataset(train_filename)

            test_dataset = test_dataset.repeat()
            training_dataset = training_dataset.repeat()

            test_dataset = test_dataset.map(_parse_function)
            if example_structure == 'triplets':
                training_dataset = training_dataset.map(_parse_triplets_function)
            elif example_structure == 'single':
                training_dataset = training_dataset.map(_parse_function)

            training_dataset = training_dataset.padded_batch(batch_size, padded_shapes=((), (), (), (), ()))
            test_dataset = test_dataset.padded_batch(batch_size, padded_shapes=((), (), (), [None], ()))

            # A reinitializable iterator is defined by its structure. We could use the
            # `output_types` and `output_shapes` properties of either `training_dataset`
            # or `validation_dataset` here, because they are compatible.
            iterator = tf.contrib.data.Iterator.from_structure(training_dataset.output_types,
                                                               training_dataset.output_shapes)

            training_init_op = iterator.make_initializer(training_dataset)
            # validation_init_op = iterator.make_initializer(test_dataset)
            # todo: fix the test set iterator
            validation_init_op = None
            next_element = iterator.get_next()

            return next_element, (training_init_op, validation_init_op)
