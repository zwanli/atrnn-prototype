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
    def __init__(self, args, ratings, embed,features_matrix,
                 tags_count, confidence_matrix, train_filename, test_filename, enabel_dropout=False, reg_lambda=0.01):
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
        self.n, self.m = ratings.shape
        self.k = args.embedding_dim
        self.training_samples_count = args.training_samples_count
        self.learning_rate = args.learning_rate
        self.maxlen = args.max_length
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)

        self.batch_size = args.batch_size

        outputs,init_ops = get_input_dataset(train_filename,test_filename, batch_size=self.batch_size)
        self.u_idx,self.v_idx, self.r, self.input_text, self.seq_lengths = outputs

        confidence = tf.constant(confidence_matrix, dtype=tf.float32, shape=confidence_matrix.shape,
                                 name='confidence')
        # Free some ram
        del confidence_matrix

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
        att_output = self.attribute_module(features_matrix,args.num_layers)

        # Free some ram
        del features_matrix

        # U matrix [num_users, embeddings_dim]
        self.U = weight_variable([self.n, self.k], 'U')
        # V matrix [num_items, embeddings_dim]
        self.V = weight_variable([self.m, self.k], 'V')



        # U, V biases
        self.U_bias = weight_variable([self.n], 'U_bias')
        self.V_bias = weight_variable([self.m], 'V_bias')

        # Users' raws form U matrix considered for the current batch [batch_size, embeddings_dim]
        self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)
        # Items' raws form V matrix considered for the current batch [batch_size, embeddings_dim]
        self.V_embed = tf.nn.embedding_lookup(self.V, self.v_idx)

        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.v_idx)


        self.F = tf.add(self.V_embed,self.G)

        self.F = tf.add(self.F,att_output)

        self.r_hat = tf.reduce_mean(tf.multiply(self.U_embed, self.F), reduction_indices=1)

        # self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
        self.r_hat = tf.add(self.r_hat, self.U_bias_embed)
        self.r_hat = tf.add(self.r_hat, self.V_bias_embed,name="R_predicted")

        # Update predicted ratings matrix
        with tf.device("/cpu:0"):
            # RNN output [num_items, embedding_dim]
            self.predicted_matrix = tf.get_variable(shape=[self.n, self.m], name='Predicted_ratings', trainable=False, dtype=tf.float32
                                       , initializer=tf.constant_initializer(0.))
            self.update_predicted_matrix = tf.scatter_nd_update(self.predicted_matrix,
                                                             indices=u_v_idx,updates=self.r_hat)

        # Tag prediction task
        tags_loss = self.tag_module(tags_count,self.k)
        # # Free some ram
        # del tags_matrix

        # Loss function
        self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.r, self.r_hat)))
        self.l2_loss =tf.nn.l2_loss(tf.subtract(self.r, self.r_hat))
        self.reg = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U)),
                          tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V)))
        # self.reg_loss = tf.add(self.l2_loss, self.reg)
        self.reg_loss = tf.add(self.l2_loss, tags_loss )


        self.MSE = tf.losses.mean_squared_error(self.r, self.r_hat,weights=confidence)


        self.RMSE = tf.sqrt(self.MSE)


        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # In case of joint loss function
        self.joint_train_step = self.optimizer.minimize(self.reg_loss)

        # In case of alternate learning method
        self.train_step_u = self.optimizer.minimize(self.reg_loss, var_list=[self.U, self.U_bias])
        self.train_step_v = self.optimizer.minimize(self.reg_loss, var_list=[self.V, self.V_bias])
        t_vars=tf.trainable_variables()
        gru_vars = [var for var in t_vars if 'gru_cell' in var.name]
        self.train_step_rnn = self.optimizer.minimize(self.reg_loss, var_list=[gru_vars])


        tf.summary.scalar("MSE", self.MSE)
        tf.summary.scalar("RMSE", self.RMSE)
        # tf.summary.scalar("MAE", self.MAE)
        # tf.summary.scalar("L2-Loss", self.l2_loss)
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

        self.mrr_10 = tf.Variable(0, trainable=False,dtype=tf.float32)
        mrr_10_sum = tf.summary.scalar('mrr@10', self.mrr_10)

        self.eval_metrics = tf.summary.merge((recall_sum,recall_10_sum,recall_50_sum,recall_100_sum,recall_200_sum,
                                             ndcg_5_sum, ndcg_10_sum,mrr_10_sum))

        # add Saver ops
        self.saver = tf.train.Saver()


    def attribute_module(self, features_matrix,  n_layers, ):
        '''

        :param input:
        :param n_layers:
        :return:
        '''
        # Implementation of a simple MLP network with one hidden layer.
        x_size = features_matrix.shape[1]

        self.features_matrix = tf.constant(features_matrix, dtype=tf.float32, shape=features_matrix.shape, name="attributes_matrix")

        # Attribute features vector
        self.input_att = tf.nn.embedding_lookup(self.features_matrix, self.v_idx)
        # self.input_att = tf.Print(self.input_att, [tf.shape(self.input_att), self.input_att],
        #                         message='Attributes', first_n=20, summarize=4)

        # Network Parameters
        # calculate the number of hidden units for each hidden layer
        # N_h = N_s / (alpha * (N_i + N_o))
        # N_i  = number of input neurons.
        # N_o = number of output neurons.
        # N_s = number of samples in training data set.
        # alpha = an arbitrary scaling factor usually 2-10.
        alpha = 2
        n_hidden_1 = int(self.training_samples_count / (alpha * (x_size + self.k)))  # 1st layer number of neurons
        n_hidden_2 = int(self.training_samples_count / (alpha * (x_size + self.k)))
        n_hidden_3 = int(self.training_samples_count / (alpha * (x_size + self.k)))  # 1st layer number of neurons
        y_size = self.k


        with tf.variable_scope('Attributes_component_%d-layers' % (n_layers)):

            # Input layer, User side
            with tf.name_scope('U_input_layer'):
                w_input = weight_variable([x_size,n_hidden_1], 'W_input')
                b_input = bias_variable(n_hidden_1, 'B_input')
                h_1 = tf.nn.relu(tf.add(tf.matmul(self.input_att, w_input), b_input))

            #Hidden layers
            for n in range(1,n_layers + 1):
                if n == 1:
                    # Hidden layer
                    with tf.name_scope('U_layer%d' % n):
                        w_h = weight_variable([n_hidden_1,n_hidden_2], 'W_%d' % n)
                        b_h = bias_variable(n_hidden_2, 'B_%d' % n)
                        h_h = tf.nn.relu(tf.add(tf.matmul(h_1,w_h),b_h),'h_%d' % n)
                else:
                    # Hidden layer
                    with tf.name_scope('U_layer%d' % n):
                        w_h = weight_variable([n_hidden_2,n_hidden_3], 'W_%d' % n)
                        b_h = bias_variable(n_hidden_3, 'B_%d' % n)
                        h_h = tf.nn.relu(tf.add(tf.matmul(h_h,w_h),b_h), 'h_%d' % n)
            with tf.name_scope('output_layer'):
                if n_layers > 2:
                    n_hidden_prev = n_hidden_3
                else:
                    n_hidden_prev = n_hidden_2
                w_U_out = weight_variable([n_hidden_prev, y_size], 'W_out')
                b_U_out = bias_variable(y_size, 'B_out')
                attribute_output = tf.nn.relu(tf.add(tf.matmul(h_h, w_U_out), b_U_out), 'Attributes_output')
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
            tags_actual = tf.nn.embedding_lookup(tags_matrix, self.v_idx) # [batch_size, max_tags]

            tags_embeddings = tf.get_variable(name="embedding", shape=[tags_count, embedding_dim])
            # tags_embeddings = tf.nn.embedding_lookup(embedding_var,tags_actual) # [batch_size, max_tags, embeding_dim]

            tags_probalities = tf.einsum('ai,bi->ab',self.F, tags_embeddings)

            # todo: add downweights for predicting the unobserved tags

            tags_loss = tf.losses.sigmoid_cross_entropy(tags_actual,tags_probalities,)
        return tags_loss


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
        r = tf.cast(context_feature['r'], tf.float32)
        abs_length = tf.cast(context_feature['abs_length'], tf.int32)
        abstract = tf.cast(sequence_feature['abstract'], tf.int32)
        return u, v, r, abstract, abs_length

def get_input_test(filenames,batch_size):
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

def get_input_dataset(train_filename,test_filename,batch_size):
    with tf.device("/cpu:0"):
        with tf.variable_scope('input'):
            # Creates a dataset that reads all of the examples from filenames.
            validation_dataset = tf.contrib.data.TFRecordDataset(test_filename)
            training_dataset = tf.contrib.data.TFRecordDataset(train_filename)

            validation_dataset = validation_dataset.repeat()
            training_dataset = training_dataset.repeat()

            validation_dataset = validation_dataset.map(_parse_function)
            training_dataset = training_dataset.map(_parse_function)

            training_dataset = training_dataset.padded_batch(batch_size, padded_shapes=((), (), (), [None], ()))
            validation_dataset = validation_dataset.padded_batch(batch_size, padded_shapes=((), (), (), [None], ()))

            # A reinitializable iterator is defined by its structure. We could use the
            # `output_types` and `output_shapes` properties of either `training_dataset`
            # or `validation_dataset` here, because they are compatible.
            iterator = tf.contrib.data.Iterator.from_structure(training_dataset.output_types,
                                                               training_dataset.output_shapes)

            training_init_op = iterator.make_initializer(training_dataset)
            validation_init_op = iterator.make_initializer(validation_dataset)

            next_element = iterator.get_next()

            return next_element, (training_init_op,validation_init_op)