import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import random
import numpy as np


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    b_init = tf.constant_initializer(0.)
    return tf.get_variable(name, shape, initializer=b_init)

class Model():
    def __init__(self, args, M, reg_lambda=0.01):
        self.args = args
        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))
        self.n, self.m = M.shape
        self.k = args.rnn_size
        self.learning_rate = args.learning_rate
        # self.batch_size = args.batch_size
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            cells.append(cell)
        self.cell = cell = rnn.MultiRNNCell(cells)

        # queue = tf.FIFOQueue()
        self.input_data = tf.placeholder(tf.int32, [None, args.seq_length],name="Input_text")
        self.seq_lengths = tf.placeholder(tf.int32,[None],name="seq_lengths")
        # self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length], name="Input_text")
        # self.seq_lengths = tf.placeholder(tf.int32, [args.batch_size], name="seq_lengths")
        # self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length],name="target_text")
        self.batch_size = tf.placeholder(tf.int32,[], name="batch_size")
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
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
            # softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            # variable_summaries(softmax_w)
            # softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            # variable_summaries(softmax_b)
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                # inputs = tf.split(, args.seq_length, 1)
                # inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
                inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        self.Yr, self.H = tf.nn.dynamic_rnn(cell,inputs,
                                            initial_state=self.initial_state,dtype=tf.float32)
        print(tf.shape(self.Yr))
        # Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
        # H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence
        self.H = tf.identity(self.H, name='H')  # just to give it a name
        self.Yr = tf.identity(self.H, name='Yr')
        _, _, self.G= tf.split(self.Yr,args.num_layers,0)
        self.G = tf.squeeze(self.G,0)
        print(tf.shape(self.G))
        self.u_idx = tf.placeholder(tf.int32, [None],name="U_matrix")
        self.v_idx = tf.placeholder(tf.int32, [None],name="V_matrix")
        self.r = tf.placeholder(tf.float32, [None],name="R_target")

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

        # add Saver ops
        self.saver = tf.train.Saver()
        # def loop(prev, _):
        #     prev = tf.matmul(prev, softmax_w) + softmax_b
        #     prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
        #     return tf.nn.embedding_lookup(embedding, prev_symbol)

        # outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        # output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])
        # self.logits = tf.matmul(output, softmax_w) + softmax_b
        # self.probs = tf.nn.softmax(self.logits)
        # loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
        #         [tf.reshape(self.targets, [-1])],
        #         [tf.ones([args.batch_size * args.seq_length])],
        #         args.vocab_size)
        # self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        # tf.summary.scalar("cost", self.cost)
        # self.final_state = last_state
        # self.lr = tf.Variable(0.0, trainable=False)
        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
        #         args.grad_clip)
        # optimizer = tf.train.AdamOptimizer(self.lr)
        # self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def build_graph(self):
        self.u_idx = tf.placeholder(tf.int32, [None])
        self.v_idx = tf.placeholder(tf.int32, [None])
        self.r = tf.placeholder(tf.float32, [None])

        self.U = weight_variable([self.n, self.k], 'U')
        self.V = weight_variable([self.m, self.k], 'V')

        self.U_bias = weight_variable([self.n], 'U_bias')
        self.V_bias = weight_variable([self.m], 'V_bias')

        self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.V_embed = tf.nn.embedding_lookup(self.V, self.v_idx)

        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.v_idx)

        self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
        self.r_hat = tf.add(self.r_hat, self.U_bias_embed)
        self.r_hat = tf.add(self.r_hat, self.V_bias_embed)

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

        tf.summary.scalar("RMSE", self.RMSE)
        tf.summary.scalar("MAE", self.MAE)
        tf.summary.scalar("L2-Loss", self.l2_loss)
        tf.summary.scalar("Reg-Loss", self.reg_loss)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        # add Saver ops
        self.saver = tf.train.Saver()

    def sample(self, sess, words, vocab, num=200, prime='first all', sampling_type=1, pick=0, width=4):
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        def beam_search_predict(sample, state):
            """Returns the updated probability distribution (`probs`) and
            `state` for a given `sample`. `sample` should be a sequence of
            vocabulary labels, with the last word to be tested against the RNN.
            """

            x = np.zeros((1, 1))
            x[0, 0] = sample[-1]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, final_state] = sess.run([self.probs, self.final_state],
                                            feed)
            return probs, final_state

        def beam_search_pick(prime, width):
            """Returns the beam search pick."""
            if not len(prime) or prime == ' ':
                prime = random.choice(list(vocab.keys()))
            prime_labels = [vocab.get(word, 0) for word in prime.split()]
            bs = BeamSearch(beam_search_predict,
                            sess.run(self.cell.zero_state(1, tf.float32)),
                            prime_labels)
            samples, scores = bs.search(None, None, k=width, maxsample=num)
            return samples[np.argmin(scores)]

        ret = ''
        if pick == 1:
            state = sess.run(self.cell.zero_state(1, tf.float32))
            if not len(prime) or prime == ' ':
                prime  = random.choice(list(vocab.keys()))
            print (prime)
            for word in prime.split()[:-1]:
                print (word)
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word,0)
                feed = {self.input_data: x, self.initial_state:state}
                [state] = sess.run([self.final_state], feed)

            ret = prime
            word = prime.split()[-1]
            for n in range(num):
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word, 0)
                feed = {self.input_data: x, self.initial_state:state}
                [probs, state] = sess.run([self.probs, self.final_state], feed)
                p = probs[0]

                if sampling_type == 0:
                    sample = np.argmax(p)
                elif sampling_type == 2:
                    if word == '\n':
                        sample = weighted_pick(p)
                    else:
                        sample = np.argmax(p)
                else: # sampling_type == 1 default:
                    sample = weighted_pick(p)

                pred = words[sample]
                ret += ' ' + pred
                word = pred
        elif pick == 2:
            pred = beam_search_pick(prime, width)
            for i, label in enumerate(pred):
                ret += ' ' + words[label] if i > 0 else words[label]
        return ret
