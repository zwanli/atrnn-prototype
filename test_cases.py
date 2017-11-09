import numpy as np
import tensorflow as tf
import math
from evaluator import Evaluator
from model import get_inputs
from model import Model
from model import get_input_test
from model import _parse_function
from model import get_input_dataset
from tensorflow.contrib import rnn
import time
import utils
from scipy.sparse import rand
from math import log
import os
import csv

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', 'logs/checkpoints',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.
      Args:
        saver: Saver.
        summary_writer: Summary writer.
        summary_op: Summary op.
      """


def evaluate(sess, filename, ckpt_dir, rating_matrix, args, embeddings, test_writer, uv_matrices=None):
    def construct_feed(bi_hid_fw, bi_hid_bw):
        return {model.init_state_fw: bi_hid_fw, model.init_state_bw: bi_hid_bw}

    evaluator = Evaluator(rating_matrix, verbose=True)

    with tf.Graph().as_default() as g:
        saver = tf.train.import_meta_graph(ckpt_dir)
        model = Model(args, rating_matrix, embeddings, filename, test=True)
        # with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        test_bi_fw = sess.run(model.init_state_fw)
        test_bi_bw = sess.run(model.init_state_bw)
        init_state = sess.run(model.initial_state)
        feed_dict = construct_feed(test_bi_fw, test_bi_bw)
        rmse_test, mae_test, summary_str = sess.run(
            [model.RMSE, model.MAE, model.summary_op], feed_dict=feed_dict)

        test_writer.add_summary(summary_str, global_step)

        # prediction_matrix = np.matmul(uv_matrices.U, uv_matrices.V.T)
        # prediction_matrix = np.add(prediction_matrix, np.reshape(uv_matrices.U_b, [-1, 1]))
        # prediction_matrix = np.add(prediction_matrix, uv_matrices.V_b)
        # rounded_predictions = utils.rounded_predictions(prediction_matrix)
        #
        # evaluator.load_top_recommendations_2(200, prediction_matrix, test_ratings)
        # recall_10 = evaluator.recall_at_x(10, prediction_matrix, parser.ratings, rounded_predictions)
        # recall_50 = evaluator.recall_at_x(50, prediction_matrix, parser.ratings, rounded_predictions)
        # recall_100 = evaluator.recall_at_x(100, prediction_matrix, parser.ratings, rounded_predictions)
        # recall_200 = evaluator.recall_at_x(200, prediction_matrix, parser.ratings, rounded_predictions)
        # recall = evaluator.calculate_recall(ratings=parser.ratings, predictions=rounded_predictions)
        # ndcg_at_five = evaluator.calculate_ndcg(5, rounded_predictions)
        # ndcg_at_ten = evaluator.calculate_ndcg(10, rounded_predictions)
        #
        # print("         | Recall@10: {0:3.4f}".format(recall_10))
        # print("         | Recall@50: {0:3.4f}".format(recall_50))
        # print("         | Recall@100: {0:3.4f}".format(recall_100))
        # print("         | Recall@200: {0:3.4f}".format(recall_200))
        # print("         | Recall: {0:3.4f}".format(recall))
        # print("         | ndcg@5: {0:3.4f}".format(ndcg_at_five))
        # print("         | ndcg@10: {0:3.4f}".format(ndcg_at_ten))

        # print("Step {0} | Train RMSE: {1:3.4f}, MAE: {2:3.4f}".format(
        #     step, rmse, mae))
        # # print("         | Valid  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
        #     rmse_valid, mae_valid))
        print("         | Test  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
            rmse_test, mae_test))

        # if best_val_rmse > rmse_test:
        #     # best_val_rmse = rmse_valid
        #     best_test_rmse = rmse_test
        #
        # if best_val_mae > rmse_test:
        #     # best_val_mae = mae_valid
        #     best_test_mae = mae_test
        #
        #
        # feed = {model.recall: recall, model.recall_10: recall_10, model.recall_50: recall_50,
        #         model.recall_100: recall_100, model.recall_200: recall_200,
        #         model.ndcg_5: ndcg_at_five, model.ndcg_10: ndcg_at_ten}
        # eval_metrics = sess.run([model.eval_metrics], feed_dict=feed)
        # test_writer.add_summary(eval_metrics[0], step)


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    b_init = tf.constant_initializer(0.)
    return tf.get_variable(name, shape, initializer=b_init)


class Model():
    def __init__(self, args, M, embed, train_filename, test_filename, reg_lambda=0.01, name='tr'):
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

        outputs, init_ops = get_input_dataset(train_filename, test_filename, batch_size=self.batch_size)
        self.u_idx, self.v_idx, self.r, self.input_text, self.seq_lengths = outputs
        self.v_idx = tf.reshape(self.v_idx, shape=[self.batch_size])
        self.train_init_op, self.validation_init_op = init_ops

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
                # with tf.name_scope('stddev'):
                #   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                # tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                # tf.summary.histogram('histogram', var)

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
        self.init_state_fw = cell_bw.zero_state(self.batch_size, tf.float32)
        self.init_state_bw = cell_fw.zero_state(self.batch_size, tf.float32)

        bi_outputs, bi_output_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=self.seq_lengths,
                                            initial_state_bw=self.init_state_bw,
                                            initial_state_fw=self.init_state_fw)
        bi_outputs = tf.concat(bi_outputs, 2)
        self.bi_output_state_fw, self.bi_output_state_bw = bi_output_state

        self.bi_output_state_fw = tf.identity(self.bi_output_state_fw, name='bi_state_fw')  # just to give it a name
        self.bi_output_state_bw = tf.identity(self.bi_output_state_bw, name='bi_state_bw')  # just to give it a name

        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells)
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        # bi_outputs = tf.stack(bi_outputs,1)
        self.Yr, self.H = tf.nn.dynamic_rnn(cell, bi_outputs, sequence_length=self.seq_lengths,
                                            initial_state=self.initial_state, dtype=tf.float32)
        # Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
        # H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence
        self.H = tf.identity(self.H, name='H')  # just to give it a name
        self.Yr = tf.identity(self.Yr, name='Yr')

        # RNN output layer:
        # avg pool layer [batch_size, embedding_dim]
        self.G = tf.reduce_mean(self.Yr, 1)

        # Update RNN output
        with tf.device("/cpu:0"):
            # RNN output [num_items, embedding_dim]
            self.RNN = tf.get_variable(shape=[self.m, self.k], name='RNN_output', trainable=False, dtype=tf.float32
                                       , initializer=tf.constant_initializer(0.))
            self.update_rnn_output = tf.scatter_update(self.RNN, self.v_idx, self.G)

        # # G matrix for current batch, [batch_size, embeddings_dim]
        self.G_embed = tf.nn.embedding_lookup(self.RNN, self.v_idx)

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

        self.F = tf.add(self.V_embed, self.G_embed)

        self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.F), reduction_indices=1)

        # self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
        self.r_hat = tf.add(self.r_hat, self.U_bias_embed)
        self.r_hat = tf.add(self.r_hat, self.V_bias_embed, name="R_predicted")

        self.RMSE = tf.sqrt(tf.losses.mean_squared_error(self.r, self.r_hat))
        self.l2_loss = tf.nn.l2_loss(tf.subtract(self.r, self.r_hat))
        self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.r, self.r_hat)))
        self.reg = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U)),
                          tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V)))
        self.reg_loss = tf.add(self.l2_loss, self.reg)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.train_step_u = self.optimizer.minimize(self.reg_loss, var_list=[self.U, self.U_bias])
        self.train_step_v = self.optimizer.minimize(self.reg_loss, var_list=[self.V, self.V_bias])

        t_vars = tf.trainable_variables()
        gru_vars = [var for var in t_vars if 'gru_cell' in var.name]
        self.train_step_rnn = self.optimizer.minimize(self.reg_loss, var_list=[gru_vars])

        tf.summary.scalar("RMSE", self.RMSE)
        tf.summary.scalar("MAE", self.MAE)
        tf.summary.scalar("L2-Loss", self.l2_loss)
        tf.summary.scalar("Reg-Loss", self.reg_loss)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()


def test_3():
    graph = tf.Graph()
    with graph.as_default():
        test_filename = '/home/wanli/data/Extended_ctr/dummy_train_0.tfrecords'
        # Creates a dataset that reads all of the examples from filenames.
        validation_dataset = tf.contrib.data.TFRecordDataset(test_filename)
        train_filename = '/home/wanli/data/Extended_ctr/dummy_test_1.tfrecords'
        training_dataset = tf.contrib.data.TFRecordDataset(train_filename)

        validation_dataset = validation_dataset.map(_parse_function)
        training_dataset = training_dataset.map(_parse_function)

        # A feedable iterator is defined by a handle placeholder and its structure. We
        # could use the `output_types` and `output_shapes` properties of either
        # `training_dataset` or `validation_dataset` here, because they have
        # identical structure.
        iterator = tf.contrib.data.Iterator.from_structure(training_dataset.output_types,
                                                           training_dataset.output_shapes)
        next_element = iterator.get_next()
        #
        # # You can use feedable iterators with a variety of different kinds of iterator
        # # (such as one-shot and initializable iterators).
        # training_iterator = training_dataset.make_one_shot_iterator()
        # validation_iterator = validation_dataset.make_initializable_iterator()

        training_init_op = iterator.make_initializer(training_dataset)
        validation_init_op = iterator.make_initializer(validation_dataset)

        u_idx_t, v_idx_t, r_t, input_t, lengths_t = next_element
        capacity = 1500
        batch_size = 128
        bucket_boundaries = [x for x in range(50, 500, 50)]
        seq_len, outputs_b = tf.contrib.training.bucket_by_sequence_length(
            lengths_t, tensors=[u_idx_t, v_idx_t, r_t, input_t, lengths_t],
            allow_smaller_final_batch=True, \
            batch_size=batch_size, bucket_boundaries=bucket_boundaries, \
            capacity=capacity, dynamic_pad=True)

        u_idx, v_idx, r, input_text, seq_lengths = outputs_b

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        # for op in tf.get_default_graph().get_operations():
        #     print (str(op.name))
        # print('--------------------')
        # for n in tf.get_default_graph().as_graph_def().node:
        #     print(n)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            for step in range(10):
                # Initialize the training dataset iterator
                tr_init = sess.run(training_init_op)
                sess.run(u_idx)
            print('Done')
        except Exception as e:
            print(e)
            # Report exceptions to the coordinator.
            coord.request_stop(e)
            print("Finished training")
        finally:
            coord.request_stop()
            coord.join(threads)


def test_2():
    graph = tf.Graph()
    with graph.as_default():
        model = Model(is_handle=True)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        # for op in tf.get_default_graph().get_operations():
        #     print (str(op.name))
        # print('--------------------')
        # for n in tf.get_default_graph().as_graph_def().node:
        #     print(n)
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        training_handle = sess.run(model.training_iterator.string_handle())
        validation_handle = sess.run(model.validation_iterator.string_handle())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        try:
            for step in range(2):
                # Initialize the training dataset iterator
                # tr_init = sess.run(model.tr)
                sess.run(model.u_idx, feed_dict={model.handle: training_handle})
                sess.run(model.u_idx, feed_dict={model.handle: validation_handle})
            print('Done')
        except Exception as e:
            # Report exceptions to the coordinator.
            print(e)
            coord.request_stop(e)
            print("Finished training")
        finally:
            coord.request_stop()
            coord.join(threads)


def test_1():
    graph = tf.Graph()
    with graph.as_default():
        model = Model()

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        # for op in tf.get_default_graph().get_operations():
        #     print (str(op.name))
        # print('--------------------')
        # for n in tf.get_default_graph().as_graph_def().node:
        #     print(n)


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            for step in range(10):
                # Initialize the training dataset
                # h = sess.partial_run_setup([model.training_init_op,model.u_idx])
                # sess.partial_run(h,model.training_init_op)
                # sess.partial_run(model.u_idx)

                print('Epoch {0}'.format(step))
                sess.run(model.training_init_op)
                sess.run(model.u_idx)

                sess.run(model.validation_init_op)
                sess.run(model.u_idx)

            print('Done')
        except Exception as e:
            print(e)
            # Report exceptions to the coordinator.
            coord.request_stop(e)
            print("Finished training")
        finally:
            coord.request_stop()
            coord.join(threads)


def test_4():
    is_handle = False
    batch_size = 10
    with tf.device("/cpu:0"):
        with tf.variable_scope('input'):
            test_filename = '/home/wanli/data/Extended_ctr/dummy_test_1.tfrecords'
            train_filename = '/home/wanli/data/Extended_ctr/dummy_train_1.tfrecords'
            example_count_train = utils.num_samples(train_filename)
            example_count_validation = utils.num_samples(test_filename)

            nb_batches_train = int(math.ceil(example_count_train / batch_size))

            print('Number of training batches {0}, number of samples {1}'.format(nb_batches_train, example_count_train))
            nb_batches_val = int(math.ceil(example_count_validation / batch_size))
            print('Number of validation batches {0}, number of samples {1}'.format(nb_batches_val,
                                                                                   example_count_validation))
            # Creates a dataset that reads all of the examples from filenames.
            validation_dataset = tf.contrib.data.TFRecordDataset(test_filename)
            training_dataset = tf.contrib.data.TFRecordDataset(train_filename)

            # validation_dataset = validation_dataset.repeat()
            # training_dataset = training_dataset.repeat()

            validation_dataset = validation_dataset.map(_parse_function)
            training_dataset = training_dataset.map(_parse_function)

            training_dataset = training_dataset.padded_batch(batch_size, padded_shapes=((), (), (), [None], ()))
            validation_dataset = validation_dataset.padded_batch(batch_size, padded_shapes=((), (), (), [None], ()))

            if not is_handle:
                # A reinitializable iterator is defined by its structure. We could use the
                # `output_types` and `output_shapes` properties of either `training_dataset`
                # or `validation_dataset` here, because they are compatible.
                iterator = tf.contrib.data.Iterator.from_structure(training_dataset.output_types,
                                                                   training_dataset.output_shapes)

                training_init_op = iterator.make_initializer(training_dataset)
                validation_init_op = iterator.make_initializer(validation_dataset)

            next_element = iterator.get_next()

            u_idx_t, v_idx_t, r_t, input_t, lengths_t = next_element

            n, m = 50, 1929
            confidence_matrix = np.ones((n, m))

            confidence = tf.get_variable(name="confidence", shape=[n, m],
                                         initializer=tf.constant_initializer(confidence_matrix), trainable=False)
            confidence_batch = tf.nn.embedding_lookup(confidence, ids=(u_idx_t, v_idx_t))

            confidence = tf.constant(confidence_matrix, dtype=tf.float32, shape=confidence_matrix.shape,
                                     name='confidence')
            u_v_idx = tf.stack([u_idx_t, v_idx_t], axis=1)
            c_g = tf.gather_nd(confidence, u_v_idx)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        try:
            for step in range(2):
                # Initialize the training dataset
                # h = sess.partial_run_setup([model.training_init_op,model.u_idx])
                # sess.partial_run(h,model.training_init_op)
                # sess.partial_run(model.u_idx)

                print('Epoch {0}'.format(step))
                print('Training .....................................')
                sess.run(training_init_op)
                for _ in range(nb_batches_train):
                    output = sess.run([u_idx_t, v_idx_t])
                    output = sess.run(c_g)
                    print(output)
                    # print(np.count_nonzero(np.asarray(output[:,0])))

                    # print('Validation .....................................')
                    # sess.run(validation_init_op)
                    # for _ in range(nb_batches_val):
                    #     input = sess.run(input_t)
                    #     print(input.shape)
            print('Done')
        except Exception as e:
            print(e)
            # Report exceptions to the coordinator.
        finally:
            print("Finshed training")


def test_5():
    with tf.device("/cpu:0"):
        with tf.variable_scope('input'):
            dataset = tf.contrib.data.Dataset.range(17)
            dataset = dataset.repeat()
            dataset = dataset.batch(5)
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        try:
            sess.run(iterator.initializer)
            for step in range(6):
                print('iteration {0}'.format(step))
                output = sess.run(next_element)
                print(output)
            print('Done')
        except Exception as e:
            print(e)
            # Report exceptions to the coordinator.
        finally:
            print("Finshed training")


def test_random_recommender(n, m, k, only_on_test_set, verbos=False):
    # np.random.seed(20)

    # ratings_matrix=np.random.randint(2,size=(n,m))
    ratings_matrix = rand(n, m, density=0.004)
    ratings_matrix.data[:] = 1

    ratings_matrix = ratings_matrix.toarray().astype(np.int32)
    if verbos:
        print('R_target')
        print(ratings_matrix, end='\n -------------------------------------\n')
    if only_on_test_set:
        num_rating = np.count_nonzero(ratings_matrix)
        idx = np.arange(num_rating)
        np.random.shuffle(idx)
        test_idx = idx[int(0.70 * num_rating):]
        nonzero_u_idx = ratings_matrix.nonzero()[0]
        nonzero_v_idx = ratings_matrix.nonzero()[1]
        u_idx = nonzero_u_idx[test_idx]
        v_idx = nonzero_v_idx[test_idx]
        test_ratings = np.zeros(shape=(n, m))
        test_ratings[u_idx, v_idx] = 1
    else:
        test_ratings = ratings_matrix.copy()
    # np.random.seed(40)

    U = np.random.randint(100, size=(n, k)) / 100
    # print(U,end='\n -------------------------------------\n')
    # np.random.seed(30)

    V = np.random.randint(100, size=(m, k)) / 100
    # print(V,end='\n -------------------------------------\n')
    # np.random.seed(21)

    rnn_output = np.random.randint(100, size=(m, k)) / 100
    # print(rnn_output,end='\n -------------------------------------\n')
    evaluator = Evaluator(ratings_matrix, verbose=True)

    prediction_matrix = np.matmul(U, np.add(V, rnn_output).T)
    sorted_prediction_matrix = np.argsort(prediction_matrix, axis=1, )

    # prediction_matrix = np.add(prediction_matrix, np.reshape(U_b, [-1, 1]))
    # prediction_matrix = np.add(prediction_matrix, V_b)
    rounded_predictions = utils.rounded_predictions(prediction_matrix)
    #
    n_top_recommednations = 200
    if only_on_test_set:
        evaluator.load_top_recommendations_2(n_top_recommednations, prediction_matrix, test_ratings)
    else:
        evaluator.new_load_top_recommendations(n_top_recommednations, prediction_matrix, test_ratings)

    recall_10 = evaluator.recall_at_x(10, prediction_matrix, ratings_matrix, rounded_predictions)
    recall_50 = evaluator.recall_at_x(50, prediction_matrix, ratings_matrix, rounded_predictions)
    # recall_100 = evaluator.recall_at_x(100, prediction_matrix, ratings_matrix, rounded_predictions)
    # recall_200 = evaluator.recall_at_x(200, prediction_matrix, ratings_matrix, rounded_predictions)
    recall = evaluator.calculate_recall(ratings=ratings_matrix, predictions=rounded_predictions)
    ndcg_at_five = evaluator.calculate_ndcg(5, rounded_predictions)
    print('ndcg@5 %f ' % ndcg_at_five)
    #
    ndcg_at_ten = evaluator.calculate_ndcg(10, rounded_predictions)
    print('ndcg@10 %f ' % ndcg_at_ten)

    ndcg_at_50 = evaluator.calculate_ndcg(50, rounded_predictions)
    print('ndcg@50 %f ' % ndcg_at_50)

    # print("         | Recall@10: {0:3.4f}".format(recall_10))
    # print("         | Recall@50: {0:3.4f}".format(recall_50))
    # print("         | Recall@100: {0:3.4f}".format(recall_100))
    # print("         | Recall@200: {0:3.4f}".format(recall_200))
    # print("         | Recall: {0:3.4f}".format(recall))
    mrr_at_ten = evaluator.calculate_mrr(10, rounded_predictions)
    print('mrr@10 %f ' % mrr_at_ten)
    if verbos:
        print('R_target')
        print(ratings_matrix, end='\n -------------------------------------\n')
        print('T')
        print(test_ratings, end='\n -------------------------------------\n')
        print('R_pred')
        print(prediction_matrix, end='\n -------------------------------------\n')
        print('R_pred sorted by item id')
        print(sorted_prediction_matrix, end='\n -------------------------------------\n')
        print('Rounded_predictions')
        print(rounded_predictions, end='\n -------------------------------------\n')


def test_metrics(verbos=False):
    np.random.seed(20)

    ratings_matrix = np.array([[1, 1, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 1, 0]])
    print('Ratings matrix:\n ', ratings_matrix, end='\n -------------------------------------\n')

    test_ratings = np.array([[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [1, 0, 0, 1, 0]])
    print('Test matrix:\n ', test_ratings, end='\n -------------------------------------\n')

    evaluator = Evaluator(ratings_matrix, verbose=True)
    prediction_matrix = np.array([[0.73, 0.03, 0.97, 0.47, 0.1], [0.21, 0.9, 0.65, 0.4, 0.51],
                                  [0.06, 0.82, 0.8, 0.3, 0.56]])
    print('Predction matrix:\n ', prediction_matrix, end='\n -------------------------------------\n')
    rounded_predictions = utils.rounded_predictions(prediction_matrix)
    print('Rounded Predction matrix:\n ', rounded_predictions, end='\n -------------------------------------\n')
    evaluator.load_top_recommendations_2(100, prediction_matrix, test_ratings)
    print('Top recomendation idx:\n', evaluator.recommendation_indices,
          end='\n -------------------------------------\n')

    sorted_prediction_matrix = np.argsort(prediction_matrix, axis=1, )

    recall_10 = evaluator.recall_at_x(10, prediction_matrix, ratings_matrix, rounded_predictions)
    recall_50 = evaluator.recall_at_x(50, prediction_matrix, ratings_matrix, rounded_predictions)
    recall_100 = evaluator.recall_at_x(100, prediction_matrix, ratings_matrix, rounded_predictions)
    recall_200 = evaluator.recall_at_x(200, prediction_matrix, ratings_matrix, rounded_predictions)
    recall = evaluator.calculate_recall(ratings=ratings_matrix, predictions=rounded_predictions)
    ndcg_at_five = evaluator.calculate_ndcg(2, rounded_predictions)
    print('ndcg@5 %f ' % ndcg_at_five)

    ndcg_at_ten = evaluator.calculate_ndcg(10, rounded_predictions)
    print('ndcg@10 %f ' % ndcg_at_ten)

    ndcg_at_50 = evaluator.calculate_ndcg(50, rounded_predictions)
    print('ndcg@50 %f ' % ndcg_at_50)

    # print("         | Recall@10: {0:3.4f}".format(recall_10))
    # print("         | Recall@50: {0:3.4f}".format(recall_50))
    # print("         | Recall@100: {0:3.4f}".format(recall_100))
    # print("         | Recall@200: {0:3.4f}".format(recall_200))
    # print("         | Recall: {0:3.4f}".format(recall))
    mrr_at_ten = evaluator.calculate_mrr(2, rounded_predictions)
    print('mrr@10 %f ' % mrr_at_ten)
    if verbos:
        print('R_target')
        print(ratings_matrix, end='\n -------------------------------------\n')
        print('T')
        print(test_ratings, end='\n -------------------------------------\n')
        print('R_pred')
        print(prediction_matrix, end='\n -------------------------------------\n')
        print('R_pred sorted by item id')
        print(sorted_prediction_matrix, end='\n -------------------------------------\n')
        print('Rounded_predictions')
        print(rounded_predictions, end='\n -------------------------------------\n')


# def dcg_at_k(scores):
#     assert scores
#     return scores[0] + sum(sc / log(ind, 2) for sc, ind in zip(scores[1:], range(2, len(scores)+1)))

def dcg_at_k(scores, method=1):
    assert scores
    if method == 0:
        return scores[0] + sum(sc / log(ind, 2) for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))
    if method == 1:
        return sum(sc / log(ind, 2) for sc, ind in zip(scores[:], range(2, len(scores) + 2)))


def ndcg_at_k(predicted_scores, user_scores):
    assert len(predicted_scores) == len(user_scores)
    idcg = dcg_at_k(sorted(user_scores, reverse=True))
    print('dcg {}, idcg {}'.format(dcg_at_k(predicted_scores), idcg))
    return (dcg_at_k(predicted_scores) / idcg) if idcg > 0.0 else 0.0


# class TestMetrics(unittest.TestCase):
#
#     def test_dcg_small(self):
#         scores = [3, 2]
#         self.assertAlmostEqual(dcg_at_k(scores), 5.0)
#
#
#     def test_dcg_large(self):
#         scores = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
#         self.assertAlmostEqual(dcg_at_k(scores), 9.6051177391888114)
#
#
#     def test_ndcg(self):
#         predicted1 = [.4, .1, .8]
#         predicted2 = [.0, .1, .4]
#         predicted3 = [.4, .1, .0]
#         actual = [.8, .4, .1, .0]
#         self.assertAlmostEqual(ndcg_at_k(predicted1, actual[:3]), 0.795, 3)
#         self.assertAlmostEqual(ndcg_at_k(predicted2, actual[:3]), 0.279, 3)
#         self.assertAlmostEqual(ndcg_at_k(predicted3, actual[:3]), 0.396, 3)

def test_tags_module():
    k = 2
    tags_matrix = np.random.randint(2, size=(4, 3))
    v_idx = [1, 2, 3]
    tags_matrix = tf.constant(tags_matrix, dtype=tf.int32, shape=tags_matrix.shape,
                              name='confidence')
    tags_actual = tf.nn.embedding_lookup(tags_matrix, v_idx)  # [batch_size, max_tags]
    embedding_var = tf.get_variable(name="embedding", shape=[tags_matrix.shape[1], k])
    tags_embeddings = tf.nn.embedding_lookup(embedding_var, tags_actual)  # [batch_size, max_tags, embeding_dim]

    f = np.random.randint(100, size=(len(v_idx), k)) / 100
    f = tf.constant(f, dtype=tf.float32, shape=f.shape)
    # f = tf.reshape(f,[tf.shape(f)[0],1,tf.shape(f)[1]])
    tags_probalities = tf.einsum('aij,aj->ai', tags_embeddings, f)
    #
    # # todo: add downweights for predicting the unobserved tags
    tags_loss = tf.losses.sigmoid_cross_entropy(tags_actual, tags_probalities)

    tags_actual = tf.to_float(tags_actual)
    tags_sigmoid = tf.nn.sigmoid(tags_probalities)
    cross_entropy = -tf.reduce_mean(
        ((tags_actual * tf.log(tags_sigmoid )) + ((1 - tags_actual) * tf.log(1 - tags_sigmoid ))),
        name='xentropy')

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    print(tags_probalities.eval())
    print(tags_actual.eval())
    loss_1 = sess.run(tags_loss)
    print(loss_1)
    loss_2= cross_entropy.eval()
    print(loss_2)
    np.testing.assert_almost_equal(loss_1, loss_2)

def test_tags_module_sparse():
    k = 2
    m = 4
    tags = 3, [[0,0],[1,1],[2,2],[2,0]]
    v_idx = [1, 2]

    def sparse_slice(indices, values, needed_row_ids):
        needed_row_ids = tf.reshape(needed_row_ids, [1, -1])
        num_rows = tf.shape(indices)[0]
        partitions = tf.cast(tf.reduce_any(tf.equal(tf.reshape(indices[:, 0], [-1, 1]), needed_row_ids), 1), tf.int32)
        rows_to_gather = tf.dynamic_partition(tf.range(num_rows), partitions, 2)[1]
        slice_indices = tf.gather(indices, rows_to_gather)
        slice_values = tf.gather(values, rows_to_gather)
        return slice_indices, slice_values

    with tf.Session().as_default():
        tags_matrix = tf.constant(np.random.randint(0,2,size=(16980,46391)), dtype=tf.int32,
                                  name='confidence')
        indices = tf.constant([[0, 0], [1, 0], [4, 0], [4, 1]],dtype=tf.int64)
        values = tf.constant([1.0, 1.0, 0.3, 0.7], dtype=tf.float32)
        batch_size = 2
        v_idx = tf.constant([0, 4],dtype=tf.int64)
        slice_indices, slice_values = sparse_slice(indices, values, v_idx)
        print(slice_indices.eval())
        print(slice_values.eval())
        tags_actual = tf.SparseTensor(slice_indices, slice_values, dense_shape=(tf.reduce_max(v_idx ),tags[0]))
        print(tags_actual.eval())

        embedding_var = tf.get_variable(name="embedding", shape=[tags[0], k])
        tf.global_variables_initializer().run()

        #item embeddings
        f = tf.constant(np.random.rand(batch_size,k), dtype=tf.float32)
        print(f.eval())
        tags_probalities = tf.matmul(f,embedding_var,transpose_b=True)
        print(tags_probalities.eval())

        loss =  tf.sparse_tensor_dense_matmul(tags_actual,tf.transpose(tags_probalities))
        print(loss.eval())
        tags_actual = tf.to_float(tags_actual)
        tags_sigmoid = tf.nn.sigmoid(tags_probalities)
        pos_prediction = tf.sparse_tensor_dense_matmul(tags_actual, tf.transpose(tf.log(tags_sigmoid)))
        print(pos_prediction.eval())
        # neg_prediction = tf.sparse_add(tags_actual,-1)
        neg_prediction = tf.transpose(tf.log(1 - tags_sigmoid)) - tf.sparse_tensor_dense_matmul(tags_actual, tf.transpose(tf.log(1 - tags_sigmoid)))
        cross_entropy = -tf.reduce_mean( pos_prediction + neg_prediction,name='xentropy')
        print(cross_entropy.eval())

    f = np.random.randint(100, size=(len(v_idx), k)) / 100
    f = tf.constant(f, dtype=tf.float32, shape=f.shape)
    tags_probalities = tf.einsum('aij,aj->ai', tags_embeddings, f)

    # todo: add downweights for predicting the unobserved tags

    tags_loss = tf.losses.sigmoid_cross_entropy(tags_actual, tags_probalities, )
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    print(tags_probalities.eval())
    print(tags_actual.eval())
    loss_1 = sess.run(tags_loss)
    print(loss_1)

def test_attributes_module():
    # Implementation of a simple MLP network with one hidden layer.
    features_matrix = np.random.randint(2, size=(20, 10))
    x_size = features_matrix.shape[1]

    # features_matrix = tf.constant(features_matrix,shape=features_matrix.shape)
    v_idx = [1, 2, 3]

    features_matrix = tf.constant(features_matrix, dtype=tf.float32, shape=features_matrix.shape,
                                  name="attributes_matrix")

    # Attribute features vector
    input_att = tf.nn.embedding_lookup(features_matrix, v_idx)
    input_att = tf.Print(input_att, [tf.shape(input_att), input_att],
                         message='Attributes', first_n=20, summarize=4)

    # Network Parameters
    # calculate the number of hidden units for each hidden layer
    # N_h = N_s / (alpha * (N_i + N_o))
    # N_i  = number of input neurons.
    # N_o = number of output neurons.
    # N_s = number of samples in training data set.
    # alpha = an arbitrary scaling factor usually 2-10.
    alpha = 5
    training_samples_count = 340000
    k = 200
    n_layers = 2

    n_hidden_1 = int(training_samples_count / (alpha * (x_size + k)))  # 1st layer number of neurons
    n_hidden_2 = int(training_samples_count / ((alpha + 1) * (x_size + k)))
    n_hidden_3 = int(training_samples_count / ((alpha + 2) * (x_size + k)))  # 1st layer number of neurons
    y_size = k

    with tf.variable_scope('Attributes_component_%d-layers' % (n_layers)):

        # Input layer, User side
        with tf.name_scope('U_input_layer'):
            w_1 = weight_variable([x_size, n_hidden_1], 'W_1')
            b_1 = bias_variable(n_hidden_1, 'B_1')
            h_1 = tf.nn.relu(tf.add(tf.matmul(input_att, w_1), b_1))

        # Hidden layers
        for n in range(2, n_layers + 1):
            if n == 2:
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
            if n_layers > 2:
                n_hidden_prev = n_hidden_3
            else:
                n_hidden_prev = n_hidden_2
            w_U_out = weight_variable([n_hidden_prev, y_size], 'W_out')
            b_U_out = bias_variable(y_size, 'B_out')
            attribute_output = tf.nn.relu(tf.add(tf.matmul(h_h, w_U_out), b_U_out), 'Attributes_output')

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    print(attribute_output.eval())
    print(tf.shape(attribute_output).eval())

    return attribute_output


def test_parse_tags():
    dataset_folder = '/home/wanli/data/Extended_ctr/citeulike_a_extended/'
    tags_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset_folder,
                        'tags.dat')

    with open(tags_file, "r", encoding='utf-8', errors='ignore') as f:
        content = f.readlines()
        tags = {}
        for t in content:
            t = t.strip()
            if t in tags:
                tags[t] += 1
            else:
                tags[t] = 1
    tags_count = len(tags)
    print('Tags vocaulary size %d' %len(tags))
    item_tags_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset_folder,
                        'item-tag.dat')
    paper_count = 16980
    tags_matrix = np.zeros((paper_count, tags_count))
    with open(item_tags_file, "r", encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter=' ')
        i =0
        for line in reader:
            count = line[0]
            # if int(count) == 0:
            #     print(i)
            for j in range(1,int(count)+1):
                tags_matrix[i][int(line[j])] = 1
            i += 1
    return tags_matrix


# def replace_row_idx(row_idx):
#     '''
#     It produces a list of increasing indices.
#     input: [0,3,5,6,6,7]
#     output: [0,1,2,3,3,4]
#
#     :return:
#     '''
#     i = 0
#     prev = 0
#     first_element = True
#     new_row_idx =[]
#     row_idx = tf.unstack(row_idx)
#     for j in row_idx:
#         if first_element:
#             new_row_idx.append(i)
#             prev = j
#             first_element = False
#             continue
#         if j == prev:
#             new_row_idx.append(i)
#         else:
#             i += 1
#             new_row_idx.append(i)
#             prev = j
#     return new_row_idx
#
# a = replace_row_idx([0,3,5,6,6,7])

def test_tf_scatter_update():
    import tensorflow as tf

    g = tf.Graph()
    with g.as_default():
        a = tf.Variable(initial_value=[[0, 0, 0, 0], [0, 0, 0, 0]])
        v = tf.constant([0,1])
        u = tf.constant([0,1])
        u_v_idx = tf.stack([u, v], axis=1)
        b = tf.scatter_nd_update(a, u_v_idx, [5,3])

        # data = tf.Variable([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [1, 2, 3, 4, 5]])
        # row = tf.gather(data, 2)
        # new_row = tf.concat([row[:2], tf.constant([0]), row[3:]], axis=0)
        # sparse_update = tf.scatter_update(data, tf.constant(2), new_row)

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a ))
        print('----------')
        print(sess.run(b))
        print('----------')

        print(sess.run(u_v_idx))
        # print (sess.run(data))
        # print('----------')
        # print (sess.run(row))
        # print('----------')
        #
        # print(sess.run(new_row))
        #
        # print('----------')
        # print(sess.run(sparse_update))

def test_loss():
    sess = tf.InteractiveSession()
    u = tf.constant(np.arange(1,6),dtype=tf.float32,shape=[2,3])
    f = tf.constant(np.arange(3,8),dtype=tf.float32,shape=[2,3])
    tf.global_variables_initializer().run()
    print(u.eval())
    print(f.eval())
    r_hat = tf.reduce_sum(tf.multiply(u, f), reduction_indices=1)
    print(tf.multiply(u, f).eval())
    print(r_hat.eval())
    confidence = tf.constant([1,1],dtype=tf.float32,shape=[2])
    r = tf.constant([28,100],dtype=tf.float32,shape=[2])
    MSE = tf.losses.mean_squared_error(r, r_hat,weights=confidence)
    RMSE = tf.sqrt(MSE)
    l2_loss = tf.nn.l2_loss(tf.multiply(confidence,tf.subtract(r, r_hat)))
    print(MSE.eval())
    print(RMSE.eval())
    print(l2_loss.eval())

def fn():
    return 1,2,3,4

def main():
    # batch_size = 1
    capacity = 1500
    # next_batch= get_input_test(filename,batch_size)

    # test_filename='/home/wanli/data/Extended_ctr/dummy_test_0.tfrecords'
    # # Creates a dataset that reads all of the examples from filenames.
    # test_dataset = tf.contrib.data.TFRecordDataset(test_filename)
    # train_filename='/home/wanli/data/Extended_ctr/dummy_test_1.tfrecords'
    # training_dataset=tf.contrib.data.TFRecordDataset(train_filename)
    #
    # test_dataset= test_dataset.repeat()
    # training_dataset =training_dataset.repeat()
    #
    # test_dataset = test_dataset.map(_parse_function)
    # training_dataset = training_dataset.map(_parse_function)
    #
    #
    # # A reinitializable iterator is defined by its structure. We could use the
    # # `output_types` and `output_shapes` properties of either `training_dataset`
    # # or `validation_dataset` here, because they are compatible.
    # iterator = tf.contrib.data.Iterator.from_structure(training_dataset.output_types,
    #                                    training_dataset.output_shapes)
    # next_element = iterator.get_next()
    #
    # training_init_op = iterator.make_initializer(training_dataset)
    # validation_init_op = iterator.make_initializer(test_dataset)
    #
    #
    #
    #
    # u_idx_t,  v_idx_t, r_t, input_t, lengths_t = next_element
    # batch_size = 128
    # bucket_boundaries = [x for x in range(50, 500, 50)]
    # seq_len, outputs_b = tf.contrib.training.bucket_by_sequence_length(
    #     lengths_t, tensors=[u_idx_t, v_idx_t, r_t, input_t, lengths_t],
    #     allow_smaller_final_batch=True, \
    #     batch_size=batch_size, bucket_boundaries=bucket_boundaries, \
    #     capacity=capacity, dynamic_pad=True)
    # u_idx,v_idx,r,input_text,seq_lengths = outputs_b
    # # v_idx = tf.squeeze(outputs_b[1], [1], name="V_matrix")
    # # r = tf.squeeze(outputs_b[2], [1], name='R_target')
    # # input_text = tf.squeeze(outputs_b[3], [1], name="Input_text")
    # # seq_lengths = tf.squeeze(outputs_b[4], [1], name="seq_lengths")
    # # u_idx, v_idx, r, input_text, seq_lengths
    # train_writer = tf.summary.FileWriter('logs/test_eval')
    # with tf.Session() as sess:
    #     tf.global_variables_initializer().run()
    #     tf.local_variables_initializer().run()
    #     train_writer.add_graph(sess.graph)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess, coord)
    #     try:
    #         # Loop forever, alternating between training and validation.
    #
    #         for step in range(20):
    #             # Run 200 steps using the training dataset. Note that the training dataset is
    #             # infinite, and we resume from where we left off in the previous `while` loop
    #             # iteration.
    #             print ('epoch {0}'.format(step))
    #             init = sess.run(training_init_op)
    #             for i in range(50):
    #                 u = sess.run(u_idx)
    #                 print('Train {0}'.format(i))
    #             # Run one pass over the validation dataset.
    #             sess.run(validation_init_op)
    #             for i in range(4):
    #                 u = sess.run(u_idx)
    #                 print('test {0}'.format(i))
    #
    #         # for i in range(2):
    #         #     u = sess.run(u_idx)
    #         #     print (u)
    #         print('Done')
    #     except Exception as e:
    #         # Report exceptions to the coordinator.
    #         coord.request_stop(e)
    #         print("Finished training")
    #     finally:
    #         coord.request_stop()
    #         coord.join(threads)
    # test_4()


    # print('=============================================\n only test set')

    # for i in range(10):
    #     print("iteration: %d" %i)
    # print('=============================================\n Include train and test sets')
    # test_top_recommendations(n,m,k,False,verbos)
    #
    # n, m, k = 20, 16980, 200
    # verbos = False
    # print("\nRandom recommender")
    # test_random_recommender(n, m, k, True, verbos)
    #
    # print("\nMetrics test")
    # test_metrics(verbos)
    #
    # print("\nndcg test ")
    # actual = [[1,1],[1,1],[1,1]] # subset of a rating matrix
    # print('Actual')
    # print(actual)
    # r = [[0,0], [1, 0], [0, 0]] # subset of a rounded predations matrix
    # print('Rounded predictions')
    # print(r)
    # print(np.mean([ndcg_at_k(x, y) for x,y in zip(r,actual)]))

    # test_tags_module()
    # test_parse_tags()
    # test_tags_module_sparse()
    # test_attributes_module()

    # test_tf_scatter_update()

    # score_file ='/home/wanli/data/Extended_ctr/citeulike_a_extended/in-matrix-item_folds/fold-1/score.npy'
    # score = np.load(score_file)
    # a = 1

    # # b = np.array(np.random.rand(2,3),dtype=np.float64)
    # b = [[ 0.76327468e-10,  0.36366175,  0.3757063 ],[ 0.5597608,   0.77586819,  0.35938623]]
    # np.save('b',b)
    # print(b)
    #
    # b= np.load('b.npy')
    #
    # print(b)

    # test_loss()

    b = fn()

if __name__ == '__main__':
    main()
