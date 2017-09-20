import numpy as np
import tensorflow as tf
import math
from evaluator import Evaluator
from model import get_inputs
from model import  Model
from model import get_input_test
from model import _parse_function
from model import get_input_dataset
from tensorflow.contrib import rnn
import time
import utils
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


def evaluate(sess,filename,ckpt_dir,rating_matrix,args,embeddings,test_writer,uv_matrices=None):
    def construct_feed(bi_hid_fw, bi_hid_bw):
        return {model.init_state_fw: bi_hid_fw, model.init_state_bw: bi_hid_bw}

    evaluator = Evaluator(rating_matrix,verbose=True)

    with tf.Graph().as_default() as g:
        saver = tf.train.import_meta_graph(ckpt_dir)
        model = Model(args,rating_matrix,embeddings,filename,test=True)
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

        u_idx, v_idx, r, input_text,  seq_lengths = outputs_b

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
                sess.run(model.u_idx,feed_dict={model.handle: training_handle})
                sess.run(model.u_idx,feed_dict={model.handle: validation_handle})
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

                print ('Epoch {0}'.format(step))
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
    is_handle =False
    batch_size = 10
    with tf.device("/cpu:0"):
        with tf.variable_scope('input'):
            test_filename = '/home/wanli/data/Extended_ctr/dummy_test_1.tfrecords'
            train_filename = '/home/wanli/data/Extended_ctr/dummy_train_1.tfrecords'
            example_count_train = utils.num_samples(train_filename)
            example_count_validation= utils.num_samples(test_filename)

            nb_batches_train = int(math.ceil(example_count_train / batch_size))

            print('Number of training batches {0}, number of samples {1}'.format(nb_batches_train,example_count_train))
            nb_batches_val = int(math.ceil(example_count_validation / batch_size))
            print('Number of validation batches {0}, number of samples {1}'.format(nb_batches_val,example_count_validation))
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

            n,m = 50,1929
            confidence_matrix = np.ones((n,m))

            confidence = tf.get_variable(name="confidence", shape=[n,m],
                                         initializer=tf.constant_initializer(confidence_matrix), trainable=False)
            confidence_batch = tf.nn.embedding_lookup(confidence, ids=(u_idx_t, v_idx_t))

            confidence = tf.constant(confidence_matrix, dtype=tf.float32, shape=confidence_matrix.shape,
                                     name='confidence')
            u_v_idx = tf.stack([u_idx_t, v_idx_t], axis=1)
            c_g = tf.gather_nd(confidence,u_v_idx)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        try:
            for step in range(2):
                # Initialize the training dataset
                # h = sess.partial_run_setup([model.training_init_op,model.u_idx])
                # sess.partial_run(h,model.training_init_op)
                # sess.partial_run(model.u_idx)

                print ('Epoch {0}'.format(step))
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

def test_top_recommendations():
    ratings_matrix=np.random.randint(2,size=(3,6))
    print(ratings_matrix,end='\n -------------------------------------\n')
    test_ratings=np.random.randint(2,size=(3,6))
    test_ratings[:, 5]=0
    test_ratings[:, 3]=0
    print(test_ratings,end='\n -------------------------------------\n')
    U=np.random.randint(100, size=(3,2)) / 1000
    # print(U,end='\n -------------------------------------\n')

    V=np.random.randint(100, size=(6,2)) / 1000
    # print(V,end='\n -------------------------------------\n')
    rnn_output = np.random.randint(100, size=(6,2)) / 1000
    # print(rnn_output,end='\n -------------------------------------\n')
    evaluator = Evaluator(ratings_matrix, verbose=True)

    prediction_matrix = np.matmul(U, np.add(V, rnn_output).T)
    print(prediction_matrix,end='\n -------------------------------------\n')

    # prediction_matrix = np.add(prediction_matrix, np.reshape(U_b, [-1, 1]))
    # prediction_matrix = np.add(prediction_matrix, V_b)
    rounded_predictions = utils.rounded_predictions(prediction_matrix)
    print(rounded_predictions,end='\n -------------------------------------\n')

    evaluator.load_top_recommendations_2(2, prediction_matrix, test_ratings)
    recall_10 = evaluator.recall_at_x(10, prediction_matrix,ratings_matrix, rounded_predictions)
    recall_50 = evaluator.recall_at_x(50, prediction_matrix, ratings_matrix, rounded_predictions)
    recall_100 = evaluator.recall_at_x(100, prediction_matrix, ratings_matrix, rounded_predictions)
    recall_200 = evaluator.recall_at_x(200, prediction_matrix, ratings_matrix, rounded_predictions)
    recall = evaluator.calculate_recall(ratings=ratings_matrix, predictions=rounded_predictions)
    ndcg_at_five = evaluator.calculate_ndcg(5, rounded_predictions)
    ndcg_at_ten = evaluator.calculate_ndcg(10, rounded_predictions)

    mrr_at_ten = evaluator.calculate_mrr(10, rounded_predictions)
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
    test_top_recommendations()

if __name__ == '__main__':
    main()
