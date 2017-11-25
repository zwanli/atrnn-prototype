import time
import tensorflow as tf
import numpy as np
import argparse
import os, errno
from evaluator import Evaluator
import pickle
from data_parser import DataParser
import utils
from model import Model
from utils import convert_to_tfrecords
from utils import convert_tripltes_to_tfrecords
from tensorflow.python import debug as tf_debug
import math
from deep_mf_model import DMF_Model
from utils import static_padding

# from Recommender_evaluator.lib import evaluator

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/wanli/data/Extended_ctr',
                        help='data directory containing input.txt')
    parser.add_argument("--dataset", "-d", type=str, default='citeulike-a',
                        help="Which dataset to use", choices=['dummy', 'citeulike-a', 'citeulike-t'])
    parser.add_argument('--embedding_dir', type=str, default='/home/wanli/data/cbow_w2v/',
                        help='GloVe embedding directory containing embeddings file')
    parser.add_argument('--embedding_dim', type=int, default=200,
                        help='dimension of the embeddings', choices=['50', '100', '200', '300'])
    parser.add_argument('--input_encoding', type=str, default=None,
                        help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')

    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--fold', type=int, default=1, help='Data fold to be used for training')
    parser.add_argument('--split', type=str, default='cold', help='The splitting strategy', choices=['warm', 'cold'])

    parser.add_argument('--example_structure', '-e', type=str, default='triplets',
                        help="Which structre of the training samples to use, single:"
                             " (u_id, v_id, r,...) or triplets (u_id,pos_id,neg_id,...)",
                        choices=['triplets', 'single'])
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')

    parser.add_argument('--confidence_mode', type=str, default='user-dependant',
                        help='Choose confidence mode', choices=['user-dependant', 'constant', 'only-positive'])

    parser.add_argument('--use_rnn', action='store_true',
                        help='Learn documents embeddings')
    parser.add_argument('--rnn_size', type=int, default=200,
                        help='size of RNN hidden state')
    parser.add_argument('--model', type=str, default='gru',
                        help='Choose the RNN cell type', choices=['rnn, gru, or lstm'])

    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--max_length', type=int, default=300,
                        help='Maximum document length')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')

    parser.add_argument('--use_att', action='store_true',
                        help='Learn attribute embeddings')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--summation', action='store_true',
                        help='Sum the attribute embeddings and the rnn output')
    parser.add_argument('--fc_layer', action='store_true',
                        help='Add a FC layer to get the joint output of the rnn and attributes embeddings')

    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--multi_task', action='store_true',
                        help='Multi-task learning')
    parser.add_argument('--mt_lambda', type=float, default=0.6,
                        help='Multi-task hyperparamter labmda')
    parser.add_argument('--gpu_mem', type=float, default=0.666,
                        help='%% of gpu memory to be allocated to this process. Default is 66.6%%')
    parser.add_argument('--save', action='store_true',
                        help='Save the model graph')
    # parser.add_argument('--decay_rate', type=float, default=0.97,
    #                     help='decay rate for rmsprop')
    # parser.add_argument('--grad_clip', type=float, default=5.,
    #                     help='clip gradients at this value')

    # parser.add_argument('--init_from', type=str, default=None,
    #                     help="""continue training from saved model at this path. Path must contain files saved by previous training process:
    #                         'config.pkl'        : configuration;
    #                         'words_vocab.pkl'   : vocabulary definitions;
    #                         'checkpoint'        : paths to model file(s) (created by tf).
    #                                               Note: this file contains absolute paths, be careful when moving files around;
    #                         'model.ckpt-*'      : file(s) with model definition (created by tf)
    #                     """)
    args = parser.parse_args()
    train(args)
    # partial_run(args)
    # train_dmf(args)


def load_abstracts(parser, dataset_folder, dataset):
    abstracts_word_idx_filename = os.path.join(dataset_folder, '{0}-abstracts_word_idx.pkl'.format(dataset))
    if os.path.exists(abstracts_word_idx_filename):
        print('Loading abstracts')
        with open(abstracts_word_idx_filename, 'rb') as f:
            parser.all_documents = pickle.load(f)
    else:
        parser.get_papar_as_word_ids()
        with open(abstracts_word_idx_filename, 'wb') as f:
            pickle.dump(parser.all_documents, f, pickle.HIGHEST_PROTOCOL)
            print("Saved abstracts")
    # delete raw data, save memory
    parser.del_raw_data()


def num_samples(path):
    c = 0
    for record in tf.python_io.tf_record_iterator(path):
        example_proto = tf.train.SequenceExample()
        example_proto.ParseFromString(record)
        c += 1
    return c


def process_input(args, parser):
    # Read text input
    # parser = DataParser(args.data_dir, args.dataset, None,
    #                     use_embeddings=True, embed_dir=args.embedding_dir, embed_dim=args.embedding_dim)

    if args.dataset == 'citeulike-a':
        dataset_folder = args.data_dir + '/citeulike_a_extended'
    elif args.dataset == 'citeulike-t':
        dataset_folder = args.data_dir + '/citeulike_t_extended'
    elif args.dataset == 'dummy':
        dataset_folder = args.data_dir + '/dummy'
    else:
        print("Warning: Given dataset not known, setting to dummy")
        dataset_folder = args.data_dir + '/citeulike_a_extended'

    # A dict that has the paths of all the training/test files for all folds
    folds_paths = {}

    split_folder = os.path.join(dataset_folder, '{0}'.format(
        'in-matrix-item_folds' if args.split == 'warm' else 'outof-matrix-item_folds'))
    if not os.path.exists(split_folder):
        raise

    parser.load_embeddings()
    # load abstracts
    load_abstracts(parser, dataset_folder, args.dataset)

    for fold in range(args.folds):
        train_folder = os.path.join(split_folder, 'fold-{}'.format(fold + 1),
                                    '{0}'.format('train'))
        try:
            os.makedirs(train_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        test_folder = os.path.join(split_folder, 'fold-{}'.format(fold + 1), '{0}'.format('test'))
        try:
            os.makedirs(test_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


        train_file = os.path.join(train_folder, args.dataset + '_train_{0}_triplets.tfrecords'.format(fold + 1))
        if os.path.exists(train_file):
            print("File already exists {0}".format(train_file))
        else:
            if not parser.train_ratings:
                # load_abstracts(parser, dataset_folder, args.dataset)
                # if args.split == 'warm':
                #     parser.split_warm_start_item(args.folds)
                # elif args.split == 'cold':
                #     parser.split_cold_start(args.folds)
                parser.load_folds(split_folder)
            convert_tripltes_to_tfrecords(train_file, parser, fold)

        test_file = os.path.join(test_folder, args.dataset + '_test_{0}.tfrecords'.format(fold + 1))
        if os.path.exists(test_file):
            print("File already exists {0}".format(test_file))
        else:
            if not parser.test_ratings:
                # load_abstracts(parser, dataset_folder, args.dataset)
                # parser.split_cold_start(5)
                # if args.split == 'warm':
                #     parser.split_warm_start_item(args.folds)
                # elif args.split == 'cold':
                #     parser.split_cold_start(args.folds)
                parser.load_folds(split_folder)
            convert_to_tfrecords(test_file, parser, fold, args.max_length, args.split, test=True)
        # Add the train and test files' paths
        folds_paths[fold + 1] = (train_file, test_file)

    sample_count = {}
    for fold in range(1, args.folds + 1):
        if not args.dataset == 'citeulike-a':
            count_tr = num_samples(folds_paths[fold][0])
            # Path of the training fold
            print('Total number of train samples in fold {0}: {1}'.format(fold, count_tr))
            count_test = num_samples(folds_paths[fold][1])
            # Path of the testing fold
            print('Total number of test  samples in fold {0}: {1}'.format(fold, count_test))
            sample_count[fold] = (count_tr, count_test)
        else:
            sample_count[fold] = load_samples_count(fold, args.split)
            print('Total number of train samples in fold {0}: {1}'.format(fold, load_samples_count(fold, args.split)[
                0]))
            print('Total number of test  samples in fold {0}: {1}'.format(fold, load_samples_count(fold, args.split)[
                1]))

    print('Finished parsing the input')
    return folds_paths, sample_count


def load_samples_count(fold, split_mode, negative_samples=False):
    if split_mode == 'warm' and negative_samples:
        return {
            1: (341976, 33998),
            2: (341976, 33998),
            3: (341976, 33998),
            4: (341976, 33998),
            5: (281454, 64259)
        }[fold]
    if split_mode == 'warm':
        return {
            1: (170988, 33998),
            2: (170988, 33998),
            3: (170988, 33998),
            4: (170988, 33998),
            5: (140727, 64259)
        }[fold]

    if split_mode == 'cold' and negative_samples:
        return {
            1: (163159 * 2, 41827),
            2: (163924 * 2, 41062),
            3: (163805 * 2, 41181),
            4: (164616 * 2, 40370),
            5: (164440 * 2, 40546)
        }[fold]
    if split_mode == 'cold':
        return {
            1: (163159, 41827),
            2: (163924, 41062),
            3: (163805, 41181),
            4: (164616, 40370),
            5: (164440, 40546)
        }[fold]


def train(args):
    # initialize the parser
    parser = DataParser(args.data_dir, args.dataset, 'attributes',
                        use_embeddings=True, embed_dir=args.embedding_dir, embed_dim=args.embedding_dim)

    # read input data
    print('--Processing input')
    dataset_path, dataset_count = process_input(args, parser)
    args.vocab_size = parser.get_vocab_size()
    confidence_mode = args.confidence_mode

    # parser.get_confidence_matrix(mode='constant',alpha=1 , beta=0.01)
    # parser.get_confidence_matrix(mode='only-positive')
    print('--Confidence mode: %s ' % confidence_mode)
    if confidence_mode == 'constant':
        confidence_matrix = parser.get_confidence_matrix(mode=confidence_mode, alpha=1, beta=0.01)
    else:
        confidence_matrix = parser.get_confidence_matrix(mode=confidence_mode)
    # parser.get_confidence_matrix()



    print('Vocabulary size {0}'.format(parser.words_count))

    best_val_rmse = np.inf
    best_val_mae = np.inf
    best_test_rmse = 0
    best_test_mae = 0

    evaluator = Evaluator(parser.get_ratings_matrix(), verbose=True)

    def construct_feed(bi_hid_fw, bi_hid_bw, dropout_0, dropout_1, dropout_2):
        return {model.init_state_fw: bi_hid_fw, model.init_state_bw: bi_hid_bw,
                model.dropout_embed_layer: dropout_0, model.dropout_bidir_layer: dropout_1,
                model.dropout_second_layer: dropout_2}
        # model.initial_state: hid_state,

    # TF configurations
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'

    n_steps = args.num_epochs
    multi_task = args.multi_task

    # The data fold that will be used for training
    fold = args.fold

    dir_prefix = '{0}-{1}-{2}-{3}-f{4}'.format(time.strftime("%d:%m-%H:%M:"), args.dataset, args.split,
                                               args.max_length, fold)
    # Checkpoints directory
    ckpt_dir = os.path.join(args.log_dir, 'checkpoints/{0}-train'.format(dir_prefix))

    path_training = dataset_path[fold][0]
    path_test = dataset_path[fold][1]

    train_sample_count = dataset_count[fold][0]
    test_sample_count = dataset_count[fold][1]

    nb_batches_train = int(math.ceil(train_sample_count / args.batch_size))
    print('Number of training batches {0}, number of samples {1}'.format(nb_batches_train, train_sample_count))
    nb_batches_val = int(math.ceil(test_sample_count / args.batch_size))
    print(
        'Number of test batches {0}, number of samples {1}'.format(nb_batches_val, test_sample_count))

    graph = tf.Graph()
    with graph.as_default():
        args.training_samples_count = train_sample_count
        model = Model(args, parser.get_ratings_matrix(), parser.get_feature_matrix(),
                      parser.get_tag_count(), confidence_matrix, path_training, path_test)

    with tf.Session(config=config, graph=graph) as sess:
        train_writer = tf.summary.FileWriter(args.log_dir + '/{0}-train'.format(dir_prefix))
        # valid_writer = tf.summary.FileWriter(args.log_dir + '/{0}-validation'.format(time.strftime(dir_prefix)))
        test_writer = tf.summary.FileWriter(args.log_dir + '/{0}-test'.format(time.strftime(dir_prefix)))
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        print('Saving graph to disk...')
        # train_writer.add_graph(sess.graph)
        # valid_writer.add_graph(sess.graph)
        # test_writer.add_graph(sess.graph)
        dropout_second_layer = 0.3
        dropout_bidir_layer = 0.5
        dropout_embed_layer = 0.1

        feed_dict = {model.abstracts_matrix_init: static_padding(parser.all_documents, maxlen=args.max_length,
                                                                 num_papers=parser.paper_count),
                     'RNN/embeddings_init:0': parser.embeddings}
        if multi_task:
            # Load the tag matrix while initializing graph variables.
            feed_dict[model.tags_matrix_init] = parser.get_tag_matrix()
            sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
        else:
            # # free some ram
            # del parser.tag_matrix
            sess.run(tf.global_variables_initializer(),feed_dict=feed_dict)
        # free some ram
        del parser.tag_matrix
        tf.local_variables_initializer().run()

        # print("Loading test ratings matrix")
        # test_ratings = utils.get_test_ratings_matrix(path_test, parser.user_count, parser.paper_count, sess)
        if args.use_rnn:
            bi_state_fw = sess.run(model.init_state_bw)
            bi_state_bw = sess.run(model.init_state_fw)
            h_state = sess.run(model.initial_state)

        try:
            for step in range(n_steps):
                # for step in range(1):
                print('{0}: Epoch {1}'.format(time.strftime("%d:%m-%H:%M:"), step))
                print('Training .....................................')

                # Initialize the training dataset iterator
                sess.run(model.train_init_op)

                if args.use_rnn:
                    feed_dict = construct_feed(bi_state_fw, bi_state_bw,
                                               dropout_embed_layer, dropout_bidir_layer, dropout_second_layer)
                else:
                    feed_dict = {}

                fetches = [model.joint_train_step,
                           # model.u_idx, model.v_idx,
                           # model.U, model.V, model.doc_embed, model.U_bias, model.V_bias, model.att_output,
                           model.RMSE, model.summary_op, model.inc_batch_pointer_op
                           ]
                if args.use_rnn:
                    fetches.extend(
                        [model.bi_output_state_fw, model.bi_output_state_bw, model.H, model.update_doc_embed])
                if args.use_att:
                    fetches.extend([model.update_att_embed, model.update_doc_att_embed])

                start = time.time()

                for batch in range(nb_batches_train):
                    # for batch in range(1):
                    # u_idx, v_idx, U, V, rnn, U_b, V_b, att_ouput, \
                    fetched = \
                        sess.run(fetches, feed_dict=feed_dict)

                    if args.use_rnn and args.use_att:
                        _, rmse, summary_str, _, bi_out_fw, bi_out_bw, final_state, doc_embed, att_embed, joint_embed = fetched
                    elif args.use_rnn:
                        _, rmse, summary_str, _, bi_out_fw, bi_out_bw, final_state, doc_embed = fetched
                    else:
                        _, rmse, summary_str, _, = fetched

                    # print every 500 iteration
                    if batch // 10 % 50 == 0:
                        print("Epoch {0}, batch {1}".format(step, batch))
                    train_writer.add_summary(summary_str, global_step=(step * nb_batches_train + batch))
                end = time.time()
                print('Epoch {0}, finished in {1}'.format(step, end - start))

                # # todo: check the condition
                # if step // 1 % 5 == 0:
                #     print('{0}:Validation ............'.format(time.strftime("%d:%m-%H:%M:")))
                #     # # save a checkpoint (every 5 epochs)
                #     # if step // 1 % 5 == 0 and step > 4:
                #     #     saved_file = model.saver.save(sess, ckpt_dir, global_step=step)
                #     #     print("Saved file: " + saved_file)
                #     if True or args.split == 'cold':
                #         # Initialize the validation dataset iterator
                #         sess.run(model.validation_init_op)
                #         if args.use_rnn:
                #             test_bi_fw = sess.run(model.init_state_fw)
                #             test_bi_bw = sess.run(model.init_state_bw)
                #             init_state = sess.run(model.initial_state)
                #             # don't dropout
                #             feed_dict = construct_feed(test_bi_fw, test_bi_bw, 0, 0, 0)
                #         else:
                #             feed_dict = {}
                #         for batch in range(2):
                #             # for batch in range(nb_batches_val):
                #             rmse_test, summary_str = sess.run(
                #                 [model.RMSE, model.summary_op], feed_dict=feed_dict)
                #             test_writer.add_summary(summary_str, global_step=(step * nb_batches_val + batch))
                #
                #         print("Step {0} | Train RMSE: {1:3.4f}".format(
                #             step, rmse))
                #         print("         | Test  RMSE: {0:3.4f}".format(
                #             rmse_test))
                #         if best_val_rmse > rmse_test:
                #             # best_val_rmse = rmse_valid
                #             best_test_rmse = rmse_test

                            # prediction_matrix = np.matmul(U, np.add(V, rnn_output).T)
                            # prediction_matrix = np.add(prediction_matrix, np.reshape(U_b, [-1, 1]))
                            # prediction_matrix = np.add(prediction_matrix, V_b)

                            # rounded_predictions = utils.rounded_predictions(prediction_matrix)
                            # evaluator.load_top_recommendations_2(200, prediction_matrix, test_ratings)
                            # recall_10 = evaluator.recall_at_x(10, prediction_matrix, parser.ratings, rounded_predictions)
                            # recall_50 = evaluator.recall_at_x(50, prediction_matrix, parser.ratings, rounded_predictions)
                            # recall_100 = evaluator.recall_at_x(100, prediction_matrix, parser.ratings, rounded_predictions)
                            # recall_200 = evaluator.recall_at_x(200, prediction_matrix, parser.ratings, rounded_predictions)
                            # recall = evaluator.calculate_recall(ratings=parser.ratings, predictions=rounded_predictions)
                            # ndcg_at_five = evaluator.calculate_ndcg(5, rounded_predictions)
                            # ndcg_at_ten = evaluator.calculate_ndcg(10, rounded_predictions)
                            # mrr_at_ten = evaluator.calculate_mrr(10, rounded_predictions)
                            #
                            # feed = {model.recall: recall, model.recall_10: recall_10, model.recall_50: recall_50,
                            #         model.recall_100: recall_100, model.recall_200: recall_200,
                            #         model.ndcg_5: ndcg_at_five, model.ndcg_10: ndcg_at_ten, model.mrr_10: mrr_at_ten}
                            # eval_metrics = sess.run([model.eval_metrics], feed_dict=feed)
                            # test_writer.add_summary(eval_metrics[0], step)
                            #
                            # print("         | Recall@10: {0:3.4f}".format(recall_10))
                            # print("         | Recall@50: {0:3.4f}".format(recall_50))
                            # print("         | Recall@100: {0:3.4f}".format(recall_100))
                            # print("         | Recall@200: {0:3.4f}".format(recall_200))
                            # print("         | Recall: {0:3.4f}".format(recall))
                            # print("         | ndcg@5: {0:3.4f}".format(ndcg_at_five))
                            # print("         | ndcg@10: {0:3.4f}".format(ndcg_at_ten))
                            # print("         | mrr@10: {0:3.4f}".format(mrr_at_ten))

                if args.use_rnn:
                    # loop rnn state around
                    h_state = final_state
                    bi_state_fw = bi_out_fw
                    bi_state_bw = bi_out_bw

            # PREDICTION:
            # In case of out-of-matrix:
            # Calculate the item embeddings by using the rnn and/or the attributes output
            print("--Calculating the predication matrix ")
            if args.use_rnn or args.use_att:
                if args.use_rnn:
                    bi_fw = bi_state_fw
                    bi_bw = bi_state_bw
                    # don't dropout
                    feed_dict = construct_feed(bi_fw, bi_bw, 0, 0, 0)
                else:
                    feed_dict = {}

                # Initialize the training dataset iterator
                sess.run(model.train_init_op)

                # for batch in range(1):
                for batch in range(nb_batches_train):
                    if args.use_rnn and args.use_att:
                        _ = sess.run(
                            [model.update_doc_att_embed], feed_dict=feed_dict)
                    elif args.use_rnn:
                        _ = sess.run(
                            [model.update_doc_embed], feed_dict=feed_dict)
                    elif args.use_att:
                        _ = sess.run(
                            [model.update_att_embed], feed_dict=feed_dict)

                # # Initialize the test dataset iterator
                # sess.run(model.validation_init_op)

                # # for batch in range(1):
                # for batch in range(nb_batches_val):
                #     if args.use_rnn and args.use_att:
                #         _ = sess.run(
                #             [model.update_doc_att_embed], feed_dict=feed_dict)
                #     elif args.use_rnn:
                #         _ = sess.run(
                #             [model.update_doc_embed], feed_dict=feed_dict)
                #     elif args.use_att:
                #         _ = sess.run(
                #             [model.update_att_embed], feed_dict=feed_dict)

            prediction_matrix = sess.run(model.get_prediction_matrix)
            fold_dir = os.path.dirname(os.path.dirname(path_training))
            predicted_ratings_file = os.path.join(fold_dir, 'score.npy')
            np.save(predicted_ratings_file, prediction_matrix)
            print("Predication matrix saved to {}".format(predicted_ratings_file))

            if args.save:
                model.saver.save(sess, fold_dir + '/{0}-model.ckpt'.format(time.strftime(dir_prefix)))
            with open(os.path.join(fold_dir, '{0}-config.pkl'.format(time.strftime(dir_prefix))), 'wb') as f:
                pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)
            # print('Best test rmse:', best_test_rmse, 'Best test mae', best_test_mae, sep=' ')
            train_writer.close()
            test_writer.close()
        except Exception as e:
            print(e)
            print("Finished training")


def train_dmf(args):
    parser = DataParser(args.data_dir, args.dataset, 'attributes',
                        use_embeddings=True, embed_dir=args.embedding_dir, embed_dim=args.embedding_dim)
    # read input data
    dataset_path, dataset_count = process_input(args, parser)
    args.vocab_size = parser.get_vocab_size()
    confidence_mode = 'user-dependant'
    # parser.get_confidence_matrix(mode='constant',alpha=1 , beta=0.01)
    # parser.get_confidence_matrix(mode='only-positive')
    confidence_matrix = parser.get_confidence_matrix(mode=confidence_mode)
    print('Confidence mode %s ' % confidence_mode)
    # parser.get_confidence_matrix()

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

    for fold in range(args.folds):
        path_training = dataset_path[fold][0]
        path_test = dataset_path[fold][1]

        example_count_train = dataset_count[fold][0]
        example_count_validation = dataset_count[fold][1]

        nb_batches_train = int(math.ceil(example_count_train / args.batch_size))
        print('Number of training batches {0}, number of samples {1}'.format(nb_batches_train,
                                                                             example_count_train))
        nb_batches_val = int(math.ceil(example_count_validation / args.batch_size))
        print(
            'Number of validation batches {0}, number of samples {1}'.format(nb_batches_val,
                                                                             example_count_validation))
    # calcate the size of the last batch, it might be smaller than the default batch_size
    last_batch_size = example_count_train % args.batch_size

    print('Vocabulary size {0}'.format(parser.words_count))

    dir_prefix = '{0}-{1}-{2}-{3}'.format(time.strftime("%d:%m-%H:%M:"), args.dataset, args.split,
                                          args.max_length)
    # Checkpoints directory
    ckpt_dir = os.path.join(args.log_dir, 'checkpoints/{0}-train'.format(dir_prefix))

    best_val_rmse = np.inf
    best_val_mae = np.inf
    best_test_rmse = 0
    best_test_mae = 0

    evaluator = Evaluator(parser.get_ratings_matrix(), verbose=True)

    graph = tf.Graph()
    with graph.as_default():
        #####TEST AREA ###############
        args.training_samples_count = example_count_train
        args.mf_num_layers = 2
        model = DMF_Model(args, parser.get_raw_ratings_matrix(), parser.embeddings, confidence_matrix,
                          path_training, path_test)
        train_writer = tf.summary.FileWriter(args.log_dir + '/{0}-train'.format(dir_prefix),
                                             graph=tf.get_default_graph())
        a = 0

        #####TEST AREA ###############
        # model = Model(args, parser.get_ratings_matrix(), parser.embeddings, confidence_matrix, path_training,
        #               path_test)
        train_writer = tf.summary.FileWriter(args.log_dir + '/{0}-train'.format(dir_prefix))
        valid_writer = tf.summary.FileWriter(args.log_dir + '/{0}-validation'.format(time.strftime(dir_prefix)))
        test_writer = tf.summary.FileWriter(args.log_dir + '/{0}-test'.format(time.strftime(dir_prefix)))

    def construct_feed(bi_hid_fw, bi_hid_bw, dropout_0, dropout_1, dropout_2):
        return {model.init_state_fw: bi_hid_fw, model.init_state_bw: bi_hid_bw,
                model.dropout_embed_layer: dropout_0, model.dropout_bidir_layer: dropout_1,
                model.dropout_second_layer: dropout_2}
        # model.initial_state: hid_state,

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'

    n_steps = args.num_epochs
    with tf.Session(config=config, graph=graph) as sess:

        print('Saving graph to disk...')
        # train_writer.add_graph(sess.graph)
        # valid_writer.add_graph(sess.graph)
        # test_writer.add_graph(sess.graph)
        dropout_second_layer = 0.2
        dropout_bidir_layer = 0.1
        dropout_embed_layer = 0.1

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        print("Loading test ratings matrix")
        test_ratings = utils.get_test_ratings_matrix(path_test, parser.user_count, parser.paper_count, sess)
        #
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        bi_state_fw = sess.run(model.init_state_bw)
        bi_state_bw = sess.run(model.init_state_fw)
        h_state = sess.run(model.initial_state)

        try:
            # for step in range(n_steps):
            for step in range(n_steps):
                print('{0}: Epoch {1}'.format(time.strftime("%d:%m-%H:%M:"), step))
                print('Training .....................................')
                # Initialize the training dataset iterator
                sess.run(model.train_init_op)
                feed_dict = construct_feed(bi_state_fw, bi_state_bw,
                                           dropout_embed_layer, dropout_bidir_layer, dropout_second_layer)

                fetches = [model.train_op,
                           model.update_rnn_output,
                           # model.U, model.V, model.RNN, model.U_bias, model.V_bias,
                           model.bi_output_state_fw, model.bi_output_state_bw, model.H,
                           model.RMSE, model.summary_op, model.inc_batch_pointer_op]
                start = time.time()

                for batch in range(nb_batches_train):
                    # for batch in range(2):
                    _, _, bi_out_fw, bi_out_bw, final_state, rmse, summary_str, _ = \
                        sess.run(fetches, feed_dict=feed_dict)
                    # print every 500 iteration
                    if batch // 10 % 50 == 0:
                        print("Epoch {0}, batch {1}".format(step, batch))
                    train_writer.add_summary(summary_str, global_step=(step * nb_batches_train + batch))
                end = time.time()
                print('Epoch {0}, finished in {1}'.format(step, end - start))
                if True:  # or step // 1 % 5 == 0:
                    print('{0}:Validation ............'.format(time.strftime("%d:%m-%H:%M:")))

                    # save a checkpoint (every 5 epochs)
                    if False and step // 1 % 5 == 0 and step > 4:
                        saved_file = model.saver.save(sess, ckpt_dir, global_step=step)
                        print("Saved file: " + saved_file)

                    # Initialize the validation dataset iterator
                    sess.run(model.validation_init_op)
                    test_bi_fw = sess.run(model.init_state_fw)
                    test_bi_bw = sess.run(model.init_state_bw)
                    init_state = sess.run(model.initial_state)
                    # don't dropout
                    feed_dict = construct_feed(test_bi_fw, test_bi_bw, 0, 0, 0)
                    for batch in range(nb_batches_val):
                        rmse_test, mae_test, summary_str = sess.run(
                            [model.RMSE, model.cross_entropy, model.summary_op], feed_dict=feed_dict)
                        test_writer.add_summary(summary_str, global_step=(step * nb_batches_val + batch))
                        #
                        # prediction_matrix = np.matmul(U, np.add(V, rnn_output).T)
                        # prediction_matrix = np.add(prediction_matrix, np.reshape(U_b, [-1, 1]))
                        # prediction_matrix = np.add(prediction_matrix, V_b)
                        # rounded_predictions = utils.rounded_predictions(prediction_matrix)
                        # evaluator.load_top_recommendations_2(200, prediction_matrix, test_ratings)
                        # recall_10 = evaluator.recall_at_x(10, prediction_matrix, parser.ratings,
                        #                                   rounded_predictions)
                        # recall_50 = evaluator.recall_at_x(50, prediction_matrix, parser.ratings,
                        #                                   rounded_predictions)
                        # recall_100 = evaluator.recall_at_x(100, prediction_matrix, parser.ratings,
                        #                                    rounded_predictions)
                        # recall_200 = evaluator.recall_at_x(200, prediction_matrix, parser.ratings,
                        #                                    rounded_predictions)
                        # recall = evaluator.calculate_recall(ratings=parser.ratings, predictions=rounded_predictions)
                        # ndcg_at_five = evaluator.calculate_ndcg(5, rounded_predictions)
                        # ndcg_at_ten = evaluator.calculate_ndcg(10, rounded_predictions)
                        #
                        # mrr_at_ten = evaluator.calculate_mrr(10, rounded_predictions)
                        #
                        # feed = {model.recall: recall, model.recall_10: recall_10, model.recall_50: recall_50,
                        #         model.recall_100: recall_100, model.recall_200: recall_200,
                        #         model.ndcg_5: ndcg_at_five, model.ndcg_10: ndcg_at_ten, model.mrr_10: mrr_at_ten}
                        # eval_metrics = sess.run([model.eval_metrics], feed_dict=feed)
                        # test_writer.add_summary(eval_metrics[0], step)
                        #
                        # print("Step {0} | Train RMSE: {1:3.4f}, MAE: {2:3.4f}".format(
                        #     step, rmse, mae))
                        # print("         | Test  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
                        #     rmse_test, mae_test))
                        # print("         | Recall@10: {0:3.4f}".format(recall_10))
                        # print("         | Recall@50: {0:3.4f}".format(recall_50))
                        # print("         | Recall@100: {0:3.4f}".format(recall_100))
                        # print("         | Recall@200: {0:3.4f}".format(recall_200))
                        # print("         | Recall: {0:3.4f}".format(recall))
                        # print("         | ndcg@5: {0:3.4f}".format(ndcg_at_five))
                        # print("         | ndcg@10: {0:3.4f}".format(ndcg_at_ten))
                        # print("         | mrr@10: {0:3.4f}".format(mrr_at_ten))
                        # if best_val_rmse > rmse_test:
                        #     # best_val_rmse = rmse_valid
                        #     best_test_rmse = rmse_test
                        #
                        # if best_val_mae > rmse_test:
                        #     # best_val_mae = mae_valid
                        #     best_test_mae = mae_test

                # loop state around
                h_state = final_state
                bi_state_fw = bi_out_fw
                bi_state_bw = bi_out_bw
            model.saver.save(sess, args.log_dir + "/{0}model.ckpt".format(time.strftime(dir_prefix)))
            print('Best test rmse:', best_test_rmse, 'Best test mae', best_test_mae, sep=' ')
            train_writer.close()
            valid_writer.close()
            test_writer.close()
        except Exception as e:
            # Report exceptions to the coordinator.
            print(e)
            # coord.request_stop(e)
            print("Finished training")


#
# def partial_run(args):
#     parser = DataParser(args.data_dir, args.dataset, 'attributes',
#                         use_embeddings=True, embed_dir=args.embedding_dir, embed_dim=args.embedding_dim)
#     # read input data
#     dataset_path, dataset_count = input(args, parser)
#     args.vocab_size = parser.get_vocab_size()
#
#     for fold in range(args.folds):
#         path_training = dataset_path[fold][0]
#         path_test = dataset_path[fold][1]
#
#         example_count_train = dataset_count[fold][0]
#         example_count_validation = dataset_count[fold][1]
#
#         nb_batches_train = int(math.ceil(example_count_train / args.batch_size))
#         print('Number of training batches {0}, number of samples {1}'.format(nb_batches_train, example_count_train))
#         nb_batches_val = int(math.ceil(example_count_validation / args.batch_size))
#         print(
#             'Number of validation batches {0}, number of samples {1}'.format(nb_batches_val, example_count_validation))
#     # calcate the size of the last batch, it might be smaller than the default batch_size
#     last_batch_size = example_count_train % args.batch_size
#
#     print('Vocabolary size {0}'.format(parser.words_count))
#     print("Uknown words {0}".format(parser.unkown_words_count))
#     print("Uknown numbers {0}".format(parser.numbers_count))
#
#     dir_prefix = '{0}-{1}-{2}-{3}'.format(time.strftime("%d:%m-%H:%M:"),args.dataset,args.split,args.max_length)
#
#     # Checkpoints directory
#     ckpt_dir = os.path.join(args.log_dir, 'checkpoints/{0}-train'.format(dir_prefix))
#
#     best_val_rmse = np.inf
#     best_val_mae = np.inf
#     best_test_rmse = 0
#     best_test_mae = 0
#
#     evaluator = Evaluator(parser.get_ratings_matrix(), verbose=True)
#
#     graph = tf.Graph()
#     with graph.as_default():
#         model = Model(args, parser.get_ratings_matrix(), parser.embeddings, path_training, path_test)
#         train_writer = tf.summary.FileWriter(args.log_dir + '/{0}-train'.format(dir_prefix))
#         valid_writer = tf.summary.FileWriter(args.log_dir + '/{0}-validation'.format(time.strftime(dir_prefix)))
#         test_writer = tf.summary.FileWriter(args.log_dir + '/{0}-test'.format(time.strftime(dir_prefix)))
#
#     def construct_feed(bi_hid_fw, bi_hid_bw):
#         return {model.init_state_fw: bi_hid_fw, model.init_state_bw: bi_hid_bw}
#         # model.initial_state: hid_state,
#
#     config = tf.ConfigProto()
#     config.gpu_options.allocator_type = 'BFC'
#
#     n_steps = args.num_epochs
#     with tf.Session(config=config, graph=graph) as sess:
#         print('Saving graph to disk...')
#         # train_writer.add_graph(sess.graph)
#         # valid_writer.add_graph(sess.graph)
#         # test_writer.add_graph(sess.graph)
#         tf.global_variables_initializer().run()
#         tf.local_variables_initializer().run()
#
#         print("Loading test ratings matrix")
#         test_ratings = utils.get_test_ratings_matrix(path_test, parser.user_count, parser.paper_count, sess)
#
#
#
#         bi_state_fw = sess.run(model.init_state_bw)
#         bi_state_bw = sess.run(model.init_state_fw)
#         h_state = sess.run(model.initial_state)
#
#         try:
#             # for step in range(n_steps):
#             for step in range(n_steps):
#                 print('{0}: Epoch {1}'.format(time.strftime("%d:%m-%H:%M:"), step))
#                 print('Training .....................................')
#                 # Initialize the training dataset iterator
#                 sess.run(model.train_init_op)
#                 feed_dict = construct_feed(bi_state_fw, bi_state_bw)
#
#                 # add dummy steps so tf won't complain and give the error
#                 # partial_run() requires empty target_list.
#                 #please refer to this issue:
#                 # https://github.com/tensorflow/tensorflow/issues/1899
#                 with tf.control_dependencies([model.train_step_rnn]):
#                     dummy_train_step_rnn = tf.constant(0)
#                 with tf.control_dependencies([model.train_step_v]):
#                     dummy_train_step_v = tf.constant(0)
#                 with tf.control_dependencies([model.train_step_u]):
#                     dummy_train_step_u = tf.constant(0)
#
#                 feeds = [model.init_state_fw, model.init_state_bw]
#                 fetches=[dummy_train_step_rnn,dummy_train_step_u,dummy_train_step_v,
#                         # model.train_step_v,model.train_step_u,model.train_step_rnn,
#                          model.update_rnn_output,
#                          model.U, model.V, model.RNN, model.U_bias, model.V_bias,
#                          model.bi_output_state_fw, model.bi_output_state_bw, model.H,
#                          model.RMSE, model.MAE, model.summary_op]
#                 start = time.time()
#
#                 for batch in range(nb_batches_train):
#                     handle = sess.partial_run_setup(fetches, feeds)
#                     sess.partial_run(handle, dummy_train_step_rnn, feed_dict=feed_dict)
#                     sess.partial_run(handle,dummy_train_step_v)
#                     _, _, U, V, rnn_output, U_b, V_b, bi_out_fw, bi_out_bw, final_state, rmse, mae, summary_str = sess.partial_run(
#                         handle,
#                         [dummy_train_step_u, model.update_rnn_output,
#                          model.U, model.V, model.RNN, model.U_bias, model.V_bias,
#                          model.bi_output_state_fw, model.bi_output_state_bw, model.H,
#                          model.RMSE, model.MAE, model.summary_op])
#                     # print every 500 iteration
#                     if batch // 10 % 50 == 0:
#                     # if True:
#                         print("Epoch {0}, batch {1}".format(step, batch))
#                     train_writer.add_summary(summary_str, global_step=(step*nb_batches_train + batch))
#                 end = time.time()
#                 print('Epoch {0}, finished in {1}'.format(step,end - start))
#                 if True:# or step // 1 % 5 == 0:
#                     print('{0}:Validation ............'.format(time.strftime("%d:%m-%H:%M:")))
#
#                     # save a checkpoint (every 5 epochs)
#                     if step // 1 % 5 == 0 and step > 4:
#                         saved_file = model.saver.save(sess, ckpt_dir, global_step=step)
#                         print("Saved file: " + saved_file)
#
#                     # Initialize the validation dataset iterator
#                     sess.run(model.validation_init_op)
#                     test_bi_fw = sess.run(model.init_state_fw)
#                     test_bi_bw = sess.run(model.init_state_bw)
#                     init_state = sess.run(model.initial_state)
#                     feed_dict = construct_feed(test_bi_fw, test_bi_bw)
#                     for batch in range(nb_batches_val):
#                         rmse_test, mae_test, summary_str = sess.run(
#                             [model.RMSE, model.MAE, model.summary_op], feed_dict=feed_dict)
#                         test_writer.add_summary(summary_str, global_step=(step*nb_batches_val + batch))
#
#
#                     prediction_matrix = np.matmul(U, np.add(V, rnn_output).T)
#                     prediction_matrix = np.add(prediction_matrix, np.reshape(U_b, [-1, 1]))
#                     prediction_matrix = np.add(prediction_matrix, V_b)
#                     rounded_predictions = utils.rounded_predictions(prediction_matrix)
#                     evaluator.load_top_recommendations_2(200, prediction_matrix, test_ratings)
#                     recall_10 = evaluator.recall_at_x(10, prediction_matrix, parser.ratings, rounded_predictions)
#                     recall_50 = evaluator.recall_at_x(50, prediction_matrix, parser.ratings, rounded_predictions)
#                     recall_100 = evaluator.recall_at_x(100, prediction_matrix, parser.ratings, rounded_predictions)
#                     recall_200 = evaluator.recall_at_x(200, prediction_matrix, parser.ratings, rounded_predictions)
#                     recall = evaluator.calculate_recall(ratings=parser.ratings, predictions=rounded_predictions)
#                     ndcg_at_five = evaluator.calculate_ndcg(5, rounded_predictions)
#                     ndcg_at_ten = evaluator.calculate_ndcg(10, rounded_predictions)
#
#                     mrr_at_ten = evaluator.calculate_mrr(10, rounded_predictions)
#
#                     feed = {model.recall: recall, model.recall_10: recall_10, model.recall_50: recall_50,
#                             model.recall_100: recall_100, model.recall_200: recall_200,
#                             model.ndcg_5: ndcg_at_five, model.ndcg_10: ndcg_at_ten, model.mrr_10: mrr_at_ten}
#                     eval_metrics = sess.run([model.eval_metrics], feed_dict=feed)
#                     test_writer.add_summary(eval_metrics[0], step)
#
#                     print("Step {0} | Train RMSE: {1:3.4f}, MAE: {2:3.4f}".format(
#                         step, rmse, mae))
#                     print("         | Test  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
#                         rmse_test, mae_test))
#                     print("         | Recall@10: {0:3.4f}".format(recall_10))
#                     print("         | Recall@50: {0:3.4f}".format(recall_50))
#                     print("         | Recall@100: {0:3.4f}".format(recall_100))
#                     print("         | Recall@200: {0:3.4f}".format(recall_200))
#                     print("         | Recall: {0:3.4f}".format(recall))
#                     print("         | ndcg@5: {0:3.4f}".format(ndcg_at_five))
#                     print("         | ndcg@10: {0:3.4f}".format(ndcg_at_ten))
#                     print("         | mrr@10: {0:3.4f}".format(mrr_at_ten))
#                     if best_val_rmse > rmse_test:
#                         # best_val_rmse = rmse_valid
#                         best_test_rmse = rmse_test
#
#                     if best_val_mae > rmse_test:
#                         # best_val_mae = mae_valid
#                         best_test_mae = mae_test
#
#                 # loop state around
#                 h_state = final_state
#                 bi_state_fw = bi_out_fw
#                 bi_state_bw = bi_out_bw
#             model.saver.save(sess, args.log_dir + "/{0}model.ckpt".format(time.strftime(dir_prefix)))
#             print('Best test rmse:', best_test_rmse, 'Best test mae', best_test_mae, sep=' ')
#             train_writer.close()
#             valid_writer.close()
#             test_writer.close()
#         except Exception as e:
#             # Report exceptions to the coordinator.
#             print(e)
#             # coord.request_stop(e)
#             print("Finished training")
#

if __name__ == '__main__':
    main()
