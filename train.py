import time
import tensorflow as tf
import numpy as np
import argparse
import os,errno
from evaluator import Evaluator
import pickle
from data_parser import DataParser
import utils
from model import Model
from model import Model_test
from utils import convert_to_tfrecords
from eval import evaluate
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/wanli/data/Extended_ctr',
                        help='data directory containing input.txt')
    parser.add_argument("--dataset", "-d", type=str, default='dummy',
                        help="Which dataset to use", choices=['dummy', 'citeulike-a', 'citeulike-t'])
    parser.add_argument('--embedding_dir', type=str, default='/home/wanli/data/glove.6B/',
                        help='GloVe embedding directory containing embeddings file')
    parser.add_argument('--embedding_dim', type=int, default=200,
                        help='dimension of the embeddings', choices=['50', '100', '200', '300'])
    parser.add_argument('--input_encoding', type=str, default=None,
                        help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--split',type=str,default='cold', help='The splitting strategy',choices=['warm','cold'])
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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--max_length', type=int, default=300,
                        help='Maximum document length')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--gpu_mem', type=float, default=0.666,
                        help='%% of gpu memory to be allocated to this process. Default is 66.6%%')
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
    # train(args)
    partial_run(args)


def load_abstracts(parser,dataset_folder):
    abstracts_word_idx_filename = os.path.join(dataset_folder,'abstracts_word_idx.pkl')
    if os.path.exists(abstracts_word_idx_filename):
        print('Loading abstracts')
        with open(abstracts_word_idx_filename, 'rb') as f:
            parser.all_documents = pickle.load(f)
    else:
        parser.get_papar_as_word_ids()
        with open(abstracts_word_idx_filename,'wb') as f:
            pickle.dump(parser.all_documents, f, pickle.HIGHEST_PROTOCOL)
            print("Saved abstracts")
    #delete raw data, save memory
    parser.del_raw_data()


def num_samples(path):
    c = 0
    for record in tf.python_io.tf_record_iterator(path):
        example_proto = tf.train.SequenceExample()
        example_proto.ParseFromString(record)
        c += 1
    return c

def load_embeddings(parser):
    embeddings_tfrecord = os.path.join(parser.dataset_folder,
                                       '{0}_{1}_embeddings.tfrecord'.format(parser.dataset, parser.embed_dim))
    if os.path.exists(embeddings_tfrecord):
        # load embeddings
        # parser.load_embeddings(filename=embeddings_tfrecord)
        parser.load_glove_embeddings()

    else:
        # load glove embeddings
        parser.load_glove_embeddings()
        if not parser.all_documents:
            load_abstracts(parser, parser.dataset_folder)
        parser.save_embeddings(embeddings_tfrecord)
        # del parser.embeddings
        # del parser.embed_vocab
        # del parser.embed_word_to_id
        # parser.load_embeddings(embeddings_tfrecord)

def input(args,parser):
    # Read text input
    # parser = DataParser(args.data_dir, args.dataset, None,
    #                     use_embeddings=True, embed_dir=args.embedding_dir, embed_dim=args.embedding_dim)


    if args.dataset == 'citeulike-a':
        dataset_folder = args.data_dir + '/citeulike_a_extended'
    elif args.dataset == 'citeulike-t':
        dataset_folder = args.data_dir + '/citeulike-t_extended'
    elif args.dataset == 'dummy':
        dataset_folder = args.data_dir + '/dummy'
    else:
        print("Warning: Given dataset not known, setting to dummy")
        dataset_folder = args.data_dir + '/dummy'

    #A dict that has the paths of all the training/test files for all folds
    all_folds_datasets ={}

    train_folder = os.path.join(dataset_folder,'{0}-{1}'.format('warm' if args.split == 'warm' else 'cold','train'))
    try:
        os.makedirs(train_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    test_folder = os.path.join(dataset_folder,'{0}-{1}'.format('warm' if args.split == 'warm' else 'cold','test'))
    try:
        os.makedirs(test_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    for fold in range(args.folds):
        train_file = os.path.join(train_folder,args.dataset + '_train_{0}.tfrecords'.format(fold))
        if os.path.exists(train_file):
            print("File already exists {0}".format(train_file))
        else:
            if not parser.train_ratings:
                load_abstracts(parser, dataset_folder)
                if args.split == 'warm':
                    parser.split_warm_start_item(args.folds)
                elif args.split == 'cold':
                    parser.split_cold_start(args.folds)
            convert_to_tfrecords(train_file, parser, fold, args.max_length)

        test_file = os.path.join(test_folder,args.dataset + '_test_{0}.tfrecords'.format(fold))
        if os.path.exists(test_file):
            print("File already exists {0}".format(test_file))
        else:
            if not parser.test_ratings:
                load_abstracts(parser, dataset_folder)
                # parser.split_cold_start(5)
                if args.split == 'warm':
                    parser.split_warm_start_item(args.folds)
                elif args.split == 'cold':
                    parser.split_cold_start(args.folds)
            convert_to_tfrecords(test_file, parser, fold, args.max_length, test=True)
        #Add the train and test files' paths
        all_folds_datasets[fold]=(train_file,test_file)

    sample_count ={}
    for fold in range(args.folds):
        count_tr = num_samples(all_folds_datasets[fold][0])
        # Path of the training fold
        print('Total number of train samples in fold {0}: {1}'.format(fold, count_tr))

        count_test = num_samples(all_folds_datasets[fold][1])
        # Path of the testing fold
        print('Total number of test  samples in fold {0}: {1}'.format(fold,count_test ))

        sample_count[fold]=(count_tr,count_test)

    load_embeddings(parser)

    print('Finished parsing the input')
    return all_folds_datasets,sample_count




def train(args):
    # #Read text input
    parser = DataParser(args.data_dir,args.dataset,'attributes',
                        use_embeddings=True,embed_dir=args.embedding_dir,embed_dim=args.embedding_dim)
    dataset_path,dataset_count = input(args,parser)

    for fold in range(args.folds):

        path_training = dataset_path[fold][0]
        path_test = dataset_path[fold][1]

        example_count_train = dataset_count[fold][0]
        example_count_validation = dataset_count[fold][1]

        nb_batches_train = int(math.ceil(example_count_train / args.batch_size))
        print('Number of training batches {0}, number of samples {1}'.format(nb_batches_train, example_count_train))
        nb_batches_val = int(math.ceil(example_count_validation / args.batch_size))
        print('Number of validation batches {0}, number of samples {1}'.format(nb_batches_val, example_count_validation))

    print('Vocabolary size {0}'.format(parser.words_count))
    print("Uknown words {0}".format(parser.unkown_words_count))
    print("Uknown numbers {0}".format(parser.numbers_count))

    dir_prefix = time.strftime("%d:%m-%H:%M:")
    # Checkpoints directory
    ckpt_dir = os.path.join(args.log_dir, 'checkpoints/{0}-train'.format(dir_prefix))

    best_val_rmse = np.inf
    best_val_mae = np.inf
    best_test_rmse = 0
    best_test_mae = 0

    evaluator = Evaluator(parser.get_ratings_matrix(), verbose=True)

    graph = tf.Graph()
    with graph.as_default():
        model = Model(args, parser.get_ratings_matrix(),parser.embeddings,path_training,path_test)
        train_writer = tf.summary.FileWriter(args.log_dir+ '/{0}-train'.format(dir_prefix))
        valid_writer = tf.summary.FileWriter(args.log_dir + '/{0}-validation'.format(time.strftime(dir_prefix)))
        test_writer = tf.summary.FileWriter(args.log_dir + '/{0}-test'.format(time.strftime(dir_prefix)))


    def construct_feed(bi_hid_fw, bi_hid_bw):
        return {model.init_state_fw: bi_hid_fw, model.init_state_bw: bi_hid_bw}
                # model.initial_state: hid_state,

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'

    n_steps = args.num_epochs
    with tf.Session(config=config,graph=graph) as sess:
        print('Saving graph to disk...')
        train_writer.add_graph(sess.graph)
        # valid_writer.add_graph(sess.graph)
        # test_writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        print("Getting test ratings matrix")
        test_ratings = utils.get_test_ratings_matrix(path_test,parser.user_count,parser.paper_count,sess)

        bi_state_fw = sess.run(model.init_state_bw)
        bi_state_bw = sess.run(model.init_state_fw)
        h_state = sess.run(model.initial_state)

        # for op in tf.get_default_graph().get_operations():
        #     print (str(op.name))
        # print('--------------------')
        # for n in tf.get_default_graph().as_graph_def().node:
        #     print(n.name)
        try:
            for step in range(n_steps):
                print('{0}: Epoch {1}'.format(time.strftime("%d:%m-%H:%M:"),step))
                print('Training .....................................')
                # Initialize the training dataset iterator
                sess.run(model.train_init_op)
                feed = construct_feed(bi_state_fw, bi_state_bw)

                start = time.time()
                for batch in range(nb_batches_train):
                    sess.run(model.train_step_v, feed_dict=feed)
                    sess.run(model.train_step_u, feed_dict=feed)
                    _,_, U, V, rnn_output, U_b, V_b, bi_out_fw, bi_out_bw, final_state, rmse, mae, summary_str = sess.run(
                        [model.train_step_rnn,model.update_rnn_output,
                         model.U,model.V, model.RNN, model.U_bias, model.V_bias,
                         model.bi_output_state_fw, model.bi_output_state_bw, model.H,
                         model.RMSE, model.MAE, model.summary_op],
                        feed_dict=feed)
                    if batch // 10 % 50 == 0:
                        print("Epoch {0}, batch {1}".format(step,batch))
                train_writer.add_summary(summary_str, step)

                end = time.time()
                print('Epoch {0}, finished in {1}'.format(step, end - start))
                if False and step // 10 % 5 == 0:
                    print('{0}:Validation ............'.format(time.strftime("%d:%m-%H:%M:" )))

                    # save a checkpoint (every 500 batches)
                    if step // 10%50 == 0 and step > 10 :
                        saved_file = model.saver.save(sess, ckpt_dir, global_step=step)
                        print("Saved file: " + saved_file)

                    #Initialize the validation dataset iterator
                    sess.run(model.validation_init_op)
                    for _ in range(nb_batches_val):
                        test_bi_fw = sess.run(model.init_state_fw)
                        test_bi_bw = sess.run(model.init_state_bw)
                        init_state = sess.run(model.initial_state)
                        feed_dict = construct_feed(test_bi_fw, test_bi_bw)
                        rmse_test, mae_test, summary_str = sess.run(
                            [model.RMSE, model.MAE, model.summary_op], feed_dict=feed_dict)

                    test_writer.add_summary(summary_str, step)

                    prediction_matrix = np.matmul(U, np.add(V,rnn_output).T)
                    prediction_matrix = np.add(prediction_matrix, np.reshape(U_b, [-1, 1]))
                    prediction_matrix = np.add(prediction_matrix, V_b)
                    rounded_predictions = utils.rounded_predictions(prediction_matrix)
                    evaluator.load_top_recommendations_2(200, prediction_matrix, test_ratings)
                    recall_10 = evaluator.recall_at_x(10, prediction_matrix, parser.ratings, rounded_predictions)
                    recall_50 = evaluator.recall_at_x(50, prediction_matrix, parser.ratings, rounded_predictions)
                    recall_100 = evaluator.recall_at_x(100, prediction_matrix, parser.ratings, rounded_predictions)
                    recall_200 = evaluator.recall_at_x(200, prediction_matrix, parser.ratings, rounded_predictions)
                    recall = evaluator.calculate_recall(ratings=parser.ratings, predictions=rounded_predictions)
                    ndcg_at_five = evaluator.calculate_ndcg(5, rounded_predictions)
                    ndcg_at_ten = evaluator.calculate_ndcg(10, rounded_predictions)

                    mrr_at_ten = evaluator.calculate_mrr(10,rounded_predictions)

                    feed = {model.recall: recall, model.recall_10: recall_10, model.recall_50: recall_50,
                            model.recall_100: recall_100, model.recall_200: recall_200,
                            model.ndcg_5: ndcg_at_five, model.ndcg_10: ndcg_at_ten, model.mrr_10: mrr_at_ten}
                    eval_metrics = sess.run([model.eval_metrics], feed_dict=feed)
                    test_writer.add_summary(eval_metrics[0], step)

                    print("Step {0} | Train RMSE: {1:3.4f}, MAE: {2:3.4f}".format(
                        step, rmse, mae))
                    print("         | Test  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
                        rmse_test, mae_test))
                    print("         | Recall@10: {0:3.4f}".format(recall_10))
                    print("         | Recall@50: {0:3.4f}".format(recall_50))
                    print("         | Recall@100: {0:3.4f}".format(recall_100))
                    print("         | Recall@200: {0:3.4f}".format(recall_200))
                    print("         | Recall: {0:3.4f}".format(recall))
                    print("         | ndcg@5: {0:3.4f}".format(ndcg_at_five))
                    print("         | ndcg@10: {0:3.4f}".format(ndcg_at_ten))
                    print("         | mrr@10: {0:3.4f}".format(mrr_at_ten))
                    if best_val_rmse > rmse_test:
                        # best_val_rmse = rmse_valid
                        best_test_rmse = rmse_test

                    if best_val_mae > rmse_test:
                        # best_val_mae = mae_valid
                        best_test_mae = mae_test

                # loop state around
                h_state = final_state
                bi_state_fw = bi_out_fw
                bi_state_bw = bi_out_bw
                # if step > 0 and (step % args.save_every == 0 or ( step == args.num_epochs - 1)):  # save for the last result
                #     checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                #     saver.save(sess, checkpoint_path, global_step=step)
                #     print("model saved to {}".format(checkpoint_path))
            model.saver.save(sess, args.log_dir + "/{0}model.ckpt".format(time.strftime(dir_prefix)))
            print('Best test rmse:', best_test_rmse, 'Best test mae', best_test_mae, sep=' ')
            # restore model
            # if args.init_from is not None:
            #     saver.restore(sess, ckpt.model_checkpoint_path)
            # for e in range(model.epoch_pointer.eval(), args.num_epochs):
            #     sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            #     data_loader.reset_batch_pointer()
            #     state = sess.run(model.initial_state)
            #     speed = 0
            #     for b in range(data_loader.pointer, data_loader.num_batches):
            #         start = time.time()
            #         x, y = data_loader.next_batch()
            #         feed = {model.input_data: x, model.targets: y, model.initial_state: state,
            #                 model.batch_time: speed}
            #         summary, train_loss, state, _, _ = sess.run([merged, model.cost, model.final_state,
            #                                                      model.train_op, model.inc_batch_pointer_op], feed)
            #         train_writer.add_summary(summary, e * data_loader.num_batches + b)
            #         speed = time.time() - start
            #         if (e * data_loader.num_batches + b) % args.batch_size == 0:
            #             print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
            #                 .format(e * data_loader.num_batches + b,
            #                         args.num_epochs * data_loader.num_batches,
            #                         e, train_loss, speed))
            #         if (e * data_loader.num_batches + b) % args.save_every == 0 \
            #                 or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
            #             checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
            #             saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
            #             print("model saved to {}".format(checkpoint_path))
            train_writer.close()
            valid_writer.close()
            test_writer.close()
        except Exception as e:
            # Report exceptions to the coordinator.
            print(e)
            # coord.request_stop(e)
            print("Finished training")

def partial_run(args):
    parser = DataParser(args.data_dir, args.dataset, 'attributes',
                        use_embeddings=True, embed_dir=args.embedding_dir, embed_dim=args.embedding_dim)
    # read input data
    dataset_path, dataset_count = input(args, parser)
    args.vocab_size = parser.get_vocab_size()

    for fold in range(args.folds):
        path_training = dataset_path[fold][0]
        path_test = dataset_path[fold][1]

        example_count_train = dataset_count[fold][0]
        example_count_validation = dataset_count[fold][1]

        nb_batches_train = int(math.ceil(example_count_train / args.batch_size))
        print('Number of training batches {0}, number of samples {1}'.format(nb_batches_train, example_count_train))
        nb_batches_val = int(math.ceil(example_count_validation / args.batch_size))
        print(
            'Number of validation batches {0}, number of samples {1}'.format(nb_batches_val, example_count_validation))

    print('Vocabolary size {0}'.format(parser.words_count))
    print("Uknown words {0}".format(parser.unkown_words_count))
    print("Uknown numbers {0}".format(parser.numbers_count))

    dir_prefix = time.strftime("%d:%m-%H:%M:")
    # Checkpoints directory
    ckpt_dir = os.path.join(args.log_dir, 'checkpoints/{0}-train'.format(dir_prefix))

    best_val_rmse = np.inf
    best_val_mae = np.inf
    best_test_rmse = 0
    best_test_mae = 0

    evaluator = Evaluator(parser.get_ratings_matrix(), verbose=True)

    graph = tf.Graph()
    with graph.as_default():
        model = Model(args, parser.get_ratings_matrix(), parser.embeddings, path_training, path_test)
        train_writer = tf.summary.FileWriter(args.log_dir + '/{0}-train'.format(dir_prefix))
        valid_writer = tf.summary.FileWriter(args.log_dir + '/{0}-validation'.format(time.strftime(dir_prefix)))
        test_writer = tf.summary.FileWriter(args.log_dir + '/{0}-test'.format(time.strftime(dir_prefix)))

    def construct_feed(bi_hid_fw, bi_hid_bw):
        return {model.init_state_fw: bi_hid_fw, model.init_state_bw: bi_hid_bw}
        # model.initial_state: hid_state,

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'

    n_steps = args.num_epochs
    with tf.Session(config=config, graph=graph) as sess:
        print('Saving graph to disk...')
        train_writer.add_graph(sess.graph)
        # valid_writer.add_graph(sess.graph)
        # test_writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        print("Loading test ratings matrix")
        test_ratings = utils.get_test_ratings_matrix(path_test, parser.user_count, parser.paper_count, sess)



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
                feed_dict = construct_feed(bi_state_fw, bi_state_bw)

                # add dummy steps so tf won't complain and give the error
                # partial_run() requires empty target_list.
                #please refer to this issue:
                # https://github.com/tensorflow/tensorflow/issues/1899
                with tf.control_dependencies([model.train_step_rnn]):
                    dummy_train_step_rnn = tf.constant(0)
                with tf.control_dependencies([model.train_step_v]):
                    dummy_train_step_v = tf.constant(0)
                with tf.control_dependencies([model.train_step_u]):
                    dummy_train_step_u = tf.constant(0)

                feeds = [model.init_state_fw, model.init_state_bw]
                fetches=[dummy_train_step_rnn,dummy_train_step_u,dummy_train_step_v,
                        # model.train_step_v,model.train_step_u,model.train_step_rnn,
                         model.update_rnn_output,
                         model.U, model.V, model.RNN, model.U_bias, model.V_bias,
                         model.bi_output_state_fw, model.bi_output_state_bw, model.H,
                         model.RMSE, model.MAE, model.summary_op]
                start = time.time()

                for batch in range(nb_batches_train):
                # for batch in range(3):
                    handle = sess.partial_run_setup(fetches, feeds)
                    sess.partial_run(handle, dummy_train_step_rnn, feed_dict=feed_dict)
                    sess.partial_run(handle,dummy_train_step_v)
                    _, _, U, V, rnn_output, U_b, V_b, bi_out_fw, bi_out_bw, final_state, rmse, mae, summary_str = sess.partial_run(
                        handle,
                        [dummy_train_step_u, model.update_rnn_output,
                         model.U, model.V, model.RNN, model.U_bias, model.V_bias,
                         model.bi_output_state_fw, model.bi_output_state_bw, model.H,
                         model.RMSE, model.MAE, model.summary_op])
                    # print every 500 iteration
                    if batch // 10 % 50 == 0:
                        print("Epoch {0}, batch {1}".format(step, batch))
                    train_writer.add_summary(summary_str, global_step=(step*nb_batches_train + batch))
                end = time.time()
                print('Epoch {0}, finished in {1}'.format(step,end - start))
                if True or step // 1 % 5 == 0:
                    print('{0}:Validation ............'.format(time.strftime("%d:%m-%H:%M:")))

                    # save a checkpoint (every 5 epochs)
                    if step // 1 % 5 == 0 and step > 4:
                        saved_file = model.saver.save(sess, ckpt_dir, global_step=step)
                        print("Saved file: " + saved_file)

                    # Initialize the validation dataset iterator
                    sess.run(model.validation_init_op)
                    test_bi_fw = sess.run(model.init_state_fw)
                    test_bi_bw = sess.run(model.init_state_bw)
                    init_state = sess.run(model.initial_state)
                    feed_dict = construct_feed(test_bi_fw, test_bi_bw)
                    for batch in range(nb_batches_val):
                        rmse_test, mae_test, summary_str = sess.run(
                            [model.RMSE, model.MAE, model.summary_op], feed_dict=feed_dict)
                        test_writer.add_summary(summary_str, global_step=(step*nb_batches_val + batch))


                    prediction_matrix = np.matmul(U, np.add(V, rnn_output).T)
                    prediction_matrix = np.add(prediction_matrix, np.reshape(U_b, [-1, 1]))
                    prediction_matrix = np.add(prediction_matrix, V_b)
                    rounded_predictions = utils.rounded_predictions(prediction_matrix)
                    evaluator.load_top_recommendations_2(200, prediction_matrix, test_ratings)
                    recall_10 = evaluator.recall_at_x(10, prediction_matrix, parser.ratings, rounded_predictions)
                    recall_50 = evaluator.recall_at_x(50, prediction_matrix, parser.ratings, rounded_predictions)
                    recall_100 = evaluator.recall_at_x(100, prediction_matrix, parser.ratings, rounded_predictions)
                    recall_200 = evaluator.recall_at_x(200, prediction_matrix, parser.ratings, rounded_predictions)
                    recall = evaluator.calculate_recall(ratings=parser.ratings, predictions=rounded_predictions)
                    ndcg_at_five = evaluator.calculate_ndcg(5, rounded_predictions)
                    ndcg_at_ten = evaluator.calculate_ndcg(10, rounded_predictions)

                    mrr_at_ten = evaluator.calculate_mrr(10, rounded_predictions)

                    feed = {model.recall: recall, model.recall_10: recall_10, model.recall_50: recall_50,
                            model.recall_100: recall_100, model.recall_200: recall_200,
                            model.ndcg_5: ndcg_at_five, model.ndcg_10: ndcg_at_ten, model.mrr_10: mrr_at_ten}
                    eval_metrics = sess.run([model.eval_metrics], feed_dict=feed)
                    test_writer.add_summary(eval_metrics[0], step)

                    print("Step {0} | Train RMSE: {1:3.4f}, MAE: {2:3.4f}".format(
                        step, rmse, mae))
                    print("         | Test  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
                        rmse_test, mae_test))
                    print("         | Recall@10: {0:3.4f}".format(recall_10))
                    print("         | Recall@50: {0:3.4f}".format(recall_50))
                    print("         | Recall@100: {0:3.4f}".format(recall_100))
                    print("         | Recall@200: {0:3.4f}".format(recall_200))
                    print("         | Recall: {0:3.4f}".format(recall))
                    print("         | ndcg@5: {0:3.4f}".format(ndcg_at_five))
                    print("         | ndcg@10: {0:3.4f}".format(ndcg_at_ten))
                    print("         | mrr@10: {0:3.4f}".format(mrr_at_ten))
                    if best_val_rmse > rmse_test:
                        # best_val_rmse = rmse_valid
                        best_test_rmse = rmse_test

                    if best_val_mae > rmse_test:
                        # best_val_mae = mae_valid
                        best_test_mae = mae_test

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


if __name__ == '__main__':
    main()
