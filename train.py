import tensorflow as tf
import argparse
import time
import os
import numpy as np
from six.moves import cPickle
from evaluator import Evaluator
from utils import textloader
from model import Model
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                       help='data directory containing input.txt')
    parser.add_argument('--embedding_dir', type=str, default='/home/wanli/data/glove.6B/',
                        help='embedding directory containing embeddings file')
    parser.add_argument('--input_encoding', type=str, default=None,
                       help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=200,
                       help='size of RNN hidden state')
    parser.add_argument('--embedding_dim', type=int, default=200,
                        help='dimension of the embeddings')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='gru',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=125,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=20,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--gpu_mem', type=float, default=0.666,
                       help='%% of gpu memory to be allocated to this process. Default is 66.6%%')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)
def construct_feed(u_idx, v_idx, r,docs,batch_size):
    return
def train(args):
    #Read text input
    data_loader = textloader(args.embedding_dir,args.batch_size)
    args.vocab_size = data_loader.vocab_size


    if os.path.exists('abstracrs_word_embeddings_dummy.pkl'):
        print('Loading abstracts')
        with open('abstracrs_word_embeddings_dummy.pkl', 'rb') as f:
            data_loader.all_documents= pickle.load(f)
    else:
        text_dir = '/home/wanli/data/Extended_ctr/citeulike_t_extended/raw-data.csv'
        data_loader.read_abstracts(text_dir)
        with open("abstracts_word_embeddings_dummy_citeulike_t_extended.pkl",'wb') as f:
            pickle.dump(data_loader.all_documents,f,pickle.HIGHEST_PROTOCOL)
            print("Saved abstracts")
    #Read ratings input
    ratings_path = '/home/wanli/data/Extended_ctr/dummy/users.dat'
    data_loader.read_dataset(ratings_path,50,1928)#CHANGE ++++++++
    # u, v = data_loader.generate_batches(10000)
    # for u,v,r,docs in data_loader.generate_batches(10000):
    #     print(len(docs))
    args.seq_length = data_loader.get_min_max_length()
    model = Model(args, data_loader.M,data_loader.embed)

    def construct_feed(u_idx, v_idx, r, docs,bi_hid_fw, bi_hid_bw, batch_size):
        return {model.u_idx: u_idx, model.v_idx: v_idx, model.r: r, model.input_data: docs,
                model.init_state_fw: bi_hid_fw, model.init_state_bw: bi_hid_bw,model.batch_size: batch_size}
                # model.initial_state: hid_state,

        # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"words_vocab.pkl")),"words_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'words_vocab.pkl'), 'rb') as f:
            saved_words, saved_vocab = cPickle.load(f)
        assert saved_words==data_loader.words, "Data and loaded model disagree on word set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    # with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
    #     cPickle.dump(args, f)
    # with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
    #     cPickle.dump((data_loader.word_to_id, data_loader.vocab), f)


    dir_prefix = time.strftime("%d:%m-%H:%M:")
    train_writer = tf.summary.FileWriter(args.log_dir+ '/{0}-train'.format(dir_prefix))
    valid_writer = tf.summary.FileWriter(args.log_dir + '/{0}-validation'.format(time.strftime(dir_prefix)))
    test_writer = tf.summary.FileWriter(args.log_dir + '/{0}-test'.format(time.strftime(dir_prefix)))
    best_val_rmse = np.inf
    best_val_mae = np.inf
    best_test_rmse = 0
    best_test_mae = 0

    evaluator = Evaluator(data_loader.M,verbose=True)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
    n_steps = args.num_epochs
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # train_writer.add_graph(sess.graph)
        # valid_writer.add_graph(sess.graph)
        # test_writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()
        model.saver = tf.train.Saver(tf.global_variables())

        bi_state_fw = sess.run(model.init_state_bw,feed_dict={model.batch_size: args.batch_size})
        bi_state_bw = sess.run(model.init_state_fw,feed_dict={model.batch_size: args.batch_size})
        h_state = sess.run(model.initial_state,feed_dict={model.batch_size: args.batch_size})

        for u, v, r, d, step in data_loader.generate_batches(n_steps):

            feed = construct_feed(u, v, r, d, bi_state_fw, bi_state_bw,args.batch_size)
            sess.run(model.train_step_v, feed_dict=feed)
            sess.run(model.train_step_u, feed_dict=feed)
            _, U,V ,U_b ,V_b , bi_out_fw, bi_out_bw, final_state, rmse, mae, summary_str = sess.run([model.train_step_rnn,
                                                                                           model.U,model.V,model.U_bias,model.V_bias,
                                                                                           model.bi_output_state_fw, model.bi_output_state_bw, model.H,
                                                                                           model.RMSE,model.MAE, model.summary_op],
                                                        feed_dict=feed)
            train_writer.add_summary(summary_str,step)

            if step % int(n_steps / 100) == 0:
                valid_u_idx, valid_v_idx, valid_m, valid_docs = data_loader.get_valid_idx()
                valid_bi_fw = sess.run(model.init_state_fw,feed_dict={model.batch_size:valid_docs.shape[0] })
                valid_bi_bw = sess.run(model.init_state_bw,feed_dict={model.batch_size:valid_docs.shape[0] })
                init_state = sess.run(model.initial_state,feed_dict={model.batch_size:valid_docs.shape[0] })
                feed_dict = construct_feed(valid_u_idx, valid_v_idx, valid_m, valid_docs,valid_bi_fw, valid_bi_bw, valid_docs.shape[0])
                rmse_valid, mae_valid, summary_str = sess.run(
                    [model.RMSE, model.MAE, model.summary_op], feed_dict=feed_dict)
                valid_writer.add_summary(summary_str, step)

                test_u_idx, test_v_idx, test_m, test_docs = data_loader.get_test_idx()
                test_bi_fw = sess.run(model.init_state_fw, feed_dict={model.batch_size: test_docs.shape[0]})
                test_bi_bw = sess.run(model.init_state_bw, feed_dict={model.batch_size: test_docs.shape[0]})
                init_state = sess.run(model.initial_state,feed_dict={model.batch_size:test_docs.shape[0] })
                feed_dict= construct_feed(test_u_idx, test_v_idx, test_m, test_docs,test_bi_fw,test_bi_bw, test_docs.shape[0])
                rmse_test, mae_test, summary_str = sess.run(
                    [model.RMSE, model.MAE, model.summary_op], feed_dict=feed_dict)
                test_writer.add_summary(summary_str, step)

                prediction_matrix = np.matmul(U,V.T)
                prediction_matrix = np.add(prediction_matrix,np.reshape(U_b,[-1,1]))
                prediction_matrix = np.add(prediction_matrix,V_b)
                rounded_predictions = data_loader.rounded_predictions(prediction_matrix)
                testM = np.zeros(data_loader.M.shape)
                testM[data_loader.nonzero_u_idx[data_loader.test_idx], data_loader.nonzero_v_idx[data_loader.test_idx]] = data_loader.M[
                    data_loader.nonzero_u_idx[data_loader.test_idx], data_loader.nonzero_v_idx[data_loader.test_idx]]

                evaluator.load_top_recommendations_2(200,prediction_matrix,testM)
                recall_10 = evaluator.recall_at_x(10, prediction_matrix, data_loader.M,rounded_predictions )
                recall_50 = evaluator.recall_at_x(50, prediction_matrix, data_loader.M, rounded_predictions)
                recall_100 = evaluator.recall_at_x(100, prediction_matrix, data_loader.M, rounded_predictions)
                recall_200 = evaluator.recall_at_x(200, prediction_matrix, data_loader.M, rounded_predictions)
                recall = evaluator.calculate_recall(ratings=data_loader.M,predictions=rounded_predictions)
                ndcg_at_five = evaluator.calculate_ndcg(5, rounded_predictions)
                ndcg_at_ten = evaluator.calculate_ndcg(10, rounded_predictions)

                print("Step {0} | Train RMSE: {1:3.4f}, MAE: {2:3.4f}".format(
                    step, rmse, mae))
                print("         | Valid  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
                    rmse_valid, mae_valid))
                print("         | Test  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
                    rmse_test, mae_test))
                print("         | Recall@10: {0:3.4f}".format(recall_10))
                print("         | Recall@50: {0:3.4f}".format(recall_50))
                print("         | Recall@100: {0:3.4f}".format(recall_100))
                print("         | Recall@200: {0:3.4f}".format(recall_200))
                print("         | Recall: {0:3.4f}".format(recall))
                print("         | ndcg@5: {0:3.4f}".format(ndcg_at_five))
                print("         | ndcg@10: {0:3.4f}".format(ndcg_at_ten))




                if best_val_rmse > rmse_valid:
                    best_val_rmse = rmse_valid
                    best_test_rmse = rmse_test

                if best_val_mae > mae_valid:
                    best_val_mae = mae_valid
                    best_test_mae = mae_test

            # loop state around
            h_state = final_state
            bi_state_fw = bi_out_fw
            bi_state_bw = bi_out_bw
            # if step > 0 and (step % args.save_every == 0 or ( step == args.num_epochs - 1)):  # save for the last result
            #     checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
            #     saver.save(sess, checkpoint_path, global_step=step)
            #     print("model saved to {}".format(checkpoint_path))
        model.saver.save(sess, args.log_dir+ "/{0}model.ckpt".format(time.strftime(dir_prefix)))
        print('Best test rmse:',best_test_rmse,'Best test mae',best_test_mae,sep=' ')
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

if __name__ == '__main__':
    main()