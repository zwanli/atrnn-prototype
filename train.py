import tensorflow as tf

import argparse
import time
import os
import numpy as np
from six.moves import cPickle

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
    parser.add_argument('--rnn_size', type=int, default=256,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='gru',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=125,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=20,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=5000,
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

    if os.path.exists('/home/wanli/repositories/atrnn_prototype/abstracrs_word_embeddings.pkl'):
        print('Loading abstracts')
        with open('/home/wanli/repositories/atrnn_prototype/abstracrs_word_embeddings.pkl', 'rb') as f:
            data_loader.all_documents= pickle.load(f)
    else:
        text_dir = '/home/wanli/data/Extended_ctr/dummy/raw-data.csv'
        data_loader.read_abstrcts(text_dir)
        with open("abstracrs_word_embeddings.pkl",'wb') as f:
            pickle.dump(data_loader.all_documents,f,pickle.HIGHEST_PROTOCOL)
            print("Saved abstracts")
    #Read ratings input
    ratings_path = '/home/wanli/data/Extended_ctr/dummy/users.dat'
    data_loader.read_dataset(ratings_path,50,1928)#CHANGE ++++++++
    # u, v = data_loader.generate_batches(10000)
    # for u,v,r,docs in data_loader.generate_batches(10000):
    #     print(len(docs))
    args.seq_length = data_loader.get_min_max_length()
    model = Model(args, data_loader.M)

    def construct_feed(u_idx, v_idx, r, docs, batch_size):
        return {model.u_idx: u_idx, model.v_idx: v_idx, model.r: r, model.input_data: docs, model.batch_size: batch_size}
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


    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.log_dir+ '/train')
    valid_writer = tf.summary.FileWriter(args.log_dir + '/validation')
    test_writer = tf.summary.FileWriter(args.log_dir + '/test')
    best_val_rmse = np.inf
    best_val_mae = np.inf
    best_test_rmse = 0
    best_test_mae = 0

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)

    n_steps = args.num_epochs
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_writer.add_graph(sess.graph)
        valid_writer.add_graph(sess.graph)
        test_writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        for u, v, r, d, step in data_loader.generate_batches(n_steps):
            # feed = {model.u_idx: u, model.v_idx: v, model.r: r, model.input_data: d}
            feed = construct_feed(u,v,r,d,args.batch_size)
            # model.Yr.eval(feed_dict={model.input_data: d})
            sess.run(model.train_step_v, feed_dict=feed)
            sess.run(model.train_step_u, feed_dict=feed)
            _, rmse, mae, summary_str = sess.run([model.train_step_rnn, model.RMSE,model.MAE, model.summary_op], feed_dict=feed)
            train_writer.add_summary(summary_str,step)


            if step % int(n_steps / 100) == 0:
                valid_u_idx, valid_v_idx, valid_m, valid_docs = data_loader.get_valid_idx()
                # feed_dict = {model.u_idx: valid_u_idx,model.v_idx: valid_v_idx, model.r:valid_m, model.input_data: valid_docs}
                feed_dict = construct_feed(valid_u_idx, valid_v_idx, valid_m, valid_docs, valid_docs.shape[0])
                rmse_valid, mae_valid, summary_str = sess.run(
                    [model.RMSE, model.MAE, model.summary_op], feed_dict=feed_dict)

                valid_writer.add_summary(summary_str, step)

                test_u_idx, test_v_idx, test_m, test_docs = data_loader.get_test_idx()
                # feed_dict = {model.u_idx:test_u_idx,model.v_idx:  test_v_idx, model.r:test_m, model.input_data: test_docs}
                feed_dict= construct_feed(test_u_idx, test_v_idx, test_m, test_docs, test_docs.shape[0])
                rmse_test, mae_test, summary_str = sess.run(
                    [model.RMSE, model.MAE, model.summary_op], feed_dict=feed_dict)

                test_writer.add_summary(summary_str, step)

                print("Step {0} | Train RMSE: {1:3.4f}, MAE: {2:3.4f}".format(
                    step, rmse, mae))
                print("         | Valid  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
                    rmse_valid, mae_valid))
                print("         | Test  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
                    rmse_test, mae_test))

                if best_val_rmse > rmse_valid:
                    best_val_rmse = rmse_valid
                    best_test_rmse = rmse_test

                if best_val_mae > mae_valid:
                    best_val_mae = mae_valid
                    best_test_mae = mae_test

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

if __name__ == '__main__':
    main()