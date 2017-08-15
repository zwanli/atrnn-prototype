import tensorflow as tf
import argparse
import time
import os
import numpy as np
from six.moves import cPickle
from evaluator import Evaluator
from data_loader import textloader
from model import Model
import pickle
from custom_runner import CustomRunner
from data_parser import DataParser
import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/wanli/data/Extended_ctr/',
                       help='data directory containing input.txt')
    parser.add_argument("--dataset", "-d",type=str, default='dummy',
                        help="Which dataset to use", choices=['dummy', 'citeulike-a', 'citeulike-t'])
    parser.add_argument('--embedding_dir', type=str, default='/home/wanli/data/glove.6B/',
                        help='GloVe embedding directory containing embeddings file')
    parser.add_argument('--embedding_dim', type=int, default=200,
                        help='dimension of the embeddings', choices=['50', '100', '200','300'])
    parser.add_argument('--input_encoding', type=str, default=None,
                       help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')

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
    parser.add_argument('--batch_size', type=int, default=128,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=300,
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
# def construct_feed(u_idx, v_idx, r,docs,batch_size):
#     return
def train(args):
    #Read text input
    parser = DataParser(args.data_dir,args.dataset,None,
                        use_embeddings=True,embed_dir=args.embedding_dir,embed_dim=args.embedding_dim)
    parser.load_embeddings()
    args.vocab_size = parser.get_vocab_size()
    if os.path.exists('abstracts_word_embeddings_{}.pkl'.format(args.dataset)):
        print('Loading abstracts')
        with open('abstracts_word_embeddings_{}.pkl'.format(args.dataset), 'rb') as f:
            parser.all_documents =  pickle.load(f)
    else:
        parser.get_papar_as_word_ids()
        with open("abstracts_word_embeddings_{}.pkl".format(args.dataset),'wb') as f:
            pickle.dump(parser.all_documents,f,pickle.HIGHEST_PROTOCOL)
            print("Saved abstracts")

    parser.split_cold_start(5)
    parser.split_warm_start(5)
    parser.generate_batches(128)
    model = Model(args, parser.get_ratings_matrix(),parser.embeddings)

    def construct_feed(u_idx, v_idx, r, docs,seq_lengths,bi_hid_fw, bi_hid_bw, batch_size):
        return {model.u_idx: u_idx, model.v_idx: v_idx, model.r: r, model.input_data: docs, model.seq_lengths:seq_lengths,
                model.init_state_fw: bi_hid_fw, model.init_state_bw: bi_hid_bw,model.batch_size: batch_size}
                # model.initial_state: hid_state,

    def get_batches(sess, coord, batch_size,bucket_boundaries, valid=False, test=False):
        with tf.device("/cpu:0"):
            custom_runner = CustomRunner(batch_size, bucket_boundaries, parser,name='v' if valid else 't')
            seq_len, outputs = custom_runner.get_outputs()
        try:
            custom_runner.enque_input(sess,valid,test)
            custom_runner.close(sess)
            # threads = tf.train.start_queue_runners(sess, coord)
            b = 0
            batches={}
            while True:
                out_lengths, (input_t, lengths_t, u_idx_t, v_idx_t, r_t) = sess.run([seq_len, outputs])
                print(len(input_t[0, 0]))
                input_t = np.squeeze(input_t, [1])
                lengths_t = np.squeeze(lengths_t, [1])
                u_idx_t = np.squeeze(u_idx_t, [1])
                v_idx_t = np.squeeze(v_idx_t, [1])
                r_t = np.squeeze(r_t, [1])
                batches[b] = (input_t, lengths_t, u_idx_t, v_idx_t, r_t)
                b += 1
        except Exception as e:
            # Report exceptions to the coordinator.
            coord.request_stop(e)
            print("Total number of {0} samples: {1}".format('',b * batch_size))
            print("Total number of {0} batches: {1}".format('',b))
        return batches

    bucket_boundaries = [x for x in range(50, 500, 50)]
    batch_size = args.batch_size


    dir_prefix = time.strftime("%d:%m-%H:%M:")
    train_writer = tf.summary.FileWriter(args.log_dir+ '/{0}-train'.format(dir_prefix))
    valid_writer = tf.summary.FileWriter(args.log_dir + '/{0}-validation'.format(time.strftime(dir_prefix)))
    test_writer = tf.summary.FileWriter(args.log_dir + '/{0}-test'.format(time.strftime(dir_prefix)))
    best_val_rmse = np.inf
    best_val_mae = np.inf
    best_test_rmse = 0
    best_test_mae = 0

    evaluator = Evaluator(parser.get_ratings_matrix(),verbose=True)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
    n_steps = args.num_epochs
    with tf.device("/cpu:0"):
        custom_runner = CustomRunner(batch_size, bucket_boundaries,parser)
        seq_len, outputs = custom_runner.get_outputs()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # train_writer.add_graph(sess.graph)
        # valid_writer.add_graph(sess.graph)
        # test_writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        train_batches = {}
        valid_batches = {}
        test_batches = {}

        try:
            custom_runner.enque_input(sess)
            custom_runner.close(sess)
            # threads = tf.train.start_queue_runners(sess, coord)
            b = 0
            while True:
                out_lengths, (input_t, lengths_t, u_idx_t, v_idx_t, r_t) = sess.run([seq_len, outputs])
                print(len(input_t[0, 0]))
                input_t = np.squeeze(input_t, [1])
                lengths_t = np.squeeze(lengths_t, [1])
                u_idx_t = np.squeeze(u_idx_t, [1])
                v_idx_t = np.squeeze(v_idx_t, [1])
                r_t = np.squeeze(r_t, [1])
                train_batches[b] = (input_t, lengths_t, u_idx_t, v_idx_t, r_t)
                b += 1
        except Exception as e:
            # Report exceptions to the coordinator.
            coord.request_stop(e)
            print("Total number of training samples: {0}".format(b * batch_size))
            print("Total number of training batches: {0}".format(b))
        finally:
            coord.request_stop()
            coord.join(threads)
        # valid_u_idx, valid_v_idx, valid_m, valid_docs = data_loader.get_valid_idx()
        # test_u_idx, test_v_idx, test_m, test_docs = data_loader.get_test_idx()
        # valid_batches = get_batches(sess,coord,valid_docs.shape[0],[0],valid=True)
        # test_batches = get_batches(sess, coord, test_docs.shape[0],[0], test=True)

        print('Finished batching ')
        # model.saver = tf.train.Saver(tf.global_variables())

        bi_state_fw = sess.run(model.init_state_bw,feed_dict={model.batch_size: args.batch_size})
        bi_state_bw = sess.run(model.init_state_fw,feed_dict={model.batch_size: args.batch_size})
        h_state = sess.run(model.initial_state,feed_dict={model.batch_size: args.batch_size})

        for step in range(n_steps):
            for d, s, u, v, r in train_batches.values():
                feed = construct_feed(u, v, r, d,s, bi_state_fw, bi_state_bw,args.batch_size)
                sess.run(model.train_step_v, feed_dict=feed)
                sess.run(model.train_step_u, feed_dict=feed)
                _, U,V ,U_b ,V_b , bi_out_fw, bi_out_bw, final_state, rmse, mae, summary_str = sess.run([model.train_step_rnn,
                                                                                               model.U,model.V,model.U_bias,model.V_bias,
                                                                                               model.bi_output_state_fw, model.bi_output_state_bw, model.H,
                                                                                               model.RMSE,model.MAE, model.summary_op],
                                                            feed_dict=feed)
                train_writer.add_summary(summary_str,step)

                cond = True
                if cond and step % int(n_steps / 100) == 0:
                    # valid_u_idx, valid_v_idx, valid_m, valid_docs = data_loader.get_valid_idx()
                    # valid_docs, valid_docs_len = data_loader.static_padding(valid_docs)
                    # valid_bi_fw = sess.run(model.init_state_fw,feed_dict={model.batch_size:valid_docs.shape[0] })
                    # valid_bi_bw = sess.run(model.init_state_bw,feed_dict={model.batch_size:valid_docs.shape[0] })
                    # init_state = sess.run(model.initial_state,feed_dict={model.batch_size:valid_docs.shape[0] })
                    # feed_dict = construct_feed(valid_u_idx, valid_v_idx, valid_m,
                    #                            valid_docs,valid_docs_len, valid_bi_fw, valid_bi_bw, valid_docs.shape[0])
                    # rmse_valid, mae_valid, summary_str = sess.run(
                    #     [model.RMSE, model.MAE, model.summary_op], feed_dict=feed_dict)
                    # valid_writer.add_summary(summary_str, step)

                    test_u_idx, test_v_idx, test_m, test_docs, test_ratings = parser.get_test_idx()
                    test_docs, test_docs_len = utils.static_padding(test_docs,maxlen=300)

                    test_bi_fw = sess.run(model.init_state_fw, feed_dict={model.batch_size: test_docs.shape[0]})
                    test_bi_bw = sess.run(model.init_state_bw, feed_dict={model.batch_size: test_docs.shape[0]})
                    init_state = sess.run(model.initial_state,feed_dict={model.batch_size:test_docs.shape[0] })
                    feed_dict= construct_feed(test_u_idx, test_v_idx, test_m,
                                              test_docs, test_docs_len, test_bi_fw,test_bi_bw, test_docs.shape[0])
                    rmse_test, mae_test, summary_str = sess.run(
                        [model.RMSE, model.MAE, model.summary_op], feed_dict=feed_dict)

                    test_writer.add_summary(summary_str, step)

                    prediction_matrix = np.matmul(U,V.T)
                    prediction_matrix = np.add(prediction_matrix,np.reshape(U_b,[-1,1]))
                    prediction_matrix = np.add(prediction_matrix,V_b)
                    rounded_predictions = utils.rounded_predictions(prediction_matrix)
                    # testM = np.zeros(parser.ratings.shape)
                    # testM[data_loader.nonzero_u_idx[data_loader.test_idx], data_loader.nonzero_v_idx[data_loader.test_idx]] = data_loader.M[
                    #     data_loader.nonzero_u_idx[data_loader.test_idx], data_loader.nonzero_v_idx[data_loader.test_idx]]

                    evaluator.load_top_recommendations_2(200,prediction_matrix,test_ratings)
                    recall_10 = evaluator.recall_at_x(10, prediction_matrix, parser.ratings,rounded_predictions )
                    recall_50 = evaluator.recall_at_x(50, prediction_matrix, parser.ratings, rounded_predictions)
                    recall_100 = evaluator.recall_at_x(100, prediction_matrix, parser.ratings, rounded_predictions)
                    recall_200 = evaluator.recall_at_x(200, prediction_matrix, parser.ratings, rounded_predictions)
                    recall = evaluator.calculate_recall(ratings=parser.ratings,predictions=rounded_predictions)
                    ndcg_at_five = evaluator.calculate_ndcg(5, rounded_predictions)
                    ndcg_at_ten = evaluator.calculate_ndcg(10, rounded_predictions)



                    feed ={model.recall:recall, model.recall_10:recall_10, model.recall_50:recall_50,
                           model.recall_100:recall_100, model.recall_200:recall_200,
                           model.ndcg_5:ndcg_at_five, model.ndcg_10:ndcg_at_ten}
                    eval_metrics = sess.run([model.eval_metrics], feed_dict=feed)
                    test_writer.add_summary(eval_metrics[0], step)

                    print("Step {0} | Train RMSE: {1:3.4f}, MAE: {2:3.4f}".format(
                        step, rmse, mae))
                    # print("         | Valid  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
                    #     rmse_valid, mae_valid))
                    print("         | Test  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
                        rmse_test, mae_test))
                    print("         | Recall@10: {0:3.4f}".format(recall_10))
                    print("         | Recall@50: {0:3.4f}".format(recall_50))
                    print("         | Recall@100: {0:3.4f}".format(recall_100))
                    print("         | Recall@200: {0:3.4f}".format(recall_200))
                    print("         | Recall: {0:3.4f}".format(recall))
                    print("         | ndcg@5: {0:3.4f}".format(ndcg_at_five))
                    print("         | ndcg@10: {0:3.4f}".format(ndcg_at_ten))




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