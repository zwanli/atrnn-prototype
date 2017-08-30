import numpy as np
import tensorflow as tf
import math
from evaluator import Evaluator
from model import get_inputs
from model import  Model
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


def evaluate(filename,rating_matrix,args,embeddings,test_writer):
    def construct_feed(bi_hid_fw, bi_hid_bw):
        return {model.init_state_fw: bi_hid_fw, model.init_state_bw: bi_hid_bw}

    with tf.Graph().as_default() as g:
        saver = tf.train.Saver()
        model = Model(args,rating_matrix,embeddings,filename)
        with tf.Session() as sess:
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

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                     start=True))
                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0

                test_bi_fw = sess.run(model.init_state_fw)
                test_bi_bw = sess.run(model.init_state_bw)
                init_state = sess.run(model.initial_state)
                feed_dict = construct_feed(test_bi_fw, test_bi_bw)
                rmse_test, mae_test, summary_str = sess.run(
                    [model.RMSE, model.MAE, model.summary_op], feed_dict=feed_dict)

                test_writer.add_summary(summary_str, step)

                prediction_matrix = np.matmul(U, V.T)
                prediction_matrix = np.add(prediction_matrix, np.reshape(U_b, [-1, 1]))
                prediction_matrix = np.add(prediction_matrix, V_b)
                rounded_predictions = utils.rounded_predictions(prediction_matrix)

                while step < num_iter and not coord.should_stop():
                    sess.run()
                    step += 1

                evaluator.load_top_recommendations_2(200, prediction_matrix, test_ratings)
                recall_10 = evaluator.recall_at_x(10, prediction_matrix, parser.ratings, rounded_predictions)
                recall_50 = evaluator.recall_at_x(50, prediction_matrix, parser.ratings, rounded_predictions)
                recall_100 = evaluator.recall_at_x(100, prediction_matrix, parser.ratings, rounded_predictions)
                recall_200 = evaluator.recall_at_x(200, prediction_matrix, parser.ratings, rounded_predictions)
                recall = evaluator.calculate_recall(ratings=parser.ratings, predictions=rounded_predictions)
                ndcg_at_five = evaluator.calculate_ndcg(5, rounded_predictions)
                ndcg_at_ten = evaluator.calculate_ndcg(10, rounded_predictions)

                feed = {model.recall: recall, model.recall_10: recall_10, model.recall_50: recall_50,
                        model.recall_100: recall_100, model.recall_200: recall_200,
                        model.ndcg_5: ndcg_at_five, model.ndcg_10: ndcg_at_ten}
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
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)