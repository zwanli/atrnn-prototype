import os
import glob
import pandas as pd
import numpy as np
from utils import read_ratings
from utils import write_ratings
from lib.utils import print_list
import csv
import argparse

breaks = range(20, 201, 20)
delimiter = " "


class Evaluator(object):
    def __init__(self, root_path, to_evaluate_folders, delimiter=" "):
        self.users_ratings = read_ratings(os.path.join(root_path, "users.dat"))
        self.root_path = root_path
        self.num_users = len(self.users_ratings)
        self.delimiter = delimiter
        self.num_items = max(max(self.users_ratings)) + 1
        self.to_evaluate_folders = to_evaluate_folders

    # calculate_metrics(hits, 100, recall_breaks, mrr_breaks, ndcg_breaks)
    def calculate_metrics_user(self, hits, num_user_test_positives, recall_breaks, mrr_breaks, ndcg_breaks):
        # Adjust the breaks lists to be 0-based:
        recall_breaks = [i - 1 for i in recall_breaks]
        mrr_breaks = [i - 1 for i in mrr_breaks]
        ndcg_breaks = [i - 1 for i in ndcg_breaks]
        iDCGs = np.cumsum(np.array([1 / np.log2(i + 2) for i in range(len(hits))]))

        # Calculate recall:
        recall = np.cumsum(hits)
        recall_at_breaks = (np.array(recall)[recall_breaks] / float(num_user_test_positives)).tolist()

        # Calculate MRR
        mrrs = [hits[i] / float(i + 1) for i in range(len(hits))]
        for i in range(1, len(mrrs)):
            mrrs[i] = max(mrrs[i], mrrs[i - 1])
        mrrs_at_breaks = np.array(mrrs)[mrr_breaks].tolist()

        # Calculate nDCG
        dcgs = [hits[i] / np.log2(i + 2) for i in range(len(hits))]
        dcgs = np.array(dcgs)
        dcgs = np.cumsum(dcgs) / iDCGs
        ndcgs_at_breaks = dcgs[ndcg_breaks].tolist()
        return recall_at_breaks + mrrs_at_breaks + ndcgs_at_breaks

    def compute_metrics_all_folds(self, base_directory, paths, splits, recall_breaks=[5, 10] + list(range(20, 201, 20)),
                                  mrr_breaks=[10],
                                  ndcg_breaks=[5, 10], folds_num=5, top=200):
        results = np.zeros(shape=(folds_num, self.num_users, len(recall_breaks) + len(mrr_breaks) + len(ndcg_breaks)))
        results_list = []
        results_header = ["Rec@" + str(i) for i in recall_breaks] + ["MRR@" + str(i) for i in mrr_breaks] + [
            "nDCG@" + str(i) for i in ndcg_breaks]
        results_header = ['{:7}'.format('fold')] + ['{:7}'.format(h) for h in results_header]
        results_list.append(results_header)

        for fold in range(folds_num):
            users_with_zero_test = 0
            score_path = "{}/score.npy".format(paths[fold])
            if not os.path.exists(score_path):
                print("Scores file {} is not available".format(score_path))
                return
            scores = np.load(score_path)
            num_users, num_items = scores.shape
            user_test = read_ratings(os.path.join(paths[fold], "test-fold_{}-users.dat".format(fold + 1)))
            results_users_file = os.path.join(paths[fold], "results-users.dat")
            print("Computing {}...\n".format(results_users_file))
            with open(results_users_file, 'w') as f:
                row_header = ['{:7}'.format('user_id')] + results_header
                f.write('{}'.format(' '.join(map(str, row_header)) + '\n'))
                for user in range(num_users):
                    # Get the test positive items
                    user_test_positive = user_test[user]
                    if len(user_test_positive) == 0:
                        users_with_zero_test += 1
                        continue

                    # Get the candidate items
                    candidate_items = np.array(splits[user, fold])

                    # Get the prediction scores for the candidate items
                    scores_u = scores[user, candidate_items]

                    # Identify the top recommendations
                    recommended_items_idx = np.argsort(scores_u)[::-1][0:top]
                    recommended_items_ids = candidate_items[recommended_items_idx]

                    # Identify the hits:
                    hits = [1 if i in user_test_positive else 0 for i in recommended_items_ids]

                    # Calculate the metrics:
                    metrics_values = self.calculate_metrics_user(hits, len(user_test_positive), recall_breaks,
                                                                 mrr_breaks,
                                                                 ndcg_breaks)

                    # Save results in the tensor
                    results[fold, user, :] = metrics_values
                    # Write results to the file
                    f.write('{}'.format(' '.join(map(str, ['{:7d}'.format(user), '{:7d}'.format(fold + 1)] + [
                        "{:7.3f}".format(i) for i in metrics_values])) + '\n'))

                print(" Fold {} Results, users with zero test items = {}: ".format(fold + 1, users_with_zero_test))
                print_list(results_header)
                print_list(['{:7d}'.format(fold + 1)] + ["{:7.3f}".format(i) for i in
                                                         np.average(results[fold], axis=(0)).tolist()])
                results_list.append(['{:7d}'.format(fold + 1)] + ["{:7.3f}".format(i) for i in
                                                                  np.average(results[fold], axis=(0)).tolist()])
        print("Average Results over all folds: ")
        print_list(results_header[1:])
        print_list(["{:7.3f}".format(i) for i in np.average(results, axis=(0, 1)).tolist()])
        results_list.append([('-' * (9 * len(results_header) - 1))])
        results_list.append(
            ['{:7}'.format('avg')] + ["{:7.3f}".format(i) for i in np.average(results, axis=(0, 1)).tolist()])

        np.save(os.path.join(base_directory, "results_matrix"), results)
        # Writing the results to a file:
        with open(os.path.join(base_directory, "evaluation_results.txt"), 'w', newline='') as f:
            for s in results_list:
                f.write('[%s]' % ', '.join(map(str, s)) + '\n')


    def score(self, folder_path):
        score_path = os.path.join(folder_path, "score")
        if os.path.exists(score_path + ".npy"):
            print("Score file [{}] already exists, exiting.".format(score_path + ".npy"))
            return

        u_path = os.path.join(folder_path, "final-U.dat")
        if not os.path.exists(u_path):
            print("U file {} is not found".format(u_path))
            return

        print("Reading U file...{}".format(u_path))
        U = pd.read_csv(u_path, sep=' ', header=None).iloc[:, 0:-1]

        v_path = os.path.join(folder_path, "final-V.dat")
        if not os.path.exists(v_path):
            print("V file {} is not found".format(v_path))
            return

        print("Reading V file...{}".format(v_path))
        V = pd.read_csv(v_path, sep=' ', header=None).iloc[:, 0:-1]

        print("Multiplication...")
        scores = np.dot(U, V.T)

        print("Saving scores file...{}".format(score_path + ".npy"))
        np.save(score_path, scores)

    def score_all(self):
        folds_folders = glob.glob(self.root_path + "/*folds/fold-*")
        folds_folders.sort()
        for f in [f for f in folds_folders]:
            print("Scoring {} ...".format(f))
            self.score(f)

    def compute_metrics_all_methods(self):
        for to_evaluate_folder in self.to_evaluate_folders:
            print("Evaluating {}: ".format(to_evaluate_folder))

            folder = os.path.join(self.root_path, to_evaluate_folder)
            if not os.path.exists(folder):
                print("Folder {} not found".format(folder))
                continue

            print("reading from {}".format(folder))
            splits_cf_path = os.path.join(folder, "splits.npy")

            print("loading {} ...\n".format(splits_cf_path))
            if not os.path.exists(splits_cf_path):
                print("File {} not found".format(splits_cf_path))
                continue
            splits = np.load(splits_cf_path)
            paths = glob.glob(folder + "/fold-*")
            paths.sort()
            self.compute_metrics_all_folds(folder, paths, splits)


def evaluate(data_directory, evaluate, score = False):
    '''

    :param data_directory:
    :param evaluate:
    :param score:
    :return:
    '''
    evaluator = Evaluator(data_directory, to_evaluate_folders=evaluate)
    # 1- score:
    if args.score:
        evaluator.score_all()
    # 2- evaluate:
    evaluator.compute_metrics_all_methods()


if __name__ == '__main__':
    data_folder = "../../data"
    to_evaluate_folders = ['user-based']
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", "-d",
                        help="The folder containing users ratings (not split) users.dat file, expecting the existence of subfolder named as specefied with -evaluate")
    parser.add_argument("--score", "-s", action="store_true", default=False,
                        help="A flag orders the code to calculate the score matrix: U.VT, default: the score is not calculated, assuming the existence of the files [data_directory]/fold-[1-5]/score.npy")
    parser.add_argument("--evaluate", "-e", nargs='+',
                        help="Space separated list of folder names which contain data to be evaluated, each folder should contain the folds folders fold-[1-5], each folder contains two files: final-U.dat and final-V.dat")
    args = parser.parse_args()
    if args.data_directory:
        data_folder = args.data_directory
    if args.evaluate:
        to_evaluate_folders = args.evaluate

    evaluator = Evaluator(data_folder, to_evaluate_folders=args.evaluate)

    # 1- score:
    if args.score:
        evaluator.score_all()

    # 2- evaluate:
    evaluator.compute_metrics_all_methods()

