import os
import random
import numpy as np
from utils import read_ratings
from utils import write_ratings
import argparse


def random_divide(lst, k):
	"""
	Randomly splits the items of a given list into k lists
	:param lst: the input list
	:param k: the number of resulting lists
	:return: 2d list
	"""
	res = []
	random.shuffle(lst)
	partition_size = len(lst) // k
	for fold in range(k):
		if fold == k - 1:
			res.append(lst[fold * partition_size:len(lst)])
		else:
			res.append(lst[fold * partition_size:fold * partition_size + partition_size])
	return res


class Splitter(object):
	def __init__(self, root_path, split_method="in-matrix-item", delimiter=" "):
		self.users_ratings = read_ratings(os.path.join(root_path, "users.dat"))
		self.root_path = root_path
		self.num_users = len(self.users_ratings)
		self.delimiter = delimiter
		self.split_method = split_method
		self.num_items = max(max(self.users_ratings)) + 1

		self.out_folder = os.path.join(self.root_path, self.split_method + "_folds")
		if not os.path.exists(self.out_folder):
			os.makedirs(self.out_folder)

		# List for split stats, the list is written to a file in the split() method:
		self.stats_list = []
		stats_header = ['{:20}'.format("Stats_of"), '{:4}'.format('fold'),'{:4}'.format('min'), '{:4}'.format('max'),'{:6}'.format('avg'),'{:3}'.format('std')]
		self.stats_list.append(stats_header)

	def items_mat_from_users_ratings(self, users_ratings):
		"""
		construct item matrix from user matrix
		:return: 2d list, row num = num_items, the ith row contains all user ids who rated item i
		"""
		items_mat = [[] for _ in range(self.num_items)]
		num_users = len(users_ratings)
		for user in range(num_users):
			item_ids = users_ratings[user]
			for item in item_ids:
				items_mat[item].append(user)
		for item in range(self.num_items):
			items_mat[item].sort()
			return items_mat

	def users_mat_from_items(self, items):
		"""
		Construct user matrix from item matrix
		:param items: 2d list, row num = num_item, the ith row contains all user ids who rated item i
		:return: 2d list, row num = num_user, the ith row contains item ids which rated by user i
		"""
		users = [[] for _ in range(self.num_users)]
		num_items = len(items)
		for i in range(num_items):
			user_ids = items[i]
			for user in user_ids:
				users[user].append(i)
		for user in range(self.num_users):
			users[user].sort()
		return users

	def read_user_and_write_item(self, item_path):
		"""
		Generates items matrix and writes it to a file
		:param item_path: the path of the itemss file, the file will be formated as following: line i has space separated list of user ids who rated item i
		"""
		item = self.items_mat_from_users_ratings(self.num_items)
		write_ratings(ratings_list=item, filename=item_path)

	def calculate_split_stats(self, users_train, users_test, items_train, items_test, fold):
		users_train_stats = [len(i) for i in users_train]
		self.stats_list.append(['{:20}'.format("pos_in_train_per_usr"), '{:4d}'.format(fold + 1),
								'{:4d}'.format(min(users_train_stats)), '{:4d}'.format(max(users_train_stats)),
								'{:6.1f}'.format(np.mean(users_train_stats)),
								'{:3.1f}'.format(np.std(users_train_stats))])
		users_test_stats = [len(i) for i in users_test]
		self.stats_list.append(['{:20}'.format("pos_in_test_per_usr"), '{:4d}'.format(fold + 1),
								'{:4d}'.format(min(users_test_stats)), '{:4d}'.format(max(users_test_stats)),
								'{:6.1f}'.format(np.mean(users_test_stats)),
								'{:3.1f}'.format(np.std(users_test_stats))])
		items_train_stats = [len(i) for i in items_train]
		self.stats_list.append(['{:20}'.format("pos_in_train_per_itm"), '{:4d}'.format(fold + 1),
								'{:4d}'.format(min(items_train_stats)), '{:4d}'.format(max(items_train_stats)),
								'{:6.1f}'.format(np.mean(items_train_stats)),
								'{:3.1f}'.format(np.std(items_train_stats))])
		items_test_stats = [len(i) for i in items_test]
		self.stats_list.append(['{:20}'.format("pos_in_test_per_itm"), '{:4d}'.format(fold + 1),
								'{:4d}'.format(min(items_test_stats)), '{:4d}'.format(max(items_test_stats)),
								'{:6.1f}'.format(np.mean(items_test_stats)),
								'{:3.1f}'.format(np.std(items_test_stats))])

	def create_all_folds_test_split_matrix(self, folds_num=5):
		"""
		Creates the splits matrix for test data, the result after invoking this method is a
		single file that saves an ndarray of shape:(num_users, num_folds, list of test ids).
		The list of test ids contains both user_positive_ids and user_fold_unrated_items
		:param folds_num: the number of folds, default 5
		:return: None
		"""
		num_users = len(self.users_ratings)
		print("Number of items: {}".format(self.num_items))
		splits = [[[] for _ in range(folds_num)] for _ in range(num_users)]
		items_ids = set(range(self.num_items))
		for user in range(num_users):
			user_items = self.users_ratings[user]
			unrated_items = list(items_ids - set(user_items))
			splits[user] = random_divide(unrated_items, folds_num)
			if user % 500 == 0:
				print("user_{}".format(user))
		for fold in range(folds_num):
			print("Calculating fold_{}".format(fold + 1))
			rated_items_test_fold = read_ratings(os.path.join(self.out_folder, "fold-{}".format(fold + 1), "test-fold_{}-users.dat".format(fold + 1)))
			for user in range(num_users):
				splits[user][fold].extend(rated_items_test_fold[user])
				splits[user][fold].sort()
		splits = np.array(splits)

		# Calculate statistics:
		for fold in range(folds_num):
			users_test_stats = [len(i) for i in splits[:,fold]]
			self.stats_list.append( ['{:20}'.format("item_in_test_per_usr"), '{:4d}'.format(fold+1), '{:4d}'.format(min(users_test_stats)),
							'{:4d}'.format(max(users_test_stats)), '{:6.1f}'.format(np.mean(users_test_stats)), '{:3.1f}'.format(np.std(users_test_stats))])

		print("Saving the splits ndarray to: {}".format("splits.cf.dat"))
		np.save(os.path.join(self.out_folder, "splits"), splits)

	def cf_split(self, folds_num=5):
		"""
		Splits the rating matrix following the in-matrix method defined in CTR, the result after invoking this method is two
		 files for each fold (cf-train-fold_id-users.dat and cf-train-fold_id-users.dat): both files have the same format,
		 as following: line i has delimiter-separated list of item ids rated by user i
		:param root_path: the path to the root folder which contains the data files
		:param folds_num: the number of folds, default 5
		:return: None
		"""
		items_mat = self.items_mat_from_users_ratings(self.users_ratings)
		train = [[[] for _ in range(self.num_items)] for _ in range(folds_num)]
		test = [[[] for _ in range(self.num_items)] for _ in range(folds_num)]
		for item in range(self.num_items):
			if item % 1000 == 0:
				print("doc_{}".format(item))
			user_ids = np.array(items_mat[item])
			n = len(user_ids)
			if n >= folds_num:
				idx = list(range(n))
				user_ids_folds = random_divide(idx, folds_num)
				for fold in range(folds_num):
					test_idx = user_ids_folds[fold]
					train_idx = [id for id in idx if id not in test_idx]
					train[fold][item].extend(user_ids[train_idx].tolist())
					test[fold][item].extend(user_ids[test_idx].tolist())
			else:
				for fold in range(folds_num):
					train[fold][item].extend(user_ids.tolist())
					test[fold][item].extend([])
		for fold in range(folds_num):
			items_train = train[fold]
			print(len(items_train))
			users_train = self.users_mat_from_items(items_train)
			for u in users_train:
				if len(u) == 0:
					print("some users contains 0 training items, split again again!")
					raise Exception("Split_Error!")
			write_ratings(users_train, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1),
															 "train-fold_{}-users.dat".format(fold + 1)),
						  delimiter=self.delimiter)
			write_ratings(items_train, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1),
															 "train-fold_{}-items.dat".format(fold + 1)),
						  delimiter=self.delimiter)

			items_test = test[fold]
			users_test = self.users_mat_from_items(items_test)

			# Storing the fold test items for all users
			write_ratings(users_test, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1),
															"test-fold_{}-users.dat".format(fold + 1)),  delimiter=self.delimiter)
			write_ratings(items_test, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1),
															"test-fold_{}-items.dat".format(fold + 1)),  delimiter=self.delimiter)
			# Calculate statistics:
			self.calculate_split_stats(users_train, users_test, items_train, items_test, fold)




	def out_of_matrix_split(self, folds_num=5):

		"""
		Splits the rating matrix following the out-of-matrix method defined in CTR, the result after invoking this method is two
		files for each fold (out_of-train-fold_id-users.dat and out_of-train-fold_id-users.dat): both files have the same format,
		as following: line i has delimiter-separated list of item ids rated by user i
		:param root_path: the path to the root folder which contains the data files
		:param folds_num: the number of folds, default 5
		:return: None
		"""
		items_ids = list(range(self.num_items))

		random.shuffle(items_ids)
		partition_size = len(items_ids) // folds_num
		for fold in range(folds_num):
			items_train_ids = set()
			items_test_ids = set()
			if fold == folds_num - 1:
				items_test_ids = set(items_ids[fold * partition_size:len(items_ids)])
			else:
				items_test_ids = set(items_ids[fold * partition_size:fold * partition_size + partition_size])
			items_train_ids = set(items_ids) - items_test_ids

			users_train = []
			users_test = []

			for user_ratings in self.users_ratings:
				tr_ratings = list(items_train_ids.intersection(user_ratings))
				if len(tr_ratings)==0:
					print("some users contains 0 training items, split again again!")
					raise Exception("Split_Error!")
				tes_ratings = list(items_test_ids.intersection(user_ratings))
				tr_ratings.sort()
				tes_ratings.sort()

				users_train.append(tr_ratings)
				users_test.append(tes_ratings)

			write_ratings(users_train, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "train-fold_{}-users.dat".format(fold + 1)), delimiter=self.delimiter)
			write_ratings(users_test,  filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "test-fold_{}-users.dat".format(fold + 1)), delimiter=self.delimiter)

			items_train = self.items_mat_from_users_ratings(users_train)
			write_ratings(items_train, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "train-fold_{}-items.dat".format(fold + 1)), delimiter=self.delimiter)
			items_test = self.items_mat_from_users_ratings(users_test)
			write_ratings(items_test, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "test-fold_{}-items.dat".format(fold + 1)), delimiter=self.delimiter)

			# Saving left out items ids:
			items_test_lst = list(items_test)
			items_test_lst.sort()
			write_ratings(items_test_lst, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "heldout-set-fold_{}-items.dat".format(fold + 1)), delimiter=self.delimiter, print_line_length=False)

			# Calculate statistics:
			self.calculate_split_stats(users_train, users_test, items_train, items_test, fold)

	def user_based_split(self, folds_num=5):
		"""
		Splits the rating matrix following the user-based method, the result after invoking this method is two
		 files for each fold (cf-train-fold_id-users.dat and cf-train-fold_id-users.dat): both files have the same format,
		 as following: line i has delimiter-separated list of item ids rated by user i
		:param root_path: the path to the root folder which contains the data files
		:param folds_num: the number of folds, default 5
		:return: None
		"""
		train = [[[] for _ in range(self.num_users)] for _ in range(folds_num)]
		test = [[[] for _ in range(self.num_users)] for _ in range(folds_num)]
		for user in range(self.num_users):
			if user % 1000 == 0:
				print("user_{}".format(user))
			items_ids = np.array(self.users_ratings[user])
			n = len(items_ids)
			if n >= folds_num:
				idx = list(range(n))
				item_ids_folds = random_divide(idx, folds_num)
				for fold in range(folds_num):
					test_idx = item_ids_folds[fold]
					train_idx = [id for id in idx if id not in test_idx]
					train[fold][user].extend(items_ids[train_idx].tolist())
					test[fold][user].extend(items_ids[test_idx].tolist())
			else:
				for fold in range(folds_num):
					train[fold][user].extend(items_ids.tolist())
					test[fold][user].extend([])

		for fold in range(folds_num):
			users_train = train[fold]
			items_train = self.items_mat_from_users_ratings(users_train)
			for u in users_train:
				if len(u) == 0:
					print("some users contains 0 training items, split again again!")
					raise Exception("Split_Error!")
			write_ratings(users_train, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "train-fold_{}-users.dat".format(fold + 1)), delimiter=self.delimiter)
			write_ratings(items_train, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "train-fold_{}-items.dat".format(fold + 1)), delimiter=self.delimiter)

			users_test = test[fold]
			items_test = self.items_mat_from_users_ratings(users_test)

			# Storing the fold test items for all users
			write_ratings(users_test, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "test-fold_{}-users.dat".format(fold + 1)), delimiter=self.delimiter)
			write_ratings(items_test, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "test-fold_{}-items.dat".format(fold + 1)), delimiter=self.delimiter)

			# Calculate statistics:
			self.calculate_split_stats(users_train, users_test, items_train, items_test, fold)


	def split(self):
		# Calculating and saving the split matrix
		if self.split_method == "user-based":
			self.user_based_split()
			self.create_all_folds_test_split_matrix()
		if self.split_method == "in-matrix-item":
			self.cf_split()
			self.create_all_folds_test_split_matrix()
		if self.split_method == "outof-matrix-item":
			self.out_of_matrix_split()
			self.create_all_folds_test_split_matrix()

		# Write split statistics:
		if len(self.stats_list)>1:
			with open(os.path.join(self.out_folder, 'stats.txt'), 'w') as f:
				for s in self.stats_list:
					f.write('{}'.format('  '.join(map(str, s)) + '\n'))

if __name__ == '__main__':
	data_folder = "../../data"
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_directory", "-d",
						help="The folder containing data, ")
	parser.add_argument("--split", "-s", choices=['user-based', 'in-matrix-item', 'outof-matrix-item'],
						help="The split strategy: uer-based, splits the ratings of each user in train/test; in-matrix-item: CTR in-matrix, outof-matrix-item: CTR out of matrix")
	args = parser.parse_args()
	if args.data_directory:
		data_folder = args.data_directory
	if args.split:
		splitter = Splitter(data_folder, split_method=args.split)
		splitter.split()
