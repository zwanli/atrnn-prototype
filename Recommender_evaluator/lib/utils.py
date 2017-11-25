import csv
import os
def read_ratings(ratings_file):
	"""
	reads the ratings_list.dat file
	:param ratings_file: the path of the ratings_list file, the file is expected to be formated as following: line i has space separated list of item ids rated by user i
	:return: 2d list, the ith row contains a list of relevant items ids to user i
	"""
	ratings_list = []
	with open(ratings_file) as f:
		for line in f:
			ratings = [int(x) for x in line.replace("\n", "").split(" ")[1:] if x!= ""]
			ratings_list.append(ratings)
	return ratings_list


def write_ratings(ratings_list, filename, delimiter=" ", print_line_length = True):
	"""
	writes user matrix to a file, the file will be formated as following: line i has delimiter-separated list of item ids rated by user i
	:param ratings_list: users 2d list, row num = num_users
	:param filename: the path of the users file
	:param delimiter: default: space
	:param print_line_length: if True: the first column of each line will record the line's length
	"""
	if not os.path.exists(os.path.dirname(filename)):
		os.makedirs(os.path.dirname(filename))

	with open(filename, 'w', newline='') as f:
		writer = csv.writer(f, delimiter=delimiter)
		for ratings in ratings_list:
			if print_line_length:
				writer.writerow([len(ratings)]+ratings)
			else:
				writer.writerow(ratings)


def print_list(lst):
	print('[%s]' % ', '.join(map(str, lst)))

