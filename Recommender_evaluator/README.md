Albert-Ludwigs-Universitaet Freiburg
Database and Information Systems group
Georges-koehler-Allee 51, 79110 Freiburg, Germany
Anas Alzogbi
email: alzoghba@informatik.uni-freiburg.de


This code provides the functionality of: (a) splitting a given users' ratings based on several split methods (in-matrix-item based, in-matrix-user based, and outof-matrix item based) into training and test datasets;
and (b) evaluating recommendation predictions over the test dataset based on several evaluation metrics (recall, ndcg, mrr)

### Rquirements:
python 3.5
pandas
numpy


### Data Split:
python3  python_code/lib/split.py
With arguments:
\-d [DATA_DIRECTORY where items.dat and users.dat files are located]
\-s [SPLIT_STRATEGY, one of the following: user-based, in-matrix-item or outof-matrix-item]

#### Expected results:
- New folder: [DATA_DIRECTORY]/[SPLIT_STRATEGY]_folds
- 5 folder, the 5 folds: [DATA_DIRECTORY]/[SPLIT_STRATEGY]_folds/fold-[1-5]
- Each fold folder contains 4 files: train-fold_[1-5]-users.dat, test-fold_[1-5]-users.dat, train-fold_[1-5]-items.dat, test-fold_[1-5]-items.dat
- File contains the splits matrix for test data: [DATA_DIRECTORY]/splits.npy:
 it saves an ndarray of shape:(num_users, num_folds, list of test ids). The list of test ids contains both user_positive_ids and user_fold_unrated_items
- File contains the split statistics: [DATA_DIRECTORY]/[SPLIT_STRATEGY]_folds/stats.txt



### Results evaluation:
python python_code/lib/evaluator.py
With arguments:
\-d [DATA_DIRECTORY The folder containing users ratings (not split) users.dat file, expecting the existence of subfolder named as specefied with -evaluate]
\-s [A flag orders the code to calculate the score matrix: U.VT, default: the score is not calculated, assuming the existence of the files [data_directory]/fold-[1-5]/score.npy, default False]
\-e Space separated list of SPLIT_STRATEGY (folder names which contain data to be evaluated), each folder should contain the folds folders fold-[1-5], each folder contains two files: final-U.dat and final-V.dat

Invokation example: python python_code/lib/evaluator.py -d data -s -e user-based_folds in-matrix-item_folds

#### Expected results:
- if the argument -s is provided: a score file will be saved for each fold under:
[DATA_DIRECTORY]/[SPLIT_STRATEGY]_folds/fold-[1-5]/score.npy

- All users results file for each fold under (to investigate the results at the user level):
[DATA_DIRECTORY]/[SPLIT_STRATEGY]_folds/fold-[1-5]/results-users.dat

- The average results for all folds over all users (provides the overview over the results):
[DATA_DIRECTORY]/[SPLIT_STRATEGY]_folds/evaluation_results.txt

- The numpy nd array results matrix dimension: folds X users X metrics (if further statistics over the results are needed):
[DATA_DIRECTORY]/[SPLIT_STRATEGY]_folds/results_matrix.npy