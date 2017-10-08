import os
import csv
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import argparse
import ntpath
import sys
import tensorflow as tf
import numpy as np
import datetime
from prettytable import PrettyTable
import pandas as pd
maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True


vocab = {}

unknown_words = {}

filtered_vocabulary = {}

glove_vocab = []
glove_word_to_id = {}
glove_embeddings = []
avg_num_embedd = []

MIN_WORD_FREQUENCY = 5

numbers_freq = 0



def load_embeddings(embed_dir, embed_dim, keep_embed = False, w2v =True):
    # Load pre-trained embeddings
    if w2v:
        f = open(os.path.join(embed_dir, 'w2v_{0}.txt'.format(embed_dim)))
    else:
        f = open(os.path.join(embed_dir, 'glove.6B.{0}d.txt'.format(embed_dim)))
    print("Loading {0} embedding data {1}  ".format(('w2v' if w2v else 'GloVe'),str(f.name)))
    #Count the number of the occurance of numbers in glove embeddings
    num_freq = 0
    global avg_num_embedd
    avg_num_embedd = np.zeros(200)
    first_line = True
    for line in f:
        if first_line:
            first_line = False
            continue
        row = line.split()
        word = row[0]
        # Add the word to the vocabulary list of the pre-trained word embeddings dataset
        glove_vocab.append(word)
        if keep_embed:
            if is_number(word):
                num_freq +=1
                avg_num_embedd += [float(val) for val in row[1:]]
        # Add the embedding of the word
            glove_embeddings.append([float(val) for val in row[1:]])
    f.close()
    #calculate the number embedding as the avg of the numbers that are in glove embeddings
    #It's a naive way to do it!!
    avg_num_embedd = avg_num_embedd / num_freq
    # Create a word to id index
    global glove_word_to_id
    glove_word_to_id = {word: i for i, word in enumerate(glove_vocab)}

    return glove_vocab, glove_word_to_id, glove_embeddings

def add_unkonwn_word(word):
    if word in unknown_words:
        unknown_words[word] += 1
    else:
        unknown_words[word] = 1


def in_glove_vocab(word):
    """
    Check if the word id exists in the vocabulary of the pre-trained embeddings .
        :returns: Word id
        :rtype int
    """
    if word in glove_word_to_id:
        return True
    else:
        add_unkonwn_word(word)
        return False


def add_word(word, vocab):
    word = word.lower()
    if word in vocab:
       vocab[word] += 1
    else:
        vocab[word] = 1
    return word

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def process_word(word):
    '''
    replace unknown words with 'unk' token, replace numbers with '<num>'. Otherwise, return word
    :param word:
    :return:
    '''
    if is_number(word):
        add_word('<num>',filtered_vocabulary)
        global numbers_freq
        numbers_freq += 1
        return '<num>'
    if in_glove_vocab(word):
        add_word(word, filtered_vocabulary)
        return word
    add_word('unk', filtered_vocabulary)
    return 'unk'


def is_less_frequent(word,threshold):
    if is_number(word) or word == 'unk' or vocab[word] > threshold:
        return False
    return True


def process_documents(path, dataset):
    """
     Parses paper raw data
     :return: A tuple of Papers' labels and abstracts, where abstracts are returned as strings
     """

    delimiter = ','
    if dataset == 'citeulike-t':
        delimiter = '\t'
    # read raw data
    print('Reading documents ...')
    with open(path, "r", encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter=delimiter)
        first_line = True
        documents = {}
        row_length = 0
        for line in reader:
            if first_line:
                labels = line
                row_length = len(line)
                first_line = False
                continue
            doc_id = line[0]
            if dataset == 'citeulike-t':
                paper = line[1]
            elif row_length > 2:
                paper = line[1]+' '+line[4]
            sentences = sent_tokenize(paper)
            sentences = [word_tokenize(x) for x in sentences]
            #add the word to the raw data vocabulary
            documents[doc_id] = [add_word(word,vocab) for sentence in sentences for word in sentence]

    # process raw data,
    # - remove less frequent words
    # - replace unknown words with 'unk' token, replace numbers with '<num>'
    print('Processing documents ...')
    c = 0
    filtered_documents={}
    for doc_id in documents:
        filtered_documents[doc_id] =[]
        for word in documents[doc_id]:
            if not is_less_frequent(word, MIN_WORD_FREQUENCY):
                # process the word, and add it to the filtered data vocabulary
                filtered_documents[doc_id].append(process_word(word))
            else:
                c +=1
    print('Words with frequency less than {0}: {1}'.format(MIN_WORD_FREQUENCY, c))

    # write the processd documents
    parent_dir, filename = ntpath.split(path)
    out_path = os.path.join(parent_dir,'processed-'+filename)
    print('Writing processed file {0} ...'.format(out_path))
    with open(out_path, "w", encoding='utf-8', errors='ignore') as f:
        writer = csv.writer(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['doc_id','title+abstract'])
        for doc_id in filtered_documents:
            writer.writerow([doc_id]+[' '.join(filtered_documents[doc_id])])
    print('Finished Writing.'.format(out_path))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def save_embeddings(filename):
    if bool(filtered_vocabulary):
        print('Writing embeddings data {0} ...'.format(filename))
        writer = tf.python_io.TFRecordWriter(filename)
        for word in filtered_vocabulary:
            # Create a feature
            if word == '<num>':
                word_id = -1
                word_embed = avg_num_embedd
            else:
                word_id = glove_word_to_id[word]
                word_embed = glove_embeddings[word_id]

            feature = {'word': _bytes_feature(tf.compat.as_bytes(word)),
                       'glove_id': _int64_feature(word_id),
                       'embed': tf.train.Feature(float_list=tf.train.FloatList(value=word_embed))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        writer.close()
        sys.stdout.flush()




def process_features(path,paper_count ):
    null_token = 'NaN'
    now = datetime.datetime.now()

    clean_file_path = path +'.cleaned'
    if os.path.exists(clean_file_path):
        with open(path, "r", encoding='utf-8', errors='ignore') as infile:
            reader = csv.reader(infile, delimiter='\t')
            i = 0
            first_line = True

            with open(clean_file_path, 'w', newline='') as outfile:
                writer = csv.writer(outfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                for line in reader:
                    if first_line:
                        row_length = len(line)
                        first_line = False
                        writer.writerow(line)
                        continue
                    if len(line) > row_length:
                        line[row_length] = ' '.join(line[row_length-1:]).replace('\t', ' ')
                        line = line[:row_length]
                    paper_id = line[0]
                    if int(paper_id) != i:
                        for _ in range(int(paper_id) - i):
                            empty_row = [str(i)]
                            empty_row.extend([null_token] * (row_length-1))
                            # empty_row = '\t'.join(empty_row)
                            writer.writerow(empty_row)
                            i += 1
                    for j, _ in enumerate(line):
                        if line[j] == '\\N':
                            line[j] = null_token
                    writer.writerow(line)
                    i += 1
                if i != paper_count:
                    for _ in range(int(paper_count) - i):
                        empty_row = [str(i)]
                        empty_row.extend([null_token] * (row_length - 1))
                        # empty_row = '\t'.join(empty_row)
                        writer.writerow(empty_row)
                        i += 1





    # Month converter
    months = ['apr','aug', 'dec' ,'feb', 'jan' ,'jul' ,'jun' ,'mar' ,'may', 'nov', 'oct', 'sep']
    month_convert_func = lambda x: x if x in months else null_token

    def number_convert_func (x):
        if is_number(x):
            return x
        else:
            print(x)
            return -1

    labels = ['doc_id', 'citeulike_id', 'type', 'pages', 'year']
    labels_dtype = {'doc_id': np.int32, 'citeulike_id': np.int32, 'type': str, 'pages': np.int32}
    convert_func= {'pages': number_convert_func, 'doc_id': number_convert_func,
                   'citeulike_id': number_convert_func}
    # labels = ['doc_id', 'citeulike_id', 'type', 'journal', 'booktitle', 'series', 'pages', 'year', 'month', 'address']
    # labels_dtype = {'doc_id': np.int32, 'citeulike_id': np.int32, 'type': str, 'journal': str, 'booktitle': str,
    #                 'series': str,
    #                 'pages': np.int32, 'month': str, 'address': str}
    # convert_func = {'month': month_convert_func, 'pages': number_convert_func, 'doc_id': number_convert_func,
    #                 'citeulike_id': number_convert_func}

    df = pd.read_table(clean_file_path, delimiter='\t', index_col = 'doc_id', usecols=labels,dtype=labels_dtype,
                         na_values='\\N',na_filter=False,
                         converters=convert_func)

    # Filter values with frequency less than min_freq
    def filter(df, tofilter_list, min_freq):
        for col in tofilter_list:
            to_keep = df[col].value_counts().reset_index(name="count").query("count > %d" % min_freq)["index"]
            to_keep = to_keep.values.tolist()
            df[col] = [x if x in to_keep else 'NaN' for x in df[col]]
        return df

    tofilter_list = []
    df = filter(df, tofilter_list, 2)

    # Convert catigorical feature into one-hot encoding
    def dummmy_df(df, todummy_list):
        for x in todummy_list:
            dummies = pd.get_dummies(df[x], prefix=x, dummy_na=True)
            df = df.drop(x, 1)
            df = pd.concat([df, dummies], axis=1)
        return df

    todummy_list = ['type']
    df = dummmy_df(df, todummy_list)

    # # features_matrix = np.asarray(feature_vec)
    # # df = pd.DataFrame(features_matrix,columns=labels)
    # # filter months
    # df.month[~df['month'].isin(months)] = null_token
    # print ('Uninque \'month\' values %d ' % df.month.nunique())
    #
    #
    #
    return labels, labels

def feature_index(feature, labels):
    for index, label in enumerate(labels):
        if label == feature:
            return index
    return -1

def get_features_distribution(feature_labels, feature_matrix):
    # if papers_presentation == 'attributes':
    #     return feature_matrix
    if feature_labels is None:
        raise
    # A dict of tuples. First element of the tuple is the unique values of a the chosen attribute, the second is
    # their frequencies
    uniqe_freq = {}
    # A dict that has the id of the missing token '\\N' for each attribute
    missing_value_id = {}

    #Number of papers that have a value for each attribute
    att_count = {}
    t = PrettyTable(['feature','# of unique values', '# of papers that have value','# of papers that have missing values'])
    for feature in feature_labels:
        uniqe_freq[feature] = np.unique(feature_matrix[:, feature_index(feature, feature_labels)], return_counts=True)
        missing_value_id[feature] = np.where(uniqe_freq[feature][0] == '\\N')

        att_count[feature] = np.sum(uniqe_freq[feature][1]) \
                             - (uniqe_freq[feature][1][missing_value_id[feature]]
                                if len(missing_value_id[feature][0]) != 0 else 0)

        t.add_row([feature,len(uniqe_freq[feature][0]),att_count[feature],(uniqe_freq[feature][1][missing_value_id[feature]]
                                if missing_value_id[feature][0] is not None else 0)])
    print (t)

    att_per_paper = {}
    for paper in feature_matrix:
        att_per_paper[paper[0]] = len(np.where(paper != '\\N')[0])

    # plt.hist(list(att_per_paper.values()))
    # plt.title("Attribute per paper")
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    #
    # #fig = plt.gcf()
    # plt.savefig('{0}attribute-per-paper-histogram_{1}'.format(dataset_folder+'/',dataset))
    # print('')
    # for feature in feature_labels:
    #     print('Number of unique {0} {1}, frequencies {2}'.format(feature,len(uniqe_freq[feature][0]), uniqe_freq[feature][1]))
    #     print('', end="")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/wanli/data/Extended_ctr',
                        help='data directory containing input.txt')
    parser.add_argument("--dataset", "-d", type=str, default='dummy',
                        help="Which dataset to use", choices=['dummy', 'citeulike-a', 'citeulike-t'])
    parser.add_argument('--embedding_dir', type=str, default='/home/wanli/data/cbow_w2v',
                        help='GloVe embedding directory containing embeddings file')
    parser.add_argument('--embedding_dim', type=int, default=200,
                        help='dimension of the embeddings', choices=['50', '100', '200', '300'])
    args = parser.parse_args()

    if args.dataset == 'citeulike-a':
        dataset_folder = args.data_dir + '/citeulike_a_extended'
    elif args.dataset == 'citeulike-t':
        dataset_folder = args.data_dir + '/citeulike_t_extended'
    elif args.dataset == 'dummy':
        dataset_folder = args.data_dir + '/dummy'
    else:
        print("Warning: Given dataset not known, setting to dummy")
        dataset_folder = args.data_dir + '/citeulike_a_extended'

    raw_data_path = os.path.join(dataset_folder, 'raw-data.csv')
    if not os.path.exists(raw_data_path):
        print("File {0} doesn't exist".format(raw_data_path))
        raise

    use_w2v = False
    _, embed_dir= ntpath.split(args.embedding_dir)
    if (embed_dir == 'cbow_w2v'):
        use_w2v = True
    load_embeddings(args.embedding_dir,args.embedding_dim,keep_embed=True,w2v=use_w2v)

    process_documents(raw_data_path,args.dataset)

    embeddings_path = os.path.join(dataset_folder, '{0}-embeddings-{1}-{2}.tfrecord'.
                                   format(args.dataset,args.embedding_dim,'w2v' if use_w2v else 'glove'))
    save_embeddings(embeddings_path)

    print('Raw data vocabulary size {0}, frequency {1}'.format(len(vocab), sum(vocab.values())))
    print('Processed data vocabulary size {0}, frequency {1}'.format(len(filtered_vocabulary),
                                                                     sum(filtered_vocabulary.values())))
    print('# of unique unknown words {0}, frequency of unknown words {1}'
          .format(len(unknown_words),sum(unknown_words.values())))
    print('Numbers frequency {0}'.format(numbers_freq))

    # #
    # paper_count={'dummy': 1929, 'citeulike-a': 16980, 'citeulike-t': 25976 }
    # features_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset_folder, 'paper_info.csv')
    #
    # labels, raw_features = process_features(features_path, paper_count=paper_count[args.dataset])
    # # # get_features_distribution(labels, raw_features)

    a =1
if __name__ == '__main__':
    main()
