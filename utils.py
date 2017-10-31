import tensorflow as tf
import os
import sys
import numpy as np
def rounded_predictions(predictions):
    """
    The method rounds up the predictions and returns a prediction matrix containing only 0s and 1s.
    :returns: predictions rounded up matrix
    :rtype: int[][]
    """
    rounded_matrix = predictions.copy()
    n_users = predictions.shape[0]
    for user in range(n_users):
        avg = sum(predictions[user]) / predictions.shape[1]
        low_values_indices = predictions[user, :] < avg
        rounded_matrix[user, :] = 1
        rounded_matrix[user, low_values_indices] = 0
    return rounded_matrix

def static_padding(docs,maxlen):
    lengths = []
    long_docs = 0
    for d in docs:
        if len(d) <= maxlen:
            lengths.append(len(d))
        else:
            long_docs +=1
            lengths.append(maxlen)
    print('Documents longer than {0} tokens: {1}'.format(maxlen,long_docs))
    padded_seq = pad_sequences(docs, maxlen=maxlen, padding='post')
    return padded_seq, lengths

def num_samples(path):
    c = 0
    for record in tf.python_io.tf_record_iterator(path):
        example_proto = tf.train.SequenceExample()
        example_proto.ParseFromString(record)
        c += 1
    return c

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def  convert_to_tfrecords(path, parser,fold, maxlen, split_method, test=False):
    """Converts a dataset to tfrecords."""
    print('Writing', path)
    writer = tf.python_io.TFRecordWriter(path)
    for u_id, v_id, rating, doc in parser.generate_samples(fold, test=test):
        context = tf.train.Features(feature={
            'u': _int64_feature(u_id),
            'v': _int64_feature(v_id),
            'r': _int64_feature(rating),
            'abs_length': _int64_feature(len(doc))
        })
        if rating == 0:
            print (u_id,v_id)
        feature_lists = tf.train.FeatureLists(feature_list={
            "abstract": _int64_feature_list(doc[:maxlen]) })
        sequence_example = tf.train.SequenceExample(
            context=context, feature_lists=feature_lists)
        writer.write(sequence_example.SerializeToString())
    writer.close()
    sys.stdout.flush()


def read_and_decode(filename_queue):
    context_feature = {'u': tf.FixedLenFeature([], tf.int64),
               'v': tf.FixedLenFeature([], tf.int64),
               'r': tf.FixedLenFeature([], tf.int64),
               'abs_length': tf.FixedLenFeature([], tf.int64)}

    sequence_feature={'abstract': tf.FixedLenSequenceFeature([], tf.int64)}

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    context_feature,sequence_feature = tf.parse_single_sequence_example(serialized_example, context_features=context_feature,
    sequence_features=sequence_feature)

    u = tf.cast(context_feature['u'], tf.int32)
    v = tf.cast(context_feature['v'], tf.int32)
    r = tf.cast(context_feature['r'], tf.float32)
    abs_length = tf.cast(context_feature['abs_length'], tf.int32)
    abstract = tf.cast(sequence_feature['abstract'], tf.int32)
    return u, v, r, abstract, abs_length


def _parse_function(sequence_example_proto):
        context_feature = {'u': tf.FixedLenFeature([], tf.int64),
                           'v': tf.FixedLenFeature([], tf.int64),
                           'r': tf.FixedLenFeature([], tf.int64),
                           'abs_length': tf.FixedLenFeature([], tf.int64)}

        sequence_feature = {'abstract': tf.FixedLenSequenceFeature([], tf.int64)}

        # Decode the record read by the reader
        context_feature, sequence_feature = tf.parse_single_sequence_example(sequence_example_proto,
                                                                             context_features=context_feature,
                                                                             sequence_features=sequence_feature)
        u = tf.cast(context_feature['u'], tf.int32)
        v = tf.cast(context_feature['v'], tf.int32)
        r = tf.cast(context_feature['r'], tf.float32)
        abs_length = tf.cast(context_feature['abs_length'], tf.int32)
        abstract = tf.cast(sequence_feature['abstract'], tf.int32)
        return u, v, r, abstract, abs_length


def get_test_ratings_matrix(filename,user_count,paper_count,sess):
    ratings = np.zeros((user_count, paper_count))
    with tf.device("/cpu:0"):
        # Creates a dataset that reads all of the examples from filenames.
        dataset = tf.contrib.data.TFRecordDataset(filename)
        dataset = dataset.map(_parse_function)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        try:
            while True:
                u,v,r,_,_ =sess.run(next_element)
                ratings[u,v]=r
        except Exception as e:
            print(e.message)
            print("Finished reading the test dataset")
    return ratings

def main():
    test_filename = '/home/wanli/data/Extended_ctr/dummy_test_1.tfrecords'
    with tf.Session() as sess:
        get_test_ratings_matrix(test_filename,50,1920,sess)

if __name__ == '__main__':
    main()
