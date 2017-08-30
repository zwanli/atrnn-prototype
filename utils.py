from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import os
import sys
def rounded_predictions(predictions):
    """
    The method rounds up the predictions and returns a prediction matrix containing only 0s and 1s.
    :returns: predictions rounded up matrix
    :rtype: int[][]
    """
    n_users = predictions.shape[0]
    for user in range(n_users):
        avg = sum(predictions[user]) / predictions.shape[1]
        low_values_indices = predictions[user, :] < avg
        predictions[user, :] = 1
        predictions[user, low_values_indices] = 0
    return predictions

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


def convert_to_tfrecords(dir, parser, name,folds,maxlen, validation=False, test=False):
    """Converts a dataset to tfrecords."""
    for fold in range(folds):
        filename = os.path.join(dir, name + '{0}_{1}.tfrecords'.format("_test" if test else "_train",fold))
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for u_id, v_id, rating, doc in parser.generate_samples(1,fold, validation=validation, test=test):
            context = tf.train.Features(feature={
                'u': _int64_feature(u_id),
                'v': _int64_feature(v_id),
                'r': _int64_feature(rating),
                'abs_length': _int64_feature(len(doc))
            })
            feature_lists = tf.train.FeatureLists(feature_list={
                "abstract": _int64_feature_list(doc[:maxlen]) })
            sequence_example = tf.train.SequenceExample(
                context=context, feature_lists=feature_lists)
            writer.write(sequence_example.SerializeToString())
            # example = tf.train.Example(feature))1
            # writer.write(example.SerializeToString())
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
        r = tf.cast(context_feature['r'], tf.int32)
        abs_length = tf.cast(context_feature['abs_length'], tf.int32)
        abstract = tf.cast(sequence_feature['abstract'], tf.int32)
        return u, v, r, abstract, abs_length


def read_tfrecoed_as_dataset(filenames):

    # Creates a dataset that reads all of the examples from filenames.
    dataset = tf.contrib.data.TFRecordDataset(filenames)

    # Repeat the input indefinitely.
    dataset = dataset.repeat()
    # Parse the record into tensors.
    dataset = dataset.map(_parse_function)
    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=10000)
    # Generate batches
    #dataset = dataset.batch(128)

    # iterator = dataset.make_initializable_iterator()
    print(dataset.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
    print(dataset.output_shapes)  # ==> "(10, ((), (100,)))"

    dataset = dataset.padded_batch(128, padded_shapes=((),(),(),[None],()))
    # Create a one-shot iterator
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    # with tf.Session() as sess:
    #     for i in range(100):
    #         record = sess.run(next_element)
    return next_element

# filename = '/home/wanli/data/Extended_ctr/dummy/warm-start-user/dummy_train_0.tfrecords'
# next_element = read_tfrecoed_as_dataset(filename)
# with tf.Session() as sess:
#     #while True:
#       # Run 200 steps using the training dataset. Note that the training dataset is
#       # infinite, and we resume from where we left off in the previous `while` loop
#       # iteration.
#       for _ in range(20):
#         _,_,_,doc,lengths = (sess.run(next_element))
#         print ('%d , %d , %d , %d ' % (len(doc[0]),lengths[0],len(doc[1]),lengths[1]))
#       # Run one pass over the validation dataset.
#
