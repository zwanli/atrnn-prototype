from keras.preprocessing.sequence import pad_sequences

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