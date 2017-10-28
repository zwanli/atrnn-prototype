import time
import tensorflow as tf
import numpy as np
import argparse
import os,errno
from evaluator import Evaluator
import pickle
from data_parser import DataParser
import utils
from model import Model
from utils import convert_to_tfrecords
from test_cases import evaluate
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/wanli/data/Extended_ctr',
                        help='data directory containing input.txt')
    parser.add_argument("--dataset", "-d", type=str, default='dummy',
                        help="Which dataset to use", choices=['dummy', 'citeulike-a', 'citeulike-t'])
    parser.add_argument('--embedding_dir', type=str, default='/home/wanli/data/glove.6B/',
                        help='GloVe embedding directory containing embeddings file')
    parser.add_argument('--embedding_dim', type=int, default=200,
                        help='dimension of the embeddings', choices=['50', '100', '200', '300'])
    parser.add_argument('--input_encoding', type=str, default=None,
                        help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--split',type=str,default='cold', help='The splitting strategy',choices=['warm','cold'])
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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--max_length', type=int, default=300,
                        help='Maximum document length')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--gpu_mem', type=float, default=0.666,
                        help='%% of gpu memory to be allocated to this process. Default is 66.6%%')
    # parser.add_argument('--decay_rate', type=float, default=0.97,
    #                     help='decay rate for rmsprop')
    # parser.add_argument('--grad_clip', type=float, default=5.,
    #                     help='clip gradients at this value')

    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    predict(args)
    # partial_run(args)


def predict(args):

    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from), " %s must be a directory" % args.init_from
        assert os.path.isfile(
            os.path.join(args.init_from, "config.pkl")), "config.pkl file does not exist in path %s" % args.init_from

        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = pickle.load(f)
        # need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        # for checkme in need_be_same:
        #     assert vars(saved_model_args)[checkme] == vars(args)[
        #         checkme], "Command line argument and saved model disagree on '%s' " % checkme

        # assert os.path.isfile(os.path.join(args.init_from,
        #                                    "words_vocab.pkl")), "words_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        # # open saved vocab/dict and check if vocabs/dicts are compatible
        # with open(os.path.join(args.init_from, 'words_vocab.pkl'), 'rb') as f:
        #     saved_words, saved_vocab = cPickle.load(f)
        # assert saved_words == data_loader.words, "Data and loaded model disagree on word set!"
        # assert saved_vocab == data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"


    # model = Model(args, rating_matrix, embeddings, confidence, train_filename=,test_filename=)
