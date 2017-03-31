# _*_coding:utf-8_*_

from log_tool import generate_logger

logger = generate_logger(__file__)

from word2id import Word2Id

import chainer
from chainer import optimizers, serializers, cuda
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L

import numpy as np
import argparse
from decimal import *
import os.path
import math
import dill
import cPickle as pickle

from tqdm import tqdm

from progressbar import ProgressBar, Percentage, Bar, ETA

xp = np

VOCAB_SIZE = 100
HIDDEN_SIZE = 50
FEATURE_SIZE = 50
EPOCH = 10
DROPOUT_RATIO = 0.5
BATCH_SIZE = 256

CACHE_PATH = "./cache/word2id_dict.pkl"
MODEL_PATH = "./serialize/seq2seq_epoch{}.model"
OPT_PATH = "./serialize/seq2seq_epoch{}.opt"

getcontext().prec = 7

START_TOKEN = "BOL"
END_TOKEN = "NEWLINE"


def main(args):
    global xp

    logger.debug("Load word2id dictionary")

    with open("CACHE_PATH", "rb") as f:
        w2i = dill.load(f)

    logger.debug("Load seq2seq model and optimizer")

    serializers.load_hdf5(MODEL_PATH.format(10), model)
    serializers.load_hdf5(OPT_PATH.format(10), optimizer)




if __name__ == "__main__":
    try:
        main(getArgs())
    except Exception as e:
        logger.exception(e.args)
