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
from numpy import * 
import argparse
from decimal import *
import os.path
import math
import dill
import cPickle as pickle

from tqdm import tqdm

xp = np

VOCAB_SIZE = 100
HIDDEN_SIZE = 50
batchsize = 256 

getcontext().prec = 7

CACHE_PATH = "./cache/word2id_dict.pkl"
MODEL_PATH = "./serialize/seq2seq_epoch{}.model"
OPT_PATH = "./serialize/seq2seq_epoch{}.opt"

def getArgs():
    """
    コマンド引数をパースします
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-c", "--correct",
        dest="correct_corpus",
        type=argparse.FileType("r"),
        help="correct corpus file",
        required=True,
    )

    parser.add_argument(
        "-w", "--wrong",
        dest="wrong_corpus",
        type=argparse.FileType("r"),
        help="wrong corpus file",
        required=True,
    )

    parser.add_argument(
        "--gpu",
        dest="gpu",
        type=int,
        default=0,
    )

    return parser.parse_args()


class LSTM(Chain):
    def __init__(self):
        super(LSTM, self).__init__(
            embed=L.EmbedID(VOCAB_SIZE, HIDDEN_SIZE),  # word embedding
            mid=L.LSTM(HIDDEN_SIZE, HIDDEN_SIZE),  # the first LSTM layer
            out=L.Linear(HIDDEN_SIZE, VOCAB_SIZE),  # the feed-forward output layer
        )

    def initialize(self):
        self.mid.reset_state()

    def __call__(self, cur_word):
        # Given the current word ID, predict the next word.
        x = self.embed(cur_word)
        h = self.mid(x)
        y = self.out(h)
        return y


def compute_loss(x_list):
    # 損失関数
    loss = 0

    # perm = np.random.permutation(len(train_id_lists)-batchsize-1)

    for i in six.moves.range(len(x_list)-1):
        cur_word_vec = xp.array([x_list[i]])
        next_word_vec = xp.array([x_list[i+1]])
        loss += model(cur_word_vec, next_word_vec)
    print("loss:{}".format(loss.data))
    return loss


def main():
    global xp
    args = getArgs()

    tokens_list = args.correct_corpus.readlines()

    model = LSTM()

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)
	import cupy
	xp = cupy

    optimizer = optimizers.SGD()
    optimizer.setup(model)

    w2i = Word2Id()

    id_lists = []

    pbar = tqdm(total=len(tokens_list))
    logger.debug("Convert token corpus to id corpus")

    for tokens in tokens_list:
        id_lists.append(xp.array(w2i.convert_id_list(tokens), dtype=xp.int32))
        pbar.update(1)
    pbar.close()

    num_of_sample = len(id_lists)

    train_id_lists = id_lists[:num_of_sample*9/10]
    test_id_lists = id_lists[num_of_sample*9/10+1:]

    for epoch in range(1, EPOCH+1):
        logger.debug('Start learning phase of {} epoch'.format(epoch))

        for id_list in train_id_lists:
            optimizer.update(compute_loss, id_list)

    def evaluation(x_list):
        total = 0
        hit = 0
        for i in xrange(len(x_list) - 1):
            cur_word_vec = xp.array([x_list[i]])
            result = lstm(cur_word_vec).data
            print(result)
            print(result.shape)
            result = xp.argmax(result)
            print(w2i.search_word_by(x_list[i+1]), w2i.search_word_by(result))
            total += 1
            # print("Answer:{} -> Predict:{}".format(x_list[i+1], result))
            if x_list[i+1] == result:
                hit += 1

        accuracy = Decimal(hit)/Decimal(total)
        print("accuracy:{}".format(accuracy))

    # 評価フェーズ
    for id_list in test_id_lists:
        evaluation(id_list)


if __name__ == "__main__":
    try:
        main(getArgs())
    except Exception as e:
        logger.exception(e.args)
