# _*_coding:utf-8_*_

from word2id import Word2Id

from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import math
import six
import numpy as np
import argparse
from decimal import *

VOCAB_SIZE = 10000
HIDDEN_SIZE = 100
batchsize = 100

getcontext().prec = 7


def getArgs():
    """
    コマンド引数をパースします
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-f", "--input",
        dest="input_file",
        type=argparse.FileType("r"),
        help="input filename"
    )

    return parser.parse_args()


class LSTM(Chain):
    def __init__(self):
        super(LSTM, self).__init__(
            embed=L.EmbedID(VOCAB_SIZE, HIDDEN_SIZE),  # word embedding
            mid=L.LSTM(HIDDEN_SIZE, HIDDEN_SIZE),  # the first LSTM layer
            out=L.Linear(HIDDEN_SIZE, VOCAB_SIZE),  # the feed-forward output layer
        )

    def reset_state(self):
        self.mid.reset_state()

    def __call__(self, cur_word):
        # Given the current word ID, predict the next word.
        x = self.embed(cur_word)
        h = self.mid(x)
        y = self.out(h)
        return y


def create_one_hot_vector(word_id):
    word_vec = np.zeros(VOCAB_SIZE, dtype=np.int32)
    word_vec[word_id] = 1
    return word_vec


def compute_loss(x_list):
    # 損失関数
    loss = 0

    # perm = np.random.permutation(len(train_id_lists)-batchsize-1)

    for i in six.moves.range(len(x_list)-1):
        cur_word_vec = np.array([x_list[i]])
        next_word_vec = np.array([x_list[i+1]])
        loss += model(cur_word_vec, next_word_vec)
    print("loss:{}".format(loss.data))
    return loss


if __name__ == "__main__":
    args = getArgs()

    tokens_list = args.input_file.readlines()

    lstm = LSTM()
    model = L.Classifier(lstm)
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    lstm.reset_state()

    w2i = Word2Id()

    id_lists = []

    for tokens in tokens_list:
        id_lists.append(np.array(w2i.convert_id_list(tokens), dtype=np.int32))

    num_of_sample = len(id_lists)

    train_id_lists = id_lists[:num_of_sample*9/10]
    test_id_lists = id_lists[num_of_sample*9/10+1:]

    # 学習フェーズ
    epoch = 0
    for id_list in train_id_lists:
        optimizer.update(compute_loss, id_list)
        epoch += 1
        if epoch == 10:
            break

    def evaluation(x_list):
        total = 0
        hit = 0
        for i in xrange(len(x_list) - 1):
            cur_word_vec = np.array([x_list[i]])
            result = lstm(cur_word_vec).data
            print(result)
            print(result.shape)
            result = np.argmax(result)
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
