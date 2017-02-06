# _*_coding:utf-8_*_

from logging import getLogger, StreamHandler, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

from word2id import Word2Id

from chainer import optimizers
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L

import numpy as np
import argparse
from decimal import *
import os.path
import dill
import cPickle as pickle

from progressbar import ProgressBar, Percentage, Bar, ETA

VOCAB_SIZE = 100
HIDDEN_SIZE = 50
FEATURE_SIZE = 50
EPOCH = 10000
DROPOUT_RATIO = 0.5

CACHE_PATH = "./cache/word2id_dict.pkl"

getcontext().prec = 7


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
         "--check_cache",
         dest="cache_flag",
         default=True,
    )

    return parser.parse_args()


class Seq2Seq(Chain):
    def __init__(self):
        super(Seq2Seq, self).__init__(
            embed=L.EmbedID(VOCAB_SIZE, FEATURE_SIZE),
            encoder=L.LSTM(FEATURE_SIZE, HIDDEN_SIZE),

            connecter=L.LSTM(HIDDEN_SIZE, VOCAB_SIZE),

            decoder=L.LSTM(VOCAB_SIZE, VOCAB_SIZE),
            output=L.Linear(VOCAB_SIZE, VOCAB_SIZE),
        )

    def encode(self, token_list, train):
        for token in token_list:
            token = Variable(np.array([[token]], dtype=np.int32))
            embed_vec = F.tanh(self.embed(token))  # Enbed層で埋め込み後, tanh関数で活性化
            input_feature = self.encoder(embed_vec)
            context = self.connecter(F.dropout(input_feature,
                                     ratio=DROPOUT_RATIO, train=train))

        return context

    def decode(self, context, teacher_data, train):
        output_feature = self.decoder(context)
        predict_data = self.output(output_feature)

        if train:
            # 学習フェーズでは教師データを使って学習
            t = np.array([teacher_data], dtype=np.int32)
            t = Variable(t)
            return F.softmax_cross_entropy(predict_data, t), predict_data
        else:
            return predict_data

    def initialize(self):
        self.encoder.reset_state()
        self.connecter.reset_state()
        self.decoder.reset_state()

    def generate(self, w2i, start_token_id, end_token_id, limit=30):
        context = self.encode([start_token_id], train=False)
        token_list = []

        for _ in range(limit):
            context = self.decode(context, teacher_data=None, train=False)
            token = w2i.search_word_by(np.argmax(context.data))
            token_list.append(token)
            if token == w2i.search_word_by(end_token_id):
                break
        return token_list


# TODO
# dillで各データセットをキャッシュとして残そうと考えたが、
# dillだとかなり遅くなるので今は実装しない
# 後ほどcPickleで再度挑戦
def check_and_load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    else:
        return None


def dump_cache(w2i_dict, correct_id_lists, wrong_id_lists):
    with open(CACHE_PATH, "wb") as f:
        pickle.dump([w2i_dict, correct_id_lists, wrong_id_lists], f)


if __name__ == "__main__":
    logger.debug('Start seq2seq script')

    args = getArgs()

    # if args.cache_flag:
    #     cache_data = check_and_load_cache()
    # else:
    #     cache_data = None

    # if cache_data is not None:
    #     logger.debug('Load cache data because we can get cache data from {}'.format(CACHE_PATH))
    #     w2i, correct_id_lists, wrong_id_lists = cache_data
    # else:
    #     logger.debug('Build dataset initially because we cannot get cache data from {}'.format(CACHE_PATH))
    w2i = Word2Id()

    correct_id_lists = []
    wrong_id_lists = []

    # 正しいトークン列を取得し、ID列に変換
    logger.debug('Build correct token dict')
    correct_tokens_list = args.correct_corpus.readlines()
    dict_progress = ProgressBar(widgets=[Bar('=', '[', ']'), ' ', Percentage(), ' ', ETA()],
                                maxval=len(correct_tokens_list)).start()
    i = 1
    for tokens in correct_tokens_list:
        correct_id_lists.append(np.array(w2i.generate_id_list_by(tokens), dtype=np.int32))
        dict_progress.update(i)
        i = i + 1

    # 誤ったトークン列を取得し、ID列に変換
    logger.debug('Build wrong token dict')
    wrong_tokens_list = args.wrong_corpus.readlines()
    dict_progress = ProgressBar(widgets=[Bar('=', '[', ']'), ' ', Percentage(), ' ', ETA()],
                                maxval=len(wrong_tokens_list)).start()
    i = 1
    for tokens in wrong_tokens_list:
        wrong_id_lists.append(np.array(w2i.generate_id_list_by(tokens), dtype=np.int32))
        dict_progress.update(i)
        i = i + 1

    # logger.debug('Generate cache of dataset')
    # dump_cache(w2i, correct_id_lists, wrong_id_lists)

    logger.debug('\nThis is word2id dict')
    w2i.show_dict()

    logger.debug('Build Sequence to sequence newral network')
    model = Seq2Seq()
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    num_of_sample = len(correct_id_lists)

    # 学習サンプルと評価サンプルに分割
    logger.debug('Generate train and test dataset')
    train_correct_lists = correct_id_lists[:num_of_sample*9/10]
    test_correct_lists = correct_id_lists[num_of_sample*9/10+1:]

    train_wrong_lists = wrong_id_lists[:num_of_sample*9/10]
    test_wrong_lists = wrong_id_lists[num_of_sample*9/10+1:]

    logger.debug(' >>> Num of training dataset: {}'.format(len(train_correct_lists)))
    logger.debug(' >>> Num of test dataset: {}'.format(len(test_correct_lists)))

    logger.debug('\n >>> Total vocab size: {}'.format(w2i.vocab_size()))

    logger.debug('Start learning phase')
    epoch = 0
    learning_progress = ProgressBar(widgets=[Bar('=', '[', ']'), ' ', Percentage(), ' ', ETA()],
                                    maxval=EPOCH).start()

    for correct_id_list, wrong_id_list in np.random.permutation(zip(train_correct_lists, train_wrong_lists)):
        model.initialize()

        context = model.encode(wrong_id_list, train=True)

        acc_loss = 0

        for correct_id in correct_id_list:
            loss, context = model.decode(context, correct_id, train=True)
            acc_loss += loss

        model.zerograds()
        acc_loss.backward()
        acc_loss.unchain_backward()
        optimizer.update()

        epoch += 1
        learning_progress.update(epoch)
        if epoch == EPOCH:
            break

    logger.debug('Start evaluation phase')
    for correct_id_list, wrong_id_list in zip(test_correct_lists, test_wrong_lists):
        correct_token_list = w2i.generate_token_list_by(correct_id_list)
        wrong_token_list = w2i.generate_token_list_by(wrong_id_list)

        start_token_id = w2i["BOL"]
        end_token_id = w2i["NEWLINE"]
        predicted_token_list = model.generate(w2i, start_token_id, end_token_id)

        logger.debug('=======================================================')
        logger.debug('INPUT (wrong tokens)')
        logger.debug(" ".join(wrong_token_list))
        logger.debug('OUTPUT (predicted_tokens)')
        logger.debug(" ".join(predicted_token_list))
        logger.debug('EXPECTED (correct tokens)')
        logger.debug(" ".join(correct_token_list))