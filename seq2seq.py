# _*_coding:utf-8_*_

from log_tool import generate_logger

logger = generate_logger(__file__)

from word2id import Word2Id

from chainer import optimizers, serializers, cuda
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L

import numpy as np
import argparse
from decimal import *
import os.path
import math
import cPickle as pickle

from tqdm import tqdm

from progressbar import ProgressBar, Percentage, Bar, ETA

VOCAB_SIZE = 100
HIDDEN_SIZE = 50
FEATURE_SIZE = 50
EPOCH = 10
DROPOUT_RATIO = 0.5

CACHE_PATH = "./cache/word2id_dict.pkl"
MODEL_PATH = "./serialize/seq2seq.model"
OPT_PATH = "./serialize/seq2seq.opt"

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

    parser.add_argument(
        "--gpu",
        dest="gpu",
        type=int,
        default=0,
    )

    return parser.parse_args()


class Seq2Seq(Chain):
    def __init__(self):
        super(Seq2Seq, self).__init__(
            embed=L.EmbedID(VOCAB_SIZE, FEATURE_SIZE),
            encoder=L.LSTM(FEATURE_SIZE, HIDDEN_SIZE),

            connecter=L.Linear(HIDDEN_SIZE, VOCAB_SIZE),

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
            t = np.array([teacher_data], dtype=np.int32)
            t = Variable(t)
            loss = F.softmax_cross_entropy(predict_data, t)
            accuracy = F.accuracy(predict_data, t)
            return loss, accuracy, predict_data
        else:
            return predict_data

    def initialize(self):
        self.encoder.reset_state()
        self.connecter.reset_state()
        self.decoder.reset_state()

    def generate(self, w2i, input_id_list, end_token_id, limit=30):
        context = self.encode(input_id_list, train=False)
        token_list = []

        for _ in range(limit):
            context = self.decode(context, teacher_data=None, train=False)
            token = w2i.search_word_by(np.argmax(context.data))
            token_list.append(token)
            if token == w2i.search_word_by(end_token_id):
                break
        return token_list

    def evaluate(self, input_id_list, end_token_id, test_id_list):
        # 評価方法よくわからんかったのでとりあえず自前で書いてみる
        context = self.encode(input_id_list, train=False)

        N = len(test_id_list)
        loss = 0.0
        accuracy = 0.0

        for test_id in test_id_list:
            context = self.decode(context, teacher_data=None, train=False)

            t = np.array([test_id], dtype=np.int32)
            t = Variable(t)
            loss += F.softmax_cross_entropy(context, t)
            accuracy += F.accuracy(context, t)

            if np.argmax(context.data) == end_token_id:
                break

        return loss, accuracy, N


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


def main(args):
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

    logger.debug('Build Sequence to sequence newral network')
    model = Seq2Seq()

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

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

    logger.debug('===================================================')
    for epoch in range(1, EPOCH+1):

        logger.debug('Start learning phase of {} epoch'.format(epoch))
        learning_progress = tqdm(total=len(train_correct_lists))

        i = 1

        for correct_id_list, wrong_id_list in np.random.permutation(
                            zip(train_correct_lists, train_wrong_lists)):

            model.initialize()

            context = model.encode(wrong_id_list, train=True)

            sum_loss = 0
            sum_accuracy = 0
            N = len(correct_id_list)

            # バッチ学習の場合には行を複数にしてsum_lossを計算した後にパラメータを更新する
            for correct_id in correct_id_list:
                loss, accuracy, context = model.decode(context, correct_id, train=True)
                sum_loss += loss
                sum_accuracy += accuracy

            model.cleargrads()  # 前の学習結果のgradが残っているので初期化する
            sum_loss.backward()  # model内でgradを計算, 設定
            sum_loss.unchain_backward()  # これ必要なの？
            sum_accuracy.backward()  # model内でgradを計算, 設定
            sum_accuracy.unchain_backward()  # これ必要なの？
            optimizer.update()  # 上で計算したgradを元にSGDを実行

            loss_data = float(cuda.to_cpu(sum_loss.data))/N
            perp = math.exp(loss_data)
            acc_data = float(cuda.to_cpu(sum_accuracy.data))/N

            learning_progress.set_description("[training] epoch={} iter={} perplexity={} accuracy={}".format(
                            epoch, i, perp, acc_data
                        ))

            learning_progress.update(1)
            i = i + 1

        learning_progress.close()
        logger.debug('Start evaluation phase of {} epoch'.format(epoch))

        N = len(zip(test_correct_lists, test_wrong_lists))
        evaluate_progress = ProgressBar(widgets=[Bar('=', '[', ']'), ' ', Percentage(), ' ', ETA()],
                                        maxval=N).start()
        j = 0
        loss = 0.0
        accuracy = 0.0
        num_of_trials = 0

        end_token_id = w2i["NEWLINE"]

        for correct_id_list, wrong_id_list in np.random.permutation(
                        zip(test_correct_lists, test_wrong_lists)):
            # correct_token_list = w2i.generate_token_list_by(correct_id_list)
            # wrong_token_list = w2i.generate_token_list_by(wrong_id_list)

            # predicted_token_list = model.generate(w2i, wrong_id_list, end_token_id)

            # logger.debug('INPUT (wrong tokens)')
            # logger.debug(" ".join(wrong_token_list))
            # logger.debug('OUTPUT (predicted_tokens)')
            # logger.debug(" ".join(predicted_token_list))
            # logger.debug('EXPECTED (correct tokens)')
            # logger.debug(" ".join(correct_token_list))

            loss, accuracy, n = model.evaluate(wrong_id_list, end_token_id, correct_id_list)

            loss += loss
            accuracy += accuracy
            num_of_trials += n

            evaluate_progress.update(j+1)
            j = j + 1

        accuracy = float(cuda.to_cpu(accuracy))/num_of_trials
        loss = float(cuda.to_cpu(loss))/num_of_trials

        logger.debug("accuracy: {}, loss: {}".format(accuracy.data, loss.data))

    logger.debug('Save result of learning')
    serializers.save_hdf5(MODEL_PATH, model)
    serializers.save_hdf5(OPT_PATH, optimizer)


if __name__ == "__main__":
    try:
        main(getArgs())
    except Exception as e:
        logger.exception(e.args)
