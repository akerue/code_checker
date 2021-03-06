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

from progressbar import ProgressBar, Percentage, Bar, ETA

xp = np

HIDDEN_SIZE = 50
FEATURE_SIZE = 50
#EPOCH = 10
DROPOUT_RATIO = 0.5
BATCH_SIZE = 256

CACHE_PATH = "./cache/word2id_dict.pkl"
MODEL_PATH = "./serialize/seq2seq_epoch{}.model"
OPT_PATH = "./serialize/seq2seq_epoch{}.opt"

getcontext().prec = 7

START_TOKEN = "BOL"
END_TOKEN = "NEWLINE"


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
    def __init__(self, vocab_size):
        feature_size = vocab_size/50
        hidden_size = vocab_size/50
        super(Seq2Seq, self).__init__(
            embed=L.EmbedID(vocab_size, feature_size),
            encoder=L.LSTM(feature_size, hidden_size),

            connecter=L.Linear(hidden_size, hidden_size),

            decoder=L.LSTM(feature_size, hidden_size),
            output=L.Linear(hidden_size, vocab_size),
        )

    def encode(self, token_list_batch):
        for tokens in token_list_batch:
            # tokens = Variable(xp.array([[tokens]], dtype=xp.int32))
            embed_vec = F.tanh(self.embed(tokens))  # Enbed層で埋め込み後, tanh関数で活性化
            state = self.encoder(embed_vec)

        return state

    def decode(self, input_data, teacher_data):
        embed_vec = F.tanh(self.embed(input_data))  # Enbed層で埋め込み後, tanh関数で活性化
        state = self.decoder(embed_vec)
        predict_data = self.output(state)

        # 実際にトークンを予測生成する場合には教師データはいらない
        if teacher_data is not None:
            loss = F.softmax_cross_entropy(predict_data, teacher_data)
            accuracy = F.accuracy(predict_data, teacher_data, ignore_label=-1)
        else:
            loss = None
            accuracy = None

        return loss, accuracy, predict_data

    def initialize(self):
        self.encoder.reset_state()
        self.decoder.reset_state()

    def generate(self, w2i, input_id_list, limit=30):
        # generateではバッチ処理は行う必要はないが、バッチとして
        # encode, decodeに入力して処理を共通化
        # バッチサイズ1のバッチとして生成

        self.initialize()

        xs = make_batch(input_id_list, train=False)
        
        state = self.encode(xs)
        context = self.connecter(F.dropout(state,
                                 ratio=DROPOUT_RATIO, train=False))
        self.decoder.h = context

        token_list = []

        # 入力の最初は"BOL"
        input_data = Variable(xp.array([w2i[START_TOKEN]], dtype=xp.int32), volatile=True)

        for _ in range(limit):
            _, _, predict_data = self.decode(input_data, teacher_data=None)

            token = w2i.search_word_by(xp.argmax(predict_data.data))
            token_list.append(token)
            if token == w2i.search_word_by(w2i[END_TOKEN]):
                break

            # 生成では予測された次の単語を入力データにする
            input_list = xp.argmax(predict_data.data, axis=1).astype(np.int32)
            input_data = Variable(input_list, volatile=True)
            
        return token_list

    def evaluate(self, w2i, input_id_list, test_id_list):
        self.initialize()

        state = self.encode(input_id_list)

        context = self.connecter(F.dropout(state,
                                 ratio=DROPOUT_RATIO, train=False))
        self.decoder.h = context

        N = len(test_id_list)
        sum_loss = 0.0
        sum_accuracy = 0.0

        # 入力の最初は"BOL"
        input_data = xp.ones(BATCH_SIZE, dtype=xp.int32)
        input_data = input_data * xp.array([w2i[START_TOKEN]], dtype=xp.int32)
        input_data = Variable(input_data)

        for test_id in test_id_list:
            loss, accuracy, predict_data = self.decode(input_data, test_id)

            sum_loss += loss
            sum_accuracy += accuracy

            # 評価では予測された次の単語を入力データにする
            input_list = xp.argmax(predict_data.data, axis=1).astype(np.int32)
            input_data = Variable(input_list)

        return sum_loss/N, sum_accuracy/N


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


def make_batch(x, train, tail=True, xp=cuda.cupy):
    # 最長のトークン列の長さを計測
    if not isinstance(x[0], (int, xp.integer)):
        N = len(x)
        max_length = -1

        for i in xrange(N):
            l = len(x[i])
            if l > max_length:
                max_length = l
    else:
        N = 1
        max_length = len(x)

    y = np.zeros((N, max_length), dtype=np.int32)

    # tailがTrueなら逆向きにパッディングしている
    if N > 1:
        # バッチサイズ1のときには必要ない
        if tail:
            for i in xrange(N):
                l = len(x[i])
                y[i, 0:max_length-l] = -1
                y[i, max_length-l:] = x[i]
        else:
            for i in xrange(N):
                l = len(x[i])
                y[i, 0:l] = x[i]
                y[i, l:] = -1

    # バッチとして使えるように転地している
    y = [Variable(xp.asarray(y[:,j]), volatile=not train)
            for j in xrange(y.shape[1])]

    return y


def main(args):
    global xp
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
        correct_id_lists.append(xp.array(w2i.generate_id_list_by(tokens), dtype=xp.int32))
        dict_progress.update(i)
        i = i + 1

    # 誤ったトークン列を取得し、ID列に変換
    logger.debug('Build wrong token dict')
    wrong_tokens_list = args.wrong_corpus.readlines()
    dict_progress = ProgressBar(widgets=[Bar('=', '[', ']'), ' ', Percentage(), ' ', ETA()],
                                maxval=len(wrong_tokens_list)).start()
    i = 1
    for tokens in wrong_tokens_list:
        wrong_id_lists.append(xp.array(w2i.generate_id_list_by(tokens), dtype=xp.int32))
        dict_progress.update(i)
        i = i + 1

    # logger.debug('Generate cache of dataset')
    # dump_cache(w2i, correct_id_lists, wrong_id_lists)

    with open(CACHE_PATH, "wb") as f:
        dill.dump(w2i, f)

    logger.debug('Build Sequence to sequence newral network')
    logger.debug(w2i.vocab_size())
    model = Seq2Seq(w2i.vocab_size())

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)
	import cupy
	xp = cupy

    optimizer = optimizers.SGD()
    optimizer.setup(model)

    num_of_sample = len(correct_id_lists)

    # 学習サンプルと評価サンプルに分割
    # 学習サンプルが全サンプルの9/10
    # 評価サンプルが全サンプルの1/10
    logger.debug('Generate train and test dataset')
    train_correct_lists = correct_id_lists[:num_of_sample*9/10]
    test_correct_lists = correct_id_lists[num_of_sample*9/10+1:]

    train_wrong_lists = wrong_id_lists[:num_of_sample*9/10]
    test_wrong_lists = wrong_id_lists[num_of_sample*9/10+1:]

    logger.debug(' >>> Num of training dataset: {}'.format(len(train_correct_lists)))
    logger.debug(' >>> Num of test dataset: {}'.format(len(test_correct_lists)))

    logger.debug('\n >>> Total vocab size: {}'.format(w2i.vocab_size()))

    # for epoch in range(1, EPOCH+1):
    epoch = 0
    while True:
        epoch += 1

        logger.debug('Start learning phase of {} epoch'.format(epoch))

        train_dataset = zip(train_correct_lists, train_wrong_lists)

        i = 1
        max_iter = len(train_dataset)/BATCH_SIZE/2 # 1時間くらいで終わるように
        learning_progress = tqdm(total=max_iter)

        for id_list_batch in chainer.iterators.SerialIterator(train_dataset, BATCH_SIZE, False, True):

            correct_id_list = [id_list[0] for id_list in id_list_batch]
            wrong_id_list = [id_list[1] for id_list in id_list_batch]

            xs = make_batch(wrong_id_list, True)
            ts = make_batch(correct_id_list, True)

            model.initialize()

            state = model.encode(xs)

            sum_loss = 0
            sum_accuracy = 0
            N = len(ts) - 1

            context = model.connecter(F.dropout(state,
                                     ratio=DROPOUT_RATIO, train=True))
            model.decoder.h = context

            # バッチ学習の場合には行を複数にしてsum_lossを計算した後にパラメータを更新する
            for correct_id, next_correct_id in zip(ts[:-1], ts[1:]):
                loss, accuracy, context = model.decode(correct_id, next_correct_id)
                sum_loss += loss
                sum_accuracy += accuracy

            sum_loss /= N
            sum_accuracy /= N

            model.cleargrads()  # 前の学習結果のgradが残っているので初期化する
            sum_loss.backward()  # model内でgradを計算, 設定
            sum_loss.unchain_backward()  # これ必要なの？
            optimizer.update()  # 上で計算したgradを元にSGDを実行

            loss_data = float(cuda.to_cpu(sum_loss.data))
            perp = math.exp(loss_data)
            acc_data = float(cuda.to_cpu(sum_accuracy.data))

            learning_progress.set_description("[training] epoch={:<2} iter={:<6} perplexity={:<5.5} accuracy={:<5.5}".format(
                            epoch, i, perp, acc_data
                        ))

            learning_progress.update(1)
            i = i + 1

            if max_iter == i:
                break

        learning_progress.close()

        logger.debug('Start evaluation phase of {} epoch'.format(epoch))

        N = len(test_correct_lists)
        max_iter = N/BATCH_SIZE

        evaluate_progress = ProgressBar(widgets=[Bar('=', '[', ']'), ' ', Percentage(), ' ', ETA()],
                                         maxval=max_iter).start()
        j = 0
        loss = 0.0
        accuracy = 0.0

        test_dataset = zip(test_correct_lists, test_wrong_lists)

        for id_list_batch in chainer.iterators.SerialIterator(test_dataset, BATCH_SIZE, False, True):
            correct_id_list = [id_list[0] for id_list in id_list_batch]
            wrong_id_list = [id_list[1] for id_list in id_list_batch]

            xs = make_batch(wrong_id_list, True)
            ts = make_batch(correct_id_list, True)

            loss, accuracy = model.evaluate(w2i, xs, ts)

            loss += loss
            accuracy += accuracy

            evaluate_progress.update(j+1)
            j = j + 1

            if max_iter == j:
                break

        accuracy = float(cuda.to_cpu(accuracy.data))/j
        loss = float(cuda.to_cpu(loss.data))/j

        logger.debug("accuracy: {}, loss: {}".format(accuracy, loss))

	logger.debug('Save result of learning')
	serializers.save_hdf5(MODEL_PATH.format(epoch), model)
	serializers.save_hdf5(OPT_PATH.format(epoch), optimizer)

        logger.debug("Try to generate predict code from random wrong code")

        index = random.randint(len(test_dataset))
        correct_id_list, wrong_id_list = test_dataset[index]

        correct_token_list = w2i.generate_token_list_by(correct_id_list)
        wrong_token_list = w2i.generate_token_list_by(wrong_id_list)

        predicted_token_list = model.generate(w2i, wrong_id_list)

        logger.debug('INPUT (wrong tokens)')
        logger.debug(" ".join(wrong_token_list))
        logger.debug('OUTPUT (predicted_tokens)')
        logger.debug(" ".join(predicted_token_list))
        logger.debug('EXPECTED (correct tokens)')
        logger.debug(" ".join(correct_token_list))

        if accuracy > 0.7:
            break


if __name__ == "__main__":
    try:
        main(getArgs())
    except Exception as e:
        logger.exception(e.args)
