# _*_coding:utf-8_*_

from operate_dat_file import import_token_from

import argparse
import random
random.seed()


def getArgs():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-f", "--input",
        dest="input_file",
        type=argparse.FileType("r"),
        required=True,
    )

    parser.add_argument(
        "-c", "--correct",
        dest="correct_file",
        type=argparse.FileType("w"),
        required=True,
    )

    parser.add_argument(
        "-w", "--wrong",
        dest="wrong_file",
        type=argparse.FileType("w"),
        required=True,
    )

    return parser.parse_args()


def insert_comma_randomly(tokens):
    index = random.randint(1, len(tokens)-1)

    tokens.insert(index, ",")

    return tokens


def generate_error_tokens(tokens, error_num=1, pair_num=10):
    """
    一つのトークン列から複数パターンのエラーが含まれたトークン列を返す
    error_numでいくつのエラーを混ぜるかを指定できる
    pair_numでいくつのエラートークン列を生成するかを指定できる
    """

    error_tokens_list = []

    # error_generators内にエラートークンを生成できる関数を登録する
    error_generators = [insert_comma_randomly]

    for i in range(pair_num):
        modified_tokens = list(tokens) # 普通に代入すると参照渡しになるので複製
        for j in range(error_num):
            modified_tokens = random.choice(error_generators)(modified_tokens)

        error_tokens_list.append(modified_tokens)

    return error_tokens_list

if __name__ == "__main__":
    args = getArgs()
    tokens_list = import_token_from(args.input_file)

    corpus = []

    for tokens in tokens_list:
        error_tokens_list = generate_error_tokens(tokens)
        corpus.append((tokens, error_tokens_list))

    for correct_token, wrong_tokens in corpus:
        for wrong_token in wrong_tokens:
            args.correct_file.write(" ".join(correct_token))
            args.wrong_file.write(" ".join(wrong_token))
