# _*_coding:utf-8_*_

import pickle
import argparse
import pprint
import kenlm

from Simplexer import Simplexer

from nltk.util import ngrams

VECTOR_FILE = "database/data_for_kenlm.dat.arpa"


def getArgs():
    """
    コマンド引数をパースします
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-n",
        dest="N",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--threshould",
        dest="threshould",
        required=True,
        type=float,
    )

    parser.add_argument(
        "-t", "--target",
        required=True,
        type=argparse.FileType("r"),
        dest="target_file"
    )

    parser.add_argument(
        "-a", "--arpa_file_path",
        required=True,
        type=str,
        dest="arpa_file_path"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()

    model = kenlm.Model(args.arpa_file_path)

    lexer = Simplexer()
    letterbook = lexer.analyze(args.target_file)

    bag_of_ngrams = ngrams(letterbook, args.N)

    line = 0
    error_count = 0
    for ngram in bag_of_ngrams:
        probabilty = 1/model.perplexity(" ".join(ngram))
        print "{0}: {1} ---> {2}".format(line, ngram, probabilty)
        if probabilty < args.threshould:
            error_count += 1
            print "error!!"
        line += 1
    print error_count
