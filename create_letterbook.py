# _*_coding:utf-8_*_

import os
import argparse
import pickle

from Simplexer import Simplexer


def getArgs():
    parser = argparse.ArgumentParser(description="")

    # デバック用にファイル一つを指定できるものを用意
    parser.add_argument(
        "-f", "--input",
        dest="input_file",
        type=argparse.FileType("r"),
        help="input filename as train data"
    )

    # 複数のプログラムが用意されたフォルダを指定すると全てのファイルを読み込む
    parser.add_argument(
        "-s", "--source",
        default=None,
        type=str,
        dest="source"
    )

    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        type=argparse.FileType("w"),
        default=None,
    )

    return parser.parse_args()

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for f in files:
            yield os.path.join(root, f)


if __name__ == "__main__":
    args = getArgs()

    lexer = Simplexer()

    if args.source is None:
        program_path_list = [args.input_file,]
    else:
        program_path_list = find_all_files(args.source)
        program_path_list = filter(lambda p: os.path.splitext(p)[1] == ".py",
                                       program_path_list)

    letterbooks = []

    for program_path in program_path_list:
        with open(program_path, "r") as f:
            letterbooks.append(lexer.analyze(f))

    if args.output_file is None:
        print(letterbooks)
    else:
        for letterbook in letterbooks:
            for letter in letterbook:
                args.output_file.write(letter + " ")
            args.output_file.write("\n")
