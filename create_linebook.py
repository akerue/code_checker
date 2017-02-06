# _*_coding:utf-8_*_

import argparse


def getArgs():
    parser = argparse.ArgumentParser(description="")

    # デバック用にファイル一つを指定できるものを用意
    parser.add_argument(
        "-f", "--input",
        dest="input_file",
        type=argparse.FileType("r"),
        help="input filename as codebook"
    )

    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        type=argparse.FileType("w"),
        default=None,
    )

    return parser.parse_args()


def create_linebooks(input_file):
    linebooks = []

    for program_tokens in input_file.readlines():
        # 行頭にはBeginnings of Lineのトークンを挿入する
        linebook = ["BOL"]
        tokens_list = program_tokens.split(" ")

        for token in tokens_list:
            linebook.append(token)

            if token == "NEWLINE":
                # 改行が来たらlinebookを登録後、初期化
                linebooks.append(linebook)
                linebook = ["BOL"]

    return linebooks


if __name__ == "__main__":
    args = getArgs()

    linebooks = create_linebooks(args.input_file)

    for linebook in linebooks:
        for token in linebook:
            args.output_file.write(token + " ")
        args.output_file.write("\n")
