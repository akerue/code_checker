# _*_coding:utf-8_*_

import sys
import pprint
import subprocess

from logging import getLogger, StreamHandler, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

from collections import OrderedDict

COMMAND_DICT = OrderedDict()

COMMAND_DICT["letterbook"] = "python create_letterbook.py -s source -o database/letterbook.dat"
COMMAND_DICT["linebook"] = "python create_linebook.py -f database/letterbook.dat -o database/linebook.dat"
COMMAND_DICT["corpus"] = "python generate_translation_corpus.py -f database/linebook.dat -c corpus/correct_tokens.dat -w corpus/wrong_tokens.dat"
COMMAND_DICT["seq2seq"] = "python seq2seq.py -c corpus/correct_tokens.dat -w corpus/wrong_tokens.dat"


def display_help():
    logger.debug("Usage: python {} COMMAND_NAME".format(argvs[0]))
    logger.debug("You can use these commands.")
    pprint.pprint(COMMAND_DICT)


if __name__ == "__main__":
    argvs = sys.argv
    argc = len(argvs)

    if argc != 2:
        display_help()
    else:
        if argvs[1] in COMMAND_DICT:
            ret = subprocess.check_output(COMMAND_DICT[argvs[1]].split(" "))
            print ret
        elif argvs[1] == "all":
            for pycmd in COMMAND_DICT.values():
                ret = subprocess.check_output(pycmd.split(" "))
                print ret
        else:
            display_help()
