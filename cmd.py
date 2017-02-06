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

COMMAND_DICT = {
        "letterbook": "python create_letterbook.py -s source -o database/letterbook.dat",
        "linebook": "python create_linebook.py -f database/letterbook.dat -o database/linebook.dat",
        "corpus": "python generate_translation_corpus.py -f database/linebook.dat -c corpus/correct_tokens.dat -w corpus/wrong_tokens.dat",
        "seq2seq": "python seq2seq.py -c corpus/correct_tokens.dat -w corpus/wrong_tokens.dat",
        }

if __name__ == "__main__":
    argvs = sys.argv
    argc = len(argvs)

    if argc != 2 or not argvs[1] in COMMAND_DICT:
        logger.debug("Usage: python {} COMMAND_NAME".format(argvs[0]))
        logger.debug("You can use these commands.")
        pprint.pprint(COMMAND_DICT)
    else:
        ret = subprocess.check_output(COMMAND_DICT[argvs[1]].split(" "))
        print ret
