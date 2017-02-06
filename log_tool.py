# _*_coding:utf-8_*_

import os
import datetime

from logging import getLogger, DEBUG
from logging import StreamHandler, FileHandler, Formatter


def generate_logger(filepath="Unknown"):
    now = datetime.datetime.now()

    exec_file = os.path.splitext(os.path.split(filepath)[-1])[0]

    logfile_path = os.path.join("log",
                                "{}_{}.log".format(
                                    now.strftime("%Y%m%d_%H%M%S"),
                                    exec_file)
                               )

    log_format = Formatter("[ %(asctime)s - %(levelname)s ] %(funcName)s in %(filename)s:  %(message)s")

    logger = getLogger(__name__)

    stdout_handler = StreamHandler()
    logfile_handler = FileHandler(logfile_path)

    stdout_handler.setLevel(DEBUG)
    logfile_handler.setLevel(DEBUG)

    stdout_handler.setFormatter(log_format)
    logfile_handler.setFormatter(log_format)

    logger.setLevel(DEBUG)

    logger.addHandler(stdout_handler)
    logger.addHandler(logfile_handler)

    return logger
