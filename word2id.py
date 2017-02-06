# _*_coding:utf-8_*_

import collections
import pprint


class Word2Id:
    def __init__(self):
        """
        始端文字ID: 0
        終端文字ID: 1
        として追加する
        """
        self.word2id_dict = collections.defaultdict(lambda: len(self.word2id_dict))
        self.word2id_dict["BOL"] = 0
        self.word2id_dict["NEWLINE"] = 1

    def __getitem__(self, key):
        return self.word2id_dict[key]

    def search_word_by(self, word_id):
        word2id_dict = dict(self.word2id_dict)

        try:
            return word2id_dict.keys()[word2id_dict.values().index(word_id)]
        except ValueError:
            return None

    def generate_id_list_by(self, tokens):
        id_list = []

        for token in tokens.split(" "):
            id_list.append(self.word2id_dict[token])

        return id_list

    def generate_token_list_by(self, id_list):
        token_list = []

        for token_id in id_list:
            token_list.append(self.search_word_by(token_id))

        return token_list

    def show_dict(self):
        pprint.pprint(self.word2id_dict)

    def vocab_size(self):
        return len(self.word2id_dict)
