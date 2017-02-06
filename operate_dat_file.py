# _*_coding:utf-8_*_


def import_token_from(dat_file):
    """
    空白区切りのトークン列をネストしたリストとして返す
    """
    dat_file = dat_file.readlines()

    dat = []

    for token_line in dat_file:
        l = []
        for token in token_line.split(" "):
            l.append(token)
        dat.append(l)

    return dat


def export_token_to(dat_file, token_list):
    """
    ネストされたリストを空白区切りのトークン文字列にしてファイルに記述する
    """
    for tokens in token_list:
        for token in tokens:
            dat_file.write(token + " ")
        dat_file.write("\n")
