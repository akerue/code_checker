# _*_coding:utf-8_*_

import ply.lex as lex


class Simplexer:

    def __init__(self):

        tokens = [
            'NUMBER',
            'PLUS',
            'MINUS',
            'TIMES',
            'DIVIDE',
            'EQUAL',
            'PERCENT',
            'ID',
            'LPAREN',
            'RPAREN',
            'LBRACK',
            'RBRACK',
            'LBRACE',
            'RBRACE',
            'COMMA',
            'DOT',
            'IN_EDGE',
            'IN_EDGE_SUB',
            'OUT_EDGE',
            'OUT_EDGE_SUB',
            'FARROW',
            'STRING',
            'COLON',
            'DOCSTRING',
            'NEWLINE',
            'TAB',
        ]

        self.REPLACED_TOKEN = (
                'TAB',
                'STRING',
                'NEWLINE',
                'NUMBER',
                'ID',
        )

        t_TAB          = r'\t'
        t_NUMBER       = r'[-]?\d'
        t_PLUS         = r'\+'
        t_MINUS        = r'-'
        t_TIMES        = r'\*'
        t_DIVIDE       = r'/'
        t_EQUAL        = r'='
        t_PERCENT      = r'%'
        t_LPAREN       = r'\('
        t_RPAREN       = r'\)'
        t_LBRACK       = r'\['
        t_RBRACK       = r'\]'
        t_LBRACE       = r'{'
        t_RBRACE       = r'}'
        t_COMMA        = r','
        t_DOT          = r'\.'
        t_IN_EDGE      = r'->'
        t_IN_EDGE_SUB  = r'~>'
        t_OUT_EDGE     = r'<-'
        t_OUT_EDGE_SUB = r'<~'
        t_FARROW       = r'=>'
        t_COLON        = r":"


        def t_DQSTRING(t):
            r'[ubr]*"[^"]+"'
            t.type = "STRING"
            return t

        def t_SQSTRING(t):
            r"[ubr]*'[^']+'"
            t.type = "STRING"
            return t

# Define a rule so we can track line numbers
        def t_newline(t):
            r'\n+'
            t.lexer.lineno += len(t.value)
            t.type = "NEWLINE"
            t.value = "\n"
            return t

# A string containing ignored characters (spaces and tabs)
        t_ignore = ' '


# Ignore Comment
        def t_COMMENT(t):
            r'\#.*'
            pass

        def t_DOCSTRING(t):
            r'"{3}\n?[.\n]+"{3}'
            pass

# Error handling rule
        def t_error(t):
            # print("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)

        reserved = {}

        for kw in __import__("keyword").kwlist:
            reserved[kw] = kw.upper()

        tokens = tokens + list(reserved.values())


# 識別子IDに対して、予約語をチェックする処理を追加
        def t_ID(t):
            r'[a-zA-Z_][a-zA-Z_0-9]*'
            t.type = reserved.get(t.value, 'ID')
            return t

        self.lexer = lex.lex()

    def analyze(self, program_file):
        letterbook = []
        for l in program_file.readlines():
            self.lexer.input(l)

            while True:
                tok = self.lexer.token()
                if not tok:
                    break
                if tok.type in self.REPLACED_TOKEN:
                    if tok.type == "NEWLINE":
                        if len(letterbook) != 0 and letterbook[-1] != "NEWLINE":
                            letterbook.append(tok.type)
                        else:
                            pass # 連続改行は無視
                    else:
                        letterbook.append(tok.type)
                else:
                    letterbook.append(tok.value)

        return letterbook
