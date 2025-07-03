import regex as re

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        self.vocab_rev = {}
        self.vocab = vocab
        for keys, values in vocab.items():
            self.vocab_rev[values] = keys
        
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else :
            self.special_tokens = []
        self.merges = merges
        
        
        tokens_cnt = len(self.vocab)

        for special_token in self.special_tokens:
            special_token = special_token.encode()
            if special_token not in self.vocab_rev:
                self.vocab_rev[special_token] = tokens_cnt
                self.vocab[tokens_cnt] = special_token
                tokens_cnt += 1

       
    # def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
    #     with open(vocab_filepath, 'rb') as f:
    #         vocab = f.read()
    #     with open(merges_filepath, 'rb') as f:
    #         merges = f.read()
    #     return cls(vocab, merges, special_tokens)

    def encode(self, text):
        tokens = []
        if text == '':
            return tokens
        if len(self.special_tokens) == 0 :
            words = re.finditer(self.PAT, text)
            for word in words:
                word_now = []
                for eee in word.group():
                    word_now += [bytes([_])for _ in list(eee.encode())]
                word_new = []
                while True:
                    min_id = len(self.vocab_rev) + 10
                    for i in range(len(word_now) - 1):
                        pattern = word_now[i] + word_now[i + 1]
                        if pattern in self.vocab_rev:
                            min_id = min(min_id, self.vocab_rev[pattern])
                    if min_id < len(self.vocab_rev):
                        i = 0
                        while i < len(word_now) - 1:
                            if word_now[i] + word_now[i + 1] == self.vocab[min_id]:
                                word_new.append(word_now[i] + word_now[i + 1])
                                i += 2
                            else :
                                word_new.append(word_now[i])
                                i += 1
                        if i == len(word_now) - 1:
                            word_new.append(word_now[i])
                        word_now, word_new = word_new, word_now
                        word_new.clear()
                    else :
                        break
                tokens += [self.vocab_rev[_] for _ in word_now]
            return tokens
        else:
            DEL = '|'.join([re.escape(_) for _ in self.special_tokens])
            DEL = '(' + DEL + ')' # capture the parenthesis
            docs = re.splititer(DEL, text)
            for doc in docs:
                if doc == '':
                    continue
                elif doc in self.special_tokens :
                    tokens.append(self.vocab_rev[doc.encode()])
                else:
                    words = re.finditer(self.PAT, doc)
                    for word in words:
                        word_now = []
                        for eee in word.group():
                            word_now += [bytes([_])for _ in list(eee.encode())]
                        
    #                    word_now = [_.encode() for _ in word.group()]
                        word_new = []
    #                    word_debug = word_now.copy()
                        while True:
                            min_id = len(self.vocab_rev) + 10
                            for i in range(len(word_now) - 1):
                                pattern = word_now[i] + word_now[i + 1]
                                if pattern in self.vocab_rev:
                                    min_id = min(min_id, self.vocab_rev[pattern])
    #                        print(min_id)
                            if min_id < len(self.vocab_rev):
                                i = 0
                                while i < len(word_now) - 1:
                                    if word_now[i] + word_now[i + 1] == self.vocab[min_id]:
                                        word_new.append(word_now[i] + word_now[i + 1])
                                        i += 2
                                    else :
                                        word_new.append(word_now[i])
                                        i += 1
                                if i == len(word_now) - 1:
                                    word_new.append(word_now[i])
                                word_now, word_new = word_new, word_now
                                word_new.clear()
                            else :
                                break
                        # word_debug, word_now = word_now, word_debug
                        # for merge in self.merges:
                        #     i = 0
                        #     while i < len(word_now) - 1:
                        #         if (word_now[i], word_now[i + 1]) == merge:
                        #             word_new.append(word_now[i] + word_now[i + 1])
                        #             i += 2
                        #         else :
                        #             word_new.append(word_now[i])
                        #             i += 1
                        #     if i == len(word_now) - 1:
                        #         word_new.append(word_now[i])
                        #     word_now, word_new = word_new, word_now
                        #     word_new.clear()
                        # print(word_now, word_debug)
                        # for i in self.merges:
                        #     if i == (b'\xf0', b'\x9f'):
                        #         print("=============")
                        # print(self.vocab_rev[(b'\xf0', b'\x9f')])
                        # assert word_now == word_debug
                        tokens += [self.vocab_rev[_] for _ in word_now]
        return tokens
    
    def encode_iterable(self, fiter): # file iterable <=> line!
        for line in fiter:
            if len(self.special_tokens) == 0 :
                docs = [line]
            else:
                DEL = '|'.join([re.escape(_) for _ in self.special_tokens])
                DEL = '(' + DEL + ')' # capture the parenthesis
                docs = re.split(DEL, line)

            for doc in docs:
                if doc == '':
                    continue
                elif doc in self.special_tokens :
                    yield(self.vocab_rev[doc.encode()])
                else:
                    words = re.finditer(self.PAT, doc)
                    for word in words:
                        word_now = []
                        for eee in word.group():
                            word_now += [bytes([_])for _ in list(eee.encode())]
                        word_new = []
                        while True:
                            min_id = len(self.vocab_rev) + 10
                            for i in range(len(word_now) - 1):
                                pattern = word_now[i] + word_now[i + 1]
                                if pattern in self.vocab_rev:
                                    min_id = min(min_id, self.vocab_rev[pattern])
    #                        print(min_id)
                            if min_id < len(self.vocab_rev):
                                i = 0
                                while i < len(word_now) - 1:
                                    if word_now[i] + word_now[i + 1] == self.vocab[min_id]:
                                        word_new.append(word_now[i] + word_now[i + 1])
                                        i += 2
                                    else :
                                        word_new.append(word_now[i])
                                        i += 1
                                if i == len(word_now) - 1:
                                    word_new.append(word_now[i])
                                word_now, word_new = word_new, word_now
                                word_new.clear()
                            else :
                                break
                        # for merge in self.merges:
                        #     i = 0
                        #     while i < len(word_now) - 1:
                        #         if (word_now[i], word_now[i + 1]) == merge:
                        #             word_new.append(word_now[i] + word_now[i + 1])
                        #             i += 2
                        #         else :
                        #             word_new.append(word_now[i])
                        #             i += 1
                        #     if i == len(word_now) - 1:
                        #         word_new.append(word_now[i])
                        #     word_now, word_new = word_new, word_now
                        #     word_new.clear()
                        for _ in word_now:
                            yield self.vocab_rev[_]
        
    def decode(self, ids):
        result = bytes([])
        for id in ids:
            result += self.vocab[id]
        return result.decode(errors='replace')