SOD_TOKEN = 0
EOD_TOKEN = 1


class Language(object):
    def __init__(self, type='full_text'):
        self.type = type
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOD", 1: "EOD"}
        self.n_words = 2

    def add_text(self, text):
        for word in text.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 0
            self.n_words += 1
        self.word2count[word] += 1
