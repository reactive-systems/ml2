"""Abstract encoder class"""

START_TOKEN = '<s>'
EOS_TOKEN = '<e>'
PAD_TOKEN = '<p>'


class Encoder():

    def __init__(self):
        self.tokens = None
        self.ids = None
        self.error = None

    def tokenize(self):
        raise NotImplementedError

    def detokenize(self):
        raise NotImplementedError

    def encode(self, input):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def load_vocabulary(self, dir):
        raise NotImplementedError

    def build_vocabulary(self, generator):
        raise NotImplementedError

    def vocabulary_to_file(self, dir):
        raise NotImplementedError

    def vocabulary_filename(self):
        raise NotImplementedError
