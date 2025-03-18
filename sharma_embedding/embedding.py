import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

class Dictionary:
    def __init__(self, filepath, encoding="latin1", OOVRandom=True):
        self.words = list()
        self.lookup = dict()
        self.OOVRandom = OOVRandom
        dictionary = list()

        for i, line in enumerate(open(filepath, encoding=encoding)): #TODO needs to be absolute path.
            line = line.strip()
            word, vec_s = line.split("  ")
            vec = [float(n) for n in vec_s.split()]
            self.lookup[word] = i
            dictionary.append(vec)
            self.words.append(word)
        self.dictionary = np.array(dictionary)
        self.norms = normalize(self.dictionary, axis=1)

    def vec(self, word):
        try:
            return self.dictionary[self.lookup[word.strip().upper()], :]
        except KeyError:
            if self.OOVRandom:
                return np.random.rand(50,)
            else:
                return np.zeros(50,)

    def word(self, vec, n=None):
        v = vec / np.linalg.norm(vec)
        dots = np.dot(self.norms, v)
        if n is None:
            return self.words[np.argmax(dots)]
        return [(self.words[x], dots[x]) for x in np.argsort(-dots)[:n]]