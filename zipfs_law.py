# encoding: utf-8
from __future__ import division

from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
import nltk, re, pprint, string

import numpy as np
import matplotlib.pyplot as plt

def plotZipfsLaw(kList, freqList, maxN=None, minN=None):

    minN = minN if minN else 0
    maxN = maxN if maxN else len(kList)

    if minN != 0 or maxN != len(kList):
        kList = kList[minN:maxN]
        freqList = freqList[minN:maxN]

    logKList    = np.log10(kList)
    logFreqList = np.log10(freqList)

    if minN == 0:
        minN =  1
        maxN += 1

    rankList = range(minN, maxN)
    logRankList = np.log10(rankList)

    fig = plt.figure()
    plt.plot(logRankList, logFreqList)
    fig.suptitle("Zipf's Law")
    plt.xlabel("log(rank)")
    plt.ylabel("log(frequency)")

    fig = plt.figure()
    plt.plot(rankList, freqList)
    fig.suptitle("Zipf's Law")
    plt.xlabel("rank")
    plt.ylabel("frequency")

    fig = plt.figure()
    plt.plot(rankList, kList)
    fig.suptitle("Zipf's Law")
    plt.xlabel("rank")
    plt.ylabel("K")

    fig = plt.figure()
    plt.plot(logRankList, logKList)
    fig.suptitle("Zipf's Law")
    plt.xlabel("log(rank)")
    plt.ylabel("log(K)")

    plt.show()


class ZipfsLawVerifier(object):

    def __init__(self, with_lemmas=False):
        # config
        self._with_lemmas = with_lemmas
        # internal representation
        self._tokenized_corpus = ""
        self._tokens_sorted_by_rank = []
        self._tokens_to_rank_map = {}
        self._freqList = []
        self._kList = []
        self._freq_dist = None

    def process_corpus(self, raw_corpus):
        cleaned_corpus = self._preprocessing(raw_corpus)
        self._tokenized_corpus = self._tokenize(cleaned_corpus)
        self._computeFrequencyDistribution()

    def _preprocessing(self, input_text):
        # delete line breaks
        output_text = input_text.translate(None, "\n")
        # delete punctuation
        output_text = output_text.translate(None, string.punctuation)
        # delete digits
        output_text = output_text.translate(None, string.digits)
        # translate from upper to lower case
        output_text = output_text.lower()

        return output_text

    def _tokenize(self, input_text):
        # This function will tokenize the text into words
        tokens = WhitespaceTokenizer().tokenize(input_text)

        # optional: test with lemmas
        if self._with_lemmas:
            lemmatizer = WordNetLemmatizer()
            tokens = [ lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    def _computeFrequencyDistribution(self):
        self._freq_dist = nltk.FreqDist(self._tokenized_corpus)

        # split sorted tokens and frequency in 2 list
        self._tokens_sorted_by_rank, self._freqList = zip(*self._freq_dist.most_common())

        # compute K=rank * freq
        self._kList = [(rank_idx+1) * self._freqList[rank_idx] for rank_idx in range(self.B())]
        # mapping (token, rank)  useful for individual queries
        self._tokens_to_rank_map = dict((self._tokens_sorted_by_rank[rank_idx], rank_idx+1) for rank_idx in range(self.B()))

    def rank(self, token):
        assert token in self._tokens_to_rank_map

        return self._tokens_to_rank_map[token]

    def K(self, token=None):
        # if no parameter is provided, returns the full list sorted by rank
        if token is None:
            return list(self._kList)

        idx = self.rank(token) - 1
        return self._kList[idx]

    def freq(self, token=None):
        # if no parameter is provided, returns the full list sorted by rank
        if token is None:
            return list(self._freqList)

        idx = self.rank(token) - 1
        return self._freqList[idx]

    def nthToken(self, n):
        assert 0 < n <= self.B()

        idx = n-1
        return self._tokens_sorted_by_rank[idx]

    def N(self):
        return self._freq_dist.B()

    def B(self):
        return self._freq_dist.B()

class ZipfsLawCharacterWiseVerifier(ZipfsLawVerifier):
    def _tokenize(self, input_text):
        output_text = input_text.translate(None, " ")

        # split the text into a list of characters
        tokens = list( output_text )
        return tokens

def main(*args, **kwargs):
    """
    Main function.
    """
    # open the corpus path passed by parameter and store its content as raw text
    with open(kwargs["corpus_path"]) as corpus_file:
        raw_text_corpus = corpus_file.read()

    # Process the corpus
    # if with_words is True, verifies Zipfs Law on words, otherwise on characters.
    with_words = kwargs.get("with_words", True)  # default: words
    if with_words:
        # if with_lemmas is True, verifies Zipfs Law on lemmas, otherwise on simple words.
        with_lemmas = kwargs.get("with_lemmas", False)  # default: without lemmas
        verifier = ZipfsLawVerifier(with_lemmas=with_lemmas)
    else:
        verifier = ZipfsLawCharacterWiseVerifier()

    verifier.process_corpus(raw_text_corpus)

    # Compute average and std deviation of K (expected ZipfsLaw's constant)
    K = verifier.K()
    kAverage = np.average(K)
    kStdDeviation = np.std(K)
    print "Average K: %s ; Standard Deviation K: %s" % (kAverage, kStdDeviation)

    frequency = verifier.freq()

    # Print top N elements in the ranking
    topN = min(verifier.B(), 50) # default 50
    print "Top %s elements in the ranking" % (topN,)
    print "{:^8} {:^8} {:^8} {:^8}".format("Rank", "Token", "K", "Frequency")
    for rank in range(1, topN + 1):
        token = verifier.nthToken(rank)
        print "{:^8} {:^8} {:^8} {:^8}".format(rank, token, verifier.K(token), verifier.freq(token))

    # Plots (ranking range can be modified)
    minN, maxN = 0, verifier.B()
    plotZipfsLaw(K, frequency, minN=minN, maxN=maxN)


if __name__ == '__main__':
    # Note: choose between english/spanish corpuses and between options words/single characters

    # english corpus
    main(corpus_path='en.txt', with_lemmas=True)

    # spanish corpus
    # the wordnet lemmas feature don't support non-ascii characters (such as á, é, etc)
    #main(corpus_path='es.txt')

    # english corpus with characters instead of words
    #main(corpus_path='en.txt', with_words=False)

    # spanish corpus with characters instead of words
    #main(corpus_path='es.txt', with_words=False)
