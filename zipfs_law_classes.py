# encoding: utf-8

from __future__ import division
import string

import nltk

from zipfs_law_functions import remove_characters

class ZipfsLawVerifier(object):
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # This is the main class containing functions necessaries for resolving the exercise
    # The parameter objext is the text that we are going to analyze
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def __init__(self, keep_punctuation=False, case_sensitive=False, keep_digits=False, with_lemmas=False):
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # config of the class
        # lemmas is a technique for formatting the text. The value false works
        # with the text without lemmatizing it
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        self._with_lemmas = with_lemmas
        self._keep_punctuation = keep_punctuation
        self._case_sensitive = case_sensitive
        self._keep_digits = keep_digits

        # internal representation
        self._tokenized_corpus = ""
        self._tokens_sorted_by_rank = []
        self._tokens_to_rank_map = {}
        self._freqList = []
        self._kList = []
        self._freq_dist = None

    def process_corpus(self, raw_corpus):
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # This is the principal function that works with the corpus applying different techniques
        #  - formatting it
        #  - tokenizen it
        #  - building the frequency distribution necessary for
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        cleaned_corpus = self._preprocessing(raw_corpus)
        self._tokenized_corpus = self._tokenize(cleaned_corpus)
        self._computeFrequencyDistribution()

    def _preprocessing(self, input_text):
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # This funtion formats the raw text for extracting the words
        # - delete line breaks
        # - delete punctuation
        # - delete numbers
        # - capitalize
        # The parameter input text is the raw text
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        #delete line breaks

        output_text = remove_characters(input_text, "\n")

        # delete punctuation?
        if not self._keep_punctuation:
            output_text = remove_characters(output_text, string.punctuation)
        # delete digits?
        if not self._keep_digits:
            output_text = remove_characters(output_text, string.digits)
        # translate from upper to lower case?
        if not self._case_sensitive:
            output_text = output_text.lower()

        return output_text

    def _tokenize(self, input_text):
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # This function will tokenize the text into words using the function from NLTK
        # Whitespacetokenizer. We make tests wit others tokenizers counting the results
        # even with the count_words function provided in auxiliar.py
        # and even counting words with other tools and the results were very similar
        # so we trust in this external function
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        tokens = nltk.WhitespaceTokenizer().tokenize(input_text)
        # optional: test with lemmas
        if self._with_lemmas:
            lemmatizer = nltk.WordNetLemmatizer()
            tokens = [ lemmatizer.lemmatize(token) for token in tokens]
        return tokens

    def _computeFrequencyDistribution(self):
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # In this function we count the words and rankerize by quantity
        # using the nltk FreqDist class and the tokenized text
        # we also compute the constant K from this list
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        self._freq_dist = nltk.FreqDist(self._tokenized_corpus)
        # split sorted tokens and frequency in 2 list
        # mostcommon() returns all the unique words in the distribution
        self._tokens_sorted_by_rank, self._freqList = zip(*self._freq_dist.most_common())
        # compute K=rank * freq
        self._kList = [(rank_idx+1) * self._freqList[rank_idx] for rank_idx in range(self.B())]
        # mapping (token, rank)  useful for individual queries
        self._tokens_to_rank_map = dict((self._tokens_sorted_by_rank[rank_idx], rank_idx+1) for rank_idx in range(self.B()))

    def rank(self, token):
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # returns rank of a word
        # parameter token is the unknown rank word
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        assert token in self._tokens_to_rank_map
        return self._tokens_to_rank_map[token]

    def K(self, token=None):
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # this function will return the list of K values or value of an item
        # if no parameter is provided, returns the full list sorted by rank
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        if token is None:
            return list(self._kList)
        idx = self.rank(token) - 1
        return self._kList[idx]

    def freq(self, token=None):
        # """""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # this function will return the distribution list or value of an item
        # if no parameter is provided, returns the full list sorted by rank
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        if token is None:
            return list(self._freqList)
        idx = self.rank(token) - 1
        return self._freqList[idx]

    def nthToken(self, n):
        # """""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # returns tokens sorted by rank
        # parameter n can determine the number of tokens
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        assert 0 < n <= self.B()
        idx = n-1
        return self._tokens_sorted_by_rank[idx]

    def N(self):
        # """""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # returns all the tokens list
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        return self._freq_dist.B()

    def B(self):
        return self._freq_dist.B()

class ZipfsLawCharacterWiseVerifier(ZipfsLawVerifier):
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # This is the class that will be used to work into character level
    # this class is composed by the procedures of the main class
    #
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def __init__(self, **kwargs):
        super(ZipfsLawCharacterWiseVerifier, self).__init__(**kwargs)

    def _tokenize(self, input_text):
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # This function will split the text and return tokens
        # The parameter is the text to split into chars
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
        tokens = super(ZipfsLawCharacterWiseVerifier, self)._tokenize(input_text)

        # concatanate every preprocessed token and split the text into a list of characters
        tokens = list( "".join(tokens))

        return tokens