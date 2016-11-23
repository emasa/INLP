# Natural Language Toolkit: Language Models
#
# Copyright (C) 2001-2016 NLTK Project
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

from __future__ import unicode_literals, division

import numpy as np
from nltk import compat

@compat.python_2_unicode_compatible
class NgramModel(object):
    """An example of how to consume NgramCounter to create a language model.
    """

    def __init__(self, ngram_counter):

        self.ngram_counter = ngram_counter
        # for convenient access save top-most ngram order ConditionalFreqDist
        self.ngrams = ngram_counter.ngrams[ngram_counter.order]
        self._ngrams = ngram_counter.ngrams
        self._order = ngram_counter.order

        self._check_against_vocab = self.ngram_counter.check_against_vocab
        self._entropy = None

    def check_context(self, context):
        """Makes sure context not longer than model's ngram order and is a tuple."""
        if len(context) >= self._order:
            raise ValueError("Context is too long for this ngram order: {0}".format(context))
        # ensures the context argument is a tuple
        return tuple(context)

    def prob(self, word, context=None):
        """Returns the conditional probability for a word given a context. if no context is
        defined returns the absolute probability for the word.

        Args:
        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: Tuple[str]
        """
        if self._order == 1 or not context:
            return self.ngram_counter.unigrams.freq(word)
        else:
            context = self.check_context(context)
            n = len(context) + 1
            return self._ngrams[n][context].freq(word)

    def logprob(self, word, context=None):
        """
        Evaluate the log probability of this word in this context.

        This implementation actually works, child classes don't have to
        redefine it.

        Args:
        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: Tuple[str]
        """
        return np.log2(self.prob(word, context))

    def _entropy_rec(self, n, ctx):
        """
        Calculate the approximate cross-entropy of the n-gram model in a recursive fashion.
        """
        words = self.ngram_counter.at_ctx(n, ctx).items()
        if n == self._order:
            p = np.array([self.prob(w, ctx) for w, _ in words])
            return np.dot(p, np.log2(p))
        else:
            p = np.array([self.prob(w, ctx) for w, _ in words])
            p_coeff = np.array([self._entropy_rec(n+1, tuple(list(ctx) + [w])) for w, _ in words])
            return np.dot(p, p_coeff)

    def entropy(self):
        """
        Calculate the approximate cross-entropy of the n-gram model.
        """
        # store internal value to avoid re-calculating the entropy
        if self._entropy is None:
            self._entropy = -self._entropy_rec(1, tuple())

        return self._entropy

    def perplexity(self):
        """
        Calculates the perplexity of the n-gram model.
        This is simply 2 ** cross-entropy.

        :param text: words to calculate perplexity of
        :type text: Iterable[str]
        """

        return np.power(2.0, self.entropy())