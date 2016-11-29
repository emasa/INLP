# Natural Language Toolkit: Language Model Counters
#
# Copyright (C) 2001-2016 NLTK Project
# Author: Ilia Kurenkov <ilia.kurenkov@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

from __future__ import unicode_literals

from collections import Counter, defaultdict
from copy import copy
from itertools import chain

from nltk.util import ngrams
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk import compat


def build_vocabulary(cutoff, *texts):
    combined_texts = chain(*texts)
    return NgramModelVocabulary(cutoff, combined_texts)


def count_ngrams(order, vocabulary, *training_texts, **counter_kwargs):
    counter = NgramCounter(order, vocabulary, **counter_kwargs)
    for text in training_texts:
        counter.train_counts(text)
    return counter


def smooth_count_ngrams(order, vocabulary, n_smooth, *tagged_training_texts, **counter_kwargs):
    counter = SmoothNgramCounter(order, vocabulary, n_smooth, **counter_kwargs)
    for tagged_text in tagged_training_texts:
        counter.train_counts(tagged_text)
    return counter


@compat.python_2_unicode_compatible
class NgramModelVocabulary(Counter):
    """Stores language model vocabulary.

    Satisfies two common language modeling requirements for a vocabulary:
    - When checking membership and calculating its size, filters items by comparing
      their counts to a cutoff value.
    - Adds 1 to its size so as to account for "unknown" tokens.
    """

    def __init__(self, unknown_cutoff, *counter_args):
        Counter.__init__(self, *counter_args)
        self.cutoff = unknown_cutoff

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, new_cutoff):
        if new_cutoff < 1:
            msg_template = "Cutoff value cannot be less than 1. Got: {0}"
            raise ValueError(msg_template.format(new_cutoff))
        self._cutoff = new_cutoff

    def __contains__(self, item):
        """Only consider items with counts GE to cutoff as being in the vocabulary."""
        return self[item] >= self.cutoff

    def __len__(self):
        """This should reflect a) filtering items by count, b) accounting for unknowns.

        The first is achieved by relying on the membership check implementation.
        The second is achieved by adding 1 to vocabulary size.
        """
        # the if-clause here looks a bit dumb, should we make it clearer?
        return sum(1 for item in self if item in self) + 1

    def __copy__(self):
        return self.__class__(self._cutoff, self)


@compat.python_2_unicode_compatible
class EmptyVocabularyError(Exception):
    pass


@compat.python_2_unicode_compatible
class NgramCounter(object):
    """Class for counting ngrams"""

    def __init__(self, order, vocabulary, unk_cutoff=None, unk_label="<UNK>", **ngrams_kwargs):
        """
        :type training_text: List[List[str]]
        """

        if order < 1:
            message = "Order of NgramCounter cannot be less than 1. Got: {0}"
            raise ValueError(message.format(order))

        self.order = order
        self.unk_label = unk_label

        # Preset some common defaults...
        self.ngrams_kwargs = {
            "pad_left": True,
            "pad_right": True,
            "left_pad_symbol": str("<s>"),
            "right_pad_symbol": str("</s>")
        }
        # While allowing whatever the user passes to override them
        self.ngrams_kwargs.update(ngrams_kwargs)
         # Set up the vocabulary
        self._set_up_vocabulary(vocabulary, unk_cutoff)

        self.ngrams = defaultdict(ConditionalFreqDist)
        self.unigrams = FreqDist()

    def _set_up_vocabulary(self, vocabulary, unk_cutoff):
        self.vocabulary = copy(vocabulary)  # copy needed to prevent state sharing
        if unk_cutoff is not None:
            # If cutoff value is provided, override vocab's cutoff
            self.vocabulary.cutoff = unk_cutoff

        if self.ngrams_kwargs['pad_left']:
            lpad_sym = self.ngrams_kwargs.get("left_pad_symbol")
            self.vocabulary[lpad_sym] = self.vocabulary.cutoff

        if self.ngrams_kwargs['pad_right']:
            rpad_sym = self.ngrams_kwargs.get("right_pad_symbol")
            self.vocabulary[rpad_sym] = self.vocabulary.cutoff

    def _enumerate_ngram_orders(self):
        return enumerate(range(self.order, 1, -1))

    def train_counts(self, training_text):
        # Note here "1" indicates an empty vocabulary!
        # See NgramModelVocabulary __len__ method for more.
        if len(self.vocabulary) <= 1:
            raise EmptyVocabularyError("Cannot start counting ngrams until "
                                       "vocabulary contains more than one item.")

        for sent in training_text:
            checked_sent = [self.check_against_vocab(word) for word in sent]

            for n in range(1, self.order+1):
                for ngram in self.to_ngrams(checked_sent, n):
                    context, word = tuple(ngram[:-1]), ngram[-1]
                    self.at_ctx(n, context)[word] += 1

    def at_ctx(self, n, ctx):
        assert n == len(ctx) + 1

        if ctx:
            return self.ngrams[n][ctx]
        else:
            return self.unigrams

    def check_against_vocab(self, word):
        if word in self.vocabulary:
            return word
        return self.unk_label

    def to_ngrams(self, sequence, n=None):
        """Wrapper around util.ngrams with useful options saved during initialization.

        :param sequence: same as nltk.util.ngrams
        :type sequence: any iterable
        """
        n = self.order if n is None else n
        return ngrams(sequence, n, **self.ngrams_kwargs)

@compat.python_2_unicode_compatible
class SmoothNgramCounter(NgramCounter):
    """Class for counting smoothed ngrams"""

    def __init__(self, order, vocabulary, n_smooth, **ngram_counter_kwargs):
        super(SmoothNgramCounter, self).__init__(order, vocabulary, **ngram_counter_kwargs)

        if 0 > n_smooth or n_smooth > self.order:
            message = "Number of smoothed components cannot be either less than 0 or greater than {1} Got: {0}"
            raise ValueError(message.format(n_smooth, self.order))

        self._n_smooth = n_smooth

    def check_against_vocab(self, tagged_word):
        word, tag = tagged_word
        if word in self.vocabulary:
            return tagged_word
        return self.unk_label, self.unk_label

    def to_ngrams(self, tagged_sequence, n=None):
        """Wrapper around util.ngrams. Smooth the first _n_smooth components using
        the POS of each word.

        :param tagged_sequence: sequence of <word, tag>
        :type sequence: any iterable
        """
        n = self.order if n is None else n

        # replace padding symbol with a tuple
        ngrams_kwargs = dict(self.ngrams_kwargs)
        ngrams_kwargs["left_pad_symbol"] = (ngrams_kwargs["left_pad_symbol"], ngrams_kwargs["left_pad_symbol"])
        ngrams_kwargs["right_pad_symbol"] = (ngrams_kwargs["right_pad_symbol"], ngrams_kwargs["right_pad_symbol"])

        tagged_ngrams = ngrams(tagged_sequence, n, **ngrams_kwargs)
        smoothed_ngrams = []
        for t_ngram in tagged_ngrams:
            words, tags = zip(*t_ngram)
            smoothed_ngram = tags[:self._n_smooth] + words[self._n_smooth:]
            smoothed_ngrams.append(smoothed_ngram)

        return smoothed_ngrams