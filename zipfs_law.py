# encoding: utf-8
from __future__ import print_function
import sys
import os

import numpy as np
import matplotlib

from zipfs_law_classes import ZipfsLawVerifier, ZipfsLawCharacterWiseVerifier
from zipfs_law_functions import plotZipfsLaw

matplotlib.interactive(True) # don't block the process thread with plots

# reading input differs between python versions
if sys.version_info < (3, 0):
    wait_for_input = raw_input
else:
    wait_for_input = input

def main(*args, **kwargs):
    #""""""""""""""""""""""""""""""""""""""""""""""""""
    #Main function. executing the program
    #""""""""""""""""""""""""""""""""""""""""""""""""""

    # if case sensitive is False, translate upper case to lower case, otherwise keep text untouched
    case_sensitive = kwargs.get("case_sensitive", False) #default: case insensitive
    # if keep_punctuation is False, remove punctuation, otherwise keep text untouched
    keep_punctuation = kwargs.get("keep_punctuation", False) #default: get rid of punctuation
    # if keep_digits is False, remove digits, otherwise keep text untouched
    keep_digits = kwargs.get("keep_digits", False) #default: get rid of digits
    # if with_lemmas is True, verifies Zipfs Law on lemmas, otherwise on simple words.
    with_lemmas = kwargs.get("with_lemmas", False)  # default: without lemmas
    # if with_words is True, verifies Zipfs Law on words, otherwise on characters.
    with_words = kwargs.get("with_words", True)  # default: words


    # open the corpus path passed by parameter and store its content as raw text
    corpus_path = kwargs["corpus_path"]
    with open(corpus_path) as corpus_file:
        raw_text_corpus = corpus_file.read()

    if with_words:
        verifier = ZipfsLawVerifier(case_sensitive=case_sensitive,
                                    keep_punctuation=keep_punctuation,
                                    keep_digits=keep_digits,
                                    with_lemmas=with_lemmas)
    else:
        verifier = ZipfsLawCharacterWiseVerifier()

    # Process the corpus
    verifier.process_corpus(raw_text_corpus)

    K = verifier.K()
    frequency = verifier.freq()

    show_stats = kwargs.get("show_stats", True) # default: True
    show_plots = kwargs.get("show_plots", False) # default: False
    show_top = kwargs.get("show_top", False) # default: False

    if show_stats:
        # Compute average and std deviation of K (expected ZipfsLaw's constant)
        # We use numpy for array calculations
        kMean, kStdDev = np.mean(K), np.std(K)
        kCoeffVar = kStdDev / kMean
        print("K distribution for {}:  Mean: {:.2f}  Stddev: {:.2f} Coeff of Variation: {:.2f}" \
              .format(os.path.basename(corpus_path), kMean, kStdDev, kCoeffVar))

    if show_plots:
        # Plots (ranking range can be modified)
        minN, maxN = 0, verifier.B() #B returns the number of items in the list
        plotZipfsLaw(K, frequency, minN=minN, maxN=maxN)

    if show_top:
        # Top N elements in the ranking
        # Tabulate
        topN = min(verifier.B(), 30)  # default 30
        print("Top %s elements in the ranking" % (topN,))
        print("{:^8} {:^8} {:^8} {:^8}".format("Rank", "Token", "K", "Frequency"))
        for rank in range(1, topN + 1):
            token = verifier.nthToken(rank)
            print("{:^8} {:^8} {:^8} {:^8}".format(rank, token, verifier.K(token), verifier.freq(token)))

        if sys.version_info < (3, 0):
            print("Matplotlib don't support non-ascii characters (such as á, é, etc) in python 2.x")
        else:
            # Plot
            verifier._freq_dist.plot(topN)

if __name__ == '__main__':
    ENGLISH_CORPUS = "en.txt"
    SPANISH_CORPUS = "es.txt"

    print("\nProcessing English corpus...")

    print("\nWith simple tokenization:")
    main(corpus_path=ENGLISH_CORPUS, case_sensitive=True, keep_digits=True, keep_punctuation=True)
    print("\nCase insensitive:")
    main(corpus_path=ENGLISH_CORPUS, case_sensitive=False, keep_digits=True, keep_punctuation=True)
    print("\nCase insensitive & remove digits & remove punctuation:")
    main(corpus_path=ENGLISH_CORPUS, case_sensitive=False, keep_digits=False, keep_punctuation=False)
    print("\nUse Lemmas & case insensitive & remove digits & remove punctuation:")
    main(corpus_path=ENGLISH_CORPUS, with_lemmas=True, case_sensitive=False, keep_digits=False, keep_punctuation=False)
    print("\nMove to char level: case insensitive & remove digits & remove punctuation:")
    main(corpus_path=ENGLISH_CORPUS, with_words=False, case_sensitive=False, keep_digits=False, keep_punctuation=False)


    print("\nProcessing Spanish corpus...")
    print("\nWith simple tokenization:")
    main(corpus_path=SPANISH_CORPUS, case_sensitive=True, keep_digits=True, keep_punctuation=True)
    print("\nCase insensitive:")
    main(corpus_path=SPANISH_CORPUS, case_sensitive=False, keep_digits=True, keep_punctuation=True)
    print("\nCase insensitive & remove digits & remove punctuation:")
    main(corpus_path=SPANISH_CORPUS, case_sensitive=False, keep_digits=False, keep_punctuation=False)

    if sys.version_info < (3,0):
        print("\nWordnet lemmatizer feature don't support non-ascii characters (such as á, é, etc) in python 2.x")
    else:
        print("\nUse Lemmas & case insensitive & remove digits & remove punctuation:")
        main(corpus_path=SPANISH_CORPUS, with_lemmas=True, case_sensitive=False, keep_digits=False,
             keep_punctuation=False)

    print("\nMove to char level: case insensitive & remove digits & remove punctuation:")
    main(corpus_path=SPANISH_CORPUS, with_words=False, case_sensitive=False, keep_digits=False, keep_punctuation=False)

    print("\nPlots & Top words:")
    wait_for_input("\nPress Enter to show plots and top words for English corpus...")
    main(corpus_path=ENGLISH_CORPUS, with_lemmas=True, show_stats=False, show_plots=True, show_top=True)

    wait_for_input("\nPress Enter to show plots and top words for English corpus in character level...")
    main(corpus_path=ENGLISH_CORPUS, with_words=False, show_stats=False, show_plots=True, show_top=True)

    wait_for_input("\nPress Enter to show plots and top words for Spanish corpus...")
    if sys.version_info < (3,0):
        print("\nWordnet lemmatizer feature don't support non-ascii characters (such as á, é, etc) in python 2.x")
        main(corpus_path=SPANISH_CORPUS, with_lemmas=False, show_stats=False, show_plots=True, show_top=True)
    else:
        print("\nUse Lemmas & case insensitive & remove digits & remove punctuation:")
        main(corpus_path=SPANISH_CORPUS, with_lemmas=True, show_stats=False, show_plots=True, show_top=True)

    wait_for_input("\nPress Enter to show plots and top words for Spanish corpus in character level...")
    main(corpus_path=SPANISH_CORPUS, with_words=False, show_stats=False, show_plots=True, show_top=True)

    wait_for_input("\nPress Enter to exit...")