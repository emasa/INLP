
import numpy as np
import matplotlib.pyplot as plt

def plotZipfsLaw(kList, freqList, maxN=None, minN=None):
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # This is an auxiliary function that we use for plotting the results
    # parameters if none is passed we plot with the max values
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""

    minN = minN if minN else 0
    maxN = maxN if maxN else len(kList)
    if minN != 0 or maxN != len(kList):
        kList = kList[minN:maxN]
        freqList = freqList[minN:maxN]
    logKList    = np.log10(kList) #we use numpy for obtaining logs
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

def remove_characters(input, chars_to_remove):
    import sys
    if sys.version_info < (3,1):
        output = input.translate(None, chars_to_remove)
    else:
        # Create a dictionary using a comprehension - this maps every character from
        # string.punctuation to None. Initialize a translation object from it.
        translator = str.maketrans({key: None for key in chars_to_remove})
        output = input.translate(translator)
    return output
