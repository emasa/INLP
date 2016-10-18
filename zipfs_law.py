from __future__ import division
from nltk.tokenize import WhitespaceTokenizer
import nltk, re, pprint, string

def tokenizeByWhiteSpaces(rawtext):
    #This function will tokenize the text into words
    #tokenizeing by whitespaces and lowercase it gives exactly the same number of words than auxiliar.py provided by Horacio so we assume it's the correct number of tokens (words)
    tokens = WhitespaceTokenizer().tokenize(rawtext)
    return tokens

def distributionList(tokens):
    #This function will create the frequency distribution of the tokens
    fdist1 = nltk.FreqDist(tokens)
  #  fdist1.plot(50)
    return fdist1

def expectedZipLawCurve(fdist1):
    import numpy as np
    import matplotlib.pyplot as plt
    MaxFreq=fdist1[fdist1.max()]
    Z = np.random.zipf(2, 50)
    index=1
    # for sample in fdist1:
    #    MaxFreq = MaxFreq / index
    #    np.append(MaxFreq)
    #    index +=1
  #  plt.plot(sorted(np[:50],reverse=True))
    plt.plot(sorted(Z,reverse=True))
    plt.show()

def computeZipfsLaw(fdist1):
    import numpy as np
    import matplotlib.pyplot as plt
    index = 1
    K = []
    freq = []
    rank = []
    logFreq = []

    for word, frequency in fdist1.most_common():
        #frequency_total = frequency / len(tokens)
        #frequency_total.update({word:frequency / len(tokens)})
        #frequency_total = np.append(word,frequency/len(tokens))
        #frequency_total = fdist1.freq(word)
        #f=K/r so K = f.r
        K.append(index * frequency)
        freq.append(frequency)
        rank.append(index)

        index  += 1
   # plt.plot(K)
  #  plt.show()
    logK = np.log10(K)
    logFreq = np.log10(freq)
    logRank = np.log10(rank)
    plt.plot(logFreq)
    #plt.axis([np.amin(logRank), np.amax(logRank), np.amin(logFreq), np.amax(logFreq)])
    plt.xscale('log')
    plt.show()
    #print (K[:1000])
    #print (logK[:1000])
def counters():
    # count total number of tokens
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    print len(raw), 'chars read'
    print len(tokens), 'tokens extracted'
    print len(fdist), 'tokens unique'
    print fdist.most_common(50)
    # count number of unique tokens
    #frequency_total.plot(50)
        #print frequency_total, 'frequenc ', frequency, 'total ', len(tokens)

   # print unique_tokens/total_tokens, 'unicos/totales'
   # print len(frequency_total)
#   print fdist1(10)
#   print fdist1(1)/total_tokens
#def frequencies():
#count of that word divided by the total number of samples

#getWordsFromFile('es.txt')
f = open('en.txt')
# We will extract words by lowercase and delete punctuation using the string function
#get a list of the stopwords
#stp = nltk.corpus.stopwords.words('english')
#from your text of words, keep only the ones NOT in stp
#filtered_text = [w for w in f if not w in stp]
raw = f.read().lower().translate(None, string.punctuation)
tokens = tokenizeByWhiteSpaces(raw)
fdist = distributionList(tokens)
expectedZipLawCurve(fdist)
computeZipfsLaw(fdist)
#counters()
#constructs an array with the list of words
#wlist=[]
#wlist.append(fdist.keys())
#print len(wlist)
