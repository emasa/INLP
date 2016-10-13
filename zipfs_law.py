from __future__ import division
from nltk.tokenize import WhitespaceTokenizer
import nltk, re, pprint, string

def tokenizeByWhiteSpaces():
    #This function will tokenize the text into words
    #We will extract words by lowercase and delete punctuation using the string function
    f = open('en.txt')
    raw = f.read().lower().translate(None, string.punctuation)
    #tokenizeing by whitespaces and lowercase it gives exactly the same number of words than auxiliar.py provided by Horacio so we assume it's the correct number of tokens (words)
    tokens = WhitespaceTokenizer().tokenize(raw)
    #count total number of tokens
    total_tokens = len(tokens)
    #count number of unique tokens
    unique_tokens = len(set(tokens))
    #create frquency distribution table and plot it
    fdist1 = nltk.FreqDist(tokens)
    #fdist1.plot(50, cumulative=False)
    print len(raw), 'chars read'
    print len(tokens), 'tokens extracted'
    print len(fdist1), 'tokens unique'
    print fdist1.most_common(50)
    #print the 50 most frequent words
   # for word, frequency in fdist1.most_common(50):
    #    print(u'{};{}'.format(word, frequency))
    print len(fdist1)
    print fdist1.N(), 'funcion N'
    import numpy as np
    #frequency_total = {}
    index = 1
    #for word, frequency in fdist1.iteritems():
    for word, frequency in fdist1.most_common():
        #frequency_total = frequency / len(tokens)
        #frequency_total.update({word:frequency / len(tokens)})
        frequency_total = np.append(word,frequency/len(tokens))
        K = index * frequency
        index  += 1
    print K
    print index

    #frequency_total.plot(50)
        #print frequency_total, 'frequenc ', frequency, 'total ', len(tokens)

   # print unique_tokens/total_tokens, 'unicos/totales'
   # print len(frequency_total)
#   print fdist1(10)
#   print fdist1(1)/total_tokens
#def frequencies():
#count of that word divided by the total number of samples

#getWordsFromFile('es.txt')
tokenizeByWhiteSpaces()