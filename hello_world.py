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
    fdist1.plot(50, cumulative=False)
    print len(raw), 'chars read'
    print len(tokens), 'tokens extracted'
    print len(fdist1), 'tokens unique'
    print fdist1.most_common(50)
    #print the 50 most frequent words
    for word, frequency in fdist1.most_common(50):
        print(u'{};{}'.format(word, frequency))

    for word, frequency in fdist1.most_common(50):
        frequency_total = frequency / len(fdist1)
        print frequency_total

    print unique_tokens/total_tokens, 'unicos/totales'
#   print fdist1(10)
#   print fdist1(1)/total_tokens
#def frequencies():
#count of that word divided by the total number of samples

def getWordsFromFile(inF):
    "get a list of words from a text file"
    lines = map(lambda x: x.replace('\n', '').lower(), open(inF).readlines())
    words = []
    for line in lines:
        for word in line.split():
            words.append(word)
    print len(words), 'words read'
    return words
    print(words)

#getWordsFromFile('es.txt')
tokenizeByWhiteSpaces()