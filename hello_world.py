from __future__ import division
from nltk.tokenize import WhitespaceTokenizer
import nltk, re, pprint

def tokenizeText():
#open text file and load it in a variable
    f = open('en.txt')
    raw = f.read()
    print len(raw), 'chars read'
    #tokenize text from file
    #tokens = nltk.word_tokenize(raw)
    unset = set(raw)
    print len(unset) , ' sets extracted'
#tokenize by whitespaces and lowercase it gives exactly the same number of words than auxiliar.py
    tokens = WhitespaceTokenizer().tokenize(raw.lower())
    print len(tokens), 'tokens extracted'
    #count total number of tokens
    total_tokens = len(tokens)
    #count number of unique tokens
    unique_tokens = len(set(tokens))
    #create frquency distribution table and plot it
    fdist1 = nltk.FreqDist(tokens)
 #   fdist1.plot(50, cumulative=False)
    print len(fdist1), 'tokens unique'
    print len(set(tokens)), 'set of tokens'
    print fdist1.most_common(10)

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
tokenizeText()