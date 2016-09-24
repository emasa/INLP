import nltk
from nltk import *
from nltk.tokenize import word_tokenize

def tokenizeText():
    var = open('en.txt')
    var2 = nltk.word_tokenize(var)
    #print len(var), 'tokens read'


def importingBrownCorpusFromNLTK(outF):
    # type: (object) -> object
    "importing tagged brown corpus from NLTK and writing on a file OutF"
    outF = open(outF,'w')
    from nltk.corpus import brown
    brown_news_tagged = brown.tagged_words(categories='news')
    print 'size', len(brown_news_tagged)
    for i in brown_news_tagged:
        outF.write(i[0]+'\t'+i[1]+'\n')
    outF.close()



def getWordsFromFile(inF):
    "get a list of words from a text file"
    lines = map(lambda x: x.replace('\n', '').lower(), open(inF).readlines())
    words = []
    for line in lines:
        for word in line.split():
            words.append(word)
    print len(words), 'words read'
    return words

#getWordsFromFile('en.txt')
tokenizeText()
