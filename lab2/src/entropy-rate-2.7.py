import re, codecs, random, math, textwrap
from collections import defaultdict, deque, Counter

def tokenize(file_path, tokenizer):
	with codecs.open(file_path, mode="r", encoding="utf-8") as file:
		for line in file:
			for token in tokenizer(line.lower().strip()):
				yield token
				
def chars(file_path):
	return tokenize(file_path, lambda s: s + " ")
	
def words(file_path):
	return tokenize(file_path, lambda s: re.findall(r"[a-zA-Z']+", s))

def markov_model(stream, model_order):
	model, stats = defaultdict(Counter), Counter()
	circular_buffer = deque(maxlen = model_order)
	
	for token in stream:
		prefix = tuple(circular_buffer)
		circular_buffer.append(token)
		if len(prefix) == model_order:
			stats[prefix] += 1.0
			model[prefix][token] += 1.0
	return model, stats


def smoothed_markov_model(tagged_stream, model_order, smooth):
    model, stats = defaultdict(Counter), Counter()
    token_circular_buffer = deque(maxlen=model_order)
    tag_circular_buffer = deque(maxlen=model_order)

    for token, tag in tagged_stream:
        token_prefix = list(token_circular_buffer)
        tag_prefix = list(tag_circular_buffer)
        prefix = tuple(tag_prefix[:smooth] + token_prefix[smooth:])

        token_circular_buffer.append(token)
        tag_circular_buffer.append(tag)

        token = token if smooth < model_order+1 else tag
        if len(prefix) == model_order:
            stats[prefix] += 1.0
            model[prefix][token] += 1.0
    return model, stats

def entropy(stats, normalization_factor):
    return -sum(proba / normalization_factor * math.log(proba / normalization_factor, 2) for proba in stats.values())

def entropy_rate(model, stats):
    normalization_factor = sum(stats.values())
    return sum(stats[prefix] * entropy(model[prefix], stats[prefix]) for prefix in stats) / normalization_factor

def pick(counter):
	sample, accumulator = None, 0
	for key, count in counter.items():
		accumulator += count
		if random.randint(0, accumulator - 1) < count:
			sample = key
	return sample
	
def generate(model, state, length):
	for token_id in range(0, length):
		yield state[0]
		state = state[1:] + (pick(model[state]), ) 

from entropy_functions import get_tagged_words_from_file
n = 3

#corpus = words("../data/en.txt")
#model, stats = markov_model(corpus, n-1)

smooth = 2
tagged_corpus = get_tagged_words_from_file('../data/taggedBrown.txt')
model, stats = smoothed_markov_model(tagged_corpus, n-1, smooth)
print "Entropy rate:", entropy_rate(model, stats)

#print textwrap.fill(" ".join(generate(model, pick(stats), 300)))

    # Copyright (C) 2013, Clement Pit--Claudel (http://pit-claudel.fr/clement/blog)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of 
# this software and associated documentation files (the "Software"), to deal in 
# the Software without restriction, including without limitation the rights to 
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so, 
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.