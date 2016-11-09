import nltk
import math
training_corpus = ['I want to knwo what this does.\n', 'and also this.\n']
unigrams = []
bigrams = []
trigrams = []
for sentence in training_corpus:
        # Tokenize a string to split off punctuation other than periods
    tokens = ['START_SYMBOL'] + nltk.word_tokenize(sentence.strip()) + ['STOP_SYMBOL']
    # tokens.insert(0, 'START_SYMBOL')
    # tokens.append('STOP_SYMBOL')
    unigrams += tokens
    bigrams += list(nltk.bigrams(tokens))
    trigrams += list(nltk.trigrams(tokens))
unigram_f = nltk.FreqDist(tokens)
bigram_f = nltk.FreqDist(bigrams)
trigram_f = nltk.FreqDist(trigrams)
print unigram_f, bigram_f, trigram_f
unigram_p = {}
bigram_p = {}
trigram_p = {}
    # division returns integers if the input are all integers. math.log cannot take 0 or negative elements.
for unigram in unigram_f:
    log_probability = math.log(unigram_f[unigram]/float(len(unigrams)), 2)
    unigram_p[unigram] = log_probability
for bigram in bigram_f:
    print bigram
    print unigram_f[bigram[0]]
    log_probability = math.log(bigram_f[bigram]/float(unigram_f[bigram[0]]), 2)
    bigram_p[bigram] = log_probability
for trigram in set(trigrams):
    log_probability = math.log(trigram_f[trigram] / float(bigram_f[trigram[:2]]), 2)
    trigram_p[trigram] = log_probability
print unigram_p
print bigram_p
print trigram_p

corpus = ['I want to not know what this does.\n']

for sentence in corpus:
    tokens = nltk.word_tokenize(sentence.strip())
    tokens.insert(0, 'START_SYMBOL')
    tokens.append('STOP_SYMBOL')
    score = 1
    for word in tokens:
        if word in unigram_p:
            score = score * unigram_p[word]
        else:
            score = -1
            break
    print score

# probability of a word given its predecessor:


