import math
import nltk
import time
import collections

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a
# newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the
# log probability of that ngram
# it is faster to loop over short lists than over very long lists, because short lists take up less memory

def calc_probabilities(training_corpus):
    unigrams_c = collections.defaultdict(int)
    bigrams_c = collections.defaultdict(int)
    trigrams_c = collections.defaultdict(int)
    for sentence in training_corpus:
        # Tokenize a string to split off punctuation other than periods
        #tokens1 = nltk.word_tokenize(sentence.strip()) + [STOP_SYMBOL]
        tokens1 = sentence.strip().split() + [STOP_SYMBOL]
        tokens2 = [START_SYMBOL] + tokens1
        tokens3 = [START_SYMBOL] + tokens2
        #print tokens1

        #count unigrams:
        for token in tokens1:
            unigrams_c[token] +=1

        # count bigrams:
        for token in nltk.bigrams(tokens2):
            bigrams_c[token] += 1

        # count trigrams:
        for token in nltk.trigrams(tokens3):
            trigrams_c[token] +=1

    # unigram probability:
    # in order to make the script faster, use dict.iteritems instead of looping over the dictionary.
    # the .iteritems creates a generator object, so instead of heaving the whole dictionary in memory (slow), it just
    # has one item at the time.
    # key in unigram_p has to be a tuple, because of the code in q1_output. if it's ot a tuple, it will only put the
    # first letter of the word in the output file
    total = sum(unigrams_c.values())
    # unigram_p = {k: math.log(unigrams_c[k]/float(total), 2) for k in unigrams_c}
    unigram_p = {(k,): math.log(float(v)/total, 2) for (k, v) in unigrams_c.iteritems()}
    # print unigram_p

    # bigram probability:
    # we need to add a value for START_SYMBOL to the unigram_c, now it's null. the number of startsymbols is the
    # number of sentences in the training corpus
    unigrams_c[START_SYMBOL] = len(training_corpus)
    bigram_p = {k: math.log(bigrams_c[k]/float(unigrams_c[k[0]]), 2) for k, v in bigrams_c.iteritems()}

    # trigram probability (MLE):
    # again, add value for the startsymbol bigram
    bigrams_c[(START_SYMBOL, START_SYMBOL)]= len(training_corpus)
    trigram_p = {k: math.log(trigrams_c[k]/float(bigrams_c[k[0:2]]), 2) for k, v in trigrams_c.iteritems()}

    return unigram_p, bigram_p, trigram_p


# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability
# of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1] + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()
    for trigram in trigrams_keys:
        outfile.write(
            'TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline
# character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc.

def score(ngram_p, n, corpus):
    scores = []
    for sentence in corpus:
        tokens = sentence.strip().split() + [STOP_SYMBOL]
        score = 0
        if n == 1:
            ngrams = [(token,) for token in tokens]
        elif n == 2:
            ngrams = nltk.bigrams([START_SYMBOL]+ tokens)
        elif n == 3:
            ngrams = nltk.trigrams([START_SYMBOL, START_SYMBOL]+tokens)
        else:
            print 'Making an n-gram model with n>3 is useless'
            break
        # the score of a sentence is calculated by product of probabilities, not by sum. However, if we have log
        # probabilities, we do sum them. This is basic algebra.
        # Usually one stores not the actual probabilities but rather their logarithm. The reason is that adding numbers
        # is faster than multiplying them. As an added benefit, you don't have to worry about the dynamic range of the
        # floating point data type you use (the actual probability could be rather close to 0 and could cause underflow,
        # but this is unlikely for its logarithm).
        for ngram in ngrams:
            try:
                p = ngram_p[ngram]
            except KeyError:
                score = MINUS_INFINITY_SENTENCE_LOG_PROB
                break
            score += p
        scores.append(score)
    return scores


# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log
# probability of that ngram
# Like score(), this function returns a python list of scores
# interpolation is P(w3|w1, w2) = lambda1 P(w3|w1, w2) + lambda2 P(w3|w2) + lambda3 P(w3) st lambda 1 + lambda 2 + lambda 3 =1
def linearscore(unigrams, bigrams, trigrams, corpus):
    l = 1.0/3
    scores = []
    for sentence in corpus:
        tokens = [START_SYMBOL, START_SYMBOL]+sentence.strip().split() + [STOP_SYMBOL]
        trigram_sentence = nltk.trigrams(tokens)
        linear_score = 0
        for trigram in trigram_sentence:
            try:
                p3 = trigrams[trigram]
            except KeyError:
                p3 = MINUS_INFINITY_SENTENCE_LOG_PROB
            try:
                p2 = bigrams[trigram[0:2]]
            except KeyError:
                p2 = MINUS_INFINITY_SENTENCE_LOG_PROB
            try:
                p1 = unigrams[trigram[2]]
            except KeyError:
                p1 = MINUS_INFINITY_SENTENCE_LOG_PROB
            # I stole the formula below, I do not know logarithms to know how to change the formula above. Also no idea
            # if the formula is right.
            linear_score += math.log(l * (2 ** p3) + l * (2 ** p2) + l * (2 ** p1), 2)
        scores.append(linear_score)
    return scores


DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'


# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'


if __name__ == "__main__": main()
