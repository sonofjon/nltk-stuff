import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

wanted = set(['inflation', 'gold', 'bank'])

with open("nltk_data/corpora/genesis/english-web.txt") as file:
    text = file.read()

words = [word.casefold() for sentence in sent_tokenize(text)
         for word in word_tokenize(sentence)]
words_fd = nltk.FreqDist(words)
bigram_fd = nltk.FreqDist(nltk.bigrams(words))
finder = BigramCollocationFinder(words_fd, bigram_fd)
bigram_measures = nltk.collocations.BigramAssocMeasures()
print(finder.nbest(bigram_measures.pmi, 5))
print(finder.score_ngrams(bigram_measures.raw_freq))
