import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

wanted = set(['inflation', 'gold', 'bank'])

with open("nltk_data/corpora/genesis/english-web.txt") as file:
    text = file.read()

words = [word.casefold() for sentence in sent_tokenize(text)
         for word in word_tokenize(sentence)]

# finder can be constructed from words directly
finder = TrigramCollocationFinder.from_words(words)
# filter words
finder.apply_word_filter(lambda w: w not in wanted)
# top n results
trigram_measures = nltk.collocations.TrigramAssocMeasures()
print(sorted(finder.nbest(trigram_measures.raw_freq, 2)))
