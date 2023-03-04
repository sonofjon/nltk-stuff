import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

frequency = {}
# document_text = open('words.txt', 'r')
document_text = open('nltk_data/corpora/genesis/english-web.txt', 'r')
# document_text = open('nltk_data/corpora/genesis/swedish.txt', 'r')
text_string = document_text.read().lower()
match_pattern = re.findall(r'\b[a-z]{3,15}\b', text_string)

finder = BigramCollocationFinder.from_words(match_pattern)
bigram_measures = nltk.collocations.BigramAssocMeasures()
print(finder.nbest(bigram_measures.pmi, 10))
