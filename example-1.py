import nltk
from nltk import bigrams
from nltk.tokenize import word_tokenize

document_text = open('~/nltk_data/corpora/genesis/english-web.txt', 'r')
text_string = document_text.read().lower()
tokens = word_tokenize(text_string)
result = bigrams(tokens)
print(result)
# close file
