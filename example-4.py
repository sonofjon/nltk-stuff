import nltk
from nltk import bigrams
from nltk import trigrams

document_text = open('nltk_data/corpora/genesis/english-web.txt', 'r')

# split the texts into tokens
tokens = nltk.word_tokenize(document_text)
tokens = [token.lower() for token in tokens if len(token) > 1] # same as unigrams
bi_tokens = bigrams(tokens)
tri_tokens = trigrams(tokens)

print [(item, tri_tokens.count(item)) for item in sorted(set(tri_tokens))]
