
# Tokens
import csv
import itertools
import nltk
import numpy as np

nltk.download("book")
vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

with open('data/reddit-comments-2015-08.csv', encoding='utf-8') as f:
    reader = csv.reader(f, skipinitialspace=True)
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

print("Parsed %d sentences." % (len(sentences)))

tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

word_frequence = nltk.FreqDist(itertools.chain(*tokenized_sentences))
vocab = word_frequence.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)

word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])


# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
