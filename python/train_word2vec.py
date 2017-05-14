import spacy
nlp = spacy.load('en') # this should take some time like 10s to load
import numpy as np
import scipy as sp
from nltk.corpus import stopwords
import pickle
from gensim.models import word2vec


from datetime import datetime

test_loc = 'test_dump.bin'
train_loc = 'train_dump.bin'

from spacy.tokens.doc import Doc
docs = []
i = 0
#print datetime.now().strftime('%Y-%m-%d %H:%M:%S')
with open(train_loc, 'rb') as file_:
    for byte_string in Doc.read_bytes(file_):
        #if i%10000 == 0: print i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        docs.append(Doc(nlp.vocab).from_bytes(byte_string))
        i += 1

i=0
with open(test_loc, 'rb') as file_:
    for byte_string in Doc.read_bytes(file_):
        if i%100000 == 0: print i, datetime.now().strftime('%H:%M:%S')
        docs.append(Doc(nlp.vocab).from_bytes(byte_string))
        i += 1        

stops = stopwords.words('english')

lemmas = []
for idx, doc in enumerate(docs):
    lemma = []
    for word in doc:
        if (not word.is_punct) and (not word.text in stops):
            if word.lemma_ == '-PRON-':
                lemma.append(unicode(word))
            else:
                lemma.append(word.lemma_)
    lemmas.append(lemma)
    if idx%100000 == 0: print idx, datetime.now().strftime('%H:%M:%S')
print lemmas[:3]

model = word2vec.Word2Vec(lemmas)
model.save("saved_vec_mod.kf")
print "done!"

# send interupt signal if successful above
with open( "save.p", "wb" ) as f:
    pickle.dump(lemmas, f)
print "done done!"



