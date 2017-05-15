import spacy
nlp = spacy.load('en') # this should take some time like 10s to load
import numpy as np
import scipy as sp
import pandas as pd
import time
from datetime import datetime

from gensim.models import word2vec

model = word2vec.Word2Vec.load("saved_vec_mod.kf")

def vec_diff(doc1,doc2):
    a = doc1.vector
    b = doc2.vector
    return a - b
    
def wmdist(sent1,sent2):
    """This requires that the parameters @sent1 @sent2 are lists of strings ideally
    lemmatized and cleaned sentences from above"""
    return model.wmdistance([x.text for x in sent1], [x.text for x in sent2])

from nltk.corpus import wordnet as wn

def wn_sim(q1doc,q2doc):
    rez = []
    for word1 in q1doc:
        max_w1 = 0.0
        for word2 in q2doc:
            wordFromList1 = wn.synsets(word1.text)
            wordFromList2 = wn.synsets(word2.text)
            if wordFromList1 and wordFromList2:
                s = wordFromList1[0].wup_similarity(wordFromList2[0])
                if (s > max_w1):
                    max_w1 = s
        rez.append(max_w1)
    return reduce(lambda x,y: x+y, rez)/len(rez)    

test_loc = 'test_dump.bin'
train_loc = 'train_dump.bin'
train = pd.read_csv('../data/train.csv')


from spacy.tokens.doc import Doc
test_docs = []
train_docs = []
i = 0

from nltk.corpus import stopwords
stop = stopwords.words('english')


def clean(doc):
    b = [x for x in doc if not x.is_punct]
    a = [x.lemma_ if not x.lemma_ == '-PRON-' else x.text for x in b ]
    return nlp(u''.join([x + ' ' for x in a if x not in stop]))

#print datetime.now().strftime('%Y-%m-%d %H:%M:%S')
with open(train_loc, 'rb') as file_:
    for byte_string in Doc.read_bytes(file_):
        if i%100000 == 0: print i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        train_docs.append(clean(Doc(nlp.vocab).from_bytes(byte_string)))
        i += 1

d = []
ct = 0
for q1,q2 in [(train_docs[2*i],train_docs[2*i+1]) for i in range(len(train_docs)/2)]:
    
    d.append({
              'sysim': q1.similarity(q2), # spacy sim
              'wmd': wmdist(q1,q2), # returns float dist
              'vectdiff': vec_diff(q1,q2) # 300 dim vect
    })
    if ct % 10000 == 0: print ct, datetime.now().strftime('%H:%M:%S')
    ct = ct+1    
features = pd.DataFrame(d)
features.to_csv('features.csv')