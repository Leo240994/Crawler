import string
import re
from nltk.tokenize import word_tokenize
import json

vocabulary = set([])
doc_names = set([])
tuples = []

def read(jsonName):
    #cargar
    file = open(jsonName, 'rt')
    data = json.load(file)
    for item in data['item']:
        #tokenizar
        tokens = word_tokenize(item['content'])
        #minuscula
        tokens = [t.lower() for t in tokens]
        words = [w for w in tokens if w.isalpha()]
        document_voc = set([])

        for w in words:
            vocabulary.add(w)
            document_voc.add(w)

        doc_names.add(item['url'])
        tuples.append((item['url'], document_voc))

