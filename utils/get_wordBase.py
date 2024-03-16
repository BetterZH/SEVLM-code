import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet

#nltk.download('omw-1.4')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(treebank_tag):
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

lemmatizer = WordNetLemmatizer() 

def lemmatize_sentence(sentence):
    lemmatized_output=[lemmatizer.lemmatize(w,get_wordnet_pos(pos)) for w,pos in nltk.pos_tag(sentence)]
    return lemmatized_output
