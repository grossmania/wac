# World After Covid
# John McLevey, Winter 2020

import csv
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 

import spacy 
nlp = spacy.load('en_core_web_lg')

from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess

r = pd.read_csv('data/from_igor/transcripts.csv', encoding='Latin1')
c = pd.read_csv('data/from_igor/codes.csv', encoding='Latin1')

who = c['Name']
custom_stops = ['covid', 'covid19', 'feel','opportunity','term','kind','negative','take','certain','important','terms','think','significant','pandemic','towards','positive','better','level','one','thing','can','around','much','need','like','go','something','lot','actually','even','see','really','things','time','many','going','change','changes','say','get','greater','certainly','now','long','also','may','first','might','new','know','done', 'wisdom']

q1c = [c for c in c.columns if 'q1' in c]
q2c = [c for c in c.columns if 'q2' in c]
q3c = [c for c in c.columns if 'q3' in c]
q4c = [c for c in c.columns if 'q4' in c]
q1c.append('Name')
q2c.append('Name')
q3c.append('Name')
q4c.append('Name')
c1 = c[q1c]
c2 = c[q2c]
c3 = c[q3c]
c4 = c[q4c]

# CUSTOM FUNCTIONS
def get_positive_codes(codes,df):
    results = {}

    for q in codes:
        if q != 'Name':
            positive = df[df[q] == 1]
            who = positive['Name'].tolist()
            results[q] = who

    return results    


def get_text(QID, posDict, responseDF):
    # prepare response dataframe for lookup
    small = r[QID]
    small.index = r.Name
    
    # perform lookups
    results = {}
    
    for k, v in posDict.items():
        text = []
        for name in v:
            text.append(small.loc[name])
        results[k] = " ".join(text).strip().replace('\xa0', '')
    
    return results


def bigram_process(texts):
    """
    Statistical model to detect bigrams. Results can be fed directly into the prep function.
    """
    words = [simple_preprocess(x, deacc=False) for x in texts if type(x) == str]  
    phrases = Phrases(words, min_count=1, threshold=0.8, scoring='npmi') # bigram model training
    bigram = Phraser(
        phrases)  # creates a leaner specialized version of the bigram model
    bigrams = list(
        bigram[words])  # concatenate words into bigrams (ie. word1_word2)
    bigrams = [' '.join(words) for words in bigrams]
    return bigrams


def prep(text, custom_stops, within_sentences=True):
    """
    This is a pre-processing function. It accepts original text in Series and returns a list of lemmatized nouns and proper nouns.
    """
    text = text.tolist() # covert Series to list
    text = bigram_process(text) # run bigram model
    docs = list(nlp.pipe(text)) 
    processed = [] 
    for doc in docs:
        if within_sentences is True:
            for sent in doc.sents:
                lemmas = [token.lemma_.lower() for token in sent if token.is_stop == False and token.pos_ in ['NOUN', 'PROPN'] and len(token) > 1]
                lemmas = [l for l in lemmas if l not in custom_stops]
                processed.append(" ".join(lemmas))
        else:
            lemmas = [token.lemma_.lower() for token in doc if token.is_stop == False and token.pos_ in ['NOUN', 'PROPN'] and len(token) > 1]
            lemmas = [l for l in lemmas if l not in custom_stops]
            processed.append(" ".join(lemmas))

    return processed 


def construct_dataframe(response_dict):
    df = pd.DataFrame.from_dict(response_dict, orient='index', columns=['Text'])
    themes = [f'({q})' for q in list(df.index)] 
    themes = [t.replace('_q1','') for t in themes]
    themes = [t.replace('_q2','') for t in themes]
    themes = [t.replace('_q3','') for t in themes]
    themes = [t.replace('_q4','') for t in themes]
    df['Theme'] = themes
    df.reset_index(drop=True, inplace=True)
    
    # pre-process text
    prepped = prep(df['Text'], custom_stops, within_sentences=False)
    
    df['Prepped'] = prepped
    return df   

def construct_edgelist(prepped_df, theme='Theme',text='Prepped', threshold = .12):
    small = prepped_df[[theme, text]]
    vect = TfidfVectorizer(stop_words='english') 
    m = vect.fit_transform(small[text])
    m2 = m.todense()
    m2[m2 < threshold] = 0

    words = vect.get_feature_names()
    tfidf_df = pd.DataFrame(m2, columns = words)
    tfidf_df.index = small[theme].tolist()
    
    words = [c for c in tfidf_df.columns]
    themes = [i for i in tfidf_df.index]

    theme_col = []
    word_col = []
    weight_col = []

    # get edges
    for t in themes:
        for w in words:
            val = tfidf_df.loc[t][w]
            if val > 0:
                theme_col.append(t) # the theme 
                word_col.append(w) # the word
                weight_col.append(val) # the TFIDF weight

    edges = pd.DataFrame([theme_col, word_col, weight_col]).T            
    edges.columns = ['Theme', 'Token', 'Weight']
    
    # hack because some words are in both sets (e.g. patience), using () to differentiate them
#     edges['Theme']= edges['Theme'].str.replace("q1","") 
#     edges['Theme']= edges['Theme'].str.replace("q2","")    
#     edges['Theme']= edges['Theme'].str.replace("q3","")    
#     edges['Theme']= edges['Theme'].str.replace("q4","")    
    
    return edges
   
def construct_network(wel):
    B = nx.Graph()
    B.add_nodes_from(wel['Theme'], bipartite=0)
    B.add_nodes_from(wel['Token'], bipartite=1)
    B.add_weighted_edges_from([wel.loc[index].tolist() for index in list(wel.index)])
    B.name = 'Theme-Token Network'
    
    theme_nodes = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    word_nodes = [n for n, d in B.nodes(data=True) if d["bipartite"] == 1]

    return B, theme_nodes, word_nodes    


def for_r(B, filepath):
    with open(filepath, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Theme','Word','TFIDF Weight'])
        for e in B.edges(data=True):
            writer.writerow([e[0], e[1], e[2]['weight']])
            
            
def bipartite_pipeline(code_text_question_dict, image_path, csv_path):
    # construct
    df = construct_dataframe(code_text_question_dict)
    el = construct_edgelist(df)
    B, theme_nodes, word_nodes = construct_network(el)
    
    # draw
    fig, ax = plt.subplots(figsize=(20,18))
    pos = nx.spring_layout(B, seed=12)
    nx.draw_networkx_nodes(B, pos=pos, nodelist=theme_nodes, node_shape='s', node_color='crimson', alpha=.6)
    nx.draw_networkx_nodes(B, pos=pos, nodelist=word_nodes, node_shape='s', node_color='darkgray', alpha=.6)
    nx.draw_networkx_edges(B, pos=pos, alpha=.8, edge_color='lightgray')
    labs = nx.draw_networkx_labels(B, pos=pos, font_size=10)
    fig.tight_layout()
    plt.axis('off')
    plt.savefig(image_path)
    plt.close()
    
    # write
    for_r(B, csv_path)    


# PROCESSING
q1 = get_positive_codes(q1c, c)
q2 = get_positive_codes(q2c, c)
q3 = get_positive_codes(q3c, c)
q4 = get_positive_codes(q4c, c)

code_text_q1 = get_text('Q1', q1, r)
code_text_q2 = get_text('Q2', q2, r)
code_text_q3 = get_text('Q3', q3, r)
code_text_q4 = get_text('Q4', q4, r)

bipartite_pipeline(code_text_q1, 'images/bipartite_q1.pdf', 'data/bipartite_q1.csv')
bipartite_pipeline(code_text_q2, 'images/bipartite_q2.pdf', 'data/bipartite_q2.csv')
bipartite_pipeline(code_text_q3, 'images/bipartite_q3.pdf', 'data/bipartite_q3.csv')
bipartite_pipeline(code_text_q4, 'images/bipartite_q4.pdf', 'data/bipartite_q4.csv')