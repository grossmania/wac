# WORLD AFTER COVID
# John McLevey
# Winter 2020 

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

import spacy 
nlp = spacy.load('en_core_web_lg')

from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess

import matplotlib.pyplot as plt

import networkx as nx
import backbone_network as bb
import community


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


def coocnet(processed, tfidf=True, threshold = 0, backbone=True): 
    """
    Accepts lists of processed texts and returns a co-occurrence network of terms within individual responses. 
    Co-occurrences are weighted by tf-idf if set to True. The adjacency matrix (projected from the incidence 
    matrix) can be thresholded. Use float for tfidf and integers for count.
    """
    if tfidf is True:
        vect = TfidfVectorizer(stop_words='english') 
    else:
        vect = CountVectorizer(stop_words='english') 
    
    m = vect.fit_transform(processed)
    words = vect.get_feature_names()
    
    # project the incidence matrix to a semantic co-occurrence matrix
    m = m.todense()
    adj_mat = m.transpose() * m
    adj_mat[adj_mat < threshold] = 0 # default is 0, so has not effect unless you provide a float or integer
    # adj_mat = np.fill_diagonal(adj_mat, 0) # zero out the diagonal
    
    adj_mat = pd.DataFrame(adj_mat)
    adj_mat.columns = words
    adj_mat.index = words
    
    G = nx.from_pandas_adjacency(adj_mat)

    if backbone is True: 
        G = bb.get_graph_backbone(G)
    G.name = 'Semantic Network'
    return G


def giant(network):
    comps = sorted(nx.connected_components(network), key=len, reverse=True)
    giant = network.subgraph(comps[0])
    return giant
    
    
def viz_giant(G, path, cd = True, labs=True):
    G = giant(G)
    
    fig, ax = plt.subplots(figsize=(14,10))
    pos = nx.spring_layout(G, seed=12)
    if cd is True:
        partition = community.best_partition(G)
        modularity = community.modularity(partition, G)
        colors = [partition[n] for n in G.nodes()]
        my_colors = plt.cm.Set2

        nx.draw_networkx_nodes(G, pos = pos, node_size = 50, node_color=colors, cmap = my_colors)
        nx.draw_networkx_edges(G, pos, edge_color = '#D4D5CE')
    
        if labs is True:
            labs = nx.draw_networkx_labels(G, pos=pos, font_size=8)
    else:
        nx.draw_networkx_nodes(G, pos = pos, node_size = 50, node_color = 'lightgray')
        nx.draw_networkx_edges(G, pos, edge_color = 'lightgray')
        if labs is True:
            labs = nx.draw_networkx_labels(G, pos=pos, font_size=8)
    plt.axis('off')
    plt.savefig(path)   
