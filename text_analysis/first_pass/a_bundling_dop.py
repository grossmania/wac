# WORLD AFTER COVID
# John McLevey
# Winter 2020 

import os
import csv

import warnings
warnings.filterwarnings("ignore")

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

import wac_utilities as wu # these are just some utility functions I wrote for this analysis

# DATA

d = pd.read_csv('data/from_igor/transcripts.csv', encoding='Latin1')
custom_stops = ['covid', 'covid19', 'feel','opportunity','term','kind','negative','take','certain','important','terms','think','significant','pandemic','towards','positive','better','level','one','thing','can','around','much','need','like','go','something','lot','actually','even','see','really','things','time','many','going','change','changes','say','get','greater','certainly','now','long','also','may','first','might','new','know','done', 'wisdom']

# ANALYSIS PIPELINE PER QUESTION

def pipeline(question, qnum, path, threshold=.12):
	"""
	Pass the question Series and output the graphs, etc.
	qnum is just the question number (for data filepath)
	If using TFIDF, threshold should be a float. If using counts, threshold should be an integer.
	"""

	# Pre-processing
	q = wu.prep(question, within_sentences=False, custom_stops=custom_stops)
	
	# network processing
	G = wu.coocnet(q, tfidf=True, threshold=threshold) # change tfidf to False if you want to use a count-based approach instead
	G.remove_edges_from(nx.selfloop_edges(G)) 
	G = wu.giant(G) # extract the giant component only
	wu.viz_giant(G, path, cd = True) # These are muh looking static visualizations. I used them a lot when iterating. 
	
	# edgelists for edge bundling in R
	with open(f"data/semantic_q{str(qnum)}.csv", 'w') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(['i','j','alpha','weight'])
		for e in G.edges(data=True):
			writer.writerow([e[0], e[1], e[2]['alpha'], e[2]['weight']])

# EXECUTE PIPELINE

pipeline(d['Q1'], 1, 'images/giant_q1.pdf')
pipeline(d['Q2'], 2, 'images/giant_q2.pdf')
pipeline(d['Q3'], 3, 'images/giant_q3.pdf')
pipeline(d['Q4'], 4, 'images/giant_q4.pdf')

full = pd.concat([d['Q1'], d['Q2'], d['Q3'], d['Q4']])
pipeline(full, 'full', 'images/giant_all.pdf', threshold=.12)

# DIFFERENCE OF PROPORTIONS

def dop(cat_a, cat_b, image_filepath, num_words=25,):
	"""
	The two questions / categories to compare.
	"""
	# pre-process text
	cat_a = wu.prep(cat_a, within_sentences=False, custom_stops=custom_stops)
	cat_b = wu.prep(cat_b, within_sentences=False, custom_stops=custom_stops)

	dpvect = CountVectorizer(stop_words="english", lowercase = True, strip_accents='ascii')
	matrix = dpvect.fit_transform([" ".join(cat_a), " ".join(cat_b)])
	wac = pd.DataFrame(matrix.toarray(), columns=dpvect.get_feature_names())

	# the n most frequently used words in each category of text
	freq_cat_a = wac.T.sort_values(0, ascending=False)[:num_words][0]
	freq_cat_b = wac.T.sort_values(1, ascending=False)[:num_words][1]

	# construct difference of proportions dataframe
	all_words = wac.sum(axis=1)
	wac = wac.iloc[:,0:].div(all_words, axis=0)
	wac.loc[2] = wac.loc[0] - wac.loc[1]
	diff = pd.DataFrame(wac.loc[2].sort_values(axis=0, ascending=False))
	diff.columns = ['DoP']
	diff['Word'] = diff.index
	# comparison dataframe
	comp = pd.concat([diff.head(num_words),diff.tail(num_words)])
	
	# visualize dop
	fig, ax = plt.subplots(figsize=(7, 9))
	ax.hlines(comp['Word'], xmin=0, xmax=comp['DoP'], linewidth=1)
	ax.plot(comp['DoP'][:num_words], comp['Word'][:num_words], "o", color='black', markersize = 6)
	ax.plot(comp['DoP'][num_words:], comp['Word'][num_words:], "o", color='black', markersize = 6)
	plt.axvline(x=0, color='black', linewidth=2)
	yticks = plt.yticks(comp['Word'])
	xlab = plt.xlabel('\nDifference of Proportions')
	fig.tight_layout()
	plt.savefig(image_filepath)

	return comp, diff

# positive vs. negative
comp, diff = dop(d['Q1'], d['Q3'], 'images/dop_consequences_pn.pdf')
comp.to_csv('data/dop_small_df_pn_consequences.csv', index=False)
diff.to_csv('data/dop_full_df_pn_consequences.csv', index=False)


# wisdom for positive vs. wisdom for negative
comp, diff = dop(d['Q2'], d['Q3'], 'images/dop_wisdom_pn.pdf')
comp.to_csv('data/dop_small_df_wisdom_pn.csv', index=False)
diff.to_csv('data/dop_full_df_wisdom_pn.csv', index=False)

# execute the edge bundling R script
os.system('Rscript c_edge_bundling.R')