
Filepaths are relative to `text_analysis`, not the root `wac` directory.

To produce the csv datasets and images for the edge bundling and difference of proportions analysis, run `python3 a_bundling_dop.py`. The Python script also triggers `c_edge_bundling.R`, which produces the D3 visualizations.

To produce the csv datasets and images for the bipartite network analysis, run `python3 b_bipartite_networks.py`. 

Igor's original data is in `data/from_igor`. To re-run with new data, just replace `transcripts.csv` and / or `coded.csv` with updated datasets. Scripts will work provided the column names don't change. All intermediary datasets (e.g. edgelists for semantic networks with tf-idf weights) are stored in `data`.

John McLevey 
December 14th, 2020

