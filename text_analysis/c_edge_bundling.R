# World After COVID Edge Bundling Plots
# John McLevey, December 2020

# This visualization script reads in the output of wac.py.
# Any modifications to the network etc. would ideally be done in 
# wac.py, but you could modify the igraph objects here if you prefer.

library(edgebundleR)
library(igraph)
library(tidyverse)

process <- function(data_path, html_path){
  edges <- read_csv(data_path) # includes alpha and weight by default
  edges_ij <- as.matrix(edges[1:2]) # igraph happiest with just ij
  g <- graph_from_edgelist(edges_ij, directed = FALSE) 
  eb <- edgebundle(g, fontsize = 8, padding=120, nodesize = c(5, 5))
  saveEdgebundle(eb, html_path, selfcontained = TRUE) # writes the html files to disk
  eb # nice to still see them in RStudio ;) 
}

process("~/Desktop/text_analysis/data/semantic_q1.csv", "~/Desktop/text_analysis/images/ebundle_semantic_q1.html")
process("~/Desktop/text_analysis/data/semantic_q2.csv", "~/Desktop/text_analysis/images/ebundle_semantic_q2.html")
process("~/Desktop/text_analysis/data/semantic_q3.csv", "~/Desktop/text_analysis/images/ebundle_semantic_q3.html")
process("~/Desktop/text_analysis/data/semantic_q4.csv", "~/Desktop/text_analysis/images/ebundle_semantic_q4.html")

# ALTERNATIVE OPTION

# The edgebundleR package is really cool because you get D3 out of the box, but 
# you can't re-order the nodes without getting into the source code. It's better
# and more informative when they are clustered by relationships (e.g. clusters / communities)
# I don't see an obvious way to do that here either, and the documentation is... extensive
# https://jokergoo.github.io/circlize_book/book/
# Plus, this is statis, so less useful in my opinion. Although for a print publication obviously
# it's the better option. Could probably tweak the visualization with a bit of time and patience ;) 

library(circlize)

chord_pipeline <- function(data_path, image_path){
  edges <- read_csv(data_path) # includes alpha and weight by default
  
  chordDiagram(edges[1:3], annotationTrack = "grid", preAllocateTracks = list(track.height = 0.1))
  
  circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {
    xlim = get.cell.meta.data("xlim")
    xplot = get.cell.meta.data("xplot")
    ylim = get.cell.meta.data("ylim")
    sector.name = get.cell.meta.data("sector.index")
    if(abs(xplot[2] - xplot[1]) < 10) {
      circos.text(mean(xlim), ylim[1], sector.name, facing = "clockwise",
                  niceFacing = TRUE, adj = c(0, 0.5))
    } else {
      circos.text(mean(xlim), ylim[1], sector.name, facing = "inside",
                  niceFacing = TRUE, adj = c(0.5, 0))
    }
  }, bg.border = NA)
}

chord_pipeline('~/Desktop/text_analysis/data/semantic_q1.csv')