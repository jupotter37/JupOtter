# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 23:10:18 2023

@author: raamc
"""

import networkx as nx
import matplotlib.pyplot as plt      

import networkx as nx

# Create an empty graph object
G = nx.Graph()

# Read the edge list file and add edges to the graph
with open('hw1_p5.txt', 'r') as file:
    for line in file:
        # Assuming the file contains an edge per line as "node1 node2"
        node1, node2 = map(int, line.split())
        G.add_edge(node1, node2)

# Now G contains the graph built from the edge list in hw1_p5.txt


nx.draw(G, with_labels=True)
plt.show()


# Assuming G is your graph object
# If not, load your graph here

centrality = nx.eigenvector_centrality(G, max_iter=500)


# Extract node indices and corresponding centrality values
nodes = list(centrality.keys())
values = list(centrality.values())

# Plotting as dots
plt.scatter(nodes, values)
plt.xlabel('Node Index')
plt.ylabel('Eigenvector Centrality')
plt.title('Eigenvector Centrality as a function of Node Index')
plt.show()

# List of edges to be deleted
edges_to_delete = [
    (8, 88), (15, 87), (25, 103), (59, 82), (111, 59),
    (169, 68), (77, 181), (82, 81), (105, 82), (86, 96)
]

# Initial number of edges
initial_num_edges = G.number_of_edges()

# Delete the edges
for edge in edges_to_delete:
    if G.has_edge(*edge):  # Check if the edge exists before trying to remove it
        G.remove_edge(*edge)

# Final number of edges
final_num_edges = G.number_of_edges()

# Calculate the fraction of edges deleted
fraction_deleted = (initial_num_edges - final_num_edges) / initial_num_edges
print(f"Fraction of edges deleted: {fraction_deleted:.2f}")


try:
    # Recompute eigenvector centrality
    centrality = nx.eigenvector_centrality(G, max_iter=500)
    
    # Extract node indices and corresponding centrality values
    nodes = list(centrality.keys())
    values = list(centrality.values())
    
    # Plotting as dots
    plt.scatter(nodes, values)
    plt.xlabel('Node Index')
    plt.ylabel('Eigenvector Centrality')
    plt.title('Eigenvector Centrality after Deleting Edges')
    plt.show()
    
except nx.PowerIterationFailedConvergence:
    print("Eigenvector centrality computation did not converge after deleting edges. Consider using a different centrality measure or increasing the number of iterations.")

import numpy as np


# Create adjacency matrices
matrix = nx.to_numpy_matrix(G)

# Find eigenvalues
eigenvalues = np.linalg.eigvals(matrix)

# Sort eigenvalues in increasing order
eigenvalues = np.sort(eigenvalues)

# Plot eigenvalues
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(eigenvalues, 'o-')
plt.title('Eigenvalues Before Deleting Edges')
plt.xlabel('Index')
plt.ylabel('Eigenvalue')

plt.tight_layout()
plt.show()
