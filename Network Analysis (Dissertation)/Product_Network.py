#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:33:32 2023

@author: Annarath

Dissertation: Market Basket Analysis using Product Network
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import leidenalg
import igraph as ig
import dataframe_image as dfi
from functools import reduce

"""
Import Data
"""

df = pd.read_csv('MBA_RetailData.csv', sep=';')
#522064 rows and 7 columns 

# Select only Itemname and BillNo to do market basket analysis
df_market = df[['BillNo', 'Itemname']]

df_market.isnull().sum() # 1455 missing itemname, not a lot hence can drop
df_market = df_market.dropna(subset=["Itemname"])

# Check Transactions and Items
df_market['Itemname'].value_counts()
# There are 4185 products

df_market['BillNo'].value_counts()
# There are 20210 transactions

"""
Data Preparation
"""
G = nx.Graph()
G.add_nodes_from(df_market['Itemname'], bipartite='products')
G.add_nodes_from(df_market['BillNo'], bipartite='transactions')
df_market['edges'] = list(zip(df_market.Itemname, df_market.BillNo))
G.add_edges_from(df_market['edges'])
type(G)
len(G.nodes()) # 24395 nodes
len(G.edges()) # 509829 edges

#Add Bipartite
#for n, d in G.nodes(data=True):
#    if type(n) == int:
 #       G.nodes[n]['bipartite'] = 'transactions'
  #  else:
   #     G.nodes[n]['bipartite'] = 'products'

#print(G.nodes(data=True))
#print(G.edges())

# Seperate product and transaction nodes
product_nodes = [n for n in G.nodes() if G.nodes[n]['bipartite'] == 'products']
transaction_nodes = [n for n in G.nodes() if G.nodes[n]['bipartite'] == 'transactions']

# Bipartite graph
#bi_graph = nx.draw(G, pos = nx.drawing.layout.bipartite_layout(G, product_nodes))

# Create Adjacency Matrix
bi_mat = nx.bipartite.biadjacency_matrix(G, row_order=transaction_nodes, column_order=product_nodes, format="coo").toarray()

# Create Data Frame
bi_mat_df = pd.DataFrame(bi_mat)
bi_mat_df['num_items'] = bi_mat_df.iloc[:,:].sum(axis=1)

# Drop all transactions that only contain one product
bi_mat_df = bi_mat_df.drop(bi_mat_df[bi_mat_df['num_items'] <= 1].index)
bi_mat_df = bi_mat_df.drop('num_items', axis=1)

df_count = pd.DataFrame(bi_mat_df.value_counts())
df_count.index.name = 'itemset'
df_count.columns = ['count']
df_count['itemset'] = df_count.index
df_count = df_count.reset_index(drop=True)

df_itemset = pd.DataFrame(df_count['itemset'].tolist())
df_itemset['count'] = df_count['count']
df_itemset['num_items'] = df_itemset.iloc[:,:-2].sum(axis=1)

# Compute Support value of each set then filter out < 0.0001 
df_itemset['support'] = df_itemset['count']/df_itemset.shape[0]

#df_itemset['support'].sort_values(ascending=False)
#df_itemset['support'].describe() #relatively low, hence should have lower threshold
#df_itemset['support'].value_counts() #17640 itemsets appeared only once in the record, it has suport value of 0.000071
#df_itemset[df_itemset['count'] == 3]
#df_itemset[df_itemset['count'] == 3]['support']
#df_itemset[df_itemset['count'] < 3]['num_items'].value_counts(ascending=False)
#df_itemset.groupby('count').count() #103 appeared only twice, support equals to 0.000113
#df_itemset[df_itemset['count'] > 2]
#df_itemset[df_itemset['count'] > 1]
#df_itemset[['count', 'support', 'num_items']]

# Drop all transactions that do not pass threshold
df_itemset = df_itemset.drop(df_itemset[df_itemset['support'] <= 0.00015].index) #86 transactions left

df_itemset_a = df_itemset.copy()
df_itemset_a = df_itemset.drop(['support', 'count', 'num_items'], axis=1)
df_itemset_b = df_itemset_a.copy()
count_list = df_itemset['count'].to_numpy()
df_itemset_b = df_itemset_b.mul(count_list, axis=0)

# Make a copy to match item names later 
df_itemset_label = df_itemset_a.copy()
df_itemset_label_b = df_itemset_b.copy()


bi_mat_support_a = np.array(df_itemset_a)
bi_mat_support_b = np.array(df_itemset_b)

# Adjacency matrix
adjacency_mat = np.matmul(bi_mat_support_a.transpose(),bi_mat_support_b)
# Diagonal represent number of transactions that product is in, hence if we want to plot product to product network we need to set it equals to 0
np.fill_diagonal(adjacency_mat, 0)

adjacency_mat_df = pd.DataFrame(adjacency_mat)

# Delete all the products that do not appear together with others in transactions
adjacency_mat_df = adjacency_mat_df.drop(adjacency_mat_df[adjacency_mat_df[:].sum() == 0].index)
adjacency_mat_df = adjacency_mat_df.loc[:, (adjacency_mat_df != 0).any(axis=0)]

"""
Network Generation
"""
# Generate Pruned Product Network
adj_prune_graph = nx.from_pandas_adjacency(adjacency_mat_df, create_using = nx.Graph)
nx.info(adj_prune_graph) # 163 nodes and 1768 edges

# Draw full network
plt.figure(figsize=(100, 100))
pos = nx.spring_layout(adj_prune_graph, seed=300, weight='weight')
nx.draw_networkx(adj_prune_graph, pos, edge_color='grey')
plt.title("Product Network without Communities", fontsize=80)
#plt.savefig("Network_00015.png", format="png")
plt.show()

# Degree Distribution
# Calculate Degree
degrees = [len(list(adj_prune_graph.neighbors(n))) for n in adj_prune_graph.nodes()]
#print(degrees)
# Plot a histogram of the degree distribution of the graph
plt.figure()
plt.hist(degrees)
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution of Product Network')
#plt.savefig("Degree_Dist")
plt.show()

1768/163

#Density
D_adj_prune = nx.density(adj_prune_graph)

"""
Community Detection: Louvain
"""
# Community based on optimal modularity
community = nx.community.louvain_communities(adj_prune_graph, seed=4500)
len(community) #there are 22 communitites found which represents baskets

# map colour for each community
def community_colours(graph, communities):
    number = len(communities[0])
    colours = ['navy','blue', 'yellow', 'purple', 'green', 'red', 'orange', 'brown', 'pink', 'olive', 'grey', 'salmon', 'gold', 'plum', 'coral', 'lime', 'royalblue', 'orchid', 'skyblue', 'peru', 'tomato', 'forestgreen'][:number]
    node_colours = []
    for node in graph:
        community_index = 0
        for comm in communities:
            if node in comm:
                node_colours.append(colours[community_index])
                break
            community_index += 1
    return node_colours

node_colour = community_colours(adj_prune_graph, community)

# Draw full network with 22 communities
plt.figure(figsize=(100, 100))
pos_lov = nx.spring_layout(adj_prune_graph, seed=300, weight='weight')
nx.draw_networkx(adj_prune_graph, pos_lov, node_color=node_colour, edge_color='grey')
plt.title("Product Network with Communities", fontsize=80)
#plt.savefig("Network_with_Community_00015.png", format="png")
plt.show()

# Take a look at each communitites
# Add community to node attributes
for c, v_c in enumerate(community):
    for v in v_c:
        # Add 1 to save 0 for external edges
        adj_prune_graph.nodes[v]['comm'] = c + 1

#Find internal edges and add their community to their attributes
for v, w, in adj_prune_graph.edges:
    if adj_prune_graph.nodes[v]['comm'] == adj_prune_graph.nodes[w]['comm']:
        # Internal edge, mark with community
        adj_prune_graph.edges[v, w]['comm'] = adj_prune_graph.nodes[v]['comm']
    else:
        # External edge, mark as 0
        adj_prune_graph.edges[v, w]['comm'] = 0

N_coms=len(community)
edges_coms=[]#edge list for each community
coms_G=[nx.Graph() for _ in range(N_coms)] #community graphs
colours = ['navy','blue', 'yellow', 'purple', 'green', 'red', 'orange', 'brown', 'pink', 'olive', 'grey', 'salmon', 'gold', 'plum', 'coral', 'lime', 'royalblue', 'orchid', 'skyblue', 'peru', 'tomato', 'forestgreen']

fig=plt.figure(figsize=(200, 260))
for i in range(N_coms):
  edges_coms.append([(u,v,d) for u,v,d in adj_prune_graph.edges(data=True) if d['comm'] == i+1])#identify edges of interest using the edge attribute
  coms_G[i].add_edges_from(edges_coms[i]) #add edges
  plt.subplot(6,4,i+1)#plot communities
  plt.title('Community '+str(i+1), fontsize=60)
  pos = nx.circular_layout(coms_G[i])
  nx.draw_networkx(coms_G[i],pos=pos,node_color=colours[i], node_size=1000, font_size=17)
  edge_labels = nx.get_edge_attributes(coms_G[i], "weight")
  nx.draw_networkx_edge_labels(coms_G[i],pos=pos, font_size=30, edge_labels=edge_labels)
#plt.savefig("Community_Network_label_00015.png", format="png")

fig=plt.figure(figsize=(32, 900))
for i in range(N_coms):
  edges_coms.append([(u,v,d) for u,v,d in adj_prune_graph.edges(data=True) if d['comm'] == i+1])#identify edges of interest using the edge attribute
  coms_G[i].add_edges_from(edges_coms[i]) #add edges
  plt.subplot(22,1,i+1)#plot communities
  plt.title('Community '+str(i+1), fontsize=60)
  pos = nx.circular_layout(coms_G[i])
  nx.draw_networkx(coms_G[i],pos=pos,node_color=colours[i], node_size=1000, font_size=17)
  edge_labels = nx.get_edge_attributes(coms_G[i], "weight")
  nx.draw_networkx_edge_labels(coms_G[i],pos=pos, font_size=30, edge_labels=edge_labels)
#plt.savefig("Community_Network_long_label_00015.png", format="png")

"""
Community Analysis
"""

# Match product name with each nodes
df_itemset_label = df_itemset_label.T
df_itemset_label['Product Name'] = df_market['Itemname']
df_itemset_label = df_itemset_label.iloc[adjacency_mat_df.index]
df_itemset_label.index.name = 'Product ID'
df_itemset_label['Product ID'] = df_itemset_label.index
df_itemset_label = df_itemset_label.reset_index(drop=True)
df_itemset_label = df_itemset_label.iloc[:,-2:]

commu_lab = pd.DataFrame()
commu_lab['Product ID'] = community
commnum = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
commu_lab['Community Number'] = commnum
commu_lab = commu_lab.explode('Product ID')

# Density within the network (how closely related products in baskets are)
D_community = [nx.density(coms_G[i]) for i in range(N_coms)]
D_community_df['Community Density'] = pd.DataFrame(D_community).round(3)
xlab = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
D_community_df['Community Number'] = xlab
D_community_df = D_community_df[['Community Density','Community Number']]
#plot
plt.figure()
plt.plot(xlab, D_community)
plt.xlabel("Community")
plt.ylabel("Density")
plt.xticks(xlab)
#plt.savefig("Density_community.png")
plt.show()

# Important Products, Degree Centrality
DC_adj_prune = nx.degree_centrality(adj_prune_graph)
DC_community = [nx.degree_centrality(coms_G[i]) for i in range(N_coms)]

# Plot degree centrality distribution
plt.figure()
plt.hist(DC_adj_prune)
plt.xlabel('Degree Centrality')
plt.ylabel('Frequency')
plt.show()

#put into table
DC_community_df = pd.DataFrame.from_dict(DC_community).T.round(3)
DC_community_df.index.name = 'Product ID'
DC_community_df['Degree Centrality'] = DC_community_df[DC_community_df.columns[:]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
DC_community_df = DC_community_df['Degree Centrality']

# Product linking one community to another, Betweenness Centrality
BC_adj_prune = nx.betweenness_centrality(adj_prune_graph)
BC_community = [nx.betweenness_centrality(coms_G[i]) for i in range(N_coms)]
BC_community_df = pd.DataFrame(BC_community).T.round(3)
BC_community_df.index.name = 'Product ID'
BC_community_df['Betweenness'] = BC_community_df[BC_community_df.columns[:]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
BC_community_df = BC_community_df['Betweenness']

# Closeness Centrality
CC_adj_prune = nx.closeness_centrality(adj_prune_graph)
CC_community = [nx.closeness_centrality(coms_G[i]) for i in range(N_coms)]
CC_community_df = pd.DataFrame(CC_community).T.round(3)
CC_community_df.index.name = 'Product ID'
CC_community_df['Closeness'] = CC_community_df[CC_community_df.columns[:]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
CC_community_df = CC_community_df['Closeness']

# Eigenvector Centrality
EC_community = [nx.eigenvector_centrality(coms_G[i]) for i in range(N_coms)]
EC_community_df = pd.DataFrame(EC_community).T.round(3)
EC_community_df.index.name = 'Product ID'
EC_community_df['Eigenvector'] = EC_community_df[EC_community_df.columns[:]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
EC_community_df = EC_community_df['Eigenvector']

#merges
data_frames = [commu_lab, df_itemset_label, DC_community_df, BC_community_df, CC_community_df, EC_community_df]
commu_label = reduce(lambda  left,right: pd.merge(left,right,on=['Product ID']), data_frames)
commu_label = pd.merge(commu_label, D_community_df, on='Community Number')
#png
#dfi.export(commu_label, 'df_itemset_label.png', max_rows=-1)

#separate communities
df1 = commu_label[:30]
df2 = commu_label[30:32]
df3 = commu_label[32:85]
df4 = commu_label[85:96]
df5 = commu_label[96:103]
df6 = commu_label[103:105]
df7 = commu_label[105:108]
df8 = commu_label[108:112]
df9 = commu_label[112:121]
df10 = commu_label[121:126]
df11 = commu_label[126:129]
df12 = commu_label[129:131]
df13 = commu_label[131:138]
df14 = commu_label[138:140]
df15 = commu_label[140:142]
df16 = commu_label[142:144]
df17 = commu_label[144:148]
df18 = commu_label[148:151]
df19 = commu_label[151:155]
df20 = commu_label[155:157]
df21 = commu_label[157:161]
df22 = commu_label[161:163]

#Export
dfi.export(df1, 'df_itemset_label_1.png', max_rows=-1)
dfi.export(df2, 'df_itemset_label_2.png', max_rows=-1)
dfi.export(df3, 'df_itemset_label_3.png', max_rows=-1)
dfi.export(df4, 'df_itemset_label_4.png', max_rows=-1)
dfi.export(df5, 'df_itemset_label_5.png', max_rows=-1)
dfi.export(df6, 'df_itemset_label_6.png', max_rows=-1)
dfi.export(df7, 'df_itemset_label_7.png', max_rows=-1)
dfi.export(df8, 'df_itemset_label_8.png', max_rows=-1)
dfi.export(df9, 'df_itemset_label_9.png', max_rows=-1)
dfi.export(df10, 'df_itemset_label_10.png', max_rows=-1)
dfi.export(df11, 'df_itemset_label_11.png', max_rows=-1)
dfi.export(df12, 'df_itemset_label_12.png', max_rows=-1)
dfi.export(df13, 'df_itemset_label_13.png', max_rows=-1)
dfi.export(df14, 'df_itemset_label_14.png', max_rows=-1)
dfi.export(df15, 'df_itemset_label_15.png', max_rows=-1)
dfi.export(df16, 'df_itemset_label_16.png', max_rows=-1)
dfi.export(df17, 'df_itemset_label_17.png', max_rows=-1)
dfi.export(df18, 'df_itemset_label_18.png', max_rows=-1)
dfi.export(df19, 'df_itemset_label_19.png', max_rows=-1)
dfi.export(df20, 'df_itemset_label_20.png', max_rows=-1)
dfi.export(df21, 'df_itemset_label_21.png', max_rows=-1)
dfi.export(df22, 'df_itemset_label_22.png', max_rows=-1)

"""
Extra Code (Trial of different unused community detection methods)
"""
#Community Detection: CNM

#community_cnm = nx.community.greedy_modularity_communities(adj_prune_graph)
#len(community_cnm) #there are 21 communitites found which represents baskets

# map colour for each community
#def community_colours(graph, communities):
 #   number = len(communities[0])
  #  colours = ['navy', 'blue', 'yellow', 'purple', 'green', 'red', 'orange', 'brown', 'pink', 'olive', 'grey', 'salmon', 'gold', 'plum', 'coral', 'lime', 'royalblue', 'orchid', 'skyblue', 'peru', 'tomato', 'forestgreen'][:number]
   # node_colours = []
    #for node in graph:
     #   community_index = 0
      #  for comm in communities:
       #     if node in comm:
        #        node_colours.append(colours[community_index])
         #       break
          #  community_index += 1
    #return node_colours

#node_colour = community_colours(adj_prune_graph, community_cnm)

# Draw full network with 20 communities
#plt.figure(figsize=(100, 100))
#pos_lov = nx.spring_layout(adj_prune_graph, seed=300, weight='weight')
#nx.draw_networkx(adj_prune_graph, pos_lov, node_color=node_colour, edge_color='grey')
#plt.title("Product Network with Communities", fontsize=80)
#plt.savefig("Network_with_Community_CNM.png", format="png")
#plt.show()

# Take a look at each communitites
# Add community to node attributes
#for c, v_c in enumerate(community_cnm):
 #   for v in v_c:
  #      # Add 1 to save 0 for external edges
   #     adj_prune_graph.nodes[v]['comm'] = c + 1

#Find internal edges and add their community to their attributes
#for v, w, in adj_prune_graph.edges:
 #   if adj_prune_graph.nodes[v]['comm'] == adj_prune_graph.nodes[w]['comm']:
  #      # Internal edge, mark with community
   #     adj_prune_graph.edges[v, w]['comm'] = adj_prune_graph.nodes[v]['comm']
    #else:
     #   # External edge, mark as 0
      #  adj_prune_graph.edges[v, w]['comm'] = 0

#N_coms=len(community_cnm)
#edges_coms=[]#edge list for each community
#coms_G=[nx.Graph() for _ in range(N_coms)] #community graphs
#colours_cnm = ['navy','blue', 'yellow', 'purple', 'green', 'red', 'orange', 'brown', 'pink', 'olive', 'grey', 'salmon', 'gold', 'plum', 'coral', 'lime', 'royalblue', 'orchid', 'skyblue', 'peru', 'tomato', 'forestgreen']

#fig=plt.figure(figsize=(100, 260))
#for i in range(N_coms):
#  edges_coms.append([(u,v,d) for u,v,d in adj_prune_graph.edges(data=True) if d['comm'] == i+1])#identify edges of interest using the edge attribute
#  coms_G[i].add_edges_from(edges_coms[i]) #add edges
#  plt.subplot(8,3,i+1)#plot communities
#  plt.title('Community '+str(i+1), fontsize=60)
#  pos = nx.circular_layout(coms_G[i])
#  nx.draw_networkx(coms_G[i],pos=pos,node_color=colours_cnm[i], node_size=1000, font_size=17)
#  edge_labels = nx.get_edge_attributes(coms_G[i], "weight")
#  nx.draw_networkx_edge_labels(coms_G[i],pos=pos, font_size=30, edge_labels=edge_labels)
#plt.savefig("Community_Network_CNM.png", format="png")

#fig=plt.figure(figsize=(32, 900))
#for i in range(N_coms):
#  edges_coms.append([(u,v,d) for u,v,d in adj_prune_graph.edges(data=True) if d['comm'] == i+1])#identify edges of interest using the edge attribute
#  coms_G[i].add_edges_from(edges_coms[i]) #add edges
#  plt.subplot(21,1,i+1)#plot communities
# plt.title('Community '+str(i+1), fontsize=60)
# pos = nx.circular_layout(coms_G[i])
# nx.draw_networkx(coms_G[i],pos=pos,node_color=colours_cnm[i], node_size=1000, font_size=17)
# edge_labels = nx.get_edge_attributes(coms_G[i], "weight")
# nx.draw_networkx_edge_labels(coms_G[i],pos=pos, font_size=30, edge_labels=edge_labels)
#plt.savefig("Community_Network_long_CNM.png", format="png")


#Community Detection: Leiden

#import leidenalg
#import igraph as ig

# because it's originally implemented in i-graph, we need to convert NetworkX graph to i-graph to
# use the original implementation
#temp_graph = ig.Graph.from_networkx(adj_prune_graph)

# function to obtain the communities calculated by leiden algorithm
#def get_leiden_communities(graph, random_state=0):
#    if isinstance(graph, (nx.Graph, nx.OrderedDiGraph, nx.DiGraph, nx.OrderedGraph)):
#        graph = ig.Graph.from_networkx(graph)
#    return list(leidenalg.find_partition(graph, partition_type=leidenalg.ModularityVertexPartition, seed=4500))

# get communities using leiden algorithm
#leiden_communities = get_leiden_communities(adj_prune_graph)
#leiden_communities = list(map(tuple, leiden_communities))
#leiden_communities = list(map(set, leiden_communities))
#len(leiden_communities)