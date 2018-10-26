# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 20:06:52 2018

@author: Mahesh
"""

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
import random
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori

#Function for Graph
def draw_graph(rules, rules_to_show):
  import networkx as nx  
  G1 = nx.DiGraph()
   
  color_map=[]
  N = 50
  colors = np.random.rand(N) 
  alpha="qwertyuiopasdfghjklzxcvbnm"
  nd=[]   
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']   
   
   
  for i in range (rules_to_show):      
    G1.add_nodes_from(["R"+str(i)])
    
     
    for a in rules.iloc[i]['antecedents']:
                
        G1.add_nodes_from([a])
        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)
       
    for c in rules.iloc[i]['consequents']:
             
            G1.add_nodes_from(c)
            
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)
 
  for node in G1:
       found_a_string = False
       for item in strs: 
           if node==item:
                found_a_string = True
       if found_a_string:
            color_map.append('yellow')
       elif(node in alpha):
           nd.append(node)
       else:
            color_map.append('green')     
 
  for i in nd:
      G1.remove_node(i)
      
  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
 
  pos = nx.spring_layout(G1, k=16, scale=10)
  nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
  for p in pos:  # raise text positions
           pos[p][1] += 0.1
  nx.draw_networkx_labels(G1, pos)
  plt.show()

#Getting the Dataset
store_data = pd.read_csv('store_data.csv',header=None)
store_data.head()
#Dataset Preprocesing
records = []  
for i in range(0, 7501):
    temp=[str(store_data.values[i,j]) for j in range(0, 20)]
    for word in list(temp):
        if word in 'nan':
            temp.remove(word)
    records.append(temp)

#Encodes database transaction data in form of a Python list of lists into a NumPy array.
oht = OnehotTransactions()
"""Via the fit method, the TransactionEncoder learns the unique labels in the dataset, 
and via the transform method, it transforms the input dataset (a Python list of lists) 
into a one-hot encoded NumPy boolean array"""
oht_ary = oht.fit(records).transform(records)
#Convert the encoded array into a pandas DataFrame:
df = pd.DataFrame(oht_ary, columns=oht.columns_)

"""Get frequent itemsets from a one-hot DataFrame Parameters
pandas DataFrame with columns ['support', 'itemsets'] of all itemsets that are >= min_support. 
Each itemset in the 'itemsets' column is of type frozenset"""
frequent_itemsets = apriori(df, min_support=0.003, use_colnames=True)

"""Generates a DataFrame of association rules including the metrics 'score', 'confidence', and 'lift'.
Metric is to evaluate if the rule is of intrest or not"""
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=3)
rules.head()
 
print (rules)

support=rules.as_matrix(columns=['support'])
confidence=rules.as_matrix(columns=['confidence'])
 
"""import seaborn as sns1
 
for i in range (len(support)):
     support[i] = support[i] 
     confidence[i] = confidence[i] 
     
plt.title('Association Rules')
plt.xlabel('support')
plt.ylabel('confidence')    
sns1.regplot(x=support, y=confidence, fit_reg=False)
 
#plt.gcf().clear()
plt.show()"""
draw_graph (rules, 10)

for i in range (len(support)):
   support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
   confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
 
plt.scatter(support, confidence, alpha=0.9, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()