import pandas as pd
import itertools
from collections import Counter
graph_path = './Scopus/'
# graph
edges = dict() # {v, u} -> count
nodes = dict() # keyword -> degree
not_used = 0

def parse(filename):
  global not_used
  df =  pd.read_csv(filename)
  keywords_col = df['Author Keywords'].values
  for keywords in keywords_col:
    if (type(keywords) != str):
      not_used += 1
      continue
    keywords_list = [x.strip().lower() for x in keywords.split(';') 
                     if x != '']
    nodes.update(Counter(keywords_list))
    comb = list(itertools.combinations(sorted(keywords_list), 2))
    edges.update(Counter(comb))
    
if __name__ == '__main__':
  parse(graph_path + 'Scopus data.csv')
  print(f"not used = {not_used}")
  edge_list = list()
  for (key, value) in edges.items():
    d = dict({'source':key[0], 'target':key[1], 'weight':value})
    edge_list.append(d)
  df = pd.DataFrame(edge_list)
  df.to_csv(graph_path + 'edges.csv',sep=',', index = False)
  nodes_list = list()
  for key in sorted(nodes, key=nodes.get, reverse=True):
    d = dict({'keyword':key, 'label': key, 'degree': nodes[key]})
    nodes_list.append(d)
  df = pd.DataFrame(nodes_list)
  df.to_csv(graph_path + 'nodes.csv',sep=',', index = False)
