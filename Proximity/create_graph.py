import pandas as pd
from itertools import combinations
import numpy
from collections import defaultdict
filename = 'author_refauthor.csv'
weight = dict()
cnt = dict()
def compute_weight(row):
  x = row['target']
  y = row['source']
  all= cnt[x] + cnt[y] - row['weight']
  return row['weight'] / all

def create_edges():
  global cnt
  df =  pd.read_csv(filename, sep = ',', usecols=['auth', 'author'], encoding='utf-8')
  cnt = df['auth'].value_counts().to_dict()

  df = pd.read_csv('intersection.csv', sep = ',', encoding='utf-8')
  df = df.rename(columns={"intersection":"weight"})
  df['weight'] = df.apply(compute_weight, axis = 1)
  df = df.sort_values('weight')
  df.to_csv("edges.csv", sep = ',', encoding='utf-8', index = False)

def create_node():
  df =  pd.read_csv(filename, sep = ',', usecols=['auth'], encoding='utf-8')
  nodes = pd.DataFrame(df.auth.unique(), columns=['id'])
  nodes.to_csv("nodes.csv", sep = ',', encoding='utf-8', index = False)
if __name__ == '__main__':
  create_node()
  create_edges()