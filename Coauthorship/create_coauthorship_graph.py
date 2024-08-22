import pandas as pd
import itertools


# Authors graph
edges = dict() # {v, u} -> weight
nodes = dict() # node_id -> name, list_papers, 

def create_node(id):
  nodes[id] = dict()
  nodes[id]['full name'] = str()
  nodes[id]['country'] = str()
  nodes[id]['papers count'] = 0

def parse(filename):
  df =  pd.read_csv(filename)
  authors_col = df['Author full names'].values
  affilation_col = df['Authors with affiliations'].values
  for i in range(len(authors_col)):
    if (type(authors_col[i]) != str or type(affilation_col[i]) != str):
      continue
    authors = [x.strip() for x in authors_col[i].split(';')]
    affilation = [x.strip() for x in affilation_col[i].split(';')]
    authors_id = list()
    '''
      Получение id и полного имени каждого автора.
      Пример того, как выглядят необработанные данные:
      "van Selm, Gijsbertus (6507865137); Dölle, Emiel (16446840900)"
    '''
    for i, author in enumerate(authors):
      rigthmost_leftbrace = author.rfind('(')
      id = author[rigthmost_leftbrace + 1:-1].strip()
      authors_id.append(id)
      full_name = author[0:rigthmost_leftbrace].strip()
      if id not in nodes:
        create_node(id)
      nodes[id]['full name'] = full_name
      nodes[id]['papers count'] += 1
      country = affilation[i].split(',')
      if (len(country) > 1):
        country = country[-1].strip().capitalize()
        if (country[-1] == '.'):
          country = country[:-1]
        if (country == 'Russia'):
          country = 'Russian federation'
        nodes[id]['country'] = country

    comb = list(itertools.combinations(sorted(authors_id), 2))
    weight = 1 / len(authors)
    for edge in comb:
      if (edge not in edges):
        edges[edge] = 0
      edges[edge] += weight

if __name__ == '__main__':
  parse('./Scopus/Scopus data.csv')
  edge_list = list()
  for (key, value) in edges.items():
    d = dict({'source':key[0], 'target':key[1], 'weight':value})
    edge_list.append(d)
  df = pd.DataFrame(edge_list)
  df.to_csv('./Scopus/edges.csv',sep=',', encoding='utf-8', index = False)
  nodes_list = list()
  for (key, value) in nodes.items():
    d = dict({'id':key})
    value['country'] = value['country']
    d.update(value)
    nodes_list.append(d)
  df = pd.DataFrame(nodes_list)
  df.to_csv('./Scopus/nodes.csv',sep=',', encoding='utf-8', index = False)
