import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import powerlaw
from collections import defaultdict, Counter
from ast import literal_eval
import inquirer
import numpy as np
from itertools import combinations
from math import ceil

converter_to = dict()
converter_from = dict()

def read_graph(nodes_filename : str, edges_filename : str) -> nx.Graph:
  """
    Возвращает граф с вершинами и ребрами из файлов.
    В nodes_filename обязательно должен быть столбец с названием id.

    params:
      nodes_filename: путь к файлу с вершинами
      edges_filename: путь к файлу с ребрами
  """
  G = nx.Graph()
  nodes = pd.read_csv(nodes_filename, sep=',', encoding='utf-8')
  data = nodes.set_index('id').to_dict('index')
  data1 = dict()
  size = 0
  for key,value in data.items():
    converter_to[size] = key
    converter_from[key] = size
    data1[size] = data[key]
    size += 1
  data = data1.items()
  edges = pd.read_csv(edges_filename, sep=',', encoding='utf-8')
  edges['source'] = edges['source'].apply(lambda x: converter_from[x])
  edges['target'] = edges['target'].apply(lambda x: converter_from[x])
  G = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr='weight', create_using= nx.Graph)
  G.add_nodes_from(data)
  return G 

def power_law_pdf(x, alpha = 3.5, x_min = 1): # p(x) = C * x ^ {-alpha}
  C = (alpha - 1) / (x_min ** (1 - alpha))
  return C * x ** (-alpha)  

def degree_distribution(G : nx.Graph):
  """
    Считает параметр степенного закона в распределении степеней графа.
    Показывает график степенного распределения.
  """
  degree = sorted([deg for (node, deg) in G.degree()])
  degree = degree[degree.count(0):]
  print(f"Количество вершин: {len(degree)}")
  fit = powerlaw.Fit(degree)
  print(f"alpha= {fit.alpha}, xmin = {fit.xmin}, D = {fit.D}\n Power_law, exponential compare = {fit.distribution_compare('power_law','exponential')}")
  alpha = fit.alpha
  xmin = fit.xmin
  degree =[x for x in degree if x >= xmin]
  MIN = np.log10(min(degree))
  MAX = np.log10(max(degree))
  print(MIN, MAX)
  number_of_bins = ceil((MAX-MIN)*10)
  hist, edges = np.histogram(degree, bins = np.logspace(MIN, MAX, number_of_bins), density=True)
  bin_centers = (edges[1:]+edges[:-1])/2.0
  # bin_centers = edges
  print(bin_centers, hist)
  print(len(hist))
  plt.scatter(bin_centers, hist, color='b', label="co-authorship network")
  fig = plt.gca()
  # fig = fit.plot_pdf(color='b', linewidth=2, label="Empirical pdf")
  fit.power_law.plot_pdf(color='b', linestyle='--', ax=fig, label=f"linear fit, slope={round(alpha, 2)}")
  # fit.plot_ccdf(color='r', linewidth=2, ax=fig, label="Empirical ccdf")
  # fit.power_law.plot_ccdf(color='r', linestyle='--', ax=fig, label="Fit ccdf")
  handles, labels = fig.get_legend_handles_labels()
  leg = fig.legend(handles, labels, loc=3)
  leg.draw_frame(False)
  fig.set_ylabel("p(k)")
  fig.set_xlabel("k")
  plt.legend()
  plt.show()

# def fixed_degree_distribution(G : nx.Graph):
#   degree = sorted([deg for (node, deg) in G.degree()])
#   degree = degree[degree.count(0):]
#   print(f"Количество вершин: {len(degree)}")
#   fit = powerlaw.Fit(degree)
#   print(f"alpha= {fit.alpha}, xmin = {fit.xmin}, D = {fit.D}\n Power_law, exponential compare = {fit.distribution_compare('power_law','exponential')}")
#   alpha = fit.alpha
#   xmin = fit.xmin
#   degree =[x for x in degree if x >= xmin]
#   MIN = np.log10(min(degree))
#   MAX = np.log10(max(degree))
#   print(MIN, MAX)
#   number_of_bins = ceil((MAX-MIN)*100)
#   hist, edges = np.histogram(degree, bins = np.logspace(MIN, MAX, number_of_bins), density=True)
#   bin_centers = (edges[1:]+edges[:-1])/2.0
#   # bin_centers = edges
#   print(bin_centers, hist)
#   print(len(hist))
#   plt.scatter(bin_centers, hist, color='b', label="Emirical pdf")
#   fig = plt.gca()
#   # fig = fit.plot_pdf(color='b', linewidth=2, label="Empirical pdf")
#   fit.power_law.plot_pdf(color='b', linestyle='--', ax=fig, label=f"Fit pdf, alpha={round(alpha, 2)}, xmin={xmin}")
#   # fit.plot_ccdf(color='r', linewidth=2, ax=fig, label="Empirical ccdf")
#   # fit.power_law.plot_ccdf(color='r', linestyle='--', ax=fig, label="Fit ccdf")
#   handles, labels = fig.get_legend_handles_labels()
#   leg = fig.legend(handles, labels, loc=3)
#   leg.draw_frame(False)
#   fig.set_ylabel("p(X), p(X >= x)")
#   fig.set_xlabel("Degree")
#   plt.xscale('log')
#   plt.yscale('log')
#   plt.show()

def centralities(G: nx.Graph):
    """
      Вычисляет центральности по степени, близости, 
        посредничеству, собственному вектору графа.
      Для каждой центральносит выделяет 20 лучших авторов и
      сохраняет в соответствующих файлах: 
        - "degree centrality.csv"
        - "closeness centrality.csv"
        - "betweenness centrality.csv"
        - "eigenvector centrality.csv"
    """
    giant_comp = take_giant_component(G)
    print(giant_comp.number_of_nodes(), giant_comp.number_of_edges())
    centrality = dict()
    centrality["degree"] = {n:d for n,d in giant_comp.degree(weight='weight')}
    print("Degree")
    centrality["closeness"] = nx.closeness_centrality(giant_comp, distance='weight')
    print("Closeness")
    centrality["betweenness"] = nx.betweenness_centrality(giant_comp, weight="weight")
    print("Betweenness")
    centrality["eigenvector"] = nx.eigenvector_centrality(giant_comp, weight="weight", max_iter=600)
    print("Eigenvector")
    for c in centrality.keys():
      answer = dict()
      for key in sorted(centrality[c], key=centrality[c].get, reverse=True):
        answer[key] = centrality[c][key]
        if (len(answer) == 20): break
      centrality[c] = answer

    # countries = nx.get_node_attributes(giant_comp, "country")
    # name = nx.get_node_attributes(giant_comp, "name")
    for c in centrality.keys():
      df = pd.DataFrame(columns=["name", c])
      i = 0
      for key, value in centrality[c].items():
        # df.loc[i] = [name[key], countries[key], value]
        df.loc[i] = [key, value]
        i += 1  
      df.to_csv(f'{c} centrality.csv', header=True,index=False)

def take_giant_component(G : nx.Graph) -> nx.Graph:
  """
    Выделяет гигансткую компоненту в графе
  """
  giant_component = max(nx.connected_components(G), key = lambda x: len(x))
  G.remove_nodes_from(list(x for x in G.nodes if x not in giant_component)) # Удалили все вершины, которые не в компоненте
  return G

def components_distribution(G : nx.Graph):
  """
    Показывает график распределения компонент в графе
  """
  components = [len(x) for x in nx.connected_components(G)]
  print(f"Количество компонент: {len(components)}")
  print(f"Количество изолированных вершин: {nx.number_of_isolates(G)}")
  comp_distrib = Counter(components)
  x = list()
  y = list()
  for w in sorted(comp_distrib):
    print(f"Len of component : {w}, count : {comp_distrib[w]}")
    x.append(w)
    y.append(comp_distrib[w] / len(components))
  plt.figure(figsize=(10,7))
  plt.scatter(x,y, c='black')
  plt.ylabel('p(k)', fontsize = 18)
  plt.xlabel('k', fontsize = 18)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xscale("log") 
  plt.yscale("log")
  plt.show()
  giant_component_stats(G)

def giant_component_stats(G : nx.Graph):
  G = take_giant_component(G)
  print(f"Диаметр графа: {nx.approximation.diameter(G)}")
  # print(f"Средняя длина пути: {nx.approximation.}")

def get_paper_cntAuthors_year(filename : str, authors_column : str, year_column : str, separator : str) -> list[{int,int}]:
  """
    Получает для каждой статьи количество авторов и год.
    Возвращает список из пары (кол-во авторов, год)

    params: 
      filename: путь к файлу статей
      authors_column: название столбца с списком авторов каждой статьи
      year_column: название столбца с годом написания статьи
      separator: символ, по которому разделяются авторы в authors_column.
        Если данные в auhots_column имеют вид ["..", ...], то следует указывать list 
  """
  df = pd.read_csv(filename, sep = ',', encoding='utf-8', usecols=[authors_column, year_column])
  df = df[(df[authors_column].notna()) & (df[year_column].notna())]
  if separator == 'list':
    df[authors_column] = df[authors_column].apply(literal_eval)
  else: 
    df[authors_column] = df[authors_column].apply(lambda x: x.split(separator))
  df = df[(df[authors_column].str.len() > 0)]
  
  df = df.reindex(columns=[authors_column, year_column])
  # [cnt_authors, year]
  return df.values.tolist()

def papers_author_distribution(filename, authors_column, year_column, separator : str):
  """
    Показывает график распределения количества авторов на статью.

    params:
      filename: путь к файлу статей
      authors_column: название столбца с списком авторов каждой статьи
      year_column: название столбца с годом написания статьи
      separator: символ, по которому разделяются авторы в authors_column.
        Если данные в auhots_column имеют вид ["..", ...], то следует указывать list 
  """
  x = list()
  y = list()
  data =  get_paper_cntAuthors_year(filename, authors_column, year_column, separator)
  # Получение только количество авторов
  data = [len(x[0]) for x in data]
  papers_count  = Counter(data)
  all_papers = len(data)
  for w in sorted(papers_count):
    print(w, papers_count[w], papers_count[w] / all_papers * 100)
    x.append(w)
    y.append(papers_count[w])
  plt.scatter(x,y, c="black")
  plt.figure(figsize=(10,7))

  plt.ylabel("Papers count", fontsize = 18)  
  plt.xlabel("Authors count", fontsize = 18)

  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)

  plt.yscale('log')
  plt.xscale('log')
  plt.legend()
  plt.show()

def papers_author_per_year_distribution(filename : str, authors_column : str, 
                                        year_column : str, separator : str):
  """
    Показывает график распределения количества авторов на статью по годам.

    params:
      filename: путь к файлу статей
      authors_column: название столбца с списком авторов каждой статьи
      year_column: название столбца с годом написания статьи
      separator: символ, по которому разделяются авторы в authors_column.
        Если данные в auhots_column имеют вид ["..", ...], то следует указывать list 
  """
  data = get_paper_cntAuthors_year(filename, authors_column, year_column, separator)
  all_papers = defaultdict(int)
  papers_count = dict()
  for cnt in range(1, 5):
    papers_count[cnt] = defaultdict(int)
  for cnt, year in data:
    cnt = min(4, len(cnt))
    all_papers[year] += 1
    papers_count[cnt][year] += 1
    
  legend = ["1", "2", "3", ">= 4"]
  plt.figure(figsize=(10,7))
  for cnt in range(1, 5):
    x = list()
    y = list()
    print(cnt)
    for w in sorted(all_papers):
      print(w, papers_count[cnt][w] / all_papers[w] * 100)
      x.append(w)
      y.append(papers_count[cnt][w] / all_papers[w] * 100)
    plt.plot(x,y, label = legend[cnt - 1])

  plt.ylabel("% of papers", fontsize = 18)  
  plt.xlabel("year", fontsize = 18)

  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)

  plt.legend()
  plt.show()

def create_cooccurrence_network(filename, keywords_column, separator):
  edges = defaultdict(int) # {v, u} -> count
  nodes = defaultdict(int) # keyword -> degree
  # Парсим данные
  df =  pd.read_csv(filename, usecols=[keywords_column])
  size = len(df) 
  df = df[df[keywords_column].notna()]
  if separator == 'list': 
    df[keywords_column] = df[keywords_column].apply(literal_eval)
  else: df[keywords_column] = df[keywords_column].apply(lambda x: 
                                                        x.split(separator))
  for keywords in df[keywords_column].values:
    if len(keywords) == 0:
      size += 1
      continue
    keywords_list = [x.strip().lower() for x in keywords 
                     if x != '' or x != '\\']
    for keyword in keywords_list: nodes[keyword] += 1
    comb = list(combinations(sorted(keywords_list), 2))
    for edge in comb: edges[edge] += 1
  print(f"not used = {size - len(df)}")
  # Сохраняем данные в файл
  edge_list = list()
  for (key, value) in edges.items():
    d = dict({'source':key[0], 'target':key[1], 'weight':value})
    edge_list.append(d)
  df = pd.DataFrame(edge_list)
  df.to_csv('co-occurrence_edges.csv',sep=',', index = False)
  nodes_list = list()
  for key in sorted(nodes, key=nodes.get, reverse=True):
    d = dict({'keyword':key, 'label': key, 'degree': nodes[key]})
    nodes_list.append(d)
  df = pd.DataFrame(nodes_list)
  df.to_csv('co-occurrence_nodes.csv',sep=',', index = False)

def interests_in_paper(papers_filename, stats_filename):
  df = pd.read_csv(papers_filename)
  years_cnt = df['Year'].values.tolist()
  years_cnt = Counter(years_cnt)

  df = pd.read_csv(stats_filename, sep = ',', encoding='utf-8')
  years_official_cnt = dict(zip(df['year'], df['count']))
  x = list()
  y = list()
  mean = 0
  for key in sorted(years_cnt):
    x.append(key)
    y.append(years_cnt[key] / years_official_cnt[key] * 100)
    mean += years_cnt[key] / years_official_cnt[key]
  mean /= len(years_cnt)
  print(f"mean = {mean * 100}")
  plt.figure(figsize=(10,7))
  plt.bar(x,y, color='b')
  plt.ylabel("%", fontsize = 18)
  plt.xlabel("years", fontsize = 18)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.show()

def russian_papers_distribution(papers_filename : str, authors_filename : str, authors_column : str, 
                                        year_column : str, separator : str):
  data =  get_paper_cntAuthors_year(papers_filename, authors_column, year_column, separator)
  df = pd.read_csv(authors_filename, sep = ',', encoding='utf-8', usecols=['id', 'country'])
  countries = dict(zip(df.id,df.country))
  years_cnt_russian = defaultdict(int)
  years_cnt_not_russian = defaultdict(int)
  years_cnt = defaultdict(int)
  for authors, year in data:
    years_cnt[year] += 1
    if all('Russia' not in countries[author] or 
           'Russia Federation' not in countries[author] for author in authors):
      years_cnt_not_russian[year] += 1
    else:
      years_cnt_russian[year] += 1
  x = list()
  y = dict()
  y['russian_papers'] = list()
  y['foreign_papers'] = list()
  # mean = 0
  for key in sorted(years_cnt):
    x.append(key)
    y['russian_papers'].append(years_cnt_russian[key] / years_cnt[key] * 100)
    y['foreign_papers'].append(years_cnt_not_russian[key] / years_cnt[key] * 100)
    # mean += years_cnt[key] / years_official_cnt[key]
  # mean /= len(years_cnt)
  # print(mean * 100)
  plt.figure(figsize=(10,7))
  dataset1 = np.array(y['russian_papers'])
  dataset2 = np.array(y['foreign_papers'])
  p1 = plt.bar(x,dataset1, color='b')
  p2 = plt.bar(x, dataset2, color='c',bottom=dataset1)
  plt.legend((p1[0], p2[0]), ('Russian papers', 'Foreign papers'), fontsize=12, ncol=4, framealpha=0, fancybox=True)
  plt.ylabel("years", fontsize = 18)
  plt.xlabel("%", fontsize = 18)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.ylim([0,110])
  plt.show()

def find_max_clique(G : nx.Graph):
  nodes = nx.approximation.max_clique(G)
  print(f"Lenght of max clique = {len(nodes)}")
  H = G.subgraph(list(nodes))
  print(list(H.edges))
  H = pd.DataFrame(list(H.edges), columns=['source', 'target'])
  H.to_csv('clique.csv', sep = ',', index = False)

def create_cross_country_table(nodes_filename : str, edges_filename : str):
  nodes = pd.read_csv(nodes_filename, sep = ',', encoding='utf-8', usecols=['id', 'country'])
  print(f"Count of countries = {nodes['country'].nunique()}")
  name_country = nodes.set_index('id').T.to_dict('list')
  edges = pd.read_csv(edges_filename, sep = ',', encoding='utf-8', usecols=['source', 'target'])
  print(f"Cnt of edges before = {len(edges)}")
  edges['source'] = edges['source'].apply(lambda x: name_country[x][0])
  edges['target'] = edges['target'].apply(lambda x: name_country[x][0])
  # edges= edges.replace({"source":name_country, "target":name_country})
  print(f"Cnt of edges after1 = {len(edges)}")
  print(edges.head())
  
  edges1 = edges.groupby(['source','target']).size().reset_index(name='count')
  edges_count = defaultdict(int)
  for index,row in edges1.iterrows():
    source = row['source']
    target = row['target']
    value = row['count']
    edges_count[(source, target)] += value
    if source != target:
      edges_count[(target, source)] += value
  edges2 = []
  for key, value in edges_count.items():
    dict1 = dict()
    dict1['source'] = key[0]
    dict1['target'] = key[1]
    dict1['count'] = value
    edges2.append(dict1)
  edges2 = pd.DataFrame(edges2)
  print(f"Cnt of edges1 = {len(edges1)}")
  print(f"Cnt of edges2 = {len(edges2)}")
  table = edges2.pivot(index='source', columns='target', values='count')
  table.to_csv('countries_table.csv',sep=',')
  edges2.to_csv('countries_list.csv',sep=',', index=False)
def print_menu():
  menu = [
    inquirer.List(
        name = 'menu',
        message = 'Выберите действие',
        choices=["1 - Read graph", "2 - Degree distribution", 
                 "3 - Get centrality of giant component",
                "4 - Components distribution", 
                "5 - Papers per author distribution", 
                "6 - Papers per author per year distribution",
                "7 - Create co-occurrence network",
                "8 - Russian papers distribution",
                "9 - Interests distribution",
                "10 - find max clique",
                "11 - create cross country table"]
  )]
  return menu

if __name__ == '__main__':
  G = nx.Graph()
  while True:
    menu_operation = int(inquirer.prompt(print_menu())['menu'].split('-')[0].strip())
    print(menu_operation)
    if menu_operation == 1:
      questions =  [ 
        inquirer.Path('nodes_file', message="Укажите путь к файлу вершин графа", path_type=inquirer.Path.FILE),
        inquirer.Path('edges_file', message="Укажите путь к файлу ребер графа", path_type=inquirer.Path.FILE)
              ]
      answer = inquirer.prompt(questions)
      G = read_graph(answer['nodes_file'], answer['edges_file'])
      print("Граф создан")
      print(f"Количество вершин: {G.number_of_nodes()}, количество ребер: {G.number_of_edges()}")
    elif menu_operation == 2:
      degree_distribution(G)
    elif menu_operation == 3:
      centralities(G)
    elif menu_operation == 4:
      components_distribution(G)
    elif menu_operation == 5:
      questions =  [inquirer.Path('path', message="Укажите путь к файлу статей", path_type=inquirer.Path.FILE),
                    inquirer.Text('authors_column', message="Укажите название столбца с авторами статьи"),
                    inquirer.Text('year_column', message="Укажите название столбца с годом статьи"),
                    inquirer.List("separator", message="Выберите разделитель", 
                                  choices = [';', ',', 'list'])]
      answer = inquirer.prompt(questions)
      papers_author_distribution(answer['path'], answer['authors_column'], answer['year_column'], answer['separator'])
    elif menu_operation == 6:
      questions =  [inquirer.Path('path', message="Укажите путь к файлу статей", path_type=inquirer.Path.FILE),
                    inquirer.Text('authors_column', message="Укажите название столбца с авторами статьи"),
                    inquirer.Text('year_column', message="Укажите название столбца с годом статьи"),
                    inquirer.List("separator", message="Выберите разделитель", 
                                  choices = [';', ',', 'list'])]
      answer = inquirer.prompt(questions)
      papers_author_per_year_distribution(answer['path'], answer['authors_column'], answer['year_column'], answer['separator'])
    elif menu_operation == 7:
      questions =  [inquirer.Path('path', message="Укажите путь к файлу статей", path_type=inquirer.Path.FILE),
                    inquirer.Text('keywords_column', message="Укажите название столбца с ключевыми словами статьи"),
                    inquirer.List("separator", message="Выберите разделитель", 
                                  choices = [';', ',', 'list'])]
      answer = inquirer.prompt(questions)
      create_cooccurrence_network(answer['path'], answer['keywords_column'], answer['separator'])
    elif menu_operation == 8:
      questions =  [inquirer.Path('path', message="Укажите путь к файлу статей", path_type=inquirer.Path.FILE),
                    inquirer.Path('path_author', message="Укажите путь к файлу авторов", path_type=inquirer.Path.FILE),
                    inquirer.Text('authors_column', message="Укажите название столбца с авторами статьи"),
                    inquirer.Text('year_column', message="Укажите название столбца с годом статьи"),
                    inquirer.List("separator", message="Выберите разделитель", 
                                  choices = [';', ',', 'list'])]
      answer = inquirer.prompt(questions)
      russian_papers_distribution(answer['path'], answer['path_author'], 
                                  answer['authors_column'], answer['year_column'], answer['separator'])
    elif menu_operation == 9:
      questions =  [ 
        inquirer.Path('papers_file', message="Укажите путь к файлу статей", path_type=inquirer.Path.FILE),
        inquirer.Path('stats_file', message="Укажите путь к файлу статистики", path_type=inquirer.Path.FILE)
              ]
      # papers_filename, stats_filename
      answer = inquirer.prompt(questions)
      interests_in_paper(answer['papers_file'], answer['stats_file'])
    elif menu_operation == 10:
      find_max_clique(G)
    elif menu_operation == 11:
      questions =  [ 
        inquirer.Path('nodes_file', message="Укажите путь к файлу вершин графа", path_type=inquirer.Path.FILE),
        inquirer.Path('edges_file', message="Укажите путь к файлу ребер графа", path_type=inquirer.Path.FILE)
              ]
      answer = inquirer.prompt(questions)
      create_cross_country_table(answer['nodes_file'], answer['edges_file'])