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

def read_graph(nodes_filename : str, edges_filename : str) -> nx.Graph:
  """
    Возвращает граф с вершинами и ребрами из файлов.
    В nodes_filename обязательно должен быть столбец с названием id.

    params:
      nodes_filename: путь к файлу с вершинами
      edges_filename: путь к файлу с ребрами
  """
  G = nx.Graph()
  edges = pd.read_csv(edges_filename, sep=',', encoding='utf-8')
  G = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr='weight', create_using= nx.Graph)
  nodes = pd.read_csv(nodes_filename, sep=',', encoding='utf-8')
  data = nodes.set_index('id').to_dict('index').items()
  G.add_nodes_from(data)
  return G

def degree_distribution(G : nx.Graph, G1 : nx.graph):
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
  number_of_bins = ceil((MAX-MIN)*50)
  hist, edges = np.histogram(degree, bins = np.logspace(MIN, MAX, number_of_bins), density=True)
  bin_centers = (edges[1:]+edges[:-1])/2.0
  plt.scatter(bin_centers, hist, color='b', label="proximity network, "r"$\theta=0$")
  fig = plt.gca()
  fit.power_law.plot_pdf(color='b', linestyle='--', ax=fig, label=f"linear fit, slope={round(alpha, 2)}")

  degree = sorted([deg for (node, deg) in G1.degree()])
  degree = degree[degree.count(0):]
  print(f"Количество вершин: {len(degree)}")
  fit = powerlaw.Fit(degree)
  print(f"alpha= {fit.alpha}, xmin = {fit.xmin}, D = {fit.D}\n Power_law, exponential compare = {fit.distribution_compare('power_law','exponential')}")
  alpha = fit.alpha
  xmin = fit.xmin
  degree =[x for x in degree if x >= xmin]
  MIN = np.log10(min(degree))
  MAX = np.log10(max(degree))
  number_of_bins = ceil((MAX-MIN)*10)
  hist, edges = np.histogram(degree, bins = np.logspace(MIN, MAX, number_of_bins), density=True)
  bin_centers = (edges[1:]+edges[:-1])/2.0
  plt.scatter(bin_centers, hist, color='r', label="proximity network, "r"$\theta=0.05$")
  # fig = plt.gca()
  fit.power_law.plot_pdf(color='r', linestyle='--', ax=fig, label=f"linear fit, alpha={round(alpha, 2)}")
  
  handles, labels = fig.get_legend_handles_labels()
  leg = fig.legend(handles, labels, loc=3)
  leg.draw_frame(False)
  fig.set_ylabel("p(k)")
  fig.set_xlabel("k")
  plt.legend()
  plt.show()

def components_distribution(G : nx.Graph, G1 : nx.Graph):
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
  fig = plt.gca()
  fig.scatter(x,y, c='blue', label="proximity network, "r"$\theta=0$")

  components = [len(x) for x in nx.connected_components(G1)]
  print(f"Количество компонент: {len(components)}")
  print(f"Количество изолированных вершин: {nx.number_of_isolates(G1)}")
  comp_distrib = Counter(components)
  x = list()
  y = list()
  for w in sorted(comp_distrib):
    print(f"Len of component : {w}, count : {comp_distrib[w]}")
    x.append(w)
    y.append(comp_distrib[w] / len(components))
  fig.scatter(x,y, c='red', label="proximity network, "r"$\theta=0.05$")
  plt.ylabel('p(k)', fontsize = 18)
  plt.xlabel('k', fontsize = 18)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xscale("log") 
  plt.yscale("log")
  plt.legend()
  plt.show()

if __name__ == '__main__':
  questions =  [ 
        inquirer.Path('nodes_file', message="Укажите путь к файлу вершин графа", path_type=inquirer.Path.FILE),
        inquirer.Path('edges_file', message="Укажите путь к файлу ребер графа", path_type=inquirer.Path.FILE)
              ]
  answer = inquirer.prompt(questions)
  G = read_graph(answer['nodes_file'], answer['edges_file'])
  print("Граф создан")
  questions =  [ 
        inquirer.Path('nodes_file', message="Укажите путь к файлу вершин графа", path_type=inquirer.Path.FILE),
        inquirer.Path('edges_file', message="Укажите путь к файлу ребер графа", path_type=inquirer.Path.FILE)
              ]
  answer = inquirer.prompt(questions)
  G1 = read_graph(answer['nodes_file'], answer['edges_file'])
  print("Граф создан")
  degree_distribution(G, G1)
  components_distribution(G, G1)