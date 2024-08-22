import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# SSRN - './graph_analyze/'
# SCOPUS - './scopus/'
graph_path = './scopus/'
nodes_info = dict()
# YEAR_STATS
years_cnt = dict()
years_official_cnt = dict()
years_cnt_not_russian = dict()
years_cnt_russian = dict()
def init(l, r):
  for i in range(l, r):
    years_cnt[i] = 0
    years_official_cnt[i] = 0
    years_cnt_not_russian[i] = 0
    years_cnt_russian[i] = 0
def parse(filename):
  df =  pd.read_csv(filename)
  year_col = df['Year'].values
  for year in year_col:
    years_cnt[year] += 1

def count_not_russian_papers(filename):
  df =  pd.read_csv(filename)
  year_col = df['Year'].values
  affilation_col = df['Affiliations'].values
  for i, year in enumerate(year_col):
    if (type(affilation_col[i]) != str):
      continue
    years_cnt[year] += 1
    countries = set([x.split(',')[-1].strip() for x in affilation_col[i].split(';')])
    if ('Russian Federation' in countries):
      years_cnt_russian[year] += 1
    else:
      years_cnt_not_russian[year] += 1

def show(x : list, y : dict, label_x : str, label_y : str, logScale = False):
  plt.figure(figsize=(10,7))
  dataset1 = np.array(y['russian_papers'])
  dataset2 = np.array(y['foreign_papers'])
  p1 = plt.bar(x,dataset1, color='b')
  p2 = plt.bar(x, dataset2, color='c',bottom=dataset1)
  plt.legend((p1[0], p2[0]), ('Russian papers', 'Foreign papers'), fontsize=12, ncol=4, framealpha=0, fancybox=True)

  plt.ylabel(label_y, fontsize = 18)
  plt.xlabel(label_x, fontsize = 18)

  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  if (logScale):
    plt.xscale("log") 
    plt.yscale("log")
  plt.ylim([0,110])
  plt.show()
# sorted(result[centrality], key=result[centrality].get, reverse=True)
if __name__ == '__main__':
  init(1992, 2025)
  count_not_russian_papers(graph_path + 'Scopus data.csv')
  # calc_stat()
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
  show(x,y,"years", "%")
