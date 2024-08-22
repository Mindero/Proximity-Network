import pandas as pd
from itertools import combinations
import numpy, re
from collections import defaultdict

def change_name(name):
  name = name.split(',')
  if len(name) == 1:
    print(name)
    return "Trash"
  first_name = name[0].lower()
  name[1] = re.split('[. ]', name[1])
  second_name = ""
  for x in name[1]:
    if x == "": continue
    second_name += x[0].lower()
  return first_name + ' ' + second_name

filename = './Scopus/Proximity/nodes.csv'
filename1 = './Scopus/nodes.csv'
save_filename = './Scopus/Proximity/nodes_with_country.csv'
df =  pd.read_csv(filename, sep = ',', encoding='utf-8')
authors = df['id'].to_list()

df = pd.read_csv(filename1, sep=',',usecols=['name', 'country'], encoding='utf-8')
df['name'] = df['name'].apply(change_name)
df = dict(zip(df.name,df.country))
df_save = dict()
for author in authors:
  author_low = author.lower()
  if author_low not in df:
    df_save[author] = "empty"
    print(author)
  else: df_save[author] = df[author_low]
df_save = pd.DataFrame(df_save.items(), columns=['name','country'])
df_save.to_csv(save_filename, sep = ',', encoding='utf-8', index=False) 