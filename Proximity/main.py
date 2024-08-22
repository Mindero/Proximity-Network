import pandas as pd
from itertools import combinations
from collections import defaultdict
filename = 'author_refauthor.csv'
weight = defaultdict(int)
def intersec(df):
  print(len(list(combinations(df['auth'].values, 2))))
  # for x,y in list(combinations(df['auth'].values, 2)):
    # print(x,y)

if __name__ == '__main__':
  df =  pd.read_csv(filename, sep = ',', usecols=['auth', 'author'])
  print(f"number of rows = {len(df)}")
  print(f"Unique: auth = {df['auth'].nunique()}, author = {df['author'].nunique()}")
  df = df.groupby("auth", as_index=False).agg(list)
  # print(len(df['auth'].values))
  # intersec(df = df)