import pandas as pd
filename = 'author_refauthor.csv'
weight = dict()
def save():
  saved_info = list()
  for (x,y), value in weight.items():
    d = dict({"target" : x, "source" : y, "intersection" : value})
    saved_info.append(d)
  df = pd.DataFrame(saved_info)
  df.to_csv("intersection.csv", sep = ',', encoding='utf-8', index = False)

def intersec(d):
  cnt = 0
  total = 33775 * 33775 # Суммарное количество итераций
  for i,x in enumerate(df['auth'].values):
    for j,y in enumerate(df['auth'].values):
      cnt += 1
      if (total - cnt) % 1000000 == 0: # Для отладки
        print(f"Rest = {total - cnt}")
      # Пропускаем пары вершин, которые уже рассматривали, и пары, состоящие из двух одинаковых вершин
      if j >= i: continue 
      # Считаем пересечение
      w = len(set(d[x]) & set(d[y]))
      if w > 0: weight[tuple({x,y})] = w
  save()

if __name__ == '__main__':
  df =  pd.read_csv(filename, sep = ',', usecols=['auth', 'author'], encoding='utf-8')
  print(f"number of rows = {len(df)}")
  print(f"Unique: auth = {df['auth'].nunique()}, author = {df['author'].nunique()}")
  # Считаем для каждого автора список авторов, на которые он ссылается
  df = df.groupby("auth", as_index=False).agg(list)
  print(len(df['auth'].values))
  # Упаковываем в словарь
  d = dict(zip(df.auth, df.author))
  intersec(d = d)
