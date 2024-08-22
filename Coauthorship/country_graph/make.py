import pandas as pd
import numpy as np
import itertools
from ast import literal_eval

graph_path = './Scopus/country_graph/'

# graph
edges = dict() # {v, u} -> count
nodes = dict() # country -> degree
nodes_country = dict()
country_names =['afghanistan', 'albania', 'algeria', 'american samoa', 'andorra', 'angola', 
                'anguilla', 'antarctica', 'antigua and barbuda', 'argentina', 'armenia', 
                'aruba', 'australia', 'austria', 'azerbaijan', 'bahamas', 'bahrain', 'bangladesh', 
                'barbados', 'belarus', 'belgium', 'belize', 'benin', 'bermuda', 'bhutan', 'bolivia', 
                'bonaire, sint eustatius and saba', 'bosnia and herzegovina', 'botswana', 'bouvet island', 
                'brazil', 'british indian ocean territory', 'brunei darussalam', 'bulgaria', 'burkina faso', 
                'burundi', 'cabo verde', 'cambodia', 'cameroon', 'canada', 'cayman islands', 
                'central african republic', 'chad', 'chile', 'china', 'christmas island', 'cocos islands', 
                'colombia', 'comoros', 'czech republic', 'democratic republic of the congo', 'congo', 'cook islands', 
                'costa rica', 'croatia', 'cuba', 'curacao', 'cyprus', 'czechia', "côte d'ivoire", 'denmark', 
                'djibouti', 'dominica', 'dominican republic', 'ecuador', 'egypt', 'el salvador', 
                'equatorial guinea', 'eritrea', 'estonia', 'eswatini', 'ethiopia', 'falkland islands', 
                'faroe islands', 'fiji', 'finland', 'france', 'french guiana', 'french polynesia', 
                'french southern territories', 'gabon', 'gambia', 'georgia', 'germany', 'ghana', 
                'gibraltar', 'greece', 'greenland', 'grenada', 'guadeloupe', 'guam', 'guatemala', 
                'guernsey', 'guinea', 'guinea-bissau', 'guyana', 'haiti', 'heard island and mcdonald islands', 
                'holy see', 'honduras', 'hong kong', 'hungary', 'iceland', 'india', 'indonesia', 
                'iran', 'iraq', 'ireland', 'isle of man', 'israel', 'italy', 'jamaica', 'japan', 'jersey', 
                'jordan', 'kazakhstan', 'kenya', 'kiribati', 'north korea', 'south korea', 'kuwait', 
                'kyrgyzstan', "lao people's democratic republic", 'latvia', 'lebanon', 'lesotho', 
                'liberia', 'libya', 'liechtenstein', 'lithuania', 'luxembourg', 'macao', 'madagascar', 
                'malawi', 'malaysia', 'maldives', 'mali', 'malta', 'marshall islands', 'martinique', 
                'mauritania', 'mauritius', 'mayotte', 'mexico', 'micronesia', 'moldova', 'monaco', 
                'mongolia', 'montenegro', 'montserrat', 'morocco', 'mozambique', 'myanmar', 'namibia', 
                'nauru', 'nepal', 'netherlands', 'new caledonia', 'new zealand', 'nicaragua', 'niger', 
                'nigeria', 'niue', 'norfolk island', 'northern mariana islands', 'norway', 'oman', 'pakistan', 
                'palau', 'palestine', 'panama', 'papua new guinea', 'paraguay', 'peru', 'philippines', 'pitcairn', 
                'poland', 'portugal', 'puerto rico', 'qatar', 'republic of north macedonia', 'romania', 'russian federation', 
                'rwanda', 'réunion', 'saint barthelemy', 'saint helena, ascension and tristan da cunha', 'saint kitts and nevis', 
                'saint lucia', 'saint martin', 'saint pierre and miquelon', 'saint vincent and the grenadines', 'samoa', 
                'san marino', 'sao tome and principe', 'saudi arabia', 'senegal', 'serbia', 'seychelles', 'sierra leone', 
                'singapore', 'sint maarten', 'slovakia', 'slovenia', 'solomon islands', 'somalia', 'south africa', 
                'south georgia and the south sandwich islands', 'south sudan', 'spain', 'sri lanka', 'sudan', 
                'suriname', 'svalbard and jan mayen', 'sweden', 'switzerland', 'syrian arab republic', 'taiwan', 
                'tajikistan', 'tanzania', 'thailand', 'timor-leste', 'togo', 'tokelau', 'tonga', 'trinidad and tobago', 
                'tunisia', 'turkey', 'turkmenistan', 'turks and caicos islands', 'tuvalu', 'uganda', 'ukraine', 
                'united arab emirates', 'united kingdom', 'united states minor outlying islands', 'united states of america', 
                'uruguay', 'uzbekistan', 'vanuatu', 'venezuela', 'viet nam', 'virgin islands (british)', 'virgin islands (u.s.)', 
                'wallis and futuna', 'western sahara', 'yemen', 'zambia', 'zimbabwe', 'aland islands']

def init():

  with open(graph_path + 'edges.csv', 'w', encoding='utf-8') as f:
    f.write('source,target,weight\n')
  with open(graph_path + 'nodes.csv', 'w', encoding='utf-8') as f:
    f.write('country, degree\n')

def parse(filename):
  df =  pd.read_csv(filename)
  affilation_col = df['Authors with affiliations'].values
  for affilation in affilation_col:
    if (type(affilation) != str):
      continue
    countries = list()
    for author in affilation.split(';'):
      country = author.split(',')[-1].strip().lower()
      if country[-1]=='.': country = country[:-1]
      if country == 'russia' or country == 'rsfsrrussia': country = 'russian federation'
      if country == 'united states' or country == 'usa': country = 'united states of america'
      for country_name in country_names:
        if country_name in country:
          country = country_name
          break
      else:
        continue
      countries.append(country)
    for country in countries:
      if country not in nodes:
        nodes[country] = 0
      nodes[country] += len(countries)
    comb = list(itertools.combinations(sorted(countries), 2))
    for edge in comb:
      if edge[0] == edge[1]:
        continue
      if edge not in edges:
        edges[edge] = 0
      edges[edge] += 1


if __name__ == '__main__':
  init()
  parse(graph_path + 'C-scopus_1991-1993.csv')
  parse(graph_path + 'C-scopus_1994-2016.csv')
  parse(graph_path + 'C-scopus_2017.csv')
  parse(graph_path + 'C-scopus_2018-2024.csv')
  edge_list = list()
  for (key, value) in edges.items():
    d = dict({'source':key[0], 'target':key[1], 'weight':value})
    edge_list.append(d)
  df = pd.DataFrame(edge_list)
  df.to_csv(graph_path + 'edges.csv', mode='a',sep=',', encoding='utf-8', index = False, header=False)
  nodes_list = list()
  for (key, value) in nodes.items():
    d = dict({'university':key, 'degree': value})
    nodes_list.append(d)
  df = pd.DataFrame(nodes_list)
  df.to_csv(graph_path + 'nodes.csv',mode='a',sep=',', encoding='utf-8', index = False, header=False)
