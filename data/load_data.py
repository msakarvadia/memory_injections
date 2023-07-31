
import pandas as pd

#LOAD DATA

data_loc = './'
file_name = "handwritten_obscure_explicit_data.csv"
#data_loc = '/content/drive/MyDrive/Research/Mechanistic Interpretability/data/handwritten_obscure_explicit_data'

data = pd.read_csv(data_loc+file_name)
data = data.drop(data.columns[[-1]], axis=1)
data = data[data['answer'] != ""]

for i in range(len(data['answer'])):
  data.loc[i, 'answer'] = ' '+ data['answer'][i]
print(data.head())

data['explicit_sentence'][0]

file_name = "multi_hop_100.csv"
multi = pd.read_csv(data_loc+file_name)
multi = multi.drop([ 'fact1', 'fact2'], axis=1)

#Need to add a " " space to the front of every answer to account for funny tokenization
for i in range(len(multi['answer'])):
  multi.loc[i, 'answer'] = ' '+ multi['answer'][i]
multi.rename(columns={"explicit_sent": "explicit_sentence", "obscure_sent": "obscure_sentence"}, inplace=True)


#defining my worksheet
file_name = "multi_hop_1000.csv"
multi_1000 = pd.read_csv(data_loc+file_name)
multi_1000 = multi_1000.drop([ 'fact1', 'fact2'], axis=1)

#Need to add a " " space to the front of every answer to account for funny tokenization
for i in range(len(multi_1000['answer'])):
  multi_1000.loc[i, 'answer'] = ' '+ multi_1000['answer'][i]

multi_1000.rename(columns={"explicit_sent": "explicit_sentence", "obscure_sent": "obscure_sentence"}, inplace=True)
