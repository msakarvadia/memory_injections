
import pandas as pd

#LOAD DATA
def get_top_words(data_loc='data/'):
    file_name = "top_word_types.csv"
    data = pd.read_csv(data_loc+file_name)
    return data

def get_handwritten_data(data_loc='data/'):
    file_name = "handwritten_obscure_explicit_data.csv"

    data = pd.read_csv(data_loc+file_name)
    data = data[data['answer'] != ""]

    for i in range(len(data['answer'])):
      data.loc[i, 'answer'] = ' '+ data['answer'][i]
    return data


def get_multi_100(data_loc='data/'):
    file_name = "multi_hop_100.csv"
    multi = pd.read_csv(data_loc+file_name)
    multi = multi.drop([ 'fact1', 'fact2'], axis=1)

    #Need to add a " " space to the front of every answer to account for funny tokenization
    for i in range(len(multi['answer'])):
      multi.loc[i, 'answer'] = ' '+ multi['answer'][i]
    multi.rename(columns={"explicit_sent": "explicit_sentence", "obscure_sent": "obscure_sentence"}, inplace=True)

    return multi

def get_multi_1000(data_loc='data/'):
    file_name = "multi_hop_1000.csv"
    multi_1000 = pd.read_csv(data_loc+file_name)
    multi_1000 = multi_1000.drop([ 'fact1', 'fact2'], axis=1)

    #Need to add a " " space to the front of every answer to account for funny tokenization
    for i in range(len(multi_1000['answer'])):
      multi_1000.loc[i, 'answer'] = ' '+ multi_1000['answer'][i]

    multi_1000.rename(columns={"explicit_sent": "explicit_sentence", "obscure_sent": "obscure_sentence"}, inplace=True)
    return multi_1000
