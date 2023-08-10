import sys
sys.path.append("../")
from data.load_data import get_handwritten_data, get_multi_100, get_multi_1000
from utils import reject_outliers, get_ans_prob, apply_edit, memory_tweaker_head_hook, head_latent_space_projector

# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import random
from pathlib import Path
from torch.utils.data import DataLoader

from jaxtyping import Float, Int
from functools import partial

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
#import dataclasses
#import datasets
#from IPython.display import HTML

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

"""# Get Models"""

device = "cuda" if torch.cuda.is_available() else "cpu"

gpt2_small = HookedTransformer.from_pretrained("gpt2-small", device=device)
gpt2_large = HookedTransformer.from_pretrained("gpt2-large", device=device)

prompt = "George Washington fought in the"
head_latent_space_projector(gpt2_small, prompt, 10, 12, aggregate_heads=True, intermediate_tokens=True)

prompt = "The first president of the United States fought in the"
head_latent_space_projector(gpt2_small, prompt, 10, 12, aggregate_heads=True, intermediate_tokens=True)

prompt = "The first president of the United States fought in the"
head_latent_space_projector(gpt2_large, prompt, 10, 20, aggregate_heads=False, intermediate_tokens=False)



#Get Data
data = get_handwritten_data('../data/')
multi = get_multi_100('../data/')
multi_1000 = get_multi_1000('../data/')


"""#Tweak Factor Analysis

We will replicate the above experiments, but accross a number of tweak factors so that we can do a sensativity analysis
"""

# We are going to define a more general purpose editing function which records more useful metrics up front so that we can do post-analysis later
def edit_heatmap(data, model, layers=12, heads=1, tweak_factor=4, k=30, print_output=True):
  num_data_points = len(data['answer'])

  data_cp = data.copy()
  data_cp['answer_prob_exp'] = 0
  data_cp['answer_prob_obs'] = 0


  for l in range(layers):
      #print("layer: ", l)
      layer_answer_edit = 'ans_prob_obs_edit_layer'+str(l)
      layer_top_k = 'topk_tok_obs_edit_layer'+str(l)
      #print(string)
      data_cp[layer_answer_edit] = 0
      data_cp[layer_top_k] = ''
      data_cp[layer_top_k] = data_cp[layer_top_k].apply(list)
      #print("here")
      #average_answer_prob_change_after_edit = 0

    #for h in range(heads):
      # this is a hacky way to hold the head number constant at head 0, bc it doesn't matter which head we inject into since they all get concatenated anyway
      h=0
      for i in range(num_data_points):
        answer = data['answer'][i]
        memory = data['explicit_entity'][i]
        prompt = data['obscure_sentence'][i]
        explicit_prompt = data['explicit_sentence'][i]
        logits, patched_logits = apply_edit(model,
                                          memory,
                                          prompt,
                                          tweak_factor=tweak_factor,
                                          layer=l,
                                          head_num=h)

        #print("Diff between logits and patched logits: ", logits==patched_logits)
        first_answer_tok = model.to_tokens(answer, prepend_bos=False)[0][0].item()
        answer_prob_before_mem = torch.nn.functional.softmax(logits[0][-1], dim=0)[first_answer_tok]

        answer_prob_after_mem = torch.nn.functional.softmax(patched_logits[0][-1], dim=0)[first_answer_tok]



        if l == 0:
          data_cp.loc[i, 'answer_prob_obs'] = answer_prob_before_mem.item()
          #explicit_ans_prob = get_ans_prob(model, answer, explicit_prompt)
          data_cp.loc[i, 'answer_prob_exp'] = get_ans_prob(model, answer, explicit_prompt)

        data_cp.loc[i, layer_answer_edit] = answer_prob_after_mem.item()


        vals, idx = torch.topk(patched_logits[0][-1], k)
        data_cp.at[i, layer_top_k]= idx.tolist()
        #data_cp.at[i, layer_top_k].append( idx.tolist())

        #print(i)
      if(print_output):
        print("layer: ", l)

        print("Average Answer Probability before edit: ", data_cp['answer_prob_obs'].mean())
        print("Average Answer probability difference after edit: ", (data_cp[layer_answer_edit] -data_cp['answer_prob_obs']).mean())
        print("Average Percent increase in Answer probability difference after edit: ", ((data_cp[layer_answer_edit] -data_cp['answer_prob_obs'])/ data_cp['answer_prob_obs']).mean() * 100)
        #print("Median Percent increase in Answer probability difference after edit: ", median_percent_difference * 100)


  return data_cp


# Function to vary the tweak factor

def tweak_factor_vary(tweak_factors, data, model=gpt2_small, layers=12, title="gpt2_small_subject_edits", data_loc = "drive/MyDrive/Research/Mechanistic Interpretability/Figures/Fig_data/"):
  for i in tweak_factors:
    full_title = title+"_tweakFactor_"+str(i)+".csv"
    print(full_title)

    data_cp = edit_heatmap(data, model, layers=layers, heads=1, tweak_factor=i)

    #data_loc = "drive/MyDrive/Research/Mechanistic Interpretability/Figures/Fig_data/"
    data_loc = "./"

    #save each dataframe with a descriptive title
    #torch.save(data_cp, data_loc+full_title)

    data_cp.to_csv(data_loc+full_title)

tweak_factors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
tweak_factor_vary(tweak_factors, data, gpt2_small, 12, title="gpt2_small_subject_edits_hand")

tweak_factors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
tweak_factor_vary(tweak_factors, data, gpt2_large, 36, title="gpt2_large_subject_edits_hand")

#tweak_factors = [6,7,8,9,10,11,12,13,14,15]
#tweak_factors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#tweak_factor_vary(tweak_factors, multi_1000, gpt2_small, 12, title="gpt2_small_subject_edits_2wmh")

#tweak_factors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#tweak_factor_vary(tweak_factors, multi_1000, gpt2_large, 36, title="gpt2_large_subject_edits_2wmh")
