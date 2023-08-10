import sys
sys.path.append("../")
from data.load_data import get_top_words, get_handwritten_data, get_multi_100, get_multi_1000

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
import dataclasses
import datasets
#from IPython.display import HTML

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import matplotlib.pyplot as plt
import argparse

torch.cuda.empty_cache()
torch.set_grad_enabled(False)

#Set up arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--tweak_factor", default=3, type=int)
parser.add_argument("--layer_number", default=7, type=int)
parser.add_argument("--dataset", default="hand",choices=["hand", "2wmh"],  type=str)
parser.add_argument("--model", default="gpt2-small", choices=["gpt2-small", "gpt2-large"],  type=str)
args = parser.parse_args()

#Get Data
data = get_handwritten_data('../data/')
multi = get_multi_100('../data/')
multi_1000 = get_multi_1000('../data/')
top_words = get_multi_1000('../data/')
top_words = get_top_words('../data/')

print(top_words)

"""# Get Models"""

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

if(args.model == "gpt2-small"):
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
else:
    model = HookedTransformer.from_pretrained("gpt2-large", device=device)
model.cfg.use_attn_result = True

def memory_tweaker_head_hook(
    attn_result: Float[torch.Tensor, "num_tokens num_heads d_model"],
    hook: HookPoint, #name of layer where we inject memory
    extra_info: str, #the string that we tokenize and then inject into memory
    model: transformer_lens.HookedTransformer, #the model from which we get the unembedding matrix from
    vocab_size: int, #size of model vocabulary
    tweak_factor: float,
    head_num: int #The number of the head we want to edit
    #cache: transformer_lens.ActivationCache #this is the
) -> Float[torch.Tensor, "batch pos d_model"]:



    #print("Hook point: ", hook.name)
    #print("head num: ", head_num)
    #tokenize string
    tok_extra_info = model.to_tokens(extra_info, prepend_bos=False)
    #print(tok_extra_info)

    #transform tokens into one-hot vector
    #TODO: switch back to zeros
    extra_memory = torch.zeros(vocab_size).to(device)
    #extra_memory = torch.ones(vocab_size)
    #TODO: need to put a one in the spot with all of the extra info tokens and mult by tweak factor
    for i in tok_extra_info:
      extra_memory[i] = 1

    #subtract bias, and apply transpose of unembeding matrix to tokenized string to get it into model's hidden dim
    #extra_memory = extra_memory - model.unembed.b_U
    extra_memory = einsum("d_vocab, d_vocab d_model-> d_model", extra_memory, model.W_U.T)

    #TODO think about how layer norm would imapct things

    #add the extra_info embedded in latent space to hook_attn_out
    #print(attn_result.shape)
    attn_result[:,:,head_num,:] = attn_result[:,:,head_num,:] + extra_memory * tweak_factor
    #attn_result[:,:,head_num,:] + extra_memory * tweak_factor
    #print(attn_result[:,:,head_num,:])

    # TODO: Add a "jiggle" feature here.

    #return this edited hook_attn_out
    return attn_result


def apply_edit(model, extra_memory, prompt, tweak_factor=4, layer=10, head_num=0 ):
  # Use functools.partial to create a temporary hook function with the position fixed
  temp_hook_fn = partial(memory_tweaker_head_hook,
                        extra_info= extra_memory, #"Barak Obama",
                        vocab_size=50257,
                        model=model,
                        tweak_factor=tweak_factor,
                        head_num=head_num)

  #Get original logits
  logits = model(prompt)

  #Get patched Logits
  #layer  = layer
  patched_logits = model.run_with_hooks(prompt,
                            fwd_hooks=[
                                        ( utils.get_act_name("result", layer),
                                        temp_hook_fn)
                                        ]
                            )
  return logits, patched_logits




'''
    Function to compute the probability of the next token (ans) completion
    given the logits or prompt. Either prompt or logits needs to be passed.
'''
def get_ans_prob(model, ans, prompt=None, logits=None):
    if logits is None and prompt is None:
        raise ValueError("Either logits or prompt needs to be provided")
    ans_token = model.to_tokens(ans)[0][1]
    if logits is None:
        tokens = model.to_tokens(prompt)
        logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    total_ans_prob = torch.nn.functional.softmax(logits, -1)[0][-1][ans_token].item()
    return total_ans_prob


def fake_injections(data=data, top_words=top_words, fake_data_type="conjunctions", limit=200, model=model, layer=6, k=30, tweak_factor=4, full_title="GPT2_small_hand_fake_inject.csv"):

  print("Experiment: ", full_title)
  data_cp = data.copy()
  data_cp['answer_prob_exp'] = 0
  data_cp['answer_prob_obs'] = 0
  num_data_points = len(data['answer'])

  subjects = list(data['explicit_entity'])
  top_5000 = top_words['Top 5000 Words'].dropna().head(limit).tolist()
  nouns = top_words['Nouns'].dropna().head(limit).tolist()
  verbs = top_words['Verbs'].dropna().head(limit).tolist()
  adjectives = top_words['Adjectives'].dropna().head(limit).tolist()
  adverbs = top_words['Adverbs'].dropna().head(limit).tolist()
  conjunctions = top_words['Conjunctions'].dropna().tolist()

  if fake_data_type=="subject":
    words=subjects
  if fake_data_type=="top_5000":
    words=top_5000
  if fake_data_type=="nouns":
    words=nouns
  if fake_data_type=="verbs":
    words=verbs
  if fake_data_type=="adjectives":
    words=adjectives
  if fake_data_type=="adverbs":
    words=adverbs
  if fake_data_type=="conjunctions":
    words=conjunctions


  counter = 0
  for s in words:

      print("subject: ", s)
      subject_answer_edit = 'ans_prob_obs_edit_subject_'+s
      subject_top_k = 'topk_tok_obs_edit_subject_'+s
      #print(string)
      data_cp[subject_answer_edit] = 0
      data_cp[subject_top_k] = ''
      data_cp[subject_top_k] = data_cp[subject_top_k].apply(list)
      #print("here")
      #average_answer_prob_change_after_edit = 0

    #for h in range(heads):
      # this is a hacky way to hold the head number constant at head 0, bc it doesn't matter which head we inject into since they all get concatenated anyway
      h=0
      for i in range(num_data_points):
        answer = data['answer'][i]
        memory = s
        prompt = data['obscure_sentence'][i]
        explicit_prompt = data['explicit_sentence'][i]
        logits, patched_logits = apply_edit(model,
                                          memory,
                                          prompt,
                                          tweak_factor=tweak_factor,
                                          layer=layer,
                                          head_num=h)

        first_answer_tok = model.to_tokens(answer, prepend_bos=False)[0][0].item()
        answer_prob_after_mem = torch.nn.functional.softmax(patched_logits[0][-1], dim=0)[first_answer_tok]
        data_cp.loc[i, subject_answer_edit] = answer_prob_after_mem.item()


        vals, idx = torch.topk(patched_logits[0][-1], k)
        data_cp.at[i, subject_top_k]= idx.tolist()

        if counter == 0:
          answer_prob_before_mem = torch.nn.functional.softmax(logits[0][-1], dim=0)[first_answer_tok]
          data_cp.loc[i, 'answer_prob_obs'] = answer_prob_before_mem.item()
          explicit_ans_prob = get_ans_prob(model, answer, explicit_prompt)
          data_cp.loc[i, 'answer_prob_exp'] = get_ans_prob(model, answer, explicit_prompt)

        #print("explicit prompt: ", explicit_prompt)
        #print("obscure prompt: ", prompt)
        #print("memory: ", s)
        #print("answer: ", answer)
        #print("explicit prob", explicit_ans_prob)
        #print("edit prob: ", answer_prob_after_mem)

      counter+=1


  data_loc = "./"
  data_cp.to_csv(data_loc+full_title)
  return data_cp


#data_cp = fake_injections(data=data, top_words=top_words, model=gpt2_small, layer=8, k=30, tweak_factor=4, full_title="GPT2_small_hand_fake_inject_layer9_tweak4.csv")

#data_cp = fake_injections(data=data, top_words=top_words, model=gpt2_large, layer=14, k=30, tweak_factor=9, full_title="GPT2_large_hand_fake_inject_layer14_tweak10.csv")

#data_cp = fake_injections(data=multi_1000, top_words=top_words, model=gpt2_small, layer=8, k=30, tweak_factor=4, full_title="GPT2_small_2wmh_fake_inject_layer8_tweak4.csv")

#data_cp = fake_injections(data=multi_1000, top_words=top_words, model=gpt2_large, layer=4, k=30, tweak_factor=8, full_title="GPT2_small_2wmh_fake_inject_layer8_tweak4.csv")

#gpt2 large, 2wmh, layer 4, tweak 8
#gpt2 large, hand, layer 14, tweak 9
#gpt2 small, 2wmh, layer 8, tweak 4
#gpt2 small, hand, layer  7 , tweak 3

#data_cp = fake_injections(data=data, top_words=top_words, fake_data_type="conjunctions", model=gpt2_small, layer=7, k=30,
#                          tweak_factor=3, full_title="GPT2_small_hand_fake_inject_layer7_tweak3_conjunctions.csv")

fake_data_types = ["top_5000", "nouns", "verbs", "adjectives", "adverbs", "conjunctions"]

for fake_data_type in fake_data_types:
    fake_injections(data=data, 
                    top_words=top_words, 
                    fake_data_type=fake_data_type, 
                    model=model, 
                    layer=args.layer_number, 
                    k=30,
                    tweak_factor=args.tweak_factor,
                     full_title=f"{args.model}_{args.dataset}_fake_inject_layer{args.layer_number}_tweak{args.tweak_factor}_{fake_data_type}.csv")
