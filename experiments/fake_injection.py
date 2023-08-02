import sys
sys.path.append("../")
from data.load_data import get_handwritten_data, get_multi_100, get_multi_1000

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
import plotly.express as px
from torch.utils.data import DataLoader

from jaxtyping import Float, Int
from typing import List, Union, Optional
from functools import partial
import copy

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

torch.set_grad_enabled(False)

#Get Data
data = get_handwritten_data('../data/')
multi = get_multi_100('../data/')
multi_1000 = get_multi_1000('../data/')


"""# Get Models"""

device = "cuda" if torch.cuda.is_available() else "cpu"

gpt2_small = HookedTransformer.from_pretrained("gpt2-small", device=device)
gpt2_large = HookedTransformer.from_pretrained("gpt2-large", device=device)
gpt2_small.cfg.use_attn_result = True
gpt2_large.cfg.use_attn_result = True


def head_latent_space_projector(model, prompt, k_tokens, num_heads, aggregate_heads=True, intermediate_tokens=True):
  # TODO: implement a way to turn off intermediate_tokens (for the sake of truncating output)
  # intermediate_tokens = boolean arg that specifies if we want to the projections for all intermediate tokens as well, not just the last one

  #This is how you change the amount of heads that are cached
  model.cfg.use_attn_result = True

  #tokenize the prompt
  tokens = model.to_tokens(prompt)
  logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)

  if aggregate_heads:

    for l in cache:
      if "hook_attn_out" in l:
        #Get the aggregate head info for a particular layer, and apply a layer norm
        #ln_final = model.ln_final(cache[l])[None, :, :]
        #logits = model.unembed(ln_final)

        head_results = cache[l][None, :, :]
        logits = model.unembed(head_results)

        topk_token_preds = torch.topk(logits, k_tokens)

        for i in range(len(tokens[0])):
          if not intermediate_tokens:
              print("LAYER: ", l)
              print("PROMPT: ", model.to_string(tokens[0][:]))
              print(model.to_string(topk_token_preds[1][0][-1].reshape(k_tokens, 1)))
              break
          print("LAYER: ", l)
          print("PROMPT: ", model.to_string(tokens[0][0:i+1]))
          print(model.to_string(topk_token_preds[1][0][i].reshape(k_tokens, 1)))
          print("---------")

  ## This section below needs to be cleaned up
  else: # This is incase we want each individual head
    for l in cache:
      if "hook_result" in l:
        #print("LAYER: ", l)
        #ln_final = model.ln_final(cache[l])[None, :, :] #blocks.10.hook_attn_out #blocks.5.hook_resid_post

        head_results = cache[l][None, :, :]

        for h in range(num_heads):
          #head_out = ln_final[:,:, h, :]
          head_out = head_results[:,:, h, :]
          logits = model.unembed(head_out)
          topk_token_preds = torch.topk(logits, k_tokens)
          for i in range(len(tokens[0])):
            if not intermediate_tokens:
              print("LAYER: ", l, "| HEAD: ", h)
              print("PROMPT: ", model.to_string(tokens[0][:]))
              print(model.to_string(topk_token_preds[1][0][-1].reshape(k_tokens, 1)))
            break
            print("LAYER: ", l, "| HEAD: ", h)
            print("PROMPT: ", model.to_string(tokens[0][0:i+1]))
            print(model.to_string(topk_token_preds[1][0][i].reshape(k_tokens, 1)))
          print("---------")


def memory_tweaker_hook(
    attn_out: Float[torch.Tensor, "num_tokens d_model"],
    hook: HookPoint, #name of layer where we inject memory
    extra_info: str, #the string that we tokenize and then inject into memory
    model: transformer_lens.HookedTransformer, #the model from which we get the unembedding matrix from
    vocab_size: int, #size of model vocabulary
    tweak_factor: float,
    #cache: transformer_lens.ActivationCache #this is the
) -> Float[torch.Tensor, "batch pos d_model"]:

    print("Hook point: ", hook.name)
    #tokenize string
    tok_extra_info = model.to_tokens(extra_info, prepend_bos=False)
    print(tok_extra_info)

    #transform tokens into one-hot vector
    extra_memory = torch.zeros(vocab_size).to(device)
    #TODO: need to put a one in the spot with all of the extra info tokens and mult by tweak factor
    for i in tok_extra_info:
      extra_memory[i] = 1

    #subtract bias, and apply transpose of unembeding matrix to tokenized string to get it into model's hidden dim
    #extra_memory = extra_memory - model.unembed.b_U
    extra_memory = einsum("d_vocab, d_vocab d_model-> d_model", extra_memory, model.W_U.T)

    #TODO think about how layer norm would imapct things

    #add the extra_info embedded in latent space to hook_attn_out
    attn_out = attn_out + extra_memory * tweak_factor

    # TODO: Add a "jiggle" feature here.

    #return this edited hook_attn_out
    return attn_out

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

  #prompt = "The first black president of the United States was a member of the"
  #Get original logits
  logits = model(prompt)

  #Get patched Logits
  layer  = layer
  patched_logits = model.run_with_hooks(prompt,
                            fwd_hooks=[
                                        ( utils.get_act_name("result", layer),
                                        temp_hook_fn)
                                        ]
                            )
  return logits, patched_logits

def interpret_logits_as_vocab(model, logits, top_k=30):
  topk_token_vals_edit, topk_token_preds_edit = torch.topk(logits, top_k)
  return model.to_string(topk_token_preds_edit[0][-1])


def print_edit_results(data, model, layer=9, head_num=8, tweak_factor=4):
  average_answer_prob_change_after_edit = 0
  data['ans_prob_obs'] = 0
  data['ans_prob_exp'] = 0
  data['ans_prob_after_edit'] = 0

  for i in range(len(data['answer'])):
    answer = data['answer'][i]
    memory = data['explicit_entity'][i]
    prompt = data['obscure_sentence'][i]

    explicit_prompt = data['explicit_sentence'][i]
    exp_logits = model(explicit_prompt)


    logits, patched_logits = apply_edit(model,
                                      memory,
                                      prompt,
                                      tweak_factor=tweak_factor,
                                      layer=layer,
                                      head_num=head_num)

    first_answer_tok = gpt2_small.to_tokens(answer, prepend_bos=False)[0][0].item()
    answer_prob_before_mem = torch.nn.functional.softmax(logits[0][-1], dim=0)[first_answer_tok]
    answer_prob_after_mem = torch.nn.functional.softmax(patched_logits[0][-1], dim=0)[first_answer_tok]
    ans_prob_exp = torch.nn.functional.softmax(exp_logits[0][-1], dim=0)[first_answer_tok]

    average_answer_prob_change_after_edit += answer_prob_after_mem - answer_prob_before_mem

    data.loc[i, 'ans_prob_obs'] = answer_prob_before_mem.item()
    data.loc[i, 'ans_prob_exp'] = ans_prob_exp.item()
    data.loc[i, 'ans_prob_after_edit'] = answer_prob_after_mem.item()

    print("Prompt: ", prompt)
    print("Answer: ", data['answer'][i])
    print("Memory: ", memory)
    print("original logits | Answer Probability: ", answer_prob_before_mem)
    print(interpret_logits_as_vocab(model, logits))
    print("edited logits| Answer Probability: ", answer_prob_after_mem)
    print(interpret_logits_as_vocab(model, patched_logits))
    print("---------------- ", i)
  print("Average Answer probability difference after edit: ", average_answer_prob_change_after_edit/len(data['answer']))
  return data

print_edit_results(data, gpt2_small)

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
"""
model =gpt2_small
answer = " Revolutionary"
explicit_prompt = "George Washington fought in the"
get_ans_prob(model, answer, explicit_prompt)

model =gpt2_small
answer = " Revolutionary"
explicit_prompt = "The first president of the United States fought in the"
get_ans_prob(model, answer, explicit_prompt)
"""

#get probs of expected answer pre and pos tedit
def edit_layerwise_tweak(d, model, layers=12, heads=1, tweak_factor=4):
  data = d.copy()
  average_answer_prob_change_after_edit = 0
  avg_edits = np.empty([layers, heads])

  data['ans_prob_exp'] = 0
  data['ans_prob_obs'] = 0

  for l in range(layers):
      print("layer: ", l)
      string = 'ans_prob_obs_edit_layer'+str(l)
      print(string)
      data[string] = 0
      average_answer_prob_change_after_edit = 0

      h=0
      for i in range(len(data['answer'])):
        answer = data['answer'][i]
        memory = data['explicit_entity'][i]
        obscure_prompt = data['obscure_sentence'][i]
        explicit_prompt = data['explicit_sentence'][i]

        #first_answer_tok = model.to_tokens(answer, prepend_bos=False)[0][0].item()

        explicit_ans_prob = get_ans_prob(model, answer, explicit_prompt)
        obscure_ans_prob = get_ans_prob(model, answer, obscure_prompt)
        #print(type(obscure_ans_prob ))

        data.loc[i, 'ans_prob_exp'] = explicit_ans_prob
        data.loc[i, 'ans_prob_obs'] = obscure_ans_prob

        logits, patched_logits = apply_edit(model,
                                          memory,
                                          obscure_prompt,
                                          tweak_factor=tweak_factor,
                                          layer=l,
                                          head_num=h)

        obscure_ans_prob_after_edit =  get_ans_prob(model, answer, logits=patched_logits) #torch.nn.functional.softmax(patched_logits[0][-1], dim=0)[first_answer_tok].item()
        data.loc[i, string] = obscure_ans_prob_after_edit

  return data

def reject_outliers(data, m = 2., cutoff=99):
    cutoff = np.percentile(data, cutoff)
    print("cutoff: ",cutoff)
    return data[data < cutoff]

def edit_heatmap(data, model, layers=12, heads=1, tweak_factor=4):
  #average_answer_prob_change_after_edit = 0
  avg_edits = np.empty([layers, heads])
  avg_percent_diffs = np.empty([layers, heads])
  avg_percent_diffs_wo_outliers = np.empty([layers, heads])
  median_percent_diffs = np.empty([layers, heads])

  for l in range(layers):
      print("layer: ", l)
    #for h in range(heads):
      # this is a hacky way to hold the head number constant at head 0, bc it doesn't matter which head we inject into since they all get concatenated anyway
      h=0
      #print("head: ", h, " layer: ", l)
      #print("Average Answer probability difference after edit: ", average_answer_prob_change_after_edit/len(data['answer']))

      orig_probs_list = np.zeros(len(data['answer']))
      edit_probs_list = np.zeros(len(data['answer']))
      for i in range(len(data['answer'])):
        answer = data['answer'][i]
        memory = data['explicit_entity'][i]
        prompt = data['obscure_sentence'][i]
        logits, patched_logits = apply_edit(model,
                                          memory,
                                          prompt,
                                          tweak_factor=tweak_factor,
                                          layer=l,
                                          head_num=h)

        #print("Diff between logits and patched logits: ", logits==patched_logits)
        first_answer_tok = model.to_tokens(answer, prepend_bos=False)[0][0].item()
        answer_prob_before_mem = torch.nn.functional.softmax(logits[0][-1], dim=0)[first_answer_tok]
        #average_original_answer_prob += answer_prob_before_mem
        answer_prob_after_mem = torch.nn.functional.softmax(patched_logits[0][-1], dim=0)[first_answer_tok]
        #average_answer_prob_change_after_edit += answer_prob_after_mem - answer_prob_before_mem

        orig_probs_list[i] = answer_prob_before_mem
        edit_probs_list[i] = answer_prob_after_mem
        #percent_difference = ((answer_prob_after_mem - answer_prob_before_mem) / torch.abs(answer_prob_before_mem))
        #average_percent_difference += percent_difference

        '''
        if l == 0:
          print("Prompt: ", prompt)
          print("Answer: ", data['answer'][i])
          print("Memory: ", memory)
          print("original logits | Answer Probability: ", answer_prob_before_mem)
          #print(interpret_logits_as_vocab(model, logits))
          print("edited logits| Answer Probability: ", answer_prob_after_mem)
          #print(interpret_logits_as_vocab(model, patched_logits))
          #print("Percent increase in Answer probability difference after edit: ", percent_difference)
          #print("Cumulative Percent difference: ", average_percent_difference)
          print("---------------- ", i)
        '''

      average_answer_prob_change_after_edit = np.average(edit_probs_list - orig_probs_list)
      average_original_answer_prob = np.average(orig_probs_list)
      average_percent_difference = np.average((edit_probs_list - orig_probs_list) / orig_probs_list)
      median_percent_difference = np.median((edit_probs_list - orig_probs_list) / orig_probs_list)



      print("Average Answer Probability before edit: ", average_original_answer_prob)
      print("Average Answer probability difference after edit: ", average_answer_prob_change_after_edit)
      print("Average Percent increase in Answer probability difference after edit: ", average_percent_difference * 100)
      print("Median Percent increase in Answer probability difference after edit: ", median_percent_difference * 100)

      #Reject outliers
      percent_difference_wo_outliers = reject_outliers(((edit_probs_list - orig_probs_list) / orig_probs_list))
      print("How many values are kept after outliers: ", len(percent_difference_wo_outliers))
      average_percent_difference_wo_outliers = np.average(percent_difference_wo_outliers)

      print("Average Percent increase in Answer probability difference w/o Outliers after edit: ", average_percent_difference_wo_outliers * 100)

      avg_edits[l,h] = average_answer_prob_change_after_edit
      avg_percent_diffs[l,h] = average_percent_difference * 100
      avg_percent_diffs_wo_outliers[l,h] = average_percent_difference_wo_outliers * 100
      median_percent_diffs[l,h] = median_percent_difference * 100


  return avg_edits, avg_percent_diffs, median_percent_diffs, avg_percent_diffs_wo_outliers


"""# Fake Injection Experiments
injecting unrelavent noise so that we can see if our method actually works

"""


def fake_injections(data=data, model=gpt2_small, layer=6, k=30, tweak_factor=4, full_title="GPT2_small_hand_fake_inject.csv"):
  data_cp = data.copy()
  data_cp['answer_prob_exp'] = 0
  data_cp['answer_prob_obs'] = 0
  num_data_points = len(data['answer'])
  subjects = list(data['explicit_entity'])


  counter = 0
  for s in subjects:

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

        #print("Diff between logits and patched logits: ", logits==patched_logits)

        first_answer_tok = model.to_tokens(answer, prepend_bos=False)[0][0].item()
        answer_prob_after_mem = torch.nn.functional.softmax(patched_logits[0][-1], dim=0)[first_answer_tok]
        data_cp.loc[i, subject_answer_edit] = answer_prob_after_mem.item()


        vals, idx = torch.topk(patched_logits[0][-1], k)
        data_cp.at[i, subject_top_k]= idx.tolist()


        if counter == 0:
          answer_prob_before_mem = torch.nn.functional.softmax(logits[0][-1], dim=0)[first_answer_tok]
          data_cp.loc[i, 'answer_prob_obs'] = answer_prob_before_mem.item()
          explicit_ans_prob = get_ans_prob(model, answer, explicit_prompt)
          #print(explicit_prompt)
          #print(answer)
          #print(explicit_ans_prob)
          data_cp.loc[i, 'answer_prob_exp'] = get_ans_prob(model, answer, explicit_prompt)

      counter+=1


  data_loc = "./"
  data_cp.to_csv(data_loc+full_title)
  return data_cp

def get_probs(data=data, model=gpt2_small, ):
  data_cp = data.copy()
  data_cp['answer_prob_exp'] = 0
  data_cp['answer_prob_obs'] = 0
  num_data_points = len(data['answer'])


  for i in range(num_data_points):
    answer = data['answer'][i]

    prompt = data['obscure_sentence'][i]
    explicit_prompt = data['explicit_sentence'][i]

    data_cp.loc[i, 'answer_prob_obs'] = get_ans_prob(model, answer, prompt)
    data_cp.loc[i, 'answer_prob_exp'] = get_ans_prob(model, answer, explicit_prompt)


  return data_cp

#data_cp = get_probs(multi_1000, gpt2_small)
#data_loc = "./"

#data_cp.to_csv(data_loc+"data_with_exp_probs.csv")

data_cp = fake_injections(data=data, model=gpt2_small, layer=9, k=30, tweak_factor=4, full_title="GPT2_small_hand_fake_inject_layer9_tweak4.csv")

data_cp = fake_injections(data=data, model=gpt2_large, layer=14, k=30, tweak_factor=10, full_title="GPT2_large_hand_fake_inject_layer14_tweak10.csv")

data_cp = fake_injections(data=multi_1000, model=gpt2_small, layer=8, k=30, tweak_factor=4, full_title="GPT2_small_2wmh_fake_inject_layer8_tweak4.csv")

