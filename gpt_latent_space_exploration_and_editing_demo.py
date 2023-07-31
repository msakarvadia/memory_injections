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

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="",title="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis, "title":title}, **kwargs).show(renderer)

"""# Get Models"""

device = "cuda" if torch.cuda.is_available() else "cpu"

gpt2_small = HookedTransformer.from_pretrained("gpt2-small", device=device)
gpt2_large = HookedTransformer.from_pretrained("gpt2-large", device=device)

"""# Interpret the Attention Head Outputs as knowledge retrievers

Hypothesis:

*   The residual stream of a transformer acts as "memory" for the transformer
*   Each attention layer plays a specific role in editing the concepts in the residual stream (these are more like ideas/themes/concepts rather than actual facts)
*   the Attention head outputs heavily influence what knowledge is retrieved from the MLPs that follow it.
*   Within an attention layer, individual heads play even more specific roles (i.e. pronoun head, proper noun head, etc.)
*   The MLPs act as knowledge stores. Based on the concepts in the residual stream, specific facts might be retrieved from the MLP and pushed into the residual stream.



What we are doing here:

*   We are going to project the latent space outputs of each attention layer back into vocabulary space so we can see what concepts a particular layer is adding to the residual stream
*   We will also do this at the individual attention-head granularity so we can emprically inspect if specific heads have specific themes/roles





"""

for name, param in gpt2_small.named_parameters():
  print(name, param.shape)

for name, param in gpt2_large.named_parameters():
  print(name, param.shape)

gpt2_small

# Function: head_latent_space_projector
# Args: agregate heads, individual heads, k tokens, model, prompt
# This function just projects the latent space output of each head in a
# transformer back into vocabulary space so a user can assess what information is being put back into memory

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

        #TODO print out the next 10 token predictions as well

prompt = "George Washington fought in the"
head_latent_space_projector(gpt2_small, prompt, 10, 12, aggregate_heads=True, intermediate_tokens=False)

prompt = "George Washington fought in the"
prompt = "St. Peter's Bacillica is in the city of"
head_latent_space_projector(gpt2_small, prompt, 10, 12, aggregate_heads=False, intermediate_tokens=False)

prompt = "The first president of the United States"
prompt = "The largest church in the world is in the city of"
head_latent_space_projector(gpt2_small, prompt, 10, 12, aggregate_heads=False, intermediate_tokens=False)

prompt = "George Washington fought in the"
head_latent_space_projector(gpt2_small, prompt, 10, 12, aggregate_heads=True, intermediate_tokens=True)

prompt = "The first president of the United States fought in the"
head_latent_space_projector(gpt2_small, prompt, 10, 12, aggregate_heads=True, intermediate_tokens=True)

prompt = "The first president of the United States fought in the"
head_latent_space_projector(gpt2_large, prompt, 10, 20, aggregate_heads=False, intermediate_tokens=False)

"""# How do we edit the memories at each layer?

We can inject concepts into the residual stream by editing the outputs of the attention layers (both in aggregate, and also at the invididual head level)

How do we do this:

We project memories from vocabulary space into the pseudo-hidden latent space of the model by applying the transpose of the unembedding matrix to the model. We then add these transformed memories directly to the outputs of an attention layer.
"""

# We define a residual stream patching hook
# We choose to act on the residual stream at the start of the layer, so we call it resid_pre
# The type annotations are a guide to the reader and are not necessary

#Args:
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

# Use functools.partial to create a temporary hook function with the position fixed
temp_hook_fn = partial(memory_tweaker_hook,
                       extra_info="George Washington",
                       vocab_size=50257,
                       model=gpt2_small,
                       tweak_factor=3)

prompt = "The first president of the United States fought in the"
#Get original logits
logits = gpt2_small(prompt)

#Get patched Logits
layer  = 10  # 9
patched_logits = gpt2_small.run_with_hooks(prompt,
                          fwd_hooks=[
                                      ( utils.get_act_name("attn_out", layer),
                                      temp_hook_fn)
                                      ]
                          )

logits

topk_token_vals, topk_token_preds = torch.topk(logits, 70)
gpt2_small.to_string(topk_token_preds[0][-1])

topk_token_vals_edit, topk_token_preds_edit = torch.topk(patched_logits, 70)
gpt2_small.to_string(topk_token_preds_edit[0][-1])

"""# How do we edit the memories at each head?

We can inject concepts into the residual stream by editing the outputs of the individual attention heads.

How do we do this:

We project memories from vocabulary space into the pseudo-hidden latent space of the model by applying the transpose of the unembedding matrix to the model. We then add these transformed memories directly to an individual attention head as the model is doing inference.
"""

# We define a residual stream patching hook
# We choose to act on the residual stream at the start of the layer, so we call it resid_pre
# The type annotations are a guide to the reader and are not necessary

#Args:
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

"""Below, simply edit the prompt, extra_info, head_number, tweak_factor, layer to adjust to your example."""

# Use functools.partial to create a temporary hook function with the position fixed
temp_hook_fn = partial(memory_tweaker_head_hook,
                       extra_info="afasd asdfjadsf ldslfa sdfasdfa alsdfjeww ANNAndsaf asdnf HELLO NNEHS dfjasdle e",
                       vocab_size=50257,
                       model=gpt2_small,
                       tweak_factor=4,
                       head_num=0)


prompt = "The leader of the United States lives in the"
#Get original logits
logits = gpt2_small(prompt)

#Get patched Logits
layer  = 9
patched_logits_a = gpt2_small.run_with_hooks(prompt,
                          fwd_hooks=[
                                      ( utils.get_act_name("result", layer),
                                      temp_hook_fn)
                                      ]
                          )
print(patched_logits_a[-1])

# Use functools.partial to create a temporary hook function with the position fixed
temp_hook_fn = partial(memory_tweaker_head_hook,
                       extra_info="afasd asdfjadsf ldslfa sdfasdfa alsdfjeww ANNAndsaf asdnf HELLO NNEHS dfjasdle e",
                       vocab_size=50257,
                       model=gpt2_small,
                       tweak_factor=4,
                       head_num=3)


prompt = "The leader of the United States lives in the"
#Get original logits
logits = gpt2_small(prompt)

#Get patched Logits
layer  = 9
patched_logits_b = gpt2_small.run_with_hooks(prompt,
                          fwd_hooks=[
                                      ( utils.get_act_name("result", layer),
                                      temp_hook_fn)
                                      ]
                          )

print(patched_logits_b)

print("Unedited top K tokens: ")
topk_token_vals, topk_token_preds = torch.topk(logits, 70)
gpt2_small.to_string(topk_token_preds[0][-1])

print("Edited top K tokens: ")
topk_token_vals_edit, topk_token_preds_edit = torch.topk(patched_logits_a, 300)
gpt2_small.to_string(topk_token_preds_edit[0][-1])

print("Edited top K tokens: ")
topk_token_vals_edit, topk_token_preds_edit = torch.topk(patched_logits_b, 300)
gpt2_small.to_string(topk_token_preds_edit[0][-1])

# Use functools.partial to create a temporary hook function with the position fixed
temp_hook_fn = partial(memory_tweaker_head_hook,
                       extra_info="Barak Obama",
                       vocab_size=50257,
                       model=gpt2_large,
                       tweak_factor=3,
                       head_num=0)

prompt = "The first black president of the United States was a member of the"
#Get original logits
logits = gpt2_large(prompt)

#Get patched Logits
layer  = 10
patched_logits = gpt2_large.run_with_hooks(prompt,
                          fwd_hooks=[
                                      ( utils.get_act_name("result", layer),
                                      temp_hook_fn)
                                      ]
                          )

torch.equal(patched_logits, logits)

print("Unedited top K tokens: ")
topk_token_vals, topk_token_preds = torch.topk(logits, 70)
gpt2_small.to_string(topk_token_preds[0][-1])

print("Edited top K tokens: ")
topk_token_vals_edit, topk_token_preds_edit = torch.topk(patched_logits, 70)
gpt2_small.to_string(topk_token_preds_edit[0][-1])

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

logits, patched_logits = apply_edit(gpt2_large, "Abe Lincoln",
                                    "George Washington fought in the",
                                    tweak_factor=4, layer=9, head_num=8)
print("original logits")
print(interpret_logits_as_vocab(gpt2_large, logits))
print("edited logits")
print(interpret_logits_as_vocab(gpt2_large, patched_logits))

torch.equal(logits, patched_logits)

logits, patched_logits = apply_edit(gpt2_large, "George Washington",
                                    "The first president of the United States fought in the",
                                    tweak_factor=4, layer=9, head_num=5)
print("original logits")
print(interpret_logits_as_vocab(gpt2_large, logits))
print("edited logits")
print(interpret_logits_as_vocab(gpt2_large, patched_logits))

logits, patched_logits = apply_edit(gpt2_large, "George Washington",
                                    "The first president of the United States fought in the",
                                    tweak_factor=4, layer=9, head_num=7)
print("original logits")
print(interpret_logits_as_vocab(gpt2_large, logits))
print("edited logits")
print(interpret_logits_as_vocab(gpt2_large, patched_logits))

logits, patched_logits = apply_edit(gpt2_small,
                                    "President",
                                    "The leader of the United States lives in the",
                                    tweak_factor=4, layer=9, head_num=8)
print("original logits")
print(interpret_logits_as_vocab(gpt2_small, logits))
print("edited logits")
print(interpret_logits_as_vocab(gpt2_small, patched_logits))

logits, patched_logits_a = apply_edit(gpt2_small,
                                    "President",
                                    "The leader of the United States lives in the",
                                    tweak_factor=40, layer=9, head_num=9)
print("original logits")
print(interpret_logits_as_vocab(gpt2_small, logits))
print("edited logits")
print(interpret_logits_as_vocab(gpt2_small, patched_logits))


#Get Data
data = get_handwritten_data()
multi = get_multi_100()
multi_1000 = get_multi_1000()
# worksheet

"""# How useful is memory editing at a specific head (if we know what to inject)

The working hypothesis here is the: the obscure (multi-hop) prompts are lacking specific memories (the additional hop) which is why their completions are not as good as explicit prompt completions.

We have a dataset of obscure prompts, explicit prompts, and respectively their obscure subject and explicit subject.

We wonder if naievely injecting the explicit subject as a memory into the obscure prompts hidden activation states will be enough to correct the final prompt.

We will measure this memory injection approach's success by counting:
1. How many places the desired next token moved up in the prediction heirarchy
2. How much the individual probability of the desired next token increases
"""

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

model =gpt2_small
answer = " Revolutionary"
explicit_prompt = "George Washington fought in the"
get_ans_prob(model, answer, explicit_prompt)

model =gpt2_small
answer = " Revolutionary"
explicit_prompt = "The first president of the United States fought in the"
get_ans_prob(model, answer, explicit_prompt)

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
    '''
    print(data)
    d = np.abs(data - np.median(data)) # if you want to remove high and low outliers
    #d = data - np.median(data) # if you want to remove just high outliers
    #print(d)
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    print(data[s<m])
    return data[s<m]
    '''

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

data = edit_layerwise_tweak(data, gpt2_small, layers=12, heads=1, tweak_factor=4)

data_5 = edit_layerwise_tweak(data, gpt2_small, layers=12, heads=1, tweak_factor=5)

for i in range(2,12,2):
  print("tweak factor: ", i)
  heat_map = edit_heatmap(data, gpt2_small, layers=12, heads=1, tweak_factor=i)

  plt.figure(figsize=(12,12))
  ax = heatmap(heat_map, annot=True, fmt=".5f", cmap = 'RdYlGn') #, vmin=0, vmax=1)
  ax.set(xlabel='Average Increase in Probability of Expected Next Token', ylabel='Edited Layer')
  plt.show()

for i in range(1,10,2):
  print("tweak factor: ", i)
  heat_map = edit_heatmap(data, gpt2_large, layers=36, heads=1, tweak_factor=i)

  plt.figure(figsize=(12,12))
  ax = heatmap(heat_map, annot=True, fmt=".5f", cmap = 'RdYlGn') #, vmin=0, vmax=1)
  ax.set(xlabel='Average Increase in Probability of Expected Next Token', ylabel='Edited Layer')
  plt.show()

heat_map, avg_percent_inc, median_percent_inc, avg_percent_inc_wo_outliers = edit_heatmap(data, gpt2_small, layers=12, heads=1, tweak_factor=4)

plt.figure(figsize=(6,12))
ax = heatmap(heat_map, annot=True, fmt=".5f", cmap = 'RdYlGn') #, vmin=0, vmax=1)
ax.set(xlabel='Average Increase in Probability of Expected Next Token', ylabel='Edited Layer')
plt.show()

# Save avg percent increase for pgt2 small handwritten
#need to mount drive before saving files!
from google.colab import drive
drive.mount('/content/drive')

data_loc = "drive/MyDrive/Research/Mechanistic Interpretability/Figures/Fig_data/"
data_name = "avg_percent_inc_wo_outliers_gpt2_small_handwritten"
torch.save(avg_percent_inc_wo_outliers, data_loc+data_name)

plt.figure(figsize=(6,12))
ax = heatmap(avg_percent_inc, annot=True, fmt=".5f", cmap = 'RdYlGn') #, vmin=0, vmax=1)
ax.set(xlabel='Average Increase in Probability of Expected Next Token', ylabel='Edited Layer')
plt.show()

plt.figure(figsize=(6,12))
ax = heatmap(median_percent_inc, annot=True, fmt=".5f", cmap = 'RdYlGn') #, vmin=0, vmax=1)
ax.set(xlabel='Median Percent Increase in Probability of Expected Next Token', ylabel='Edited Layer')
plt.show()

plt.figure(figsize=(6,12))
ax = heatmap(avg_percent_inc_wo_outliers, annot=True, fmt=".5f", cmap = 'RdYlGn') #, vmin=0, vmax=1)
ax.set(xlabel='Avger Percent Increase in Probability w/o outliers of Expected Next Token', ylabel='Edited Layer')
plt.show()

data

heat_map_L, avg_percent_inc_L, median_percent_inc_L, avg_percent_inc_wo_outliers_L = edit_heatmap(data, gpt2_large, layers=36, heads=1, tweak_factor=4)

plt.figure(figsize=(6,12))
ax = heatmap(heat_map_L, annot=True, fmt=".5f", cmap = 'RdYlGn') #, vmin=0, vmax=1)
ax.set(xlabel='Average Increase in Probability of Expected Next Token', ylabel='Edited Layer')
plt.show()

plt.figure(figsize=(6,12))
ax = heatmap(heat_map_L, annot=True, fmt=".5f", cmap = 'RdYlGn') #, vmin=0, vmax=1)
ax.set(xlabel='Average Increase in Probability of Expected Next Token', ylabel='Edited Layer')
plt.show()

plt.figure(figsize=(6,12))
ax = heatmap(avg_percent_inc_wo_outliers_L, annot=True, fmt=".5f", cmap = 'RdYlGn') #, vmin=0, vmax=1)
ax.set(xlabel='Avger Percent Increase in Probability w/o outliers of Expected Next Token', ylabel='Edited Layer')
plt.show()

plt.figure(figsize=(6,12))
ax = heatmap(avg_percent_inc_wo_outliers_L, annot=True, fmt=".5f", cmap = 'RdYlGn') #, vmin=0, vmax=1)
ax.set(xlabel='Avger Percent Increase in Probability w/o outliers of Expected Next Token', ylabel='Edited Layer')
plt.show()

# Save avg percent increase for pgt2 large handwritten
from google.colab import drive
drive.mount('/content/drive')

data_loc = "drive/MyDrive/Research/Mechanistic Interpretability/Figures/Fig_data/"
data_name = "avg_percent_inc_wo_outliers_gpt2_large_handwritten"
torch.save(avg_percent_inc_wo_outliers_L, data_loc+data_name)

data = print_edit_results(data, gpt2_small, tweak_factor=4)

data = print_edit_results(multi  , gpt2_small)

data = print_edit_results(data, gpt2_large, layer=26, head_num=17, tweak_factor=4)

data = print_edit_results(multi, gpt2_large, layer=26, head_num=17, tweak_factor=4)

"""#Tweak Factor Analysis

We will replicate the above experiments, but accross a number of tweak factors so that we can do a sensativity analysis
"""

from google.colab import drive
drive.mount('/content/drive')

# We are going to define a more general purpose editing function which records more useful metrics up front so that we can do post-analysis later
def edit_heatmap(data, model, layers=12, heads=1, tweak_factor=4, k=30, print_output=False):
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

data_cp = edit_heatmap(data, gpt2_small, print_output=True)

data_cp

# Function to vary the tweak factor

def tweak_factor_vary(tweak_factors, data, model=gpt2_small, layers=12, title="gpt2_small_subject_edits", data_loc = "drive/MyDrive/Research/Mechanistic Interpretability/Figures/Fig_data/"):
  for i in tweak_factors:
    full_title = title+"_tweakFactor_"+str(i)+".csv"
    print(full_title)

    data_cp = edit_heatmap(data, model, layers=layers, heads=1, tweak_factor=i)

    data_loc = "drive/MyDrive/Research/Mechanistic Interpretability/Figures/Fig_data/"

    #save each dataframe with a descriptive title
    #torch.save(data_cp, data_loc+full_title)

    data_cp.to_csv(data_loc+full_title)

tweak_factors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
tweak_factor_vary(tweak_factors, data, gpt2_small, 12, title="gpt2_small_subject_edits_hand")

tweak_factors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
tweak_factor_vary(tweak_factors, data, gpt2_large, 36, title="gpt2_large_subject_edits_hand")

tweak_factors = [6,7,8,9,10,11,12,13,14,15]
#tweak_factors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
tweak_factor_vary(tweak_factors, multi_1000, gpt2_small, 12, title="gpt2_small_subject_edits_2wmh")

tweak_factors = [3,4,5,6,7,8,9,10,11,12,13,14,15]
#tweak_factors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
tweak_factor_vary(tweak_factors, multi_1000, gpt2_large, 36, title="gpt2_large_subject_edits_2wmh")

"""# Fake Injection Experiments
injecting unrelavent noise so that we can see if our method actually works




"""

list(data['explicit_entity'])

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


  data_loc = "drive/MyDrive/Research/Mechanistic Interpretability/Figures/Fig_data/"
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

data_cp = get_probs(multi_1000, gpt2_small)
data_loc = "drive/MyDrive/Research/Mechanistic Interpretability/Figures/Fig_data/"


data_cp.to_csv(data_loc+"data_with_exp_probs.csv")

data_cp = fake_injections(data=data, model=gpt2_small, layer=9, k=30, tweak_factor=4, full_title="GPT2_small_hand_fake_inject_layer9_tweak4.csv")

data_cp = fake_injections(data=data, model=gpt2_large, layer=14, k=30, tweak_factor=10, full_title="GPT2_large_hand_fake_inject_layer14_tweak10.csv")

data_cp = fake_injections(data=multi_1000, model=gpt2_small, layer=8, k=30, tweak_factor=4, full_title="GPT2_large_hand_fake_inject_layer8_tweak4.csv")

model.to_string([8372, 1578, 7840, 2520, 12108, 10183, 8211])

data_cp

np.mean(reject_outliers((data_cp['ans_prob_obs_edit_subject_George Washington'] - data_cp['answer_prob_obs'] ) / data_cp['answer_prob_obs']))

np.mean(reject_outliers((data_cp['ans_prob_obs_edit_subject_The president'] - data_cp['answer_prob_obs'] ) / data_cp['answer_prob_obs']))

np.mean(reject_outliers((data_cp['ans_prob_obs_edit_subject_St. Peter\'s Basilica'] - data_cp['answer_prob_obs'] ) / data_cp['answer_prob_obs']))

(data_cp['ans_prob_obs_edit_subject_George Washington'] - data_cp['answer_prob_obs'] ) / data_cp['answer_prob_obs']

"""# Highest KL divergence head memory injection (injecting based on highest KL divergence of specific prompts, not averaged over dataset)

Now lets only inject the explicit subject at a head that has the highest KL divergence between the hidden activations of explicit vs obscure prompts
"""

def kl_div_headwise_vocab_diff_plotter(explicit_prompt, obscure_prompt, num_layers, num_heads, model, plot=False):
  num_layers = num_layers
  num_heads = num_heads

  #This is how you change the amount of heads that are cached
  model.cfg.use_attn_result = True

  kl_loss = nn.KLDivLoss(reduction="batchmean")

  explicit_tokens = model.to_tokens(explicit_prompt)
  obscure_tokens = model.to_tokens(obscure_prompt)

  explicit_logits, explicit_cache = model.run_with_cache(explicit_tokens)
  obscure_logits, obscure_cache = model.run_with_cache(obscure_tokens)

  divs = torch.zeros([num_layers, num_heads])
  for layer in range(num_layers):
    position = -1 #last token index

    explicit_head_results = explicit_cache[utils.get_act_name("result", layer)]
    obscure_head_results = obscure_cache[utils.get_act_name("result", layer)]

    for head in range(num_heads):

      log_prob_explicit = F.log_softmax(model.unembed(explicit_head_results[:, :, head, :]), dim=1)[:,position,:]
      log_prob_obscure = F.softmax(model.unembed(obscure_head_results[:, :, head, :]), dim=1)[:,position,:]
      divs[layer][head] = (kl_loss(log_prob_explicit, log_prob_obscure).item())

  if(plot):
    imshow(divs, xaxis="head", yaxis="Layer")
  return divs

def print_kl_div_edit_results(data, model, num_layers=12, num_heads=12, tweak_factor=4, k=5):
  average_answer_prob_change_after_edit = 0
  for i in range(len(data['answer'])):
    answer = data['answer'][i]
    memory = data['explicit_entity'][i]
    explicit_prompt = data['explicit_sentence'][i]
    obscure_prompt = data['obscure_sentence'][i]

    #Get the correct head, layer num
    divs = kl_div_headwise_vocab_diff_plotter(explicit_prompt=explicit_prompt,
                                   obscure_prompt=obscure_prompt,
                                   num_layers=num_layers,
                                   num_heads=num_heads,
                                   model=model,
                                   plot=False)

    abs_divs = torch.abs(divs) #Rows = layers, Columns = heads
    max_div = torch.argmax(torch.flatten(abs_divs))
    layer = max_div // (num_layers)
    head_num = max_div % (num_heads)


    # patch memory
    logits, patched_logits = apply_edit(model,
                                      memory,
                                      obscure_prompt,
                                      tweak_factor=tweak_factor,
                                      layer=layer,
                                      head_num=head_num)


    first_answer_tok = gpt2_small.to_tokens(answer, prepend_bos=False)[0][0].item()
    answer_prob_before_mem = torch.nn.functional.softmax(logits[0][-1], dim=0)[first_answer_tok]
    answer_prob_after_mem = torch.nn.functional.softmax(patched_logits[0][-1], dim=0)[first_answer_tok]
    average_answer_prob_change_after_edit += answer_prob_after_mem - answer_prob_before_mem

    print("Explicit Prompt: ", explicit_prompt)
    print("Obscure Prompt: ", obscure_prompt)
    print("Layer: ", layer, "Head Number: ", head_num)
    print("Answer: ", data['answer'][i])
    print("Memory: ", memory)
    print("original logits | Answer Probability: ", answer_prob_before_mem)
    print(interpret_logits_as_vocab(model, logits))
    print("edited logits | Answer Probability: ", answer_prob_after_mem)
    print(interpret_logits_as_vocab(model, patched_logits))
    print("---------------- ", i)
  print("Average Answer probability difference after edit: ", average_answer_prob_change_after_edit/len(data['answer']))

print_targeted_edit_results(data, gpt2_small)

"""*  I tried to inject just he subject token naievly at the head with the highest KL diveregnce and I am empiraclly seeing that the desired next token probability goes down. Why? bc the heads we are injecting subjects into are not subject heads

#Can we do a much more targeted memory injection?

For both the handwritten dataset, and the script generated dataset (for hotpotqa) we see that a niaeve editing technique works. I wonder now if we can do a more targeted editing approach where we isolate the top 1 head in a network where there is the greatest divergence between an obscure and explicit pompt and edit it with the top vocabulary token from the explicit prompts hidden space. (we can replicate this experiment but instead just inject the explicit subject as a memory; I expect this approach to perform worse than injecting the top explicit vocab from the corresponding head).
"""

def get_top_k_vocab_at_head(head, layer, model, prompt, k=5):
  #This is how you change the amount of heads that are cached
  model.cfg.use_attn_result = True

  tokens = model.to_tokens(prompt)
  logits, cache = model.run_with_cache(tokens)

  position = -1 #last token index

  head_results = cache[utils.get_act_name("result", layer)]
  probs = F.log_softmax(model.unembed(head_results[:, :, head, :]), dim=1)[:,position,:]

  vals, indices = torch.topk(probs, k)

  return model.to_string(indices)[0]

get_top_k_vocab_at_head(8, 9, gpt2_small, "The first president of the united states fought in the", k=5)

def print_targeted_edit_results(data, model, num_layers=12, num_heads=12, tweak_factor=4, k=30):
  average_answer_prob_change_after_edit = 0
  for i in range(len(data['answer'])):
    answer = data['answer'][i]
    #memory = data['explicit_entity'][i]
    explicit_prompt = data['explicit_sentence'][i]
    obscure_prompt = data['obscure_sentence'][i]

    #Get the correct head, layer num
    divs = kl_div_headwise_vocab_diff_plotter(explicit_prompt=explicit_prompt,
                                   obscure_prompt=obscure_prompt,
                                   num_layers=num_layers,
                                   num_heads=num_heads,
                                   model=model,
                                   plot=False)

    abs_divs = torch.abs(divs) #Rows = layers, Columns = heads
    max_div = torch.argmax(torch.flatten(abs_divs))
    layer = max_div // (num_layers)
    head_num = max_div % (num_heads)

    #Get the explicit memories at a specific head to insert into the obscure hidden activations
    memory = get_top_k_vocab_at_head(head_num.item(), layer.item(), model, explicit_prompt, k=k)

    # patch memory
    logits, patched_logits = apply_edit(model,
                                      memory,
                                      obscure_prompt,
                                      tweak_factor=tweak_factor,
                                      layer=layer,
                                      head_num=head_num)


    first_answer_tok = gpt2_small.to_tokens(answer, prepend_bos=False)[0][0].item()
    answer_prob_before_mem = torch.nn.functional.softmax(logits[0][-1], dim=0)[first_answer_tok]
    answer_prob_after_mem = torch.nn.functional.softmax(patched_logits[0][-1], dim=0)[first_answer_tok]
    average_answer_prob_change_after_edit += answer_prob_after_mem - answer_prob_before_mem

    print("Explicit Prompt: ", explicit_prompt)
    print("Obscure Prompt: ", obscure_prompt)
    print("Layer: ", layer, "Head Number: ", head_num)
    print("Answer: ", data['answer'][i])
    print("Memory: ", memory)
    print("original logits | Answer Probability: ", answer_prob_before_mem)
    print(interpret_logits_as_vocab(model, logits))
    print("edited logits | Answer Probability: ", answer_prob_after_mem)
    print(interpret_logits_as_vocab(model, patched_logits))
    print("---------------- ", i)
  print("Average Answer probability difference after edit: ", average_answer_prob_change_after_edit/len(data['answer']))

print_targeted_edit_results(data, gpt2_small, tweak_factor=2)

print_targeted_edit_results(multi, gpt2_large, num_layers=36, num_heads=20)

"""Note:

*  I tried to inject just he subject token naievly at the head with the highest KL diveregnce and I am empiraclly seeing that the desired next token probability goes down. Why? bc the heads we are injecting subjects into are not subject heads
* I inject the relavent vocab from the head with the highest KL divergence into the activation stream of the obscure prompt; this too actually decreases probability of next prompt completion
* What we are getting at is that the main thing networks are missing is the extra "hop" of the explicit subject, and when that is injected at an attention head that deals with proper nouns and subject, we get the biggest increase in probability
* we can probably do some studies about injecting multi-hop prompts into the specific subject head
"""
