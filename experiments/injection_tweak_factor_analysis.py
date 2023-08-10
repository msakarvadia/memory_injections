import sys
sys.path.append("../")
from data.load_data import get_handwritten_data, get_multi_100, get_multi_1000
from utils import reject_outliers, get_ans_prob, apply_edit, memory_tweaker_head_hook

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

"""
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
"""

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


"""
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
"""



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