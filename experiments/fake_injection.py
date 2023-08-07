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
          print(explicit_prompt)
          print(answer)
          print("explicit prob", explicit_ans_prob)
          print("edit prob: ", answer_prob_after_mem)
          data_cp.loc[i, 'answer_prob_exp'] = get_ans_prob(model, answer, explicit_prompt)

      counter+=1


  data_loc = "./"
  data_cp.to_csv(data_loc+full_title)
  return data_cp


data_cp = fake_injections(data=data, model=gpt2_small, layer=9, k=30, tweak_factor=4, full_title="GPT2_small_hand_fake_inject_layer9_tweak4.csv")

data_cp = fake_injections(data=data, model=gpt2_large, layer=14, k=30, tweak_factor=10, full_title="GPT2_large_hand_fake_inject_layer14_tweak10.csv")

#data_cp = fake_injections(data=multi_1000, model=gpt2_small, layer=8, k=30, tweak_factor=4, full_title="GPT2_small_2wmh_fake_inject_layer8_tweak4.csv")

