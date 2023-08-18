import sys
sys.path.append("../../")
from data.load_data import get_top_words, get_handwritten_data, get_multi_100, get_multi_1000
from utils import reject_outliers, get_ans_prob, apply_edit, memory_tweaker_head_hook, head_latent_space_projector
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import transformer_lens.utils as utils
import argparse
import random
import os
import pandas as pd

#set random seed!
random.seed(42)

torch.set_grad_enabled(False)

#Set up arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="hand",choices=["hand", "2wmh"],  type=str)
parser.add_argument("--model_name", default="gpt2-small", choices=["gpt2-small", "gpt2-large"],  type=str)
parser.add_argument("--save_dir", default="pos_results", type=str)
args = parser.parse_args()

"""# Get Models"""

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

if(args.model_name == "gpt2-small"):
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    num_layers=12
    num_heads=12
else:
    model = HookedTransformer.from_pretrained("gpt2-large", device=device)
    num_layers=36
    num_heads=20

model.cfg.use_attn_result = True

#Get Data
data = get_handwritten_data('../../data/')
multi = get_multi_100('../../data/')
multi_1000 = get_multi_1000('../../data/')
top_words = get_top_words('../../data/')

if(args.dataset == "data"):
    data = data
if(args.dataset == "2wmh"):
    data = multi_1000

#Get list of layers to cache (all the attn layers)
hook_name = 'result'
model_name = args.model_name

hook_ids = []
for i in range(num_layers):
  hook_ids.append(utils.get_act_name(hook_name, i))

#func to project attn layers back into vocab space
def interp_attn_layer(head_results, num_heads, k_tokens=50, model=model):
  heads = []
  for h in range(num_heads):
          head_out = head_results[:, :, h, :]
          with torch.no_grad():
            logits = model.unembed(head_out)
          heads.append(logits[0][-1].cpu().numpy())
  return heads

#func to grab all of the cached outputs of attn heads
def get_caches_for_prompt(prompt, cache_df, hook_ids, model=model, model_name=args.model_name, num_heads=num_heads):
  with torch.no_grad():
      logits, cache = model.run_with_cache(prompt, names_filter=hook_ids, remove_batch_dim=False, prepend_bos=False)

  layers=[0]*num_heads
  layer=0
  models= [model_name]*num_heads
  head_nums = range(num_heads)
  prompts = [prompt]*num_heads

  for i in cache:
    heads = interp_attn_layer(cache[i], num_heads, k_tokens=50, model=model)
    
    #need to append to pandas df
    df = pd.DataFrame({'layer': layers, 'prompt':prompts, 'head': head_nums, 'head_output':heads, 'model':models})

    #do the appending
    cache_df = pd.concat([cache_df, df])

    layer+=1
    layers=[layer]*num_heads

  return cache_df

#define the dataframe to hold all the cached activations
cache_df = pd.DataFrame(columns=['prompt', 'layer', 'head', 'head_output', 'model'])

#Need to loop over full dataset
    #for both "explicit_sentence" and "obscure_sentence"
for i in data['explicit_sentence']:
    cache_df = get_caches_for_prompt(i, cache_df, hook_ids, model=model, model_name=model_name, num_heads=num_heads)
#cache_df = get_caches_for_prompt("My name is", cache_df, hook_ids, model=model, model_name=model_name, num_heads=num_heads)

full_title=f"{args.model_name}_{args.dataset}_attn_head_outputs.csv"
print(full_title)
cache_df.to_csv(full_title)
