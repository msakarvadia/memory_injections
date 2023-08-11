import sys
sys.path.append("../../")
from data.load_data import get_top_words, get_handwritten_data, get_multi_100, get_multi_1000
from utils import reject_outliers, get_ans_prob, apply_edit, memory_tweaker_head_hook, head_latent_space_projector
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import argparse
import random

torch.set_grad_enabled(False)

#Set up arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--memory_dataset", default="nouns",choices=["subject","top_5000", "nouns","verbs", "adjective", "adverbs", "conjunctions"],  type=str, help="the category to choose fake memories from")
parser.add_argument("--dataset", default="hand",choices=["hand", "2wmh"],  type=str)
parser.add_argument("--model", default="gpt2-small", choices=["gpt2-small", "gpt2-large"],  type=str)
parser.add_argument("--num_layers", default=12, choices=[12, 36],  type=str, help="number of layers in your model")
parser.add_argument("--tweak_factors", default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], type=list, help="list of tweak factors to test")
args = parser.parse_args()
#TODO: specify save directory
#TODO: add extra column per layer to specify the memory that was injected

"""# Get Models"""

device = "cuda" if torch.cuda.is_available() else "cpu"

if(args.model == "gpt2-small"):
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
else:
    model = HookedTransformer.from_pretrained("gpt2-large", device=device)

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

def get_words(data=data, fake_data_type=args.memory_dataset):

  subjects = list(data['explicit_entity'])
  top_5000 = top_words['Top 5000 Words'].dropna().tolist()
  nouns = top_words['Nouns'].dropna().tolist()
  verbs = top_words['Verbs'].dropna().tolist()
  adjectives = top_words['Adjectives'].dropna().tolist()
  adverbs = top_words['Adverbs'].dropna().tolist()
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
  return words

# We are going to define a more general purpose editing function which records more useful metrics up front so that we can do post-analysis later
def edit_heatmap(data, model, layers=12, heads=1, tweak_factor=4, k=30, print_output=True):
  num_data_points = len(data['answer'])

  data_cp = data.copy()
  data_cp['answer_prob_exp'] = 0
  data_cp['answer_prob_obs'] = 0

  memories = get_words()

  for l in range(layers):
      layer_answer_edit = 'ans_prob_obs_edit_layer'+str(l)
      layer_top_k = 'topk_tok_obs_edit_layer'+str(l)
      data_cp[layer_answer_edit] = 0
      data_cp[layer_top_k] = ''
      data_cp[layer_top_k] = data_cp[layer_top_k].apply(list)

    #for h in range(heads):
      # this is a hacky way to hold the head number constant at head 0, bc it doesn't matter which head we inject into since they all get concatenated anyway
      h=0
      for i in range(num_data_points):
        answer = data['answer'][i]
        #randomly choose word from "memories"
        memory = random.choice(memories)
        print(memory)
        prompt = data['obscure_sentence'][i]
        explicit_prompt = data['explicit_sentence'][i]
        logits, patched_logits = apply_edit(model,
                                          memory,
                                          prompt,
                                          tweak_factor=tweak_factor,
                                          layer=l,
                                          head_num=h)

        first_answer_tok = model.to_tokens(answer, prepend_bos=False)[0][0].item()
        answer_prob_before_mem = torch.nn.functional.softmax(logits[0][-1], dim=0)[first_answer_tok]
        answer_prob_after_mem = torch.nn.functional.softmax(patched_logits[0][-1], dim=0)[first_answer_tok]

        if l == 0:
          data_cp.loc[i, 'answer_prob_obs'] = answer_prob_before_mem.item()
          data_cp.loc[i, 'answer_prob_exp'] = get_ans_prob(model, answer, explicit_prompt)

        data_cp.loc[i, layer_answer_edit] = answer_prob_after_mem.item()

        vals, idx = torch.topk(patched_logits[0][-1], k)
        data_cp.at[i, layer_top_k]= idx.tolist()

        #print(i)
      if(print_output):
        print("layer: ", l)

        print("Average Answer Probability before edit: ", data_cp['answer_prob_obs'].mean())
        print("Average Answer probability difference after edit: ", (data_cp[layer_answer_edit] -data_cp['answer_prob_obs']).mean())
        print("Average Percent increase in Answer probability difference after edit: ", ((data_cp[layer_answer_edit] -data_cp['answer_prob_obs'])/ data_cp['answer_prob_obs']).mean() * 100)

  return data_cp

# Function to vary the tweak factor
def tweak_factor_vary(tweak_factors=args.tweak_factors, data=args.dataset, model=args.model, layers=args.num_layers, title="gpt2_small_subject_edits", data_loc = "./"):
  for i in tweak_factors:
    #specify title of file automatically
    full_title=f"{args.model}_{args.dataset}_pos_inject_tweak{i}_{args.memory_dataset}.csv"
    print(full_title)

    data_cp = edit_heatmap(data, model, layers=layers, heads=1, tweak_factor=i)

    data_cp.to_csv(data_loc+full_title)

#Experiments

tweak_factors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
tweak_factor_vary(args.tweak_factors, data, model, args.num_layers, title="gpt2_small_subject_edits_hand")
