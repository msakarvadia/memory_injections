import sys
import os
sys.path.append("../../")
from data.load_data import get_handwritten_data, get_multi_100, get_multi_1000
from utils import (reject_outliers, get_ans_prob, apply_edit,
                head_latent_space_projector,
                memory_tweaker_unembed_head_hook,
                memory_tweaker_embed_head_hook)
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from transformers import LlamaForCausalLM, LlamaTokenizer



# We are going to define a more general purpose editing function which records more useful metrics up front so that we can do post-analysis later
def edit_heatmap(data, model, dtype, hook_func, layers=12, heads=1, tweak_factor=4, k=30, print_output=True):
  num_data_points = len(data['answer'])

  data_cp = data.copy()
  data_cp['answer_prob_exp'] = 0
  data_cp['answer_prob_obs'] = 0


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
        memory = data['explicit_entity'][i]
        prompt = data['obscure_sentence'][i]
        explicit_prompt = data['explicit_sentence'][i]
        logits, patched_logits = apply_edit(model,
                                          memory,
                                          prompt,
                                          hook_func=hook_func,
                                          dtype=dtype,
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
def tweak_factor_vary(tweak_factors, data, model, layers, hook_func, dtype, title="gpt2_small_subject_edits"):
  for i in tweak_factors:
    full_title = title+"_tweakFactor_"+str(i)+".csv"
    print(full_title)

    data_cp = edit_heatmap(data, model, dtype, hook_func, layers=layers, heads=1, tweak_factor=i)

    data_loc = "./"
    data_cp.to_csv(data_loc+full_title)

def get_model(model_name:str, dtype, device):
        if ("gpt" or "mistral" or "eleuther") in model_name:
        #if ("gpt" in model_name) or ("mistral" in model_name) or ("eluthur" in model_name):
            model = HookedTransformer.from_pretrained(model_name, 
                                                    device=device,
                                                    dtype=dtype)
            print(model.cfg)
        if "llama" in model_name:
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            """
            hf_model = LlamaForCausalLM.from_pretrained(model_name, 
                                                    low_cpu_mem_usage=True,
                                                    torch_dtype=torch.float16
                                                   )
            """

            #Load model into TransformerLens template
            model = HookedTransformer.from_pretrained(model_name, 
                                                   #  hf_model=hf_model, 
                                                     tokenizer=tokenizer,
                                                      device="cpu", 
                                                    # device="cuda", 
                                                     fold_ln=False, 
                                                     center_writing_weights=False, 
                                                     center_unembed=False,
                                                    # n_devices=4
                                                    ) #can load model onto multiple devices but have trouble running hooks
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        print(model.cfg)
        print(model.generate("The capital of Germany is", max_new_tokens=20, temperature=0))

        #cache individual outputs of attention heads
        model.cfg.use_attn_result = True
        return model

def namestr(obj, namespace):
    return [str(name) for name in namespace if namespace[name] is obj]

#Experiments
if __name__=="__main__":

    #Get Data
    data = get_handwritten_data('../../data/')
    multi = get_multi_100('../../data/')
    multi_1000 = get_multi_1000('../../data/')
    datasets = [multi_1000, data]

    models = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-7b-hf",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-j-6B",
    "EleutherAI/gpt-neox-20b",
    "gpt2-xl",
    "gpt2-small",
    "gpt2-large",
    ]

    models_need_more_compute = [
    #"mistralai/Mistral-7B-v0.1", 
    #"mistralai/Mistral-7B-Instruct-v0.1",
    #"meta-llama/Llama-2-70b-chat-hf",
    #"meta-llama/Llama-2-70b-hf",
    #"meta-llama/Llama-2-13b-chat-hf",
    #"meta-llama/Llama-2-13b-hf",
    #"meta-llama/Llama-2-13b-hf",
    ]
    
    hook_types = [memory_tweaker_unembed_head_hook,
                memory_tweaker_embed_head_hook]

    torch.set_grad_enabled(False)

    """# Get Models"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    #iterate over models
    for model_name in models:
        #get model
        model = get_model(model_name, dtype=dtype, device=device)

        for d in datasets:
            #iterate over hook types

            for hook in hook_types:
                #TODO geneate descriptive title/data output location
                save_dir =namestr(d, globals())[0]+"/"+model_name+"/"+namestr(hook, globals())[0]+"/"
                #make save_dir if it doesn't exist
                os.makedirs(save_dir, exist_ok=True)
                print(namestr(hook, globals())[0])
                print(namestr(d, globals())[0])
                print(model_name)

                # define all tweak factor ranges we are interested in
                tweak_factors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                for i in tweak_factors:
                    full_title = save_dir+"tweak_"+str(i)+".csv"
                    print(full_title)


                    data_cp = edit_heatmap(data, model, dtype, hook, layers=model.cfg.n_layers, heads=1, tweak_factor=i)

                    data_cp.to_csv(full_title)
                #tweak_factor_vary(tweak_factors=tweak_factors,data=d, model=model, dtype=dtype, hook_func=hook, layers=model.cfg.n_layers, title=model_name+"_subject_edits_hand")

