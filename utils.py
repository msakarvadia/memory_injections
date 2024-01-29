import torch
from jaxtyping import Float, Int
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
import transformer_lens
from fancy_einsum import einsum
from functools import partial
import transformer_lens.utils as utils

from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from transformers import LlamaForCausalLM, LlamaTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

def reject_outliers(data, cutoff=99):
    cutoff = np.percentile(data, cutoff)
    #print("cutoff: ",cutoff)
    return data[data < cutoff]

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

def interpret_logits_as_vocab(model, logits, top_k=30):
  topk_token_vals_edit, topk_token_preds_edit = torch.topk(logits, top_k)
  return model.to_string(topk_token_preds_edit[0][-1])

def apply_edit(model, extra_memory, prompt, dtype, hook_func, tweak_factor=4, layer=10, head_num=0):
  # Use functools.partial to create a temporary hook function with the position fixed
  temp_hook_fn = partial(hook_func,
                        extra_info= extra_memory, #"Barak Obama",
                        vocab_size=model.cfg.d_vocab, #TODO set this to be model.cfg.vocab_size
                        model=model,
                        tweak_factor=tweak_factor,
                        head_num=head_num,
                        dtype=dtype,
                        )

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

def memory_tweaker_unembed_head_hook(
    attn_result: Float[torch.Tensor, "num_tokens num_heads d_model"],
    hook: HookPoint, #name of layer where we inject memory
    extra_info: str, #the string that we tokenize and then inject into memory
    model: transformer_lens.HookedTransformer, #the model from which we get the unembedding matrix from
    vocab_size: int, #size of model vocabulary
    tweak_factor: float,
    head_num: int, #The number of the head we want to edit
    dtype, #The torch dtype we want to use
) -> Float[torch.Tensor, "batch pos d_model"]:

    tok_extra_info = model.to_tokens(extra_info, prepend_bos=False)

    #extra_memory = torch.zeros(vocab_size)
    extra_memory = torch.zeros(vocab_size, dtype=dtype).to(device)
    for i in tok_extra_info:
      extra_memory[i] = 1

    #extra_memory = einsum("d_vocab, d_vocab d_model-> d_model", extra_memory, W_U.T)
    extra_memory = einsum("d_vocab, d_vocab d_model-> d_model", extra_memory, model.W_U.T) #this line works

    #add the extra_info embedded in latent space to hook_attn_out
    attn_result[:,:,head_num,:] = attn_result[:,:,head_num,:] + extra_memory * tweak_factor

    return attn_result

def memory_tweaker_embed_head_hook(
    attn_result: Float[torch.Tensor, "num_tokens num_heads d_model"],
    hook: HookPoint, #name of layer where we inject memory
    extra_info: str, #the string that we tokenize and then inject into memory
    model: transformer_lens.HookedTransformer, #the model from which we get the unembedding matrix from
    vocab_size: int, #size of model vocabulary
    tweak_factor: float,
    head_num: int, #The number of the head we want to edit
    dtype, #The torch dtype we want to use
) -> Float[torch.Tensor, "batch pos d_model"]:

    tok_extra_info = model.to_tokens(extra_info, prepend_bos=False)

    #extra_memory = torch.zeros(vocab_size)
    extra_memory = torch.zeros(vocab_size, dtype=dtype).to(device)
    for i in tok_extra_info:
      extra_memory[i] = 1

    extra_memory = einsum("d_vocab, d_vocab d_model-> d_model", extra_memory, model.W_E) #this line works

    #add the extra_info embedded in latent space to hook_attn_out
    attn_result[:,:,head_num,:] = attn_result[:,:,head_num,:] + extra_memory * tweak_factor

    return attn_result

def memory_layer_encoding_hook(
    attn_result: Float[torch.Tensor, "num_tokens num_heads d_model"],
    hook: HookPoint, #name of layer where we inject memory
    #extra_info: str, #the string that we tokenize and then inject into memory
    extra_info: Float[torch.Tensor, "d_model"], #the string that we tokenize and then inject into memory
    model: transformer_lens.HookedTransformer, #the model from which we get the unembedding matrix from
    vocab_size: int, #size of model vocabulary
    tweak_factor: float,
    head_num: int, #The number of the head we want to edit
    dtype,
    #cache: transformer_lens.ActivationCache #this is the
) -> Float[torch.Tensor, "batch pos d_model"]:

    #logits, cache = model.run_with_cache(extra_info, remove_batch_dim=True)
    #encoded_memory = cache[utils.get_act_name("attn_out", layer_encode)][-1]
    #encoded_memory.to(device)

    attn_result[:,:,head_num,:] = attn_result[:,:,head_num,:] + extra_info * tweak_factor

    #return this edited hook_attn_out
    return attn_result

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
    """ This function is how you grab the name of a variable"""
    return [str(name) for name in namespace if namespace[name] is obj]
