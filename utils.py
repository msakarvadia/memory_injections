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